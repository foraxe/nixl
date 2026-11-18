// Experimental ABI-3 facade for vLLM row264 over four NIXL-backed rails.
// Each process must explicitly select this library and the same rail/QP layout.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void ck(cudaError_t rc, const char *where) {
  if (rc != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(rc));
}
template <class F> int guard(F f) {
  try {
    f();
    error.clear();
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
struct Api {
  void *lib;
  const char *(*last)();
  void *(*create)(uint32_t, uint16_t, const char *, const char *, uint32_t);
  int (*reg)(void *, int, const uint64_t *, const uint64_t *, uint32_t);
  int64_t (*metadata)(void *, char *, size_t);
  int (*connect)(void *, const char *, size_t, const uint64_t *,
                 const uint64_t *, uint32_t, uint32_t);
  int (*destroy)(void *);
  int (*consumed)(void *, uint64_t);
  void *(*group_create)(void **, uint32_t, uint32_t, uint32_t);
  int (*qualify)(void *, uint32_t);
  int (*pipeline)(void *, uint32_t);
  int (*weights)(void *, uint32_t, uint32_t, uint32_t, uint32_t);
  int (*fetch)(void *, uint64_t, uint32_t, size_t, size_t, uint32_t, uint32_t,
               uint64_t, uint64_t, uint64_t, uint64_t, int);
  int (*fetch_compacted)(void *, uint64_t, uint32_t, size_t, size_t, uint32_t,
                         uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, int,
                         uint64_t, uint64_t, uint64_t);
  int (*fetch_adaptive)(void *, uint64_t, uint32_t, size_t, size_t, uint32_t,
                        uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, int,
                        uint64_t, uint64_t, uint64_t, uint64_t);
  int (*group_destroy)(void *);
  template <class T> T sym(const char *name) {
    auto p = dlsym(lib, name);
    if (!p)
      throw std::runtime_error(std::string("missing bridge symbol: ") + name);
    return reinterpret_cast<T>(p);
  }
  Api() {
    const char *path = std::getenv("ENGRAM_ABI3_BASE_LIBRARY");
    if (!path || !*path)
      throw std::runtime_error("ENGRAM_ABI3_BASE_LIBRARY is required");
    lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!lib)
      throw std::runtime_error(dlerror());
    last = sym<decltype(last)>("engram_last_error");
    create = sym<decltype(create)>("engram_create_on_rail");
    reg = sym<decltype(reg)>("engram_register");
    metadata = sym<decltype(metadata)>("engram_metadata");
    connect = sym<decltype(connect)>("engram_connect");
    destroy = sym<decltype(destroy)>("engram_destroy");
    consumed = sym<decltype(consumed)>("engram_record_consumed");
    group_create = sym<decltype(group_create)>("engram_multi_group_create");
    qualify = sym<decltype(qualify)>("engram_multi_qualify_depth2");
    pipeline = sym<decltype(pipeline)>("engram_read_group_pipeline");
    weights = sym<decltype(weights)>("engram_multi_set_weights");
    fetch = sym<decltype(fetch)>("engram_read_group_fetch");
    fetch_compacted =
        sym<decltype(fetch_compacted)>("engram_read_group_fetch_compacted");
    fetch_adaptive = sym<decltype(fetch_adaptive)>("engram_read_group_fetch_adaptive");
    group_destroy = sym<decltype(group_destroy)>("engram_read_group_destroy");
  }
  void check(int rc, const char *where) {
    if (rc < 0)
      throw std::runtime_error(std::string(where) + ": " + last());
  }
};
Api &api() {
  static Api instance;
  return instance;
}
struct Context {
  uint32_t gpu = 0, qps = 0;
  std::array<void *, 4> child{};
  void *group = nullptr;
  uint64_t *stats = nullptr;
  uint64_t *dedup_table = nullptr;
  uint32_t *adaptive_flag = nullptr;
  int64_t *representatives = nullptr, *query_list = nullptr;
  uint64_t arena_address = 0, arena_bytes = 0;
  uint32_t dedup_capacity = 0, dedup_slots = 0;
  cudaEvent_t retired{};
  cudaStream_t last_stream{};
  bool dedup_enabled = false, adaptive_enabled = false, workspace = false, has_work = false;
  uint32_t active = 0;
  bool host = false, registered = false, connected = false, failed = false;
  bool setup_failed = false;
  std::mutex mutex;
};
// Failed partial construction must not free resources whose retirement is
// uncertain.
std::mutex quarantine_mutex;
std::vector<Context *> quarantine;
void keep(Context *c) {
  std::lock_guard lock(quarantine_mutex);
  quarantine.push_back(c);
}
uint32_t qps() {
  const char *s = std::getenv("ENGRAM_ABI3_RAIL_QPS");
  if (!s || !*s)
    throw std::runtime_error("ENGRAM_ABI3_RAIL_QPS is required");
  char *end = nullptr;
  auto v = std::strtoul(s, &end, 10);
  if (*end || !v || v > 32)
    throw std::runtime_error("invalid QP count");
  return uint32_t(v);
}
std::array<std::string, 4> rails() {
  const char *s = std::getenv("ENGRAM_ABI3_RAILS");
  if (!s)
    throw std::runtime_error("ENGRAM_ABI3_RAILS is required");
  std::string a(s);
  std::array<std::string, 4> out;
  size_t start = 0;
  for (int i = 0; i < 4; ++i) {
    auto end = a.find(',', start);
    if (i < 3 && end == std::string::npos)
      throw std::runtime_error("four rails required");
    if (i == 3 && end != std::string::npos)
      throw std::runtime_error("four rails required");
    out[i] = a.substr(start, end == std::string::npos ? end : end - start);
    if (out[i].empty())
      throw std::runtime_error("empty rail");
    for (int j = 0; j < i; ++j)
      if (out[i] == out[j])
        throw std::runtime_error("duplicate rail");
    start = end == std::string::npos ? a.size() : end + 1;
  }
  return out;
}
__global__ void aggregate(const uint64_t *lanes, uint32_t n, uint64_t *out) {
  if (blockIdx.x || threadIdx.x)
    return;
  for (int j = 0; j < 8; ++j)
    out[j] = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const auto *p = lanes + i * 8;
    if (p[0])
      out[0] = p[0];
    out[1] += p[1];
    out[2] += p[2];
    out[3] = max(out[3], p[3]);
    out[4] += p[4];
    out[5] += p[5];
    out[6] += p[6];
    if (!i || p[7] < out[7])
      out[7] = p[7];
  }
}
// Test-only stratified sample: 512 evenly spaced adjacent pairs.
__global__ void sample_duplicate_rate(const int64_t *ids, uint32_t n, uint64_t lo,
                                      uint64_t hi, uint32_t *flag) {
  __shared__ unsigned long long keys[2048];
  __shared__ uint32_t valid, duplicate;
  for (uint32_t i = threadIdx.x; i < 2048; i += blockDim.x) keys[i] = 0;
  if (threadIdx.x == 0) { valid = 0; duplicate = 0; }
  __syncthreads();
  for (uint32_t sample = threadIdx.x; sample < 1024; sample += blockDim.x) {
    const uint64_t pairs = (uint64_t(n) + 1) / 2;
    const uint64_t q = ((sample / 2) * pairs / 512) * 2 + (sample & 1);
    if (q >= n) continue;
    const int64_t row = ids[q];
    if (row < 0 || uint64_t(row) < lo || uint64_t(row) >= hi) continue;
    atomicAdd(&valid, 1u);
    const unsigned long long key = uint64_t(row) + 1;
    uint32_t slot = uint32_t(uint64_t(row) * 2654435761ull) & 2047;
    auto seen = atomicCAS(&keys[slot], 0ull, key);
    while (seen && seen != key) { slot = (slot + 1) & 2047; seen = atomicCAS(&keys[slot], 0ull, key); }
    if (seen == key) atomicAdd(&duplicate, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0) *flag = valid && uint64_t(duplicate) * 10 >= valid;
}
// Each entry combines a 32-bit row key and its unique representative query.
__global__ void compact_ids(const int64_t *ids, uint32_t n, uint64_t lo,
                            uint64_t hi, unsigned long long *table,
                            uint32_t slots, int64_t *representatives,
                            int64_t *query_list, const uint32_t *adaptive_flag) {
  const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n)
    return;
  const int64_t row = ids[q];
  if (adaptive_flag && !*adaptive_flag) return;
  if (row < 0 || uint64_t(row) < lo || uint64_t(row) >= hi) {
    representatives[q] = -1;
    return;
  }
  uint32_t slot = uint32_t(uint64_t(row) * 2654435761ull) & (slots - 1);
  const unsigned long long key = ((uint64_t(row) + 1) << 32) | uint64_t(q);
  auto seen = atomicCAS(&table[slot], 0ull, key);
  while (seen && (seen >> 32) != (uint64_t(row) + 1)) {
    slot = (slot + 1) & (slots - 1);
    seen = atomicCAS(&table[slot], 0ull, key);
  }
  if (!seen) {
    representatives[q] = q;
    const auto dst = atomicAdd(&table[slots], 1ull);
    query_list[dst] = q;
  } else
    representatives[q] = int64_t(uint32_t(seen));
}
// Representatives are never rewritten, so copies to duplicate slots do not
// race.
__global__ void restore_rows(uint8_t *payload, const int64_t *reps, uint32_t n,
                             const uint64_t *status, const uint32_t *adaptive_flag) {
  if (status[0] || (adaptive_flag && !*adaptive_flag)) return;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t warp = threadIdx.x / 32;
  for (uint32_t q = blockIdx.x * 8 + warp; q < n; q += gridDim.x * 8) {
    const int64_t rep = reps[q];
    if (rep < 0 || uint64_t(rep) == q) continue;
    auto *dst = payload + size_t(q) * 264;
    const auto *src = payload + size_t(rep) * 264;
    if ((reinterpret_cast<uintptr_t>(payload) & 7) == 0) {
      auto *d = reinterpret_cast<uint64_t *>(dst);
      const auto *source = reinterpret_cast<const uint64_t *>(src);
      d[lane] = source[lane];
      if (lane == 0) d[32] = source[32];
    } else {
      for (uint32_t byte = lane; byte < 264; byte += 32) dst[byte] = src[byte];
    }
  }
}

void recordEvent(Context &c, cudaStream_t stream) {
  cudaStreamCaptureStatus capture;
  ck(cudaStreamIsCapturing(stream, &capture), "query workspace capture");
  ck(cudaEventRecordWithFlags(c.retired, stream,
                              capture == cudaStreamCaptureStatusNone
                                  ? cudaEventRecordDefault
                                  : cudaEventRecordExternal),
     "record workspace retirement");
}
constexpr char magic[] = "NIXL4R1";
void append32(std::vector<char> &v, uint32_t n) {
  for (int i = 0; i < 4; ++i)
    v.push_back(char((n >> (8 * i)) & 255));
}
uint32_t read32(const char *data, size_t len, size_t &off) {
  if (off > len || len - off < 4)
    throw std::runtime_error("truncated rail metadata");
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i)
    v |= uint32_t(uint8_t(data[off++])) << (8 * i);
  return v;
}
} // namespace
extern "C" const char *engram_last_error() { return error.c_str(); }
extern "C" uint32_t engram_abi_version() { return 3; }
extern "C" void *engram_create(uint32_t gpu, uint16_t port, const char *name) {
  Context *c = nullptr;
  int rc = guard([&] {
    if (!name || port > 65532)
      throw std::runtime_error("invalid context args");
    auto devs = rails();
    auto &b = api();
    c = new Context;
    c->gpu = gpu;
    c->qps = qps();
    if (const char *flag = std::getenv("ENGRAM_ABI3_DEDUP")) {
      if (std::strcmp(flag, "0") && std::strcmp(flag, "1") && std::strcmp(flag, "auto"))
        throw std::runtime_error("ENGRAM_ABI3_DEDUP must be 0, 1, or auto");
      c->workspace = std::strcmp(flag, "0") != 0;
      c->dedup_enabled = c->workspace;
      c->adaptive_enabled = std::strcmp(flag, "auto") == 0;
    }
    for (uint32_t i = 0; i < 4; ++i) {
      auto n = std::string(name) + "-rail" + std::to_string(i);
      c->child[i] = b.create(gpu, port + i, n.c_str(), devs[i].c_str(), c->qps);
      if (!c->child[i])
        throw std::runtime_error(std::string("create rail: ") + b.last());
    }
  });
  if (rc < 0 && c) {
    bool closed = true;
    try {
      auto &b = api();
      for (auto h : c->child)
        if (h && b.destroy(h) < 0)
          closed = false;
    } catch (...) {
      closed = false;
    }
    if (closed)
      delete c;
    else
      keep(c);
    c = nullptr;
  }
  return c;
}
extern "C" int engram_register(void *p, int host, const uint64_t *addresses,
                               const uint64_t *lengths, uint32_t n) {
  return guard([&] {
    auto *c = static_cast<Context *>(p);
    if (!c || c->registered || !n)
      throw std::runtime_error("invalid registration");
    std::lock_guard lock(c->mutex);
    try {
      for (auto h : c->child)
        api().check(api().reg(h, host, addresses, lengths, n), "register rail");
      c->host = host != 0;
      if (!c->host) {
        if (n != 1)
          throw std::runtime_error("dedup reader requires one arena");
        c->arena_address = addresses[0];
        c->arena_bytes = lengths[0];
      }
      c->registered = true;
    } catch (...) {
      c->failed = true;
      throw;
    }
  });
}
extern "C" int64_t engram_metadata(void *p, char *out, size_t capacity) {
  int64_t size = -1;
  int rc = guard([&] {
    auto *c = static_cast<Context *>(p);
    if (!c || !c->registered)
      throw std::runtime_error("not registered");
    std::lock_guard lock(c->mutex);
    std::vector<char> blob(magic, magic + sizeof(magic) - 1);
    append32(blob, c->qps);
    for (auto h : c->child) {
      auto n = api().metadata(h, nullptr, 0);
      if (n < 0 || uint64_t(n) > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("rail metadata size");
      append32(blob, uint32_t(n));
      auto pos = blob.size();
      blob.resize(pos + size_t(n));
      if (api().metadata(h, blob.data() + pos, size_t(n)) != n)
        throw std::runtime_error("rail metadata export");
    }
    size = blob.size();
    if (out) {
      if (capacity < blob.size())
        throw std::runtime_error("metadata buffer too small");
      std::memcpy(out, blob.data(), blob.size());
    }
  });
  return rc < 0 ? -1 : size;
}
extern "C" int engram_connect(void *p, const char *data, size_t len,
                              const uint64_t *addresses,
                              const uint64_t *lengths, uint32_t n,
                              uint32_t remote_gpu) {
  return guard([&] {
    auto *c = static_cast<Context *>(p);
    if (!c || !c->registered || c->host || c->connected || c->setup_failed || !data ||
        len < sizeof(magic) - 1 || std::memcmp(data, magic, sizeof(magic) - 1))
      throw std::runtime_error("invalid multi-rail metadata");
    std::lock_guard lock(c->mutex);
    size_t off = sizeof(magic) - 1;
    if (read32(data, len, off) != c->qps)
      throw std::runtime_error("peer QP count mismatch");
    try {
      for (auto h : c->child) {
        auto sz = read32(data, len, off);
        if (off > len || sz > len - off)
          throw std::runtime_error("truncated rail blob");
        api().check(
            api().connect(h, data + off, sz, addresses, lengths, n, remote_gpu),
            "connect rail");
        off += sz;
      }
      if (off != len)
        throw std::runtime_error("trailing rail metadata");
    } catch (...) {
      // No read group or data operation exists yet. Make the context
      // destroy-only; each child still performs its own teardown checks.
      c->setup_failed = true;
      throw;
    }
    try {
      ck(cudaSetDevice(c->gpu), "select GPU");
      ck(cudaMalloc(reinterpret_cast<void **>(&c->stats),
                    4 * c->qps * 8 * sizeof(uint64_t)),
         "allocate lane stats");
      c->group = api().group_create(c->child.data(), 4, c->qps, 1024);
      if (!c->group)
        throw std::runtime_error(std::string("create rail group: ") +
                                 api().last());
      api().check(api().qualify(c->group, 512),
                  "qualify depth2 (MCST hardware unsupported)");
      api().check(api().pipeline(c->group, 2), "set depth");
      const char *w = std::getenv("ENGRAM_ABI3_WEIGHTS");
      if (w && *w) {
        unsigned a, b, d, e;
        char tail;
        if (std::sscanf(w, "%u,%u,%u,%u%c", &a, &b, &d, &e, &tail) != 4)
          throw std::runtime_error("invalid rail weights");
        api().check(api().weights(c->group, a, b, d, e), "set rail weights");
      }
      if (c->workspace) {
        for (uint32_t i = 0; i < n; ++i)
          if (lengths[i] / 264 > uint64_t{0xffffffff})
            throw std::runtime_error(
                "dedup supports at most 32-bit row indices");
        const uint64_t capacity = c->arena_bytes / 264;
        if (!capacity || capacity > (uint64_t{1} << 29))
          throw std::runtime_error("invalid dedup workspace capacity");
        c->dedup_capacity = uint32_t(capacity);
        uint32_t slots = 2;
        while (uint64_t(slots) < capacity * 2)
          slots *= 2;
        c->dedup_slots = slots;
        ck(cudaMalloc(reinterpret_cast<void **>(&c->dedup_table),
                      size_t(slots + 1) * sizeof(uint64_t)),
           "allocate dedup table");
        ck(cudaMalloc(reinterpret_cast<void **>(&c->representatives),
                      size_t(capacity) * sizeof(int64_t)),
           "allocate representatives");
        ck(cudaMalloc(reinterpret_cast<void **>(&c->query_list),
                      size_t(capacity) * sizeof(int64_t)),
           "allocate compacted queries");
        ck(cudaMalloc(reinterpret_cast<void **>(&c->adaptive_flag), sizeof(uint32_t)),
           "allocate adaptive flag");
        ck(cudaEventCreateWithFlags(&c->retired, cudaEventDisableTiming),
           "create workspace retirement event");
      }
      c->connected = true;
    } catch (...) {
      c->failed = true;
      throw;
    }
  });
}
extern "C" int engram_fetch_row264(void *p, uint64_t ids, uint32_t n,
                                   size_t payload, size_t, size_t dump,
                                   uint32_t wi, uint32_t si, uint32_t window,
                                   uint64_t status, uint64_t stream,
                                   uint64_t lo, uint64_t hi) {
  return guard([&] {
    auto *c = static_cast<Context *>(p);
    if (!c || !c->connected || c->failed || !ids || !status || !n || wi != si ||
        lo > hi)
      throw std::runtime_error("invalid multi-rail fetch");
    std::lock_guard lock(c->mutex);
    try {
      auto s = reinterpret_cast<cudaStream_t>(stream);
      if (c->workspace && c->has_work && c->last_stream != s) {
        cudaStreamCaptureStatus capture;
        ck(cudaStreamIsCapturing(s, &capture), "query workspace capture");
        if (capture != cudaStreamCaptureStatusNone)
          throw std::runtime_error("warm workspace on capture stream");
        ck(cudaStreamWaitEvent(s, c->retired, 0), "order workspace reuse");
      }
      const bool dedup = c->dedup_enabled && n >= 1024;
      if (dedup) {
        if (n > c->dedup_capacity || hi > uint64_t{0xffffffff})
          throw std::runtime_error("dedup index or capacity unsupported");
        ck(cudaMemsetAsync(c->dedup_table, 0,
                           size_t(c->dedup_slots + 1) * sizeof(uint64_t), s),
           "clear dedup workspace");
        if (c->adaptive_enabled) {
          sample_duplicate_rate<<<1, 256, 0, s>>>(reinterpret_cast<const int64_t *>(ids), n, lo, hi, c->adaptive_flag);
          ck(cudaGetLastError(), "sample duplicate rate");
        }
        compact_ids<<<(n + 127) / 128, 128, 0, s>>>(
            reinterpret_cast<const int64_t *>(ids), n, lo, hi,
            reinterpret_cast<unsigned long long *>(c->dedup_table),
            c->dedup_slots, c->representatives, c->query_list,
            c->adaptive_enabled ? c->adaptive_flag : nullptr);
        ck(cudaGetLastError(), "compact IDs");
        if (c->adaptive_enabled) {
          api().check(api().fetch_adaptive(c->group, ids, n, payload, dump, wi, window,
              reinterpret_cast<uint64_t>(c->stats), stream, lo, hi, 0,
              reinterpret_cast<uint64_t>(c->representatives), reinterpret_cast<uint64_t>(c->query_list),
              reinterpret_cast<uint64_t>(c->dedup_table + c->dedup_slots), reinterpret_cast<uint64_t>(c->adaptive_flag)),
              "adaptive multi-rail fetch");
        } else {
          api().check(api().fetch_compacted(c->group, ids, n, payload, dump, wi, window,
              reinterpret_cast<uint64_t>(c->stats), stream, lo, hi, 0,
              reinterpret_cast<uint64_t>(c->representatives), reinterpret_cast<uint64_t>(c->query_list),
              reinterpret_cast<uint64_t>(c->dedup_table + c->dedup_slots)), "compacted multi-rail fetch");
        }
      } else {
        api().check(api().fetch(c->group, ids, n, payload, dump, wi, window,
                                reinterpret_cast<uint64_t>(c->stats), stream,
                                lo, hi, 0),
                    "multi-rail fetch");
      }
      uint32_t cap = std::min(n, n <= 1024 ? c->qps : 4 * c->qps),
               chunk = (n + cap - 1) / cap;
      c->active = (n + chunk - 1) / chunk;
      aggregate<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
          c->stats, c->active, reinterpret_cast<uint64_t *>(status));
      ck(cudaGetLastError(), "aggregate lane status");
      if (dedup) {
        restore_rows<<<std::min(256u, (n + 7) / 8), 256, 0, s>>>(
            reinterpret_cast<uint8_t *>(c->arena_address + payload),
            c->representatives, n, reinterpret_cast<const uint64_t *>(status),
            c->adaptive_enabled ? c->adaptive_flag : nullptr);
        ck(cudaGetLastError(), "restore duplicate rows");
      }
      if (c->workspace) {
        c->last_stream = s;
        c->has_work = true;
        recordEvent(*c, s);
      }
    } catch (...) {
      c->failed = true;
      throw;
    }
  });
}
extern "C" int engram_fetch(void *, uint64_t, uint32_t, size_t, size_t, size_t,
                            uint32_t, uint32_t, uint32_t, uint64_t, uint64_t,
                            uint64_t, uint64_t) {
  return guard([] {
    throw std::runtime_error("multi-rail fixture supports row264 only");
  });
}
extern "C" int engram_record_consumed(void *p, uint64_t stream) {
  return guard([&] {
    auto *c = static_cast<Context *>(p);
    if (!c || !c->connected || c->failed || !c->active)
      throw std::runtime_error("invalid consumer retirement");
    std::lock_guard lock(c->mutex);
    try {
      for (uint32_t i = 0; i < 4 && (i * c->qps) < c->active; ++i)
        api().check(api().consumed(c->child[i], stream),
                    "record rail consumer");
      if (c->workspace) {
        auto s = reinterpret_cast<cudaStream_t>(stream);
        if (!c->has_work || c->last_stream != s)
          throw std::runtime_error("invalid workspace consumer stream");
        recordEvent(*c, s);
      }
    } catch (...) {
      c->failed = true;
      throw;
    }
  });
}
extern "C" int engram_destroy(void *p) {
  if (!p)
    return 0;
  return guard([&] {
    auto *c = static_cast<Context *>(p);
    std::unique_lock lock(c->mutex);
    if (c->failed)
      throw std::runtime_error("failed multi-rail context retained");
    if (c->group) {
      api().check(api().group_destroy(c->group), "destroy rail group");
      c->group = nullptr;
    }
    for (auto &h : c->child)
      if (h) {
        api().check(api().destroy(h), "destroy rail");
        h = nullptr;
      }
    if (c->stats) {
      ck(cudaFree(c->stats), "free lane stats");
      c->stats = nullptr;
    }
    if (c->retired) {
      ck(cudaEventDestroy(c->retired), "destroy workspace event");
      c->retired = nullptr;
    }
    if (c->adaptive_flag) {
      ck(cudaFree(c->adaptive_flag), "free adaptive flag");
      c->adaptive_flag = nullptr;
    }
    if (c->query_list) {
      ck(cudaFree(c->query_list), "free compacted queries");
      c->query_list = nullptr;
    }
    if (c->representatives) {
      ck(cudaFree(c->representatives), "free representatives");
      c->representatives = nullptr;
    }
    if (c->dedup_table) {
      ck(cudaFree(c->dedup_table), "free dedup table");
      c->dedup_table = nullptr;
    }
    lock.unlock();
    delete c;
  });
}

