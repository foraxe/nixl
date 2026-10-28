// SPDX-License-Identifier: Apache-2.0
// Private shared-Engram experiment. No PyTorch C++ ABI dependency.
#include <cuda_runtime.h>
#include <gpu/nixl_device.cuh>
#include <nixl.h>
#include "../gpu_remote_kv/experimental_get.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

namespace {
thread_local std::string last_error;
void check(cudaError_t rc, const char *where) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(rc));
}
void require(nixl_status_t rc, const char *where) {
    if (rc != NIXL_SUCCESS) throw std::runtime_error(std::string(where) + ": " + std::to_string(rc));
}
template<class F> int guarded(F fn) {
    try { fn(); last_error.clear(); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1; }
}

struct Context {
    uint32_t gpu;
    std::unique_ptr<nixlAgent> agent;
    nixlBackendH *backend = nullptr;
    nixl_opt_args_t hint;
    std::unique_ptr<nixl_reg_dlist_t> registration;
    nixlMemViewH local = nullptr, remote = nullptr;
    size_t arena_bytes = 0;
    uint64_t arena_address = 0;
    uint64_t *trace = nullptr;
    uint64_t *cost_trace = nullptr;
    cudaEvent_t retired{};
    cudaStream_t stream{};
    uint64_t *stats = nullptr;
    bool has_work = false, poisoned = false;
};

cudaError_t record_retirement(Context &context, cudaStream_t stream) {
    cudaStreamCaptureStatus capture;
    const auto rc = cudaStreamIsCapturing(stream, &capture);
    if (rc != cudaSuccess) return rc;
    return cudaEventRecordWithFlags(context.retired, stream,
        capture == cudaStreamCaptureStatusNone ? cudaEventRecordDefault : cudaEventRecordExternal);
}

__device__ uint64_t timer_ns() {
    uint64_t n;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(n));
    return n;
}
__device__ bool wait_ticket(doca_gpu_dev_verbs_qp *qp, uint64_t ticket) {
    const uint64_t until = timer_ns() + 8000000000ull;
    int rc;
    do {
        rc = nixl::gpu::impl::gpunetio::compat::pollOne(qp, ticket);
        if (rc == 0) return true;
        if (rc != EBUSY) return false;
    } while (timer_ns() < until);
    return false;
}
__device__ bool scalar_get(nixlMemViewH local, nixlMemViewH remote,
                           size_t dst, size_t index, size_t src, size_t bytes,
                           size_t dump, uint64_t *stats) {
    nixl::gpu::xferStatusH status{};
    const auto start = timer_ns();
    auto rc = nixl::gpu::experimental::get({local, 0, dst}, {remote, index, src},
                                          bytes, {local, 0, dump}, &status);
    const auto posted = timer_ns();
    stats[1] += posted - start;
    if (rc != NIXL_IN_PROG) return false;
    const uint64_t until = posted + 8000000000ull;
    do {
        rc = nixl::gpu::getXferStatus<nixl::gpu::level_t::THREAD>(status);
        if (rc == NIXL_SUCCESS) { stats[2] += timer_ns() - posted; ++stats[4]; return true; }
        if (rc != NIXL_IN_PROG) return false;
    } while (timer_ns() < until);
    return false;
}

__global__ void fetch_rows_scalar(const int64_t *ids, uint32_t count, size_t payload_offset,
                                  size_t scale_offset, size_t dump_offset,
                                  uint32_t weight_index, uint32_t scale_index,
                                  nixlMemViewH local, nixlMemViewH remote,
                                  uint64_t row_lo, uint64_t row_hi, uint64_t *stats) {
    namespace gp = nixl::gpu::impl::gpunetio;
    for (int i = 0; i < 8; ++i) stats[i] = 0;
    const auto begin = timer_ns();
    auto *lv = static_cast<const gp::GpunetioDeviceView *>(
        nixl::gpu::impl::viewWrapper(local)->backend_memview);
    auto *rv = static_cast<const gp::GpunetioDeviceView *>(
        nixl::gpu::impl::viewWrapper(remote)->backend_memview);
    if (!lv || !rv || lv->count != 1 || weight_index >= rv->count || scale_index >= rv->count ||
        lv->context != rv->context || !lv->context || !lv->context->lane) {
        stats[0] = 1; return;
    }
    const auto &weight = rv->elems[weight_index];
    const auto &scale = rv->elems[scale_index];
    for (uint32_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || uint64_t(ids[i]) >= weight.length / 256 ||
            uint64_t(ids[i]) >= scale.length / 8) { stats[0] = 2; return; }
    }
    for (uint32_t i = 0; i < count; ++i) {
        const auto row = uint64_t(ids[i]);
        if (row < row_lo || row >= row_hi) continue;
        if (!scalar_get(local, remote, payload_offset + size_t(i) * 256, weight_index,
                        row * 256, 256, dump_offset, stats) ||
            !scalar_get(local, remote, scale_offset + size_t(i) * 8, scale_index,
                        row * 8, 8, dump_offset, stats)) {
            stats[0] = 3; return;
        }
    }
    auto *qp = lv->context->lane->qp;
    stats[5] = qp->need_dump ? stats[4] : 0;
    stats[6] = stats[4];
    stats[7] = qp->sq_wqe_num;
    stats[3] = timer_ns() - begin;
}
// One resident CTA prepares disjoint WQEs and inspects disjoint CQEs. Only the
// leader reserves/publishes the FIFO range; there is no cross-CTA busy wait.
template<bool PackedRow = false, bool CompactPending = false, uint32_t CompletionDepth = 1, bool TerminalWait = false>
__device__ __forceinline__ void fetch_rows_body(const int64_t *ids, uint32_t count,
                                      size_t payload_offset, size_t scale_offset,
                                      size_t dump_offset, uint32_t weight_index,
                                      uint32_t scale_index, nixlMemViewH local,
                                      nixlMemViewH remote, uint32_t requested_window,
                                      uint64_t row_lo, uint64_t row_hi,
                                      uint64_t *stats, uint64_t *trace, uint64_t *cost, const int64_t *representatives = nullptr,
                                      uint32_t query_base = 0, uint32_t total_queries = 0, const int64_t *query_list = nullptr) {
    namespace gp = nixl::gpu::impl::gpunetio;
    __shared__ uint32_t error, window;
    __shared__ uint32_t active_count, selected[1024], pending[1024], warp_counts[4];
    __shared__ uint64_t begin, base, post_start, posted, generation, poll_start, builder_start, publish_start;
    __shared__ uint64_t queued_base[2], queued_at[2];
    __shared__ uint32_t queued_ops[2], queued_id[2];
    const auto tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < 8; ++i) stats[i] = 0;
        error = 0;
        begin = timer_ns();
        if (cost) for (int i = 0; i < 4; ++i) cost[i] = 0;
        if (trace) {
            trace[0] = begin; trace[1] = trace[2] = trace[3] = 0;
            uint32_t sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
            trace[4] = sm;
        }
    }
    __syncthreads();
    auto *lv = static_cast<const gp::GpunetioDeviceView *>(
        nixl::gpu::impl::viewWrapper(local)->backend_memview);
    auto *rv = static_cast<const gp::GpunetioDeviceView *>(
        nixl::gpu::impl::viewWrapper(remote)->backend_memview);
    if (!lv || !rv || lv->count != 1 || weight_index >= rv->count || scale_index >= rv->count ||
        lv->context != rv->context || !lv->context || !lv->context->lane) {
        if (tid == 0) stats[0] = 1;
        return;
    }
    const auto &dst = lv->elems[0];
    const auto &weight = rv->elems[weight_index];
    const auto &scale = rv->elems[scale_index];
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        const auto q = query_list ? query_list[i] : int64_t(i);
        if (q < 0 || uint64_t(q) >= (total_queries ? total_queries : count)) { atomicExch(&error, 2u); continue; }
        if (ids[q] < 0 || uint64_t(ids[q]) >= weight.length / (PackedRow ? 264 : 256) ||
            (!PackedRow && uint64_t(ids[q]) >= scale.length / 8)) atomicExch(&error, 2u);
        if (representatives && uint64_t(ids[q]) >= row_lo && uint64_t(ids[q]) < row_hi) {
            const auto r = representatives[q];
            const auto total = total_queries ? total_queries : count;
            if (r < 0 || uint64_t(r) >= total || (ids - query_base)[r] != ids[q] ||
                (representatives - query_base)[r] != r) atomicExch(&error, 2u);
        }
    }
    __syncthreads();
    if (error) { if (tid == 0) stats[0] = error; return; }
    auto *lane = lv->context->lane;
    auto phase = gp::deviceAtomic(lane->phase);
    auto *qp = lane->qp;
    if (tid == 0) {
        uint32_t observed = gp::gpunetio_lane_phase_idle;
        if (!phase.compare_exchange_strong(observed, gp::gpunetio_lane_phase_posting,
                                           cuda::memory_order_acq_rel, cuda::memory_order_acquire)) {
            error = 4;
        } else if (gp::deviceAtomic(lane->failed).load(cuda::memory_order_relaxed) || !qp ||
                   qp->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
            error = 5;
            gp::deviceAtomic(lane->failed).store(1, cuda::memory_order_relaxed);
            phase.store(gp::gpunetio_lane_phase_failed, cuda::memory_order_release);
        } else {
            const uint32_t capacity = min(qp->sq_wqe_num, doca_gpu_dev_verbs_qp_get_cq_sq(qp)->cqe_num);
            window = capacity < 4 ? 0 : min(requested_window,
                min(1024u, (capacity - 1) / (PackedRow ? 1 : 2)));
            stats[7] = qp->sq_wqe_num;
            if (!window || (CompactPending && window < 4) ||
                (CompletionDepth > 1 && (qp->need_dump || CompletionDepth * window * (PackedRow ? 1 : 2) >= capacity)) || gp::deviceAtomic(lane->generation).load(cuda::memory_order_relaxed) == ~uint64_t{0}) {
                error = 6;
                phase.store(gp::gpunetio_lane_phase_idle, cuda::memory_order_release);
            } else {
                generation = gp::deviceAtomic(lane->generation).fetch_add(1, cuda::memory_order_relaxed) + 1;
            }
        }
    }
    __syncthreads();
    if (error) { if (tid == 0) stats[0] = error; return; }
    auto *cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
    const bool full_range = query_list == nullptr && representatives == nullptr && row_lo == 0 && row_hi >= weight.length / (PackedRow ? 264 : 256);
    auto retire = [&](uint64_t done_base, uint32_t done_ops, uint64_t posted_at, uint32_t batch_id) {
        if (tid == 0) poll_start = timer_ns();
        __syncthreads();
        if constexpr (TerminalWait) {
            if (tid == 0) {
                int rc;
                do {
                    rc = gp::compat::nixl_gpunetio_dev_priv_poll_one_cq_at<>(cq, done_base + done_ops - 1);
                    if (rc == 0) break;
                    if (rc != EBUSY || timer_ns() - posted_at >= 8000000000ull) { error = 7; break; }
                } while (true);
            }
            __syncthreads();
        }
        for (uint32_t j = tid; j < done_ops; j += blockDim.x) {
            int rc;
            do {
                rc = gp::compat::nixl_gpunetio_dev_priv_poll_one_cq_at<>(cq, done_base + j);
                if (rc == 0) break;
                if (TerminalWait || rc != EBUSY || timer_ns() - posted_at >= 8000000000ull) {
                    atomicExch(&error, 7u);
                    break;
                }
            } while (true);
        }
        __syncthreads();
        if (tid == 0) {
            if (!error) {
                doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
                doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                    &cq->cqe_ci, done_base + done_ops);
                stats[4] += done_ops;
            }
            stats[2] += timer_ns() - poll_start;
            if (trace && batch_id < 128) trace[9 + batch_id * 4] = timer_ns();
            if (!error && qp->need_dump) {
                const auto dump_start = timer_ns();
                const auto idx = doca_gpu_dev_verbs_reserve_wq_slots<>(qp, 1);
                doca_gpu_dev_verbs_wqe_prepare_dump(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, idx), idx,
                    DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, dst.base + dump_offset, dst.key, 1);
                doca_gpu_dev_verbs_mark_wqes_ready<>(qp, idx, idx);
                doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(qp, idx + 1);
                const auto dump_posted = timer_ns();
                ++stats[5]; ++stats[6];
                stats[1] += dump_posted - dump_start;
                if (!wait_ticket(qp, idx)) error = 7;
                stats[2] += timer_ns() - dump_posted;
            }
        }
        __syncthreads();
    };
    uint32_t head = 0, tail = 0, inflight = 0;
    uint32_t pending_cursor = (tid / 32) * 32, pending_lane = 0;
    for (uint32_t start = 0; start < count;) {
        const uint32_t n = min(window, count - start);
        if (tid == 0) {
            post_start = timer_ns();
            active_count = 0;
        }
        __syncthreads();
        if (full_range) {
            for (uint32_t j = tid; j < n; j += blockDim.x) selected[j] = start + j;
            if (tid == 0) active_count = n;
        } else if constexpr (CompactPending) {
            // Each warp carries selected requests across tiles without CTA barriers.
            const auto lane_id = tid & 31;
            const auto warp_id = tid / 32;
            const auto quota = window / 4;
            uint32_t num_pending = 0;
            while (pending_cursor + pending_lane < count && num_pending < quota) {
                const auto scan = min(32u - pending_lane,
                    min(count - pending_cursor - pending_lane, quota - num_pending));
                const auto candidate = pending_cursor + pending_lane + lane_id;
                const auto query = query_list ? (lane_id < scan ? uint32_t(query_list[candidate]) : 0u) : candidate;
                const bool owned = lane_id < scan && uint64_t(ids[query]) >= row_lo &&
                                   uint64_t(ids[query]) < row_hi &&
                                   (!representatives || representatives[query] == int64_t(query_base + query));
                const auto bits = __ballot_sync(0xffffffff, owned);
                if (owned) pending[warp_id * quota + num_pending +
                    __popc(bits & ((1u << lane_id) - 1))] = query;
                __syncwarp();
                num_pending += __popc(bits);
                pending_lane += scan;
                if (pending_lane == 32) {
                    pending_cursor += blockDim.x;
                    pending_lane = 0;
                }
            }
            if (lane_id == 0) warp_counts[warp_id] = num_pending;
            __syncthreads();
            uint32_t prefix = 0;
            for (uint32_t w = 0; w < warp_id; ++w) prefix += warp_counts[w];
            for (uint32_t j = lane_id; j < num_pending; j += 32)
                selected[prefix + j] = pending[warp_id * quota + j];
            if (tid == 0) active_count = warp_counts[0] + warp_counts[1] +
                                        warp_counts[2] + warp_counts[3];
        } else {
            const auto lane_id = tid & 31;
            for (uint32_t offset = 0; offset < n; offset += blockDim.x) {
                const auto j = offset + tid;
                const auto query = query_list ? (j < n ? uint32_t(query_list[start+j]) : 0u) : start+j;
                const bool owned = j < n && uint64_t(ids[query]) >= row_lo &&
                                   uint64_t(ids[query]) < row_hi &&
                                   (!representatives || representatives[query] == int64_t(query_base+query));
                const auto bits = __ballot_sync(0xffffffff, owned);
                if (bits) {
                    uint32_t slot = lane_id == 0 ? atomicAdd(&active_count, __popc(bits)) : 0;
                    slot = __shfl_sync(0xffffffff, slot, 0);
                    if (owned) selected[slot + __popc(bits & ((1u << lane_id) - 1))] = query;
                }
            }
        }
        __syncthreads();
        if constexpr (CompactPending) { if (full_range) start += n; }
        else start += n;
        if (!active_count) {
            if constexpr (CompactPending) { if (!full_range) break; }
            continue;
        }
        const uint32_t operations = (PackedRow ? 1 : 2) * active_count;
        if (tid == 0) {
            uint64_t at = 0;
            if (cost) { at = timer_ns(); cost[0] += at - post_start; }
            base = doca_gpu_dev_verbs_reserve_wq_slots<>(qp, operations);
            if (cost) { builder_start = timer_ns(); cost[1] += builder_start - at; }
        }
        __syncthreads();
        for (uint32_t j = tid; j < operations; j += blockDim.x) {
            const auto query = selected[PackedRow ? j : j / 2];
            const auto row = uint64_t(ids[query]);
            const bool payload = PackedRow || (j & 1) == 0;
            const auto &src = payload ? weight : scale;
            const uint32_t bytes = PackedRow ? 264 : (payload ? 256 : 8);
            const auto local_offset = (payload ? payload_offset : scale_offset) + size_t(query) * bytes;
            const auto idx = base + j;
            doca_gpu_dev_verbs_wqe_prepare_read(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, idx), idx,
                DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, src.base + row * bytes, src.key,
                dst.base + local_offset, dst.key, bytes);
        }
        __syncthreads();
        if (tid == 0) {
            if (cost) { publish_start = timer_ns(); cost[2] += publish_start - builder_start; }
            doca_gpu_dev_verbs_mark_wqes_ready<>(qp, base, base + operations - 1);
            doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(qp, base + operations);
            ++stats[6];
            posted = timer_ns();
            if (cost) cost[3] += posted - publish_start;
            if (trace) {
                if (stats[6] == 1) trace[1] = posted;
                trace[2] = posted;
                const auto batch = stats[6] - 1;
                if (batch < 128) { trace[8 + batch * 4] = posted;
                    trace[10 + batch * 4] = operations; trace[11 + batch * 4] = post_start; }
            }
            stats[1] += posted - post_start;
        }
        if constexpr (CompletionDepth == 1) {
            __syncthreads();
            retire(base, operations, posted, uint32_t(stats[6] - 1));
        } else {
            if (tid == 0) {
                queued_base[tail] = base; queued_ops[tail] = operations;
                queued_at[tail] = posted; queued_id[tail] = uint32_t(stats[6] - 1);
            }
            __syncthreads();
            tail = (tail + 1) % CompletionDepth; ++inflight;
            if (inflight == CompletionDepth) {
                retire(queued_base[head], queued_ops[head], queued_at[head], queued_id[head]);
                head = (head + 1) % CompletionDepth; --inflight;
            }
        }
        if (error) break;
    }
    if constexpr (CompletionDepth > 1) {
        while (inflight && !error) {
            retire(queued_base[head], queued_ops[head], queued_at[head], queued_id[head]);
            head = (head + 1) % CompletionDepth; --inflight;
        }
    }
    if (tid == 0) {
        stats[0] = error;
        if (!error) {
            gp::deviceAtomic(lane->last_success_generation).store(generation, cuda::memory_order_relaxed);
            phase.store(gp::gpunetio_lane_phase_idle, cuda::memory_order_release);
        } else {
            gp::deviceAtomic(lane->failed).store(1, cuda::memory_order_relaxed);
            phase.store(gp::gpunetio_lane_phase_failed, cuda::memory_order_release);
        }
        stats[3] = timer_ns() - begin;
        if (trace) { trace[3] = timer_ns(); trace[5] = stats[1]; trace[6] = stats[2]; trace[7] = stats[6]; }
    }
}

template<bool PackedRow = false, bool CompactPending = false>
__global__ void fetch_rows_cooperative(const int64_t *ids, uint32_t count,
    size_t payload_offset, size_t scale_offset, size_t dump_offset,
    uint32_t weight_index, uint32_t scale_index, nixlMemViewH local,
    nixlMemViewH remote, uint32_t window, uint64_t lo, uint64_t hi, uint64_t *stats, uint64_t *trace, uint64_t *cost) {
    fetch_rows_body<PackedRow, CompactPending>(ids, count, payload_offset, scale_offset,
        dump_offset, weight_index, scale_index, local, remote, window, lo, hi, stats, trace, cost);
}
struct LaneViews { nixlMemViewH local, remote; uint64_t *trace; uint64_t *cost; };
struct ReadGroup {
    std::vector<Context *> contexts;
    LaneViews *device_views = nullptr;
    uint32_t gpu = 0;
    uint64_t stagger_ns = 0;
    uint32_t completion_depth = 1;
    bool terminal_wait = false;
};
template<bool Pending, uint32_t CompletionDepth = 1, bool TerminalWait = false>
__global__ void fetch_group_rows(const LaneViews *views, const int64_t *ids,
    uint32_t count, size_t payload, size_t dump, uint32_t table, uint32_t window,
    uint64_t lo, uint64_t hi, uint64_t *stats, uint64_t stagger_ns, const int64_t *representatives, const int64_t *query_list, const uint64_t *list_count) {
    const auto lane = blockIdx.x;
    const uint64_t work = query_list ? *list_count : count;
    if (work > count) { if (threadIdx.x == 0) stats[lane*8] = 2; return; }
    const auto chunk = uint32_t((work + gridDim.x - 1) / gridDim.x);
    const auto start = lane * chunk;
    const auto local_count = start < work ? min(chunk,uint32_t(work-start)) : 0;
    if (!query_list && start >= count) return;
    if (stagger_ns) {
        if (threadIdx.x == 0) { const auto until = timer_ns() + lane * stagger_ns;
            while (timer_ns() < until) __nanosleep(256); }
        __syncthreads();
    }
    fetch_rows_body<true, Pending, CompletionDepth, TerminalWait>(query_list ? ids : ids + start, local_count,
        payload + (query_list ? 0 : size_t(start) * 264), 0, dump + lane, table, table,
        views[lane].local, views[lane].remote, window, lo, hi, stats + lane * 8, views[lane].trace, views[lane].cost,
        representatives ? representatives + (query_list ? 0 : start) : nullptr,
        query_list ? 0 : start, count, query_list ? query_list + start : nullptr);
}
} // namespace

extern "C" const char *engram_last_error() { return last_error.c_str(); }
extern "C" uint32_t engram_abi_version() { return 3; }
extern "C" void *engram_create(uint32_t gpu, uint16_t port, const char *name) {
    Context *result = nullptr;
    guarded([&] {
        check(cudaSetDevice(gpu), "set device");
        auto c = std::make_unique<Context>();
        c->gpu = gpu;
        nixlAgentConfig config;
        config.useProgThread = true;
        config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        c->agent = std::make_unique<nixlAgent>(name, config);
        nixl_b_params_t params{{"native_device_api", "true"}, {"gpu_devices", std::to_string(gpu)},
                               {"oob_port", std::to_string(port)}};
        for (auto item : {std::pair{"GPUNETIO_DEVICE_API_RDMA", "network_devices"},
                          std::pair{"GPUNETIO_DEVICE_API_OOB_INTERFACE", "oob_interface"},
                          std::pair{"GPUNETIO_DEVICE_API_GID_INDEX", "gid_index"}}) {
            if (const char *value = std::getenv(item.first)) params[item.second] = value;
        }
        require(c->agent->createBackend("GPUNETIO", params, c->backend), "create backend");
        c->hint.backends.push_back(c->backend);
        check(cudaEventCreateWithFlags(&c->retired, cudaEventDisableTiming), "create retirement event");
        result = c.release();
    });
    return result;
}
extern "C" int engram_register(void *handle, int host, const uint64_t *addresses,
                                const uint64_t *lengths, uint32_t count) {
    return guarded([&] {
        auto &c = *static_cast<Context *>(handle);
        if (c.registration || !count || (!host && count != 1)) throw std::runtime_error("invalid registration");
        auto desc = std::make_unique<nixl_reg_dlist_t>(host ? DRAM_SEG : VRAM_SEG);
        for (uint32_t i = 0; i < count; ++i) desc->addDesc({addresses[i], lengths[i], c.gpu, {}});
        require(c.agent->registerMem(*desc, &c.hint), "register memory");
        c.registration = std::move(desc);
        if (!host) {
            nixl_local_dlist_t local(VRAM_SEG);
            local.addDesc({addresses[0], lengths[0], c.gpu});
            require(c.agent->prepMemView(local, c.local, &c.hint), "local view");
            c.arena_bytes = lengths[0];
            c.arena_address = addresses[0];
        }
    });
}
extern "C" int64_t engram_metadata(void *handle, char *out, size_t capacity) {
    int64_t size = -1;
    guarded([&] {
        nixl_blob_t md;
        require(static_cast<Context *>(handle)->agent->getLocalMD(md), "export metadata");
        size = md.size();
        if (out) {
            if (capacity < md.size()) throw std::runtime_error("metadata buffer too small");
            std::memcpy(out, md.data(), md.size());
        }
    });
    return last_error.empty() ? size : -1;
}
extern "C" int engram_connect(void *handle, const char *md, size_t md_size,
                               const uint64_t *addresses, const uint64_t *lengths,
                               uint32_t count, uint32_t remote_gpu) {
    return guarded([&] {
        auto &c = *static_cast<Context *>(handle);
        if (c.remote || !c.local || !count) throw std::runtime_error("invalid connect state");
        std::string peer;
        require(c.agent->loadRemoteMD(nixl_blob_t(md, md_size), peer), "load metadata");
        require(c.agent->makeConnection(peer, &c.hint), "connect owner");
        nixl_remote_dlist_t remote(DRAM_SEG);
        for (uint32_t i = 0; i < count; ++i) remote.addDesc({addresses[i], lengths[i], remote_gpu, peer});
        require(c.agent->prepMemView(remote, c.remote, &c.hint), "remote view");
    });
}
static int fetch_impl(void *handle, uint64_t ids, uint32_t count,
                             size_t payload_offset, size_t scale_offset, size_t dump_offset,
                             uint32_t weight_index, uint32_t scale_index,
                             uint32_t window_rows, uint64_t stats, uint64_t stream,
                             uint64_t row_lo, uint64_t row_hi, bool packed, bool pending = false) {
    return guarded([&] {
        auto &c = *static_cast<Context *>(handle);
        auto s = reinterpret_cast<cudaStream_t>(stream);
        if ((pending && (window_rows < 4 || window_rows > 1024 || window_rows % 4)) ||
            !c.remote || !c.local || c.poisoned || !count || !ids || !stats || row_lo > row_hi ||
            payload_offset > c.arena_bytes ||
            scale_offset > c.arena_bytes || dump_offset >= c.arena_bytes ||
            size_t(count) * (packed ? 264 : 256) > c.arena_bytes - payload_offset ||
            (!packed && size_t(count) * 8 > c.arena_bytes - scale_offset) ||
            (packed && (!window_rows || weight_index != scale_index)))
            throw std::runtime_error("invalid fetch state, range, or stream");
        const auto payload_end = payload_offset + size_t(count) * (packed ? 264 : 256);
        const auto scale_end = scale_offset + size_t(count) * 8;
        if ((!packed && !(payload_end <= scale_offset || scale_end <= payload_offset)) ||
            (dump_offset >= payload_offset && dump_offset < payload_end) ||
            (!packed && dump_offset >= scale_offset && dump_offset < scale_end))
            throw std::runtime_error("overlapping READ destination or DUMP ranges");
        if (c.has_work && c.stream != s) {
            cudaStreamCaptureStatus capture;
            check(cudaStreamIsCapturing(s, &capture), "query capture status");
            if (capture != cudaStreamCaptureStatusNone)
                throw std::runtime_error("warm lookup on capture stream before capturing");
            const auto retired = cudaEventQuery(c.retired);
            if (retired == cudaErrorNotReady) {
                check(cudaStreamWaitEvent(s, c.retired, 0), "order lookup streams");
            } else {
                check(retired, "query prior lookup retirement");
            }
        }
        c.stream = s;
        c.stats = reinterpret_cast<uint64_t *>(stats);
        if (pending) {
            fetch_rows_cooperative<true, true><<<1, 128, 0, s>>>(reinterpret_cast<const int64_t *>(ids), count,
                payload_offset, scale_offset, dump_offset, weight_index, scale_index,
                c.local, c.remote, window_rows, row_lo, row_hi, c.stats, c.trace, c.cost_trace);
        } else if (packed) {
            fetch_rows_cooperative<true><<<1, 128, 0, s>>>(reinterpret_cast<const int64_t *>(ids), count,
                payload_offset, scale_offset, dump_offset, weight_index, scale_index,
                c.local, c.remote, window_rows, row_lo, row_hi, c.stats, c.trace, c.cost_trace);
        } else if (window_rows) {
            fetch_rows_cooperative<false><<<1, 128, 0, s>>>(reinterpret_cast<const int64_t *>(ids), count,
                payload_offset, scale_offset, dump_offset, weight_index, scale_index,
                c.local, c.remote, window_rows, row_lo, row_hi, c.stats, c.trace, c.cost_trace);
        } else {
            fetch_rows_scalar<<<1, 1, 0, s>>>(reinterpret_cast<const int64_t *>(ids), count,
                payload_offset, scale_offset, dump_offset, weight_index, scale_index,
                c.local, c.remote, row_lo, row_hi, c.stats);
        }
        check(cudaGetLastError(), "launch fetch");
        c.has_work = true;
        const auto rc = record_retirement(c, s);
        if (rc != cudaSuccess) c.poisoned = true;
        check(rc, "record retirement event");
    });
}
// ABI3 entry point stays unchanged; packed rows are an additive capability.
extern "C" int engram_fetch(void *handle, uint64_t ids, uint32_t count,
                             size_t payload_offset, size_t scale_offset, size_t dump_offset,
                             uint32_t weight_index, uint32_t scale_index,
                             uint32_t window_rows, uint64_t stats, uint64_t stream,
                             uint64_t row_lo, uint64_t row_hi) {
    return fetch_impl(handle, ids, count, payload_offset, scale_offset, dump_offset,
                      weight_index, scale_index, window_rows, stats, stream, row_lo, row_hi, false);
}
extern "C" int engram_fetch_row264(void *handle, uint64_t ids, uint32_t count,
                                    size_t payload_offset, size_t scale_offset, size_t dump_offset,
                                    uint32_t weight_index, uint32_t scale_index,
                                    uint32_t window_rows, uint64_t stats, uint64_t stream,
                                    uint64_t row_lo, uint64_t row_hi) {
    return fetch_impl(handle, ids, count, payload_offset, scale_offset, dump_offset,
                      weight_index, scale_index, window_rows, stats, stream, row_lo, row_hi, true);
}
extern "C" int engram_fetch_row264_pending(void *handle, uint64_t ids, uint32_t count,
                                    size_t payload_offset, size_t scale_offset, size_t dump_offset,
                                    uint32_t weight_index, uint32_t scale_index,
                                    uint32_t window_rows, uint64_t stats, uint64_t stream,
                                    uint64_t row_lo, uint64_t row_hi) {
    return fetch_impl(handle, ids, count, payload_offset, scale_offset, dump_offset,
                      weight_index, scale_index, window_rows, stats, stream, row_lo, row_hi, true, true);
}
extern "C" int engram_set_cost_trace(void *handle, uint64_t ptr) {
    return guarded([&] { static_cast<Context *>(handle)->cost_trace = reinterpret_cast<uint64_t *>(ptr); });
}
extern "C" int engram_set_trace(void *handle, uint64_t ptr) {
    return guarded([&] { static_cast<Context *>(handle)->trace = reinterpret_cast<uint64_t *>(ptr); });
}
extern "C" int engram_read_group_terminal(void *handle, int enabled) {
    return guarded([&] { static_cast<ReadGroup *>(handle)->terminal_wait = enabled != 0; });
}
extern "C" int engram_read_group_pipeline(void *handle, uint32_t depth) {
    return guarded([&] { if (depth != 1 && depth != 2) throw std::runtime_error("unsupported completion depth");
        static_cast<ReadGroup *>(handle)->completion_depth = depth; });
}
extern "C" int engram_read_group_stagger(void *handle, uint64_t delay_ns) {
    return guarded([&] { if (delay_ns > 1000000) throw std::runtime_error("diagnostic delay too large");
        static_cast<ReadGroup *>(handle)->stagger_ns = delay_ns; });
}
// Setup-only borrowed views: one CTA per independent QP, no grid-wide wait.
extern "C" void *engram_read_group_create(void **handles, uint32_t count) {
    ReadGroup *result = nullptr;
    guarded([&] {
        if (!handles || count == 0 || count > 16) throw std::runtime_error("invalid READ group");
        auto group = std::make_unique<ReadGroup>();
        std::vector<LaneViews> views;
        for (uint32_t i = 0; i < count; ++i) {
            auto *c = static_cast<Context *>(handles[i]);
            if (!c || !c->local || !c->remote || c->poisoned ||
                (i && (c->gpu != group->contexts[0]->gpu ||
                       c->arena_address != group->contexts[0]->arena_address ||
                       c->arena_bytes != group->contexts[0]->arena_bytes)) ||
                std::find(group->contexts.begin(), group->contexts.end(), c) != group->contexts.end())
                throw std::runtime_error("invalid/duplicate READ lane");
            group->contexts.push_back(c);
            views.push_back({c->local, c->remote, c->trace, c->cost_trace});
        }
        group->gpu = group->contexts[0]->gpu;
        check(cudaSetDevice(group->gpu), "select READ group device");
        check(cudaMalloc(reinterpret_cast<void **>(&group->device_views), count * sizeof(LaneViews)), "allocate READ views");
        const auto rc = cudaMemcpy(group->device_views, views.data(), count * sizeof(LaneViews), cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) { cudaFree(group->device_views); check(rc, "publish READ views"); }
        result = group.release();
    });
    return result;
}
static int read_group_fetch_impl(void *handle, uint64_t ids, uint32_t count,
    size_t payload, size_t dump, uint32_t table, uint32_t window, uint64_t stats,
    uint64_t stream, uint64_t lo, uint64_t hi, int pending, uint64_t representative_ptr = 0, uint64_t query_list_ptr = 0, uint64_t list_count_ptr = 0) {
    return guarded([&] {
        auto *group = static_cast<ReadGroup *>(handle);
        if (!group || !ids || !stats || !count || lo > hi || !window ||
            ((query_list_ptr || list_count_ptr) && (!query_list_ptr || !list_count_ptr || !representative_ptr)) ||
            (pending && (window < 4 || window > 1024 || window % 4)))
            throw std::runtime_error("invalid grouped READ");
        auto active = std::min<uint32_t>(count, group->contexts.size());
        const auto chunk = (count - 1) / active + 1;
        active = (count - 1) / chunk + 1;
        const auto end = payload + size_t(count) * 264;
        if (end < payload) throw std::runtime_error("READ range overflow");
        const auto s = reinterpret_cast<cudaStream_t>(stream);
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(s, &capture), "query READ group capture");
        for (uint32_t i = 0; i < active; ++i) {
            auto &c = *group->contexts[i];
            if (c.poisoned || !c.local || !c.remote || end > c.arena_bytes ||
                dump < end || dump >= c.arena_bytes || i >= c.arena_bytes - dump)
                throw std::runtime_error("invalid READ group lane/range");
            if (c.has_work && c.stream != s) {
                if (capture != cudaStreamCaptureStatusNone)
                    throw std::runtime_error("warm grouped READ on capture stream first");
                check(cudaStreamWaitEvent(s, c.retired, 0), "order READ group retirement");
            }
        }
        if (group->terminal_wait && group->completion_depth == 2) {
            if (pending) fetch_group_rows<true, 2, true><<<active, 128, 0, s>>>(group->device_views,
                reinterpret_cast<const int64_t *>(ids), count, payload, dump, table, window,
                lo, hi, reinterpret_cast<uint64_t *>(stats), group->stagger_ns, reinterpret_cast<const int64_t *>(representative_ptr), reinterpret_cast<const int64_t *>(query_list_ptr), reinterpret_cast<const uint64_t *>(list_count_ptr));
            else fetch_group_rows<false, 2, true><<<active, 128, 0, s>>>(group->device_views,
                reinterpret_cast<const int64_t *>(ids), count, payload, dump, table, window,
                lo, hi, reinterpret_cast<uint64_t *>(stats), group->stagger_ns, reinterpret_cast<const int64_t *>(representative_ptr), reinterpret_cast<const int64_t *>(query_list_ptr), reinterpret_cast<const uint64_t *>(list_count_ptr));
        } else if (group->terminal_wait) {
            if (pending) fetch_group_rows<true, 1, true><<<active, 128, 0, s>>>(group->device_views,
                reinterpret_cast<const int64_t *>(ids), count, payload, dump, table, window,
                lo, hi, reinterpret_cast<uint64_t *>(stats), group->stagger_ns, reinterpret_cast<const int64_t *>(representative_ptr), reinterpret_cast<const int64_t *>(query_list_ptr), reinterpret_cast<const uint64_t *>(list_count_ptr));
            else fetch_group_rows<false, 1, true><<<active, 128, 0, s>>>(group->device_views,
                reinterpret_cast<const int64_t *>(ids), count, payload, dump, table, window,
                lo, hi, reinterpret_cast<uint64_t *>(stats), group->stagger_ns, reinterpret_cast<const int64_t *>(representative_ptr), reinterpret_cast<const int64_t *>(query_list_ptr), reinterpret_cast<const uint64_t *>(list_count_ptr));
        } else if (group->completion_depth == 2) {
            if (pending) fetch_group_rows<true, 2><<<active, 128, 0, s>>>(group->device_views,
                reinterpret_cast<const int64_t *>(ids), count, payload, dump, table, window,
                lo, hi, reinterpret_cast<uint64_t *>(stats), group->stagger_ns, reinterpret_cast<const int64_t *>(representative_ptr), reinterpret_cast<const int64_t *>(query_list_ptr), reinterpret_cast<const uint64_t *>(list_count_ptr));
            else fetch_group_rows<false, 2><<<active, 128, 0, s>>>(group->device_views,
                reinterpret_cast<const int64_t *>(ids), count, payload, dump, table, window,
                lo, hi, reinterpret_cast<uint64_t *>(stats), group->stagger_ns, reinterpret_cast<const int64_t *>(representative_ptr), reinterpret_cast<const int64_t *>(query_list_ptr), reinterpret_cast<const uint64_t *>(list_count_ptr));
        } else if (pending) fetch_group_rows<true><<<active, 128, 0, s>>>(group->device_views,
            reinterpret_cast<const int64_t *>(ids), count, payload, dump, table, window,
            lo, hi, reinterpret_cast<uint64_t *>(stats), group->stagger_ns, reinterpret_cast<const int64_t *>(representative_ptr), reinterpret_cast<const int64_t *>(query_list_ptr), reinterpret_cast<const uint64_t *>(list_count_ptr));
        else fetch_group_rows<false><<<active, 128, 0, s>>>(group->device_views,
            reinterpret_cast<const int64_t *>(ids), count, payload, dump, table, window,
            lo, hi, reinterpret_cast<uint64_t *>(stats), group->stagger_ns, reinterpret_cast<const int64_t *>(representative_ptr), reinterpret_cast<const int64_t *>(query_list_ptr), reinterpret_cast<const uint64_t *>(list_count_ptr));
        const auto launch = cudaGetLastError();
        for (uint32_t i = 0; i < active; ++i) {
            auto &c = *group->contexts[i];
            c.has_work = true; c.stream = s;
            c.stats = reinterpret_cast<uint64_t *>(stats) + i * 8;
            if (launch != cudaSuccess) c.poisoned = true;
        }
        cudaError_t retirement = cudaSuccess;
        for (uint32_t i = 0; i < active; ++i) {
            auto &c = *group->contexts[i];
            const auto rc = record_retirement(c, s);
            if (rc != cudaSuccess) { c.poisoned = true; retirement = rc; }
        }
        check(launch, "launch grouped READ");
        check(retirement, "record READ group retirement");
    });
}
extern "C" int engram_read_group_fetch(void *handle, uint64_t ids, uint32_t count,
    size_t payload, size_t dump, uint32_t table, uint32_t window, uint64_t stats,
    uint64_t stream, uint64_t lo, uint64_t hi, int pending) {
    return read_group_fetch_impl(handle,ids,count,payload,dump,table,window,stats,stream,lo,hi,pending);
}
extern "C" int engram_read_group_fetch_mapped(void *handle, uint64_t ids, uint32_t count,
    size_t payload, size_t dump, uint32_t table, uint32_t window, uint64_t stats,
    uint64_t stream, uint64_t lo, uint64_t hi, int pending, uint64_t representatives) {
    return read_group_fetch_impl(handle,ids,count,payload,dump,table,window,stats,stream,lo,hi,pending,representatives);
}
extern "C" int engram_read_group_fetch_compacted(void *handle, uint64_t ids, uint32_t count,
    size_t payload, size_t dump, uint32_t table, uint32_t window, uint64_t stats,
    uint64_t stream, uint64_t lo, uint64_t hi, int pending, uint64_t representatives,
    uint64_t query_list, uint64_t list_count) {
    return read_group_fetch_impl(handle,ids,count,payload,dump,table,window,stats,stream,lo,hi,pending,representatives,query_list,list_count);
}
extern "C" int engram_read_group_destroy(void *handle) {
    return guarded([&] {
        auto *group = static_cast<ReadGroup *>(handle);
        if (!group) return;
        check(cudaSetDevice(group->gpu), "select READ group device");
        check(cudaDeviceSynchronize(), "retire READ group");
        check(cudaFree(group->device_views), "free READ views");
        delete group;
    });
}
extern "C" int engram_test_corrupt_remote_key(void *handle) {
    return guarded([&] {
        auto &c = *static_cast<Context *>(handle);
        if (!c.remote || c.has_work) throw std::runtime_error("fault injection requires a fresh test context");
        using Wrapper = std::remove_cv_t<std::remove_reference_t<decltype(*nixl::gpu::impl::viewWrapper(c.remote))>>;
        Wrapper wrapper{};
        check(cudaMemcpy(&wrapper, c.remote, sizeof(wrapper), cudaMemcpyDeviceToHost), "read test wrapper");
        nixl::gpu::impl::gpunetio::GpunetioDeviceView view{};
        check(cudaMemcpy(&view, wrapper.backend_memview, sizeof(view), cudaMemcpyDeviceToHost), "read test view");
        nixl::gpu::impl::gpunetio::GpunetioViewElem elem{};
        check(cudaMemcpy(&elem, view.elems, sizeof(elem), cudaMemcpyDeviceToHost), "read test key");
        elem.key ^= 0x7fff0000u;
        check(cudaMemcpy(const_cast<nixl::gpu::impl::gpunetio::GpunetioViewElem *>(view.elems), &elem, sizeof(elem), cudaMemcpyHostToDevice), "install test bad key");
    });
}
extern "C" int engram_destroy(void *handle) {
    if (!handle) return 0;
    return guarded([&] {
        auto *c = static_cast<Context *>(handle);
        check(cudaSetDevice(c->gpu), "select context device");
        if (c->poisoned) throw std::runtime_error("failed context quarantined; retain buffers until process exit");
        if (c->has_work) {
            // Teardown also drains consumers of raw scratch and graph replays;
            // a captured-but-unreplayed event is not the right teardown fence.
            check(cudaDeviceSynchronize(), "retire lookup and consumer GPU work");
            uint64_t error = 0;
            check(cudaMemcpy(&error, c->stats, sizeof(error), cudaMemcpyDeviceToHost), "read completion error");
            if (error) {
                c->poisoned = true;
                throw std::runtime_error("failed request quarantined; retain buffers until process exit");
            }
        }
        if (c->remote) c->agent->releaseMemView(c->remote);
        if (c->local) c->agent->releaseMemView(c->local);
        if (c->registration) require(c->agent->deregisterMem(*c->registration, &c->hint), "deregister memory");
        check(cudaEventDestroy(c->retired), "destroy retirement event");
        delete c;
    });
}

extern "C" int engram_record_consumed(void *handle, uint64_t stream) {
    return guarded([&] {
        auto &c = *static_cast<Context *>(handle);
        const auto s = reinterpret_cast<cudaStream_t>(stream);
        if (!c.has_work || c.stream != s || c.poisoned)
            throw std::runtime_error("consumer must follow lookup on the same stream");
        const auto rc = record_retirement(c, s);
        if (rc != cudaSuccess) c.poisoned = true;
        check(rc, "record consumer retirement");
    });
}
