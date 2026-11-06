// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// ABI-3 shared-Engram adapter using the NIXL device GET API. No PyTorch C++ ABI dependency.
#include <cuda_runtime.h>
#include <new>
#include <gpu/nixl_device.cuh>
#include <gpu/nixl_device_batch.cuh>
#include <nixl.h>


#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef GET_PERF_TERMINAL_GATE
#define GET_PERF_TERMINAL_GATE 0
#endif
#ifndef COOP_FAIL_STAGE
#define COOP_FAIL_STAGE 0
#endif

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
    uint32_t stats_lanes = 1, pool_count = 1;
    std::string peer;
    std::vector<uint64_t> remote_addresses, remote_lengths;
    uint32_t remote_gpu = 0;
    bool has_work = false, poisoned = false;
    bool in_pool_group = false;
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
__device__ bool scalar_get(nixlMemViewH local, nixlMemViewH remote,
                           size_t dst, size_t index, size_t src, size_t bytes,
                           size_t dump, uint64_t *stats) {
    nixl::gpu::xferStatusH status{};
    const auto start = timer_ns();
    auto rc = nixl::gpu::get({remote, index, src}, {local, 0, dst}, {local, 0, dump},
                             bytes, &status);
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
    for (int i = 0; i < 8; ++i) stats[i] = 0;
    const auto begin = timer_ns();
    nixl::gpu::readSession session{};
    if (session.bind(local, remote) != NIXL_SUCCESS) { stats[0] = 1; return; }
    const size_t weight_length = session.sourceLength(weight_index);
    const size_t scale_length = session.sourceLength(scale_index);
    if (!weight_length || !scale_length || !session.destinationLength(0) ||
        session.acquire() != NIXL_SUCCESS) { stats[0] = 1; return; }
    const bool needs_dump = session.needsConsistency();
    const uint32_t queue_capacity = session.sendQueueCapacity();
    if (session.finish() != NIXL_SUCCESS) { session.fail(); stats[0] = 1; return; }
    for (uint32_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || uint64_t(ids[i]) >= weight_length / 256 ||
            uint64_t(ids[i]) >= scale_length / 8) { stats[0] = 2; return; }
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
    stats[5] = needs_dump ? stats[4] : 0;
    stats[6] = stats[4];
    stats[7] = queue_capacity;
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
    __shared__ uint32_t error, window;
    __shared__ uint32_t active_count, selected[1024], pending[1024], warp_counts[4];
    __shared__ uint64_t begin, base, post_start, posted, poll_start, builder_start, publish_start;
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
    __shared__ nixl::gpu::readSession session;
    __shared__ uint64_t weight_length, scale_length;
    if (tid == 0) {
        ::new (static_cast<void *>(&session)) nixl::gpu::readSession{};
        if (session.bind(local, remote) != NIXL_SUCCESS) error = 1;
        else {
            weight_length = session.sourceLength(weight_index);
            scale_length = session.sourceLength(scale_index);
            if (!weight_length || !scale_length || !session.destinationLength(0)) error = 1;
        }
    }
    __syncthreads();
    if (error) { if (tid == 0) stats[0] = error; return; }
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        const auto q = query_list ? query_list[i] : int64_t(i);
        if (q < 0 || uint64_t(q) >= (total_queries ? total_queries : count)) { atomicExch(&error, 2u); continue; }
        if (ids[q] < 0 || uint64_t(ids[q]) >= weight_length / (PackedRow ? 264 : 256) ||
            (!PackedRow && uint64_t(ids[q]) >= scale_length / 8)) atomicExch(&error, 2u);
        if (representatives && uint64_t(ids[q]) >= row_lo && uint64_t(ids[q]) < row_hi) {
            const auto r = representatives[q];
            const auto total = total_queries ? total_queries : count;
            if (r < 0 || uint64_t(r) >= total || (ids - query_base)[r] != ids[q] ||
                (representatives - query_base)[r] != r) atomicExch(&error, 2u);
        }
    }
    __syncthreads();
    if (error) { if (tid == 0) stats[0] = error; return; }
    if (tid == 0) {
        const auto rc = session.acquire();
        if (rc != NIXL_SUCCESS) error = 4;
        else {
            const uint32_t capacity = session.capacity();
            window = capacity < 4 ? 0 : min(requested_window,
                min(1024u, (capacity - 1) / (PackedRow ? 1 : 2)));
            stats[7] = capacity;
            if (!window || (CompactPending && window < 4) ||
                (CompletionDepth > 1 && (session.needsConsistency() ||
                 CompletionDepth * window * (PackedRow ? 1 : 2) >= capacity))) {
                error = 6;
                if (session.finish() != NIXL_SUCCESS) session.fail();
            }
        }
    }
    __syncthreads();
    if (error) { if (tid == 0) stats[0] = error; return; }
    const bool full_range = query_list == nullptr && representatives == nullptr && row_lo == 0 && row_hi >= weight_length / (PackedRow ? 264 : 256);
    auto retire = [&](uint64_t done_base, uint32_t done_ops, uint64_t posted_at, uint32_t batch_id) {
        if (tid == 0) poll_start = timer_ns();
        __syncthreads();
        if constexpr (TerminalWait) {
            if (tid == 0) {
                int rc;
                do {
                    rc = session.poll(done_base + done_ops - 1);
                    if (rc == NIXL_SUCCESS) break;
                    if (rc != NIXL_IN_PROG || timer_ns() - posted_at >= 8000000000ull) { error = 7; break; }
                } while (true);
            }
            __syncthreads();
        }
        for (uint32_t j = tid; j < done_ops; j += blockDim.x) {
            int rc;
            do {
                rc = session.poll(done_base + j);
                if (rc == NIXL_SUCCESS) break;
                if (TerminalWait || rc != NIXL_IN_PROG || timer_ns() - posted_at >= 8000000000ull) {
                    atomicExch(&error, 7u);
                    break;
                }
            } while (true);
        }
        __syncthreads();
        if (tid == 0) {
            if (!error) {
                if (session.retire(done_base, done_ops) != NIXL_SUCCESS) error = 7;
                stats[4] += done_ops;
            }
            stats[2] += timer_ns() - poll_start;
            if (trace && batch_id < 128) trace[9 + batch_id * 4] = timer_ns();
            if (!error && session.needsConsistency()) {
                const auto dump_start = timer_ns();
                uint64_t idx = 0;
                if (session.postConsistency(0, dump_offset, idx) != NIXL_IN_PROG) error = 7;
                const auto dump_posted = timer_ns();
                ++stats[5]; ++stats[6];
                stats[1] += dump_posted - dump_start;
                nixl_status_t dump_rc = NIXL_IN_PROG;
                while (!error && (dump_rc = session.poll(idx)) == NIXL_IN_PROG) {
                    if (timer_ns() - dump_posted >= 8000000000ull) { error = 7; break; }
                }
                if (!error && (dump_rc != NIXL_SUCCESS || session.retire(idx, 1) != NIXL_SUCCESS)) error = 7;
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
            if (session.reserve(operations, base) != NIXL_SUCCESS) error = 6;
            if (cost) { builder_start = timer_ns(); cost[1] += builder_start - at; }
        }
        __syncthreads();
        if (error) break;
        for (uint32_t j = tid; j < operations; j += blockDim.x) {
            const auto query = selected[PackedRow ? j : j / 2];
            const auto row = uint64_t(ids[query]);
            const bool payload = PackedRow || (j & 1) == 0;
            const uint32_t bytes = PackedRow ? 264 : (payload ? 256 : 8);
            const auto local_offset = (payload ? payload_offset : scale_offset) + size_t(query) * bytes;
            if (session.prepare(base + j, payload ? weight_index : scale_index, row * bytes,
                                0, local_offset, bytes) != NIXL_SUCCESS) atomicExch(&error, 6u);
        }
        __syncthreads();
        if (tid == 0) {
            if (cost) { publish_start = timer_ns(); cost[2] += publish_start - builder_start; }
            if (!error && session.publish(base, operations) != NIXL_IN_PROG) error = 6;
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
        __syncthreads();
        if (error) break;
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
            if (session.finish() != NIXL_SUCCESS) { error = 7; stats[0] = error; session.fail(); }
        } else {
            session.fail();
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
    fetch_rows_body<PackedRow, CompactPending, 1, bool(GET_PERF_TERMINAL_GATE)>(ids, count, payload_offset, scale_offset,
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
    uint32_t cooperative_ctas = 0;
    bool pooled = false;
    std::vector<nixlMemViewH> owned_views;
};
// Research control: stripe a prepared set of independent QPs inside ONE CTA.
// Per-READ error checking and FIFO retirement are retained for every session.
__global__ void fetch_group_one_cta(const LaneViews *views, uint32_t lanes,
    const int64_t *ids,uint32_t count,size_t payload,size_t dump,uint32_t table,
    uint32_t requested_window,uint64_t *stats) {
    using Session=nixl::gpu::readSession;
    __shared__ __align__(16) unsigned char storage[64*sizeof(Session)];
    __shared__ uint64_t base[64];
    __shared__ uint32_t operations[64],failed;
    auto *sessions=reinterpret_cast<Session*>(storage);
    const unsigned tid=threadIdx.x,chunk=(count+lanes-1)/lanes;
    const unsigned group_size=(lanes+gridDim.x-1)/gridDim.x;
    const unsigned lane_base=blockIdx.x*group_size;
    if(lane_base>=lanes)return;
    lanes=min(group_size,lanes-lane_base);views+=lane_base;stats+=lane_base*8;
    __shared__ uint64_t phase;
    const unsigned window=min(requested_window,chunk);
    if(tid==0)failed=0;
    for(unsigned j=tid;j<lanes*8;j+=blockDim.x)stats[j]=0;
    __syncthreads();
    if(tid<lanes){::new(sessions+tid)Session{};
        if(sessions[tid].bind(views[tid].local,views[tid].remote)!=NIXL_SUCCESS||
           sessions[tid].acquire()!=NIXL_SUCCESS)atomicExch(&failed,1u);
        stats[tid*8+7]=sessions[tid].capacity();}
    __syncthreads();
    for(unsigned offset=0;offset<chunk;offset+=window){
        if(__syncthreads_or(failed!=0))break;
        if(tid==0)phase=timer_ns();
        if(tid<lanes){const unsigned start=(lane_base+tid)*chunk+offset;
            operations[tid]=start<count?min(window,min(chunk-offset,count-start)):0;
            if(operations[tid]&&sessions[tid].reserve(operations[tid],base[tid])!=NIXL_SUCCESS)atomicExch(&failed,2u);}
        if(__syncthreads_or(failed!=0))break;
        for(unsigned j=tid;j<lanes*window;j+=blockDim.x){const unsigned q=j/window,k=j%window;
            if(k<operations[q]){const unsigned query=(lane_base+q)*chunk+offset+k;
                if(COOP_FAIL_STAGE==3&&blockIdx.x==0&&j==0){atomicExch(&failed,3u);continue;}
                if(ids[query]<0||uint64_t(ids[query])>=8192||
                   sessions[q].prepare(base[q]+k,table,size_t(ids[query])*264,
                       0,payload+size_t(query)*264,264)!=NIXL_SUCCESS)atomicExch(&failed,3u);}}
        if(__syncthreads_or(failed!=0))break;
        if(tid<lanes&&operations[tid]&&sessions[tid].publish(base[tid],operations[tid])!=NIXL_IN_PROG)atomicExch(&failed,4u);
        const int publish_failed=__syncthreads_or(failed!=0);
        if(tid==0){stats[1]+=timer_ns()-phase;phase=timer_ns();}
        // Test-only arrival skew: another warp may observe an error while this
        // warp is delayed. The decision above must remain uniform, not re-read failed.
        if(COOP_FAIL_STAGE==5&&tid>=32&&tid<64){const auto until=timer_ns()+1000000;while(timer_ns()<until)__nanosleep(32);}
        if(publish_failed)break;
        for(unsigned j=tid;j<lanes*window;j+=blockDim.x){const unsigned q=j/window,k=j%window;
            if(k<operations[q]){const auto deadline=timer_ns()+8000000000ULL;nixl_status_t rc;
                if(COOP_FAIL_STAGE==5&&blockIdx.x==0&&j==0){atomicExch(&failed,5u);continue;}
                do{rc=sessions[q].poll(base[q]+k);}while(rc==NIXL_IN_PROG&&timer_ns()<deadline);
                if(rc!=NIXL_SUCCESS)atomicExch(&failed,5u);}}
        if(__syncthreads_or(failed!=0))break;
        if(tid<lanes&&operations[tid]){auto&s=sessions[tid];
            if(s.retire(base[tid],operations[tid])!=NIXL_SUCCESS)atomicExch(&failed,6u);
            else{stats[tid*8+4]+=operations[tid];++stats[tid*8+6];}
            if(s.needsConsistency()){uint64_t ticket;auto rc=s.postConsistency(0,dump+lane_base+tid,ticket);
                if(rc==NIXL_IN_PROG){const auto deadline=timer_ns()+8000000000ULL;
                    ++stats[tid*8+5];
                    do{rc=s.poll(ticket);}while(rc==NIXL_IN_PROG&&timer_ns()<deadline);
                    if(rc!=NIXL_SUCCESS||s.retire(ticket,1)!=NIXL_SUCCESS)atomicExch(&failed,7u);
                }else atomicExch(&failed,7u);}}
        __syncthreads();
        if(tid==0)stats[2]+=timer_ns()-phase;
    }
    const int prior_failed=__syncthreads_or(failed!=0);
    if(tid<lanes){if(!prior_failed&&sessions[tid].finish()!=NIXL_SUCCESS)atomicExch(&failed,8u);}
    __syncthreads();
    if(tid<lanes){stats[tid*8]=failed;if(failed)sessions[tid].fail();}
}
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
        if (const char *value = std::getenv("GPUNETIO_DEVICE_API_RELAXED_ORDERING"))
            params["native_relaxed_ordering"] = value;
        if (const char *value = std::getenv("GPUNETIO_NATIVE_QPS")) {
            params["native_qp_count"] = value;
            c->pool_count = std::stoul(value);
        }
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
        c.peer = peer;
        c.remote_gpu = remote_gpu;
        c.remote_addresses.assign(addresses, addresses + count);
        c.remote_lengths.assign(lengths, lengths + count);
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
        if(c.in_pool_group)throw std::runtime_error("context is borrowed by a prepared pool group");
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
        c.stats_lanes = 1;
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
extern "C" int engram_read_group_single_cta(void *handle, int enabled) {
    return guarded([&] { auto*g=static_cast<ReadGroup *>(handle);
        if(!g||enabled<1||uint32_t(enabled)>g->contexts.size())throw std::runtime_error("invalid cooperative CTA count");
        g->cooperative_ctas = enabled; });
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
        if (!handles || count == 0 || count > 64) throw std::runtime_error("invalid READ group");
        auto group = std::make_unique<ReadGroup>();
        std::vector<LaneViews> views;
        for (uint32_t i = 0; i < count; ++i) {
            auto *c = static_cast<Context *>(handles[i]);
            if (!c || !c->local || !c->remote || c->poisoned || c->in_pool_group ||
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
// One prepared backend, one registration and one completion event per rank.
// Channel views are ordinary NIXL views prepared once; the GPU API is unchanged.
extern "C" void *engram_pool_group_create(void *handle, uint32_t count) {
    ReadGroup *result = nullptr;
    guarded([&] {
        auto &c = *static_cast<Context *>(handle);
        if(!count || count != c.pool_count || !c.local || !c.remote || c.has_work || c.in_pool_group)
            throw std::runtime_error("invalid prepared pool");
        auto group = std::make_unique<ReadGroup>();
        group->pooled = true;group->gpu = c.gpu;
        std::vector<LaneViews> views;
        try {
            for(uint32_t channel=0;channel<count;++channel) {
                nixlMemViewH local=c.local,remote=c.remote;
                if(channel) {
                    auto hint=c.hint;hint.customParam="native_channel="+std::to_string(channel);
                    nixl_local_dlist_t l(VRAM_SEG);l.addDesc({c.arena_address,c.arena_bytes,c.gpu});
                    require(c.agent->prepMemView(l,local,&hint),"prepare pool destination view");
                    group->owned_views.push_back(local);
                    nixl_remote_dlist_t r(DRAM_SEG);
                    for(size_t i=0;i<c.remote_addresses.size();++i)
                        r.addDesc({c.remote_addresses[i],c.remote_lengths[i],c.remote_gpu,c.peer});
                    require(c.agent->prepMemView(r,remote,&hint),"prepare pool source view");
                    group->owned_views.push_back(remote);
                }
                group->contexts.push_back(&c);views.push_back({local,remote,c.trace,c.cost_trace});
            }
            check(cudaMalloc(reinterpret_cast<void **>(&group->device_views),count*sizeof(LaneViews)),"allocate prepared pool");
            check(cudaMemcpy(group->device_views,views.data(),count*sizeof(LaneViews),cudaMemcpyHostToDevice),"publish prepared pool");
        } catch(...) {
            if(group->device_views)cudaFree(group->device_views);
            for(auto view:group->owned_views)c.agent->releaseMemView(view);
            throw;
        }
        c.in_pool_group = true;
        result=group.release();
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
        if (group->cooperative_ctas) {
            if(pending||representative_ptr||query_list_ptr||lo!=0||hi!=8192||
               group->completion_depth!=1||group->terminal_wait||group->stagger_ns)
                throw std::runtime_error("single-CTA research control requires full packed rows");
            fetch_group_one_cta<<<min(active,group->cooperative_ctas),128,0,s>>>(group->device_views,active,
                reinterpret_cast<const int64_t*>(ids),count,payload,dump,table,window,
                reinterpret_cast<uint64_t*>(stats));
        } else if (group->terminal_wait && group->completion_depth == 2) {
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
        for (uint32_t i = 0; i < (group->pooled ? 1u : active); ++i) {
            auto &c = *group->contexts[i];
            c.has_work = true; c.stream = s;
            c.stats = reinterpret_cast<uint64_t *>(stats) + i * 8;
            c.stats_lanes = group->pooled ? active : 1;
            if (launch != cudaSuccess) c.poisoned = true;
        }
        cudaError_t retirement = cudaSuccess;
        for (uint32_t i = 0; i < (group->pooled ? 1u : active); ++i) {
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
        if(group->pooled) {
            auto &c=*group->contexts[0];
            if(c.has_work){std::vector<uint64_t> stats(c.stats_lanes*8);
                check(cudaMemcpy(stats.data(),c.stats,stats.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost),"read pool completion errors");
                for(uint32_t q=0;q<c.stats_lanes;++q)if(stats[q*8]){c.poisoned=true;throw std::runtime_error("failed pool quarantined; retain views and buffers");}}
            for(auto view:group->owned_views)c.agent->releaseMemView(view);
        }
        check(cudaFree(group->device_views), "free READ views");
        if(group->pooled)group->contexts[0]->in_pool_group=false;
        delete group;
    });
}
extern "C" int engram_destroy(void *handle) {
    if (!handle) return 0;
    return guarded([&] {
        auto *c = static_cast<Context *>(handle);
        if(c->in_pool_group)throw std::runtime_error("destroy the prepared pool group first");
        check(cudaSetDevice(c->gpu), "select context device");
        if (c->poisoned) throw std::runtime_error("failed context quarantined; retain buffers until process exit");
        if (c->has_work) {
            // Teardown also drains consumers of raw scratch and graph replays;
            // a captured-but-unreplayed event is not the right teardown fence.
            check(cudaDeviceSynchronize(), "retire lookup and consumer GPU work");
            std::vector<uint64_t> errors(c->stats_lanes*8);
            check(cudaMemcpy(errors.data(), c->stats, errors.size()*sizeof(uint64_t), cudaMemcpyDeviceToHost), "read completion error");
            bool failed=false;for(uint32_t q=0;q<c->stats_lanes;++q)failed=failed||errors[q*8]!=0;
            if (failed) {
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
