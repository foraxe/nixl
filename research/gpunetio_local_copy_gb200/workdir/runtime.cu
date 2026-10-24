// SPDX-License-Identifier: Apache-2.0
#include "local_copy.cuh"
#include "verbs/verbs.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

using nixl::doca::verbs::qp;
using nixl::doca::verbs::mr;
extern doca_verbs_context *open_ib_device(char *name);
extern doca_error_t create_verbs_ah_attr(doca_verbs_context *, uint32_t,
                                       doca_verbs_addr_type, doca_verbs_ah_attr **);
namespace {
thread_local std::string error;
std::atomic_uint64_t next_cookie{1};
void check(doca_error_t e, const char *op) {
    if (e != DOCA_SUCCESS) throw std::runtime_error(std::string(op) + ": " + doca_error_get_descr(e));
}
void check(cudaError_t e, const char *op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}
struct DeviceGuard {
    int old;
    explicit DeviceGuard(int gpu) { check(cudaGetDevice(&old), "get device"); check(cudaSetDevice(gpu), "set device"); }
    ~DeviceGuard() { cudaSetDevice(old); }
};
struct Attr {
    doca_verbs_qp_attr *p = nullptr;
    Attr() { check(doca_verbs_qp_attr_create(&p), "QP attr"); }
    ~Attr() { doca_verbs_qp_attr_destroy(p); }
};

// Same INIT/RTR/RTS setup as E, with local peer QPN/GID and no socket handshake.
void connect(doca_verbs_qp *q, uint32_t peer, doca_verbs_ah_attr *ah, int mtu) {
    Attr a;
#define SET(field, value) check(doca_verbs_qp_attr_set_##field(a.p, value), #field)
    SET(port_num, 1);
    SET(allow_remote_read, 1);
    SET(allow_remote_write, 0);
    SET(next_state, DOCA_VERBS_QP_STATE_INIT);
    check(doca_verbs_qp_modify(q, a.p, DOCA_VERBS_QP_ATTR_NEXT_STATE |
        DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
        DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM), "QP INIT");
    SET(path_mtu, static_cast<doca_mtu_size>(mtu));
    SET(rq_psn, 0);
    SET(dest_qp_num, peer);
    SET(ah_attr, ah);
    SET(min_rnr_timer, 1);
    SET(max_dest_rd_atomic, 16);
    SET(next_state, DOCA_VERBS_QP_STATE_RTR);
    check(doca_verbs_qp_modify(q, a.p, DOCA_VERBS_QP_ATTR_NEXT_STATE |
        DOCA_VERBS_QP_ATTR_PATH_MTU | DOCA_VERBS_QP_ATTR_RQ_PSN |
        DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_AH_ATTR |
        DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER | DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC), "QP RTR");
    SET(sq_psn, 0);
    SET(ack_timeout, 12);
    SET(retry_cnt, 2);
    SET(rnr_retry, 1);
    SET(max_rd_atomic, 16);
    SET(next_state, DOCA_VERBS_QP_STATE_RTS);
    check(doca_verbs_qp_modify(q, a.p, DOCA_VERBS_QP_ATTR_NEXT_STATE |
        DOCA_VERBS_QP_ATTR_SQ_PSN | DOCA_VERBS_QP_ATTR_ACK_TIMEOUT |
        DOCA_VERBS_QP_ATTR_RETRY_CNT | DOCA_VERBS_QP_ATTR_RNR_RETRY |
        DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC), "QP RTS");
#undef SET
}

struct Runtime {
    int gpu = 0;
    doca_gpu *gdev = nullptr;
    doca_dev *dev = nullptr;
    doca_verbs_context *context = nullptr;
    doca_verbs_pd *pd = nullptr;
    doca_verbs_ah_attr *ah = nullptr;
    std::unique_ptr<qp> requester, responder;
    std::unique_ptr<mr> host_mr, gpu_mr, dump_mr;
    void *dump = nullptr;
    local_copy::Copy *view = nullptr;
    local_copy::Recovery *recovery = nullptr;
    std::thread controller;
    std::thread cleanup_worker;
    std::atomic_int cleanup_state{0};
    std::string cleanup_error;
    std::atomic_bool stop{false};
    std::atomic_bool recovery_failed{false};
#ifdef LOCAL_COPY_TESTING
    std::atomic_bool fail_reset{false};
#endif

    void init(void *src, size_t src_size, void *dst, size_t dst_size,
              int gpu_id, const char *nic, int gid_index) {
        gpu = gpu_id;
        DeviceGuard guard(gpu);
        if (!src || !dst || !src_size || !dst_size || src_size > UINT32_MAX || dst_size > UINT32_MAX)
            throw std::invalid_argument("nonempty registered tensor ranges must fit uint32");
        cudaPointerAttributes sa{}, da{};
        check(cudaPointerGetAttributes(&sa, src), "host pointer");
        check(cudaPointerGetAttributes(&da, dst), "GPU pointer");
        if (sa.type != cudaMemoryTypeHost || da.type != cudaMemoryTypeDevice || da.device != gpu)
            throw std::invalid_argument("requires pinned host source and owner-GPU device destination");
        context = open_ib_device(const_cast<char *>(nic));
        if (!context) throw std::runtime_error("cannot open NIC");
        check(doca_verbs_pd_create(context, &pd), "PD");
        auto *ibpd = doca_verbs_bridge_verbs_pd_get_ibv_pd(pd);
        check(doca_rdma_bridge_open_dev_from_pd(ibpd, &dev), "DOCA device");
        char pci[32];
        check(cudaDeviceGetPCIBusId(pci, sizeof(pci), gpu), "GPU PCI");
        check(doca_gpu_create(pci, &gdev), "DOCA GPU");
        ibv_port_attr port{};
        ibv_gid gid{};
        if (ibv_query_port(ibpd->context, 1, &port) ||
            ibv_query_gid(ibpd->context, 1, gid_index, &gid))
            throw std::runtime_error("cannot query NIC port/GID");
        check(create_verbs_ah_attr(context, gid_index,
            port.link_layer == IBV_LINK_LAYER_INFINIBAND ? DOCA_VERBS_ADDR_TYPE_IB_NO_GRH :
                DOCA_VERBS_ADDR_TYPE_IPv4, &ah), "AH");
        doca_verbs_gid local_gid;
        memcpy(local_gid.raw, gid.raw, sizeof(gid.raw));
        check(doca_verbs_ah_attr_set_gid(ah, local_gid), "local GID");
        if (port.link_layer == IBV_LINK_LAYER_INFINIBAND)
            check(doca_verbs_ah_attr_set_dlid(ah, port.lid), "local LID");
        requester = std::make_unique<qp>(gdev, dev, context, pd, 128, 16,
            DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB);
        responder = std::make_unique<qp>(gdev, dev, context, pd, 128, 16,
            DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB);
        // IB MTU enum starts at 1; DOCA's starts at 0.
        connect(requester->get_qp(), doca_verbs_qp_get_qpn(responder->get_qp()), ah, port.active_mtu - 1);
        connect(responder->get_qp(), doca_verbs_qp_get_qpn(requester->get_qp()), ah, port.active_mtu - 1);
        host_mr = std::make_unique<mr>(nullptr, src, src_size, 1, ibpd, false);
        gpu_mr = std::make_unique<mr>(gdev, dst, dst_size, 1, ibpd, true);
        check(cudaMalloc(&dump, 4096), "dump allocation");
        dump_mr = std::make_unique<mr>(gdev, dump, 4096, 1, ibpd, true);
        local_copy::Recovery *mapped;
        check(cudaHostAlloc(&recovery, sizeof(*recovery), cudaHostAllocMapped), "recovery allocation");
        new (recovery) local_copy::Recovery{};
        check(cudaHostGetDevicePointer(&mapped, recovery, 0), "recovery map");
        int ordering;
        auto ce = cuDeviceGetAttribute(&ordering, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, gpu);
        if (ce != CUDA_SUCCESS) throw std::runtime_error("cannot query RDMA ordering");
        local_copy::Copy copy{};
        copy.qp = requester->get_qp_gpu_dev();
        copy.recovery = mapped;
        copy.src_base = reinterpret_cast<uintptr_t>(src);
        copy.dst_base = reinterpret_cast<uintptr_t>(dst);
        copy.src_size = src_size; copy.dst_size = dst_size;
        copy.rkey = host_mr->get_rkey(); copy.lkey = gpu_mr->get_lkey();
        copy.dump = {reinterpret_cast<uintptr_t>(dump), dump_mr->get_lkey()};
        copy.mcst = ordering < CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER;
        copy.cookie = next_cookie.fetch_add(1, std::memory_order_relaxed);
        check(cudaMalloc(&view, sizeof(copy)), "copy context");
        check(cudaMemcpy(view, &copy, sizeof(copy), cudaMemcpyHostToDevice), "publish copy context");
        controller = std::thread([this] {
            while (!stop.load(std::memory_order_acquire)) {
                if (std::atomic_ref<int>(recovery->fault).load(std::memory_order_acquire)) {
                    try {
#ifdef LOCAL_COPY_TESTING
                        if (fail_reset.load()) throw std::runtime_error("injected reset failure");
#endif
                        reset();
                        std::atomic_ref<int>(recovery->stopped).store(1, std::memory_order_release);
                    } catch (...) {
                        std::atomic_ref<int>(recovery->stopped).store(-1, std::memory_order_release);
                        recovery_failed.store(true, std::memory_order_release);
                    }
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    void reset() {
        Attr a;
        check(doca_verbs_qp_attr_set_next_state(a.p, DOCA_VERBS_QP_STATE_RST), "reset attr");
        for (auto *q : {requester.get(), responder.get()})
            if (q && q->get_qp())
                check(doca_verbs_qp_modify(q->get_qp(), a.p, DOCA_VERBS_QP_ATTR_NEXT_STATE), "QP reset");
    }

    void cleanup() {
        DeviceGuard guard(gpu);
        check(cudaDeviceSynchronize(), "finish copy users");
        stop.store(true, std::memory_order_release);
        if (controller.joinable()) controller.join();
        reset();
        if ((requester && !requester->close()) || (responder && !responder->close()))
            throw std::runtime_error("QP close failed: retaining registrations and buffers");
        requester.reset(); responder.reset();
        host_mr.reset(); gpu_mr.reset(); dump_mr.reset();
        if (view) { check(cudaFree(view), "free context"); view = nullptr; }
        if (dump) { check(cudaFree(dump), "free dump"); dump = nullptr; }
        if (recovery) { check(cudaFreeHost(recovery), "free recovery"); recovery = nullptr; }
        if (ah) { check(doca_verbs_ah_attr_destroy(ah), "destroy AH"); ah = nullptr; }
        if (dev) { check(doca_dev_close(dev), "close NIC"); dev = nullptr; }
        if (pd) { check(doca_verbs_pd_destroy(pd), "destroy PD"); pd = nullptr; }
        if (context) { check(doca_verbs_context_destroy(context), "destroy context"); context = nullptr; }
        if (gdev) { check(doca_gpu_destroy(gdev), "destroy GPU handle"); gdev = nullptr; }
    }

    void close() {
        // Keep all CUDA/DOCA cleanup in a retained worker: a failed close cannot
        // free caller buffers or trap the caller inside a driver/SDK API.
        if (!cleanup_worker.joinable()) {
            cleanup_state.store(0, std::memory_order_release);
            cleanup_worker = std::thread([this] {
                try { cleanup(); cleanup_state.store(1, std::memory_order_release); }
                catch (const std::exception &e) {
                    cleanup_error = e.what();
                    cleanup_state.store(-1, std::memory_order_release);
                }
            });
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (cleanup_state.load(std::memory_order_acquire) == 0) {
            if (recovery_failed.load(std::memory_order_acquire) ||
                std::chrono::steady_clock::now() > deadline)
                throw std::runtime_error("cleanup not complete: context and tensors quarantined; close refused");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cleanup_worker.join();
        if (cleanup_state.load(std::memory_order_acquire) != 1)
            throw std::runtime_error(cleanup_error);
    }
};
} // namespace

extern "C" const char *lc_error() { return error.c_str(); }
extern "C" void *lc_open(void *src, size_t src_size, void *dst, size_t dst_size,
                         int gpu, const char *nic, int gid) {
    auto *runtime = new Runtime;
    try { runtime->init(src, src_size, dst, dst_size, gpu, nic, gid); return runtime; }
    catch (const std::exception &e) {
        error = e.what();
        try { runtime->close(); delete runtime; } catch (...) { /* retain on unsafe cleanup */ }
        return nullptr;
    }
}
extern "C" void *lc_device(void *handle) { return static_cast<Runtime *>(handle)->view; }
extern "C" int lc_close(void *handle) {
    try { auto *r = static_cast<Runtime *>(handle); r->close(); delete r; return 0; }
    catch (const std::exception &e) { error = e.what(); return -1; }
}
#ifdef LOCAL_COPY_TESTING
extern "C" void lc_test_fail_reset(void *handle) { static_cast<Runtime *>(handle)->fail_reset.store(true); }
extern "C" int lc_test_state(void *handle) {
    return std::atomic_ref<int>(static_cast<Runtime *>(handle)->recovery->stopped).load(std::memory_order_acquire);
}
#endif
