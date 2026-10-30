/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NIXL contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime.h>
#include <nixl.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using Clock = std::chrono::steady_clock;
constexpr int kCount = 4096, kWarm = 128;

void
nx(nixl_status_t s, const char *m) {
    if (s != NIXL_SUCCESS) {
        throw std::runtime_error(std::string(m) + " status=" + std::to_string(s));
    }
}

void
cu(cudaError_t s, const char *m) {
    if (s != cudaSuccess) {
        throw std::runtime_error(std::string(m) + ": " + cudaGetErrorString(s));
    }
}

int
main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "usage: ordinary-notif NIC OOB_INTERFACE GID_INDEX BASE_PORT\n";
        return 2;
    }
    try {
        int port = std::stoi(argv[4]);
        if (port < 1 || port > 65534) {
            throw std::invalid_argument("invalid base port");
        }
        std::array<std::unique_ptr<nixlAgent>, 2> a;
        std::array<void *, 2> mem{};
        std::array<nixl_reg_dlist_t, 2> reg{nixl_reg_dlist_t(VRAM_SEG), nixl_reg_dlist_t(VRAM_SEG)};
        std::array<nixl_blob_t, 2> md;
        std::array<std::string, 2> peer;
        for (int d = 0; d < 2; ++d) {
            cu(cudaSetDevice(d), "select");
            cu(cudaMalloc(&mem[d], 4096), "allocate");
            nixlAgentConfig cfg;
            cfg.useProgThread = false;
            a[d] = std::make_unique<nixlAgent>(d == 0 ? "ordinary-src" : "ordinary-dst", cfg);
            nixl_mem_list_t types;
            nixl_b_params_t params;
            nx(a[d]->getPluginParams("GPUNETIO", types, params), "params");
            params["network_devices"] = argv[1];
            params["oob_interface"] = argv[2];
            params["gid_index"] = argv[3];
            params["gpu_devices"] = std::to_string(d);
            params["data_qp_count"] = "1";
            params["oob_port"] = std::to_string(port + d);
            nixlBackendH *backend = nullptr;
            nx(a[d]->createBackend("GPUNETIO", params, backend), "backend");
            reg[d].addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(mem[d]), 4096, d));
            nx(a[d]->registerMem(reg[d]), "register");
            nx(a[d]->getLocalMD(md[d]), "metadata");
        }
        for (int d = 0; d < 2; ++d) {
            cu(cudaSetDevice(d), "select");
            nx(a[d]->loadRemoteMD(md[1 - d], peer[d]), "connect");
        }
        std::atomic<bool> started{false}, stop{false}, finished{false};
        std::atomic<int> failure{0};
        std::set<std::string> received;
        std::thread receiver([&] {
            if (cudaSetDevice(1) != cudaSuccess) {
                failure = 1;
                finished = true;
                return;
            }
            started = true;
            while (!stop.load() && received.size() < kCount) {
                nixl_notifs_t msgs;
                auto status = a[1]->getNotifs(msgs);
                if (status != NIXL_SUCCESS) {
                    failure = int(status);
                    break;
                }
                for (auto &[name, items] : msgs) {
                    if (name != peer[1]) {
                        failure = 1;
                        break;
                    }
                    for (auto &msg : items) {
                        if (!received.insert(msg).second) {
                            failure = 1;
                            break;
                        }
                    }
                }
                if (failure.load()) {
                    break;
                }
                std::this_thread::yield();
            }
            finished = true;
        });
        while (!started.load() && !finished.load()) {
            std::this_thread::yield();
        }
        std::vector<double> times;
        times.reserve(kCount - kWarm);
        cu(cudaSetDevice(0), "sender");
        for (int i = 0; i < kCount && !failure.load(); ++i) {
            auto t = Clock::now();
            auto status = a[0]->genNotif(peer[0], "message-" + std::to_string(i));
            auto end = Clock::now();
            if (status != NIXL_SUCCESS) {
                failure = int(status);
                break;
            }
            if (i >= kWarm) {
                times.push_back(std::chrono::duration<double, std::micro>(end - t).count());
            }
        }
        auto deadline = Clock::now() + std::chrono::seconds(30);
        while (!finished.load() && Clock::now() < deadline) {
            std::this_thread::yield();
        }
        stop = true;
        receiver.join();
        if (failure.load() || times.size() != kCount - kWarm || received.size() != kCount) {
            throw std::runtime_error("notification count or status mismatch");
        }
        for (int i = 0; i < kCount; ++i) {
            if (!received.contains("message-" + std::to_string(i))) {
                throw std::runtime_error("notification identity mismatch");
            }
        }
        std::sort(times.begin(), times.end());
        auto pct = [&](int rank) { return times[(rank * times.size() + 99) / 100 - 1]; };
        std::cout << std::fixed << std::setprecision(2) << "ORDINARY_NOTIF_PASS sent=" << kCount
                  << " received=" << received.size() << " timed=" << times.size()
                  << " mean_us=" << std::accumulate(times.begin(), times.end(), 0.0) / times.size()
                  << " p50_us=" << pct(50) << " p99_us=" << pct(99) << std::endl;
        for (int d = 0; d < 2; ++d) {
            cu(cudaSetDevice(d), "select");
            nx(a[d]->deregisterMem(reg[d]), "deregister");
            a[d].reset();
            cu(cudaFree(mem[d]), "free");
        }
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "ORDINARY_NOTIF_FAIL " << e.what() << std::endl;
        return 1;
    }
}
