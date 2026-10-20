/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public-API-only synchronous-versus-asynchronous SEND replay.
#define main qualifier_original_main
#include "qp_progress_replay.cu"
#undef main

namespace {
constexpr size_t kReceives = 4096; // Matches the unchanged backend's initial RQ.
bool Pressure() { return Env("SEND_PRESSURE").value_or("0") == "1"; }
std::string Message(size_t round) {
    return "round-" + std::to_string(round) + std::string(7900, 'n');
}

void Drain(Endpoint &target, size_t round, bool pressure) {
    size_t received = 0;
    const size_t expected = pressure ? kReceives + 1 : 1;
    const auto deadline = Clock::now() + std::chrono::seconds(20);
    while (received < expected && Clock::now() < deadline) {
        nixl_notifs_t notifications;
        CheckNixl(target.agent.getNotifs(notifications), "receive SEND replay notification");
        for (const auto &[peer, messages] : notifications) {
            EXPECT_EQ(peer, "qp-progress-source");
            for (const auto &message : messages) {
                const auto wanted = pressure && received < kReceives ?
                    "prefill-" + std::to_string(received) : Message(round);
                EXPECT_EQ(message, wanted);
                ++received;
            }
        }
    }
    ASSERT_EQ(received, expected);
}

void TargetSend(const Config &config) {
    DeviceMemory a_memory(kSmallBytes, 0), b_memory(kSmallBytes, 1);
    VerifyCounter a_counter(0), b_counter(1);
    Endpoint a(config, "qp-progress-a", config.target_a_port, a_memory);
    Endpoint b(config, "qp-progress-b", config.target_b_port, b_memory);
    Control control(config);
    PublishMetadata(config, a, a_memory, b, b_memory);
    control.Send('R');
    if (Pressure()) { control.Expect('F'); control.Send('G'); }
    for (int round = 0; round < kWarmup + kMeasured; ++round) {
        if (Pressure()) {
            control.Expect('S');
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            Drain(a, round, true);
        }
        control.Expect('D');
        if (!Pressure()) Drain(a, round, false);
        ASSERT_EQ(a_counter.Verify(a_memory, SeedA(round), EpochA(round)), 0U);
        ASSERT_EQ(b_counter.Verify(b_memory, SeedB(round), EpochB(round)), 0U);
        control.Send('V');
    }
    control.Expect('C'); control.Send('K');
}

void SourceSend(const Config &config) {
    DeviceMemory a_memory(kSmallBytes, 0), b_memory(kSmallBytes, 0);
    Endpoint source(config, "qp-progress-source", config.source_port, a_memory);
    auto registration = Registration(b_memory);
    CheckNixl(source.agent.registerMem(registration, &source.options), "register B source");
    Control control(config);
    control.Expect('R');
    const auto remote = LoadMetadata(config, source.agent);
    if (Pressure()) {
        for (size_t i = 0; i < kReceives; ++i)
            CheckNixl(source.agent.genNotif("qp-progress-a", "prefill-" + std::to_string(i), &source.options), "prefill receiver RQ");
        control.Send('F'); control.Expect('G');
    }
    std::vector<uint64_t> a_windows, b_windows;
    size_t pending_at_b = 0;
    for (int round = 0; round < kWarmup + kMeasured; ++round) {
        Fill(a_memory, SeedA(round), EpochA(round));
        Fill(b_memory, SeedB(round), EpochB(round));
        CheckCuda(cudaStreamSynchronize(0), "prepare SEND replay payloads");
        nixlXferReqH *a = nullptr, *b = nullptr;
        auto notification = source.options;
        notification.notif = Message(round);
        CheckNixl(source.agent.createXferReq(NIXL_WRITE, TransferList(a_memory),
            RemoteList(remote.a_data, kSmallBytes, remote.a_epoch, 0), "qp-progress-a", a, &notification), "create A+notification");
        CheckNixl(source.agent.createXferReq(NIXL_WRITE, TransferList(b_memory),
            RemoteList(remote.b_data, kSmallBytes, remote.b_epoch, 1), "qp-progress-b", b, &source.options), "create independent B");
        const uint64_t a_start = NowNsec();
        auto a_status = source.agent.postXferReq(a);
        const uint64_t b_start = NowNsec();
        auto b_status = source.agent.postXferReq(b);
        ASSERT_TRUE(a_status == NIXL_SUCCESS || a_status == NIXL_IN_PROG);
        ASSERT_TRUE(b_status == NIXL_SUCCESS || b_status == NIXL_IN_PROG);
        if (Pressure()) control.Send('S');
        if (b_status == NIXL_IN_PROG) b_status = Wait(source.agent, b, kTimeout);
        const uint64_t b_done = NowNsec();
        ASSERT_EQ(b_status, NIXL_SUCCESS);
        a_status = source.agent.getXferStatus(a);
        if (round >= kWarmup && a_status == NIXL_IN_PROG) ++pending_at_b;
        if (a_status == NIXL_IN_PROG) a_status = Wait(source.agent, a, kTimeout);
        const uint64_t a_done = NowNsec();
        ASSERT_EQ(a_status, NIXL_SUCCESS);
        if (round >= kWarmup) {
            a_windows.push_back(a_done - a_start);
            b_windows.push_back(b_done - b_start);
        }
        Release(source.agent, a); Release(source.agent, b);
        control.Send('D'); control.Expect('V');
    }
    std::ostringstream report;
    report << "{\"result\":\"PASS\",\"pressure\":" << int(Pressure())
        << ",\"warmup\":" << kWarmup << ",\"measured\":" << kMeasured
        << ",\"a_p50_ns\":" << Percentile(a_windows, .5)
        << ",\"a_p99_ns\":" << Percentile(a_windows, .99)
        << ",\"b_p50_ns\":" << Percentile(b_windows, .5)
        << ",\"b_p99_ns\":" << Percentile(b_windows, .99)
        << ",\"a_pending_at_b\":" << pending_at_b << ",\"b_windows_ns\":[";
    for (size_t i = 0; i < b_windows.size(); ++i) report << (i ? "," : "") << b_windows[i];
    report << "]}\n";
    AtomicWrite(config.coord / "send_replay.json", report.str());
    CheckNixl(source.agent.deregisterMem(registration, &source.options), "deregister B source");
    control.Send('C'); control.Expect('K');
}
} // namespace

TEST(QpProgress, NotificationSendIsolation) {
    const auto config = GetConfig();
    if (auto problem = ConfigProblem(config, false)) GTEST_SKIP() << *problem;
    if (Pressure()) ASSERT_EQ(kWarmup + kMeasured, 1);
    if (IsTarget(config)) TargetSend(config); else SourceSend(config);
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv); return RUN_ALL_TESTS();
}
