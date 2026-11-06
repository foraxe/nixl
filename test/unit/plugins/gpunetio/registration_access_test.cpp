// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "verbs/registration_access.h"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nixl::doca::verbs;

int
main() {
    const auto check = [](bool ok) {
        if (!ok) throw std::runtime_error("registration access policy mismatch");
    };
    ibv_mr dummy{};
    // Test the callback policy used by BOTH DMA-BUF and PeerDirect paths.
    for (const bool relaxed : {false, true}) {
        for (const bool initial_success : {false, true}) {
            for (const bool fallback_success : {false, true}) {
                std::vector<int> access;
                const auto result = registerMrWithOrdering([&](int flags) -> ibv_mr * {
                    access.push_back(flags);
                    return (access.size() == 1 ? initial_success : fallback_success) ? &dummy : nullptr;
                }, relaxed);
                check(access[0] == (strictMrAccess | (relaxed ? IBV_ACCESS_RELAXED_ORDERING : 0)));
                const bool retry = relaxed && !initial_success;
                check(access.size() == size_t(retry ? 2 : 1));
                if (retry) check(access[1] == strictMrAccess);
                check(result.mr == ((initial_success || (retry && fallback_success)) ? &dummy : nullptr));
                check(result.relaxed == (relaxed && initial_success));
            }
        }
    }
    std::cout << "PASS strict/relaxed registration, strict retry, and complete failure\n";
}
