// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_GPUNETIO_REGISTRATION_ACCESS_H
#define NIXL_GPUNETIO_REGISTRATION_ACCESS_H

#include <infiniband/verbs.h>

namespace nixl::doca::verbs {

inline constexpr int strictMrAccess = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

struct MrRegistration {
    ibv_mr *mr;
    bool relaxed;
};

// Use the same retry policy for DMA-BUF and PeerDirect registrations. An
// unsupported optional optimization must not make a valid strict MR fail.
template<class Register>
MrRegistration
registerMrWithOrdering(Register &&reg, bool prefer_relaxed) {
    const int access = strictMrAccess | (prefer_relaxed ? IBV_ACCESS_RELAXED_ORDERING : 0);
    if (auto *mr = reg(access)) {
        return {mr, prefer_relaxed};
    }
    return {prefer_relaxed ? reg(strictMrAccess) : nullptr, false};
}

} // namespace nixl::doca::verbs
#endif
