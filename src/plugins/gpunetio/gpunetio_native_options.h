// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_GPUNETIO_NATIVE_OPTIONS_H
#define NIXL_GPUNETIO_NATIVE_OPTIONS_H
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace nixl::gpunetio {
inline constexpr uint32_t maxNativeQps = 32;

inline uint32_t
parseNativeQps(std::string_view value) {
    if (value.empty()) {
        return 1;
    }
    uint32_t count = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), count);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || count == 0 ||
        count > maxNativeQps) {
        throw std::invalid_argument("native_qp_count must be an integer in [1, 32]");
    }
    return count;
}

inline bool
parseNativeChannel(std::string_view value, uint32_t count, uint32_t &channel) {
    channel = 0;
    if (value.empty()) {
        return count != 0;
    }
    constexpr std::string_view prefix = "native_channel=";
    if (value.substr(0, prefix.size()) != prefix) {
        return false;
    }
    value.remove_prefix(prefix.size());
    const auto result = std::from_chars(value.data(), value.data() + value.size(), channel);
    return result.ec == std::errc{} && result.ptr == value.data() + value.size() && channel < count;
}
} // namespace nixl::gpunetio
#endif
