# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gpunetio_oob_endpoint_test = executable(
    'gpunetio_oob_endpoint_test',
    'gpunetio_oob_endpoint_test.cpp',
    include_directories: include_directories('../../../../src/plugins/gpunetio'),
    install: true,
)
test('gpunetio_oob_endpoint', gpunetio_oob_endpoint_test)

gpunetio_oob_socket_test = executable(
    'gpunetio_oob_socket_test',
    'gpunetio_oob_socket_test.cpp',
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
        include_directories('../../../../src/plugins/gpunetio'),
    ],
    dependencies: [gpunetio_backend_interface, cuda_dep, absl_log_dep] + plugin_gpunetio_deps,
    install: true,
)
test('gpunetio_oob_socket', gpunetio_oob_socket_test)

if not cuda_dep.found()
    message('CUDA not found, skipping gpunetio_qp_progress_gtest build')
    subdir_done()
endif

gpunetio_qp_progress_gtest = executable(
    'gpunetio_qp_progress_gtest',
    'qp_progress_gtest.cu',
    include_directories: [nixl_inc_dirs, utils_inc_dirs, gtest_inc_dirs],
    dependencies: [nixl_dep, nixl_infra, nixl_common_dep, cuda_dep, gtest_dep, gmock_dep],
    link_with: [nixl_build_lib],
    install: true,
)

# The executable is always registered. Without the paired-role environment its
# GoogleTests explicitly SKIP, so Meson exposes rather than silently omits it.
test('gpunetio_qp_progress_gtest', gpunetio_qp_progress_gtest, is_parallel: false, timeout: 180)
