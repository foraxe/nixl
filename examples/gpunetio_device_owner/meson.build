# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
deps = [nixl_dep, nixl_infra, cuda_dep, dependency('doca-gpunetio')]
incs = [nixl_inc_dirs, nixl_device_inc_dirs]

executable('owner-endpoint-exclusive',
    'owner_endpoint.cu',
    dependencies: deps,
    include_directories: incs,
    cpp_args: ['-DNIXL_GPUNETIO_DEVICE_API_HOST=1', '-DNIXL_GPUNETIO_DEVICE_API_OWNER=1'],
    cuda_args: ['-rdc=true', '-DNIXL_ENABLE_GPUNETIO_DEVICE_API=1', '-DNIXL_GPUNETIO_DEVICE_API_PUBLIC=1'],
    install: false)

executable('owner-endpoint-gpu-sharing-control',
    'owner_endpoint.cu',
    dependencies: deps,
    include_directories: incs,
    cpp_args: ['-DNIXL_GPUNETIO_DEVICE_API_HOST=1', '-DNIXL_GPUNETIO_DEVICE_API_OWNER=1'],
    cuda_args: ['-rdc=true', '-DNIXL_ENABLE_GPUNETIO_DEVICE_API=1', '-DNIXL_GPUNETIO_DEVICE_API_PUBLIC=1', '-DNIXL_GPUNETIO_OWNER_GPU_SHARING_CONTROL=1'],
    install: false)
