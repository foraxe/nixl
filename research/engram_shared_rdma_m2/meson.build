# Private ctypes shim: registration/setup on CPU, row IDs and READ submission on GPU.
if get_option('gpunetio_device_api_profile') == 'legacy-doca31'
  shared_library('engram_native', 'engram_native.cu',
    include_directories: [nixl_inc_dirs, nixl_device_inc_dirs, utils_inc_dirs],
    dependencies: [nixl_dep, nixl_infra, cuda_dep, doca_gpunetio_dep],
    cuda_args: ['-DNIXL_ENABLE_GPUNETIO_DEVICE_API=1',
                '-DNIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31=1',
                '-DALLOW_EXPERIMENTAL_API', '-DDOCA_ALLOW_EXPERIMENTAL_API'],
    build_by_default: false,
    install: false)
endif
