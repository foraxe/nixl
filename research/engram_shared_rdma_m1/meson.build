# Private functional MVP harness; no production or benchmark target.
if get_option('gpunetio_device_api_profile') == 'legacy-doca31'
    engram_m1_device_args = [
        '-DNIXL_ENABLE_GPUNETIO_DEVICE_API=1',
        '-DNIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31=1',
        '-DALLOW_EXPERIMENTAL_API',
        '-D DOCA_ALLOW_EXPERIMENTAL_API',
    ]
else
    engram_m1_device_args = [
        '-DNIXL_ENABLE_GPUNETIO_DEVICE_API=1',
        '-DNIXL_GPUNETIO_DEVICE_API_PUBLIC=1',
        '-DALLOW_EXPERIMENTAL_API',
        '-D DOCA_ALLOW_EXPERIMENTAL_API',
    ]
endif

executable('shared_engram_m1', 'shared_engram_m1.cu',
    include_directories: [nixl_inc_dirs, nixl_device_inc_dirs, utils_inc_dirs],
    dependencies: [nixl_dep, nixl_infra, cuda_dep, doca_gpunetio_dep],
    cuda_args: engram_m1_device_args,
    build_by_default: false,
    install: false)
