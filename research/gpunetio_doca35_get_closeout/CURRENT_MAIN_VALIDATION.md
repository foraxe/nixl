# Current-main source qualification

Published GET head `df0d104093f6f6a67932b13073fbe357a460aa2a` is based on main `635278fe6b954b4d737bf114ed40f8dd16acc575`. The eleven commits are range-diff equivalent to the pinned e56 candidate; GPUNETIO and device source files match byte-for-byte. Upstream core/build changes are retained. Performance samples elsewhere in this package remain explicitly e56 results.

Fresh qualification on H20/CUDA13.0/DOCA3.5.0098:

- Complete release/Werror builds: enabled294/294, disabled294/294.
- Full unit suite250/250 passed in one invocation using the complete fresh runtime stack. Separate earlier selections also passed246 CPU-only and4 GPU/host-memory cases; native pool options and netif discovery2/2 passed.
- Current-head four-QP pinned Host and VRAM scalar/CTA reads, data/guards/completions, FIFO pipeline/ring wrap, lane ownership and independent Graph checks passed.
- Real invalid-rkey FIFO CQ failure, sticky refusal, a healthy independent lane and failed-resource retention passed in a disposable process.
- Disabled consumer compiled without DOCA include paths and ran its CUDA stub checks with no DOCA library path.
- Copyright, changed-file pre-commit hooks and whitespace passed.

`results/current-main-validation.json` records binary hashes and exact native PASS lines. Native checks use a matching debugoptimized test stack with inherited test warnings allowed; production release builds retain Werror. The disabled consumer needs a GPU for its kernel; it is not a host-only test.

The initial selective test build omitted an adjacent CUDA device-ops DSO. A mixed-old-helper run is excluded; the full fresh stack was retried only after identifying and building that missing runtime target. Runtime checks then explicitly prioritized and verified all fresh NIXL helper DSOs. Earlier missing-mock/PIC/RDMA test-build setup errors were resolved without production source changes; their logs remain in the private recovery package.

Ordinary stream-pool/attached cross-node regression evidence comes from e56 and was not repeated as a new current-head result. MCST-required hardware, peer-crash recovery, full framework outer-Graph lifecycle and model serving remain unqualified. See the root README for the separate application-level multi-rail performance comparison.
