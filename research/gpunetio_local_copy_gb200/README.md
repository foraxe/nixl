# GPU-kernel local copy: GB200 handoff snapshot

Start with [GB200_HANDOFF.md](GB200_HANDOFF.md).

This directory contains the standalone workdir, compact H20 measurements, exact
NIXL E and NCCL source snapshots, and the original design context. No tar,
H20 binary, SDK install, PyTorch environment or large profiler trace is required
to inspect the handoff. GB200 porting and native GIN comparison remain to do.

```bash
git clone --depth 1 --branch gpunetio/kernel-local-copy-gb200-20261024 https://github.com/foraxe/nixl.git
cd nixl/research/gpunetio_local_copy_gb200
```

The branch itself is based on E e932593300366073dfd5f3333ef1d42f6abfbc0a.
The nested dependencies/nixl-e tree is the same frozen build input, retained to
preserve the self-contained handoff layout. NCCL is pinned in SOURCE_PINS.json.
Dependency source/license files are preserved. No GB200 performance is claimed.

Public-copy sanitization removes internal pod/node/IP/UUID and personal-path
identifiers from reports and logs. Numeric measurements and implementation
source are unchanged. Original private recovery records remain locally; the
public version's placeholders are not executable host access instructions.
Measured library hashes remain historical binary identities, not included binaries.
PACKAGE_SHA256SUMS describes this public folder, including sanitized documents.
