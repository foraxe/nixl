"""Subprocess-only reset-failure injection; leaves live kernel quarantined until process exit."""
import ctypes as C
import argparse
import json
import os
from pathlib import Path
import time
import weakref
import torch
import local_copy

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=7)
parser.add_argument("--nic", default="mlx5_bond_0")
parser.add_argument("--gid", type=int, default=3)
parser.add_argument("--library", default=str(Path(__file__).with_name("liblocal_copy_testing.so")))
args = parser.parse_args()
torch.cuda.set_device(args.gpu)
src = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
dst = torch.empty(4096, dtype=torch.uint8, device=f"cuda:{args.gpu}")
result = torch.zeros(6, dtype=torch.int64, device=dst.device)
copy = local_copy.LocalCopy(src, dst, nic=args.nic, gid=args.gid, library=args.library)
copy.lib.lc_test_fail_reset.argtypes = [C.c_void_p]
copy.lib.lc_test_state.argtypes = [C.c_void_p]
copy.lib.lc_test_fail_reset(copy.handle)
probe = copy.lib.lc_probe
probe.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_int,
                  C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
assert probe(copy.device_handle, src.data_ptr(), dst.data_ptr(), 1, 1, 1, 1, 1, 11,
             result.data_ptr(), torch.cuda.current_stream().cuda_stream) == 0
deadline = time.monotonic() + 3
while copy.lib.lc_test_state(copy.handle) != -1:
    assert time.monotonic() < deadline
    time.sleep(0.001)
start = time.monotonic()
try:
    copy.close()
    raise AssertionError("unsafe successful close")
except RuntimeError as e:
    assert "quarantined" in str(e), e
    message = str(e)
elapsed = time.monotonic() - start
assert elapsed < 5.5 and copy.handle
source_ref, dest_ref = weakref.ref(src), weakref.ref(dst)
del src, dst, copy
assert source_ref() is not None and dest_ref() is not None
assert local_copy._quarantine
print(json.dumps(dict(status="PASS", scenario="injected reset failure",
                      close_seconds=elapsed, retained=True, message=message)), flush=True)
# No CUDA synchronization/finalizers: this disposable process owns the gated kernel.
os._exit(0)
