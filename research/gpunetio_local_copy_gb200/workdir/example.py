import ctypes as C
import torch
from local_copy import LocalCopy

torch.cuda.set_device(7)
src = torch.arange(128, dtype=torch.float32, pin_memory=True)
dst = torch.empty(128, dtype=torch.float32, device="cuda:7")
out = torch.empty_like(dst)
error = torch.empty((), dtype=torch.int32, device=dst.device)

with LocalCopy(src, dst) as copy:
    launch = copy.lib.lc_example
    launch.argtypes = [C.c_void_p] * 6
    assert launch(copy.device_handle, src.data_ptr(), dst.data_ptr(), out.data_ptr(),
                  error.data_ptr(), torch.cuda.current_stream().cuda_stream) == 0
    torch.cuda.synchronize()
    assert error.item() == 0
    torch.testing.assert_close(out.cpu(), src * 1.5, rtol=0, atol=0)
print("PASS: one kernel issued GET, waited, and transformed 128 tensor elements")
