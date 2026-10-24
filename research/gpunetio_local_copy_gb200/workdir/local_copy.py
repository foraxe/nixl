"""One-time tensor preparation; device Copy is defined in local_copy.cuh."""
import ctypes as C
from pathlib import Path

_quarantine = []


class LocalCopy:
    def __init__(self, src, dst, nic="mlx5_bond_0", gid=3, library=None):
        if src.device.type != "cpu" or not src.is_pinned() or not src.is_contiguous():
            raise ValueError("source must be a contiguous pinned CPU tensor")
        if dst.device.type != "cuda" or not dst.is_contiguous():
            raise ValueError("destination must be a contiguous CUDA tensor")
        self.src, self.dst = src, dst
        self.lib = C.CDLL(str(library or Path(__file__).with_name("liblocal_copy.so")))
        self.lib.lc_error.restype = C.c_char_p
        self.lib.lc_open.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                                    C.c_int, C.c_char_p, C.c_int]
        self.lib.lc_open.restype = C.c_void_p
        self.lib.lc_device.argtypes = [C.c_void_p]
        self.lib.lc_device.restype = C.c_void_p
        self.lib.lc_close.argtypes = [C.c_void_p]
        self.lib.lc_close.restype = C.c_int
        self.handle = self.lib.lc_open(src.data_ptr(), src.numel() * src.element_size(),
            dst.data_ptr(), dst.numel() * dst.element_size(), dst.device.index, nic.encode(), gid)
        if not self.handle:
            raise RuntimeError(self.lib.lc_error().decode())
        self.device_handle = self.lib.lc_device(self.handle)

    def close(self):
        if self.handle:
            self.device_handle = None
            if self.lib.lc_close(self.handle):
                raise RuntimeError(self.lib.lc_error().decode())
            self.handle = None
            self.device_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "handle", None):
            try:
                self.close()
            except Exception:
                # Failed teardown must not drop the tensors backing live DMA.
                _quarantine.append((self.lib, self.handle, self.src, self.dst))
