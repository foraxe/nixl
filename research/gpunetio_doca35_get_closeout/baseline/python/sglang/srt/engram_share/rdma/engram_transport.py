"""ctypes bridge to the private NIXL native-GET shim; buffers remain Python-owned."""

import ctypes as C
import uuid

import torch


class Context:
    def __init__(self, library, gpu, port):
        self.gpu = gpu
        # The backend is dlopened later and must see the same host/Abseil symbols.
        self.library = C.CDLL(str(library), mode=C.RTLD_GLOBAL)
        lib = self.library
        lib.engram_abi_version.restype = C.c_uint32
        if lib.engram_abi_version() != 3:
            raise RuntimeError("shared Engram shim/wrapper ABI mismatch")
        lib.engram_last_error.restype = C.c_char_p
        lib.engram_create.argtypes = [C.c_uint32, C.c_uint16, C.c_char_p]
        lib.engram_create.restype = C.c_void_p
        lib.engram_register.argtypes = [
            C.c_void_p,
            C.c_int,
            C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint64),
            C.c_uint32,
        ]
        lib.engram_metadata.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        lib.engram_metadata.restype = C.c_int64
        lib.engram_connect.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
            C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint64),
            C.c_uint32,
            C.c_uint32,
        ]
        lib.engram_destroy.argtypes = [C.c_void_p]
        lib.engram_record_consumed.argtypes = [C.c_void_p, C.c_uint64]
        lib.engram_fetch_row264.argtypes = [
            C.c_void_p,
            C.c_uint64,
            C.c_uint32,
            C.c_size_t,
            C.c_size_t,
            C.c_size_t,
            C.c_uint32,
            C.c_uint32,
            C.c_uint32,
            C.c_uint64,
            C.c_uint64,
            C.c_uint64,
            C.c_uint64,
        ]
        self.name = "shared-engram-" + uuid.uuid4().hex
        self.handle = lib.engram_create(gpu, port, self.name.encode())
        if not self.handle:
            raise RuntimeError(lib.engram_last_error().decode())
        self.buffers = []

    def check(self, rc):
        if rc < 0:
            raise RuntimeError(self.library.engram_last_error().decode())
        return rc

    def register(self, tensors, host):
        if self.buffers:
            raise RuntimeError("context already has a registration")
        if any(not t.is_contiguous() for t in tensors):
            raise ValueError("registration tensors must be contiguous")
        if host and any(t.is_cuda or not t.is_pinned() for t in tensors):
            raise ValueError("owner tensors must be pinned Host allocations")
        if not host and (len(tensors) != 1 or not tensors[0].is_cuda):
            raise ValueError("reader requires one CUDA arena")
        self.addresses = (C.c_uint64 * len(tensors))(*(t.data_ptr() for t in tensors))
        self.lengths = (C.c_uint64 * len(tensors))(
            *(t.numel() * t.element_size() for t in tensors)
        )
        self.check(
            self.library.engram_register(
                self.handle, int(host), self.addresses, self.lengths, len(tensors)
            )
        )
        self.buffers = list(tensors)

    def metadata(self):
        size = self.check(self.library.engram_metadata(self.handle, None, 0))
        out = C.create_string_buffer(size)
        self.check(self.library.engram_metadata(self.handle, out, size))
        return out.raw

    def connect_raw(self, blob, addresses, lengths, remote_gpu):
        if (
            not addresses
            or len(addresses) != len(lengths)
            or any(n <= 0 for n in lengths)
        ):
            raise ValueError("invalid remote registration descriptors")
        addr = (C.c_uint64 * len(addresses))(*addresses)
        size = (C.c_uint64 * len(lengths))(*lengths)
        md = C.create_string_buffer(blob)
        self.check(
            self.library.engram_connect(
                self.handle, md, len(blob), addr, size, len(addresses), remote_gpu
            )
        )
        self.remote_count = len(addresses)

    def close(self):
        if self.handle:
            self.check(self.library.engram_destroy(self.handle))
            self.handle = None
            self.buffers.clear()

    def fetch_row264(self, ids, payload, dump, table, stats, lo, hi):
        if (
            ids.dtype != torch.int64
            or not ids.is_cuda
            or not ids.is_contiguous()
            or not 0 < ids.numel()
            or stats.dtype != torch.uint64
            or stats.numel() != 8
            or not stats.is_cuda
            or not stats.is_contiguous()
            or ids.device.index != self.gpu
            or stats.device.index != self.gpu
            or not 0 <= table < self.remote_count
            or not 0 <= lo <= hi
        ):
            raise ValueError("invalid graph READ inputs")
        stream = torch.cuda.current_stream(self.gpu)
        self.check(
            self.library.engram_fetch_row264(
                self.handle,
                ids.data_ptr(),
                ids.numel(),
                payload,
                0,
                dump,
                table,
                table,
                512,
                stats.data_ptr(),
                stream.cuda_stream,
                lo,
                hi,
            )
        )
        ids.record_stream(stream)
        stats.record_stream(stream)
        self.last_buffers = (ids, stats)

    def retire_idle_peer(self, name):
        if not self.handle or not isinstance(name, str) or not name:
            raise ValueError("retirement needs a live owner and peer name")
        fn = self.library.engram_retire_idle_peer
        fn.argtypes = [C.c_void_p, C.c_char_p]
        self.check(fn(self.handle, name.encode()))

    def record_consumed(self):
        self.check(
            self.library.engram_record_consumed(
                self.handle, torch.cuda.current_stream(self.gpu).cuda_stream
            )
        )
