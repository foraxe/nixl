"""Opt-in 4 x 4-QP adapter for the frozen SGLang ReadPlan.

Owner metadata must come from the matching four-rail ABI3 facade. A single
borrowed group is shared by layers; graph ownership remains with ReadPlan.
"""
import ctypes as C
import struct
import uuid

import torch
from sglang.srt.engram_share.rdma import dedup_rows
from sglang.srt.engram_share.rdma.engram_transport import Context
from sglang.srt.engram_share.rdma.read_group import ReadGroup
from sglang.srt.engram_share.rdma.read_plan import ReadPlan, _merge_stats


def unpack_metadata(blob):
    blob = bytes(blob)
    if not blob.startswith(b"NIXL4R1"):
        raise ValueError("expected four-rail owner metadata")
    off = 7
    def take32():
        nonlocal off
        if off + 4 > len(blob):
            raise ValueError("truncated four-rail metadata")
        v = struct.unpack_from("<I", blob, off)[0]
        off += 4
        return v
    if take32() != 4:
        raise ValueError("owner must use four QPs per rail")
    parts = []
    for _ in range(4):
        n = take32()
        if not n or off + n > len(blob):
            raise ValueError("invalid rail metadata length")
        parts.append(blob[off:off+n])
        off += n
    if off != len(blob):
        raise ValueError("trailing rail metadata")
    return parts


def _rail_context(library, gpu, port, rail, cls=Context):
    # Set up the same ctypes interface as Context, but create its native handle
    # on an explicitly named rail with a four-channel native pool.
    c = object.__new__(cls)
    c.gpu = gpu
    c.library = C.CDLL(str(library), mode=C.RTLD_GLOBAL)
    lib = c.library
    lib.engram_abi_version.restype = C.c_uint32
    lib.engram_last_error.restype = C.c_char_p
    if lib.engram_abi_version() != 3:
        raise RuntimeError("shared Engram shim/wrapper ABI mismatch")
    lib.engram_create_on_rail.argtypes = [C.c_uint32, C.c_uint16, C.c_char_p, C.c_char_p, C.c_uint32]
    lib.engram_create_on_rail.restype = C.c_void_p
    lib.engram_register.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64), C.c_uint32]
    lib.engram_connect.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64), C.c_uint32, C.c_uint32]
    lib.engram_destroy.argtypes = [C.c_void_p]
    lib.engram_record_consumed.argtypes = [C.c_void_p, C.c_uint64]
    lib.engram_fetch_row264.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32, C.c_size_t, C.c_size_t, C.c_size_t, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint64]
    lib.engram_metadata.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    lib.engram_metadata.restype = C.c_int64
    c.name = "shared-engram-" + uuid.uuid4().hex
    c.handle = lib.engram_create_on_rail(gpu, port, c.name.encode(), rail.encode(), 4)
    if not c.handle:
        raise RuntimeError(lib.engram_last_error().decode())
    c.buffers = []
    return c


class FirstRailContext(Context):
    """Independent decode context for a four-rail owner's metadata.

    Uses channel zero of its own first-rail pool, with separately registered
    buffers and retirement state. It does not borrow the prefetch group.
    """
    def __new__(cls, library, gpu, port, rail):
        return _rail_context(library, gpu, port, rail, cls)

    def __init__(self, library, gpu, port, rail):
        # Native context is initialized in __new__, without a default backend.
        pass

    def connect_raw(self, blob, addresses, lengths, remote_gpu):
        return super().connect_raw(unpack_metadata(blob)[0], addresses, lengths, remote_gpu)


def _group_view(contexts, handle, lanes):
    # Reuse ReadGroup's argument checking and fetch/stream retention unchanged.
    g = ReadGroup.__new__(ReadGroup)
    g.contexts = [c for c in contexts for _ in range(4)][:lanes]
    g.gpu, g.lib, g.handle, g.active = contexts[0].gpu, contexts[0].library, handle, 0
    args = [C.c_void_p, C.c_uint64, C.c_uint32, C.c_size_t, C.c_size_t,
            C.c_uint32, C.c_uint32, C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint64, C.c_int]
    g.lib.engram_read_group_fetch.argtypes = args
    g.lib.engram_read_group_fetch_mapped.argtypes = args + [C.c_uint64]
    g.lib.engram_read_group_fetch_compacted.argtypes = args + [C.c_uint64] * 3
    g.lib.engram_read_group_destroy.argtypes = [C.c_void_p]
    return g


class MultiRailReadPlan(ReadPlan):
    def __init__(self, library, gpu, port, lanes=16, max_queries=196608,
                 graph_sizes=(49152,), *, rails, weights=None, dispatch="delivered"):
        if lanes != 16 or len(rails) != 4 or len(set(rails)) != 4 or any(not x for x in rails):
            raise ValueError("requires 16 lanes and four distinct rails")
        if port < 0 or port > 65532:
            raise ValueError("four consecutive ports required")
        if weights is not None and (len(weights) != 4 or any(not isinstance(x, int) or x <= 0 for x in weights) or sum(weights) > 10000):
            raise ValueError("invalid rail weights")
        if dispatch not in ("delivered", "optimized"):
            raise ValueError("dispatch must be delivered or optimized")
        self.dispatch = dispatch
        self.gpu, self.lanes, self.maximum = gpu, lanes, max_queries
        self.graph_sizes, self.weights = set(graph_sizes), weights
        self.graph_replays, self.last_active = 0, 0
        self.contexts, self.groups, self.plans, self.last_map = [], {}, {}, {}
        self.stream, self.supports_row264, self._group = None, True, None
        try:
            for i, rail in enumerate(rails):
                self.contexts.append(_rail_context(library, gpu, port+i, rail))
        except BaseException:
            for c in self.contexts:
                c.close()
            raise

    def connect_raw(self, blob, addresses, lengths, remote_gpu):
        if self._group is not None:
            raise RuntimeError("already connected")
        for c, part in zip(self.contexts, unpack_metadata(blob)):
            c.connect_raw(part, addresses, lengths, remote_gpu)
        self.layers = len(addresses)
        self.span = self.arena.numel() // self.layers
        self.stats = torch.zeros((16, 8), dtype=torch.uint64, device="cuda")
        self.workspace = []
        for _ in range(self.layers):
            scratch, reps = dedup_rows.workspace(self.maximum, "cuda")
            self.workspace.append(dict(ids=torch.empty(self.maximum, dtype=torch.int64, device="cuda"),
                                       scratch=scratch, reps=reps, work=torch.empty_like(reps)))
        lib = self.contexts[0].library
        lib.engram_multi_group_create.argtypes = [C.POINTER(C.c_void_p), C.c_uint32, C.c_uint32, C.c_uint32]
        lib.engram_multi_group_create.restype = C.c_void_p
        lib.engram_read_group_destroy.argtypes = [C.c_void_p]
        lib.engram_multi_qualify_depth2.argtypes = [C.c_void_p, C.c_uint32]
        lib.engram_read_group_pipeline.argtypes = [C.c_void_p, C.c_uint32]
        lib.engram_multi_set_small_cutoff.argtypes = [C.c_void_p, C.c_uint32]
        lib.engram_multi_set_weights.argtypes = [C.c_void_p] + [C.c_uint32] * 4
        handles = (C.c_void_p * 4)(*(c.handle for c in self.contexts))
        self._group = lib.engram_multi_group_create(handles, 4, 4, 4095)
        if not self._group:
            raise RuntimeError(lib.engram_last_error().decode())
        check = self.contexts[0].check
        check(lib.engram_multi_qualify_depth2(self._group, 512))
        check(lib.engram_read_group_pipeline(self._group, 2))
        check(lib.engram_multi_set_small_cutoff(self._group, 16 if self.dispatch == "optimized" else 4095))
        if self.weights:
            check(lib.engram_multi_set_weights(self._group, *self.weights))
        for li in range(self.layers):
            for q in (1, 4, 16):
                self.groups[(li, q)] = _group_view(self.contexts, self._group, q)

    def fetch(
        self,
        ids,
        count,
        payload_offset,
        scale_offset,
        dump_offset,
        window_rows,
        stats,
        weight_index=0,
        scale_index=0,
        row_lo=0,
        row_hi=2**63 - 1,
        row264=False,
    ):
        if not row264 or weight_index != scale_index:
            raise ValueError("planned reader needs row264")
        if count != ids.numel() or count > self.maximum:
            raise ValueError("planned query count")
        current = torch.cuda.current_stream(self.gpu)
        if self.stream is None:
            self.stream = current.cuda_stream
        if self.stream != current.cuda_stream:
            raise RuntimeError("all planned operations must stay on oneproducerstream")
        ids.record_stream(current)
        li = weight_index
        q = (self.lanes if count >= 4096 else
         4 if self.dispatch == "optimized" and count > 16 else 1)
        self.last_active = q
        dedup = count >= 4096
        w = self.workspace[li]
        self.last_map[li] = w["reps"][:count] if dedup else None
        key = (
            li,
            count,
            q,
            payload_offset,
            dump_offset,
            window_rows,
            row_lo,
            row_hi,
            stats.data_ptr(),
        )
        if count in self.graph_sizes:
            stable = w["ids"][:count]
            stable.copy_(ids)
            if key not in self.plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "prime the communication plan before outercapture"
                    )
                for _ in range(2):
                    self._enqueue(
                        li,
                        q,
                        stable,
                        payload_offset,
                        dump_offset,
                        window_rows,
                        row_lo,
                        row_hi,
                        dedup,
                    )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=current):
                    self._enqueue(
                        li,
                        q,
                        stable,
                        payload_offset,
                        dump_offset,
                        window_rows,
                        row_lo,
                        row_hi,
                        dedup,
                    )
                    _merge_stats[(1,)](self.stats, stats, q, num_warps=1)
                self.plans[key] = (graph, stats)
            self.plans[key][0].replay()
            self.graph_replays += 1
        else:
            self._enqueue(
                li,
                q,
                ids,
                payload_offset,
                dump_offset,
                window_rows,
                row_lo,
                row_hi,
                dedup,
            )
            _merge_stats[(1,)](self.stats, stats, q, num_warps=1)

    def record_consumed(self):
        # Four lanes share the first rail context; large reads use all rails.
        count = 4 if self.last_active == 16 else int(self.last_active != 0)
        for c in self.contexts[:count]:
            c.record_consumed()

    def close(self):
        torch.cuda.synchronize()
        self.plans.clear()
        self.groups.clear()
        if self._group:
            self.contexts[0].check(self.contexts[0].library.engram_read_group_destroy(self._group))
            self._group = None
        for c in self.contexts:
            c.close()
