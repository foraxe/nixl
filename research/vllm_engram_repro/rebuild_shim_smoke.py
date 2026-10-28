"""Exercise the rebuilt ABI-3 shim with synthetic pinned Host rows.

Two GPUs are required. This is a correctness-only local-network smoke, not a
performance test or a substitute for a cross-node/model-level validation.
"""
import argparse
import ctypes as c

p = argparse.ArgumentParser()
p.add_argument("--library", required=True)
p.add_argument("--owner-port", type=int, required=True)
p.add_argument("--reader-port", type=int, required=True)
a = p.parse_args()
cuda = c.CDLL("libcudart.so.12")
lib = c.CDLL(a.library, mode=c.RTLD_GLOBAL)
P, U, Z = c.c_void_p, c.c_uint64, c.c_size_t
for name in ("cudaSetDevice", "cudaDeviceSynchronize", "cudaMalloc", "cudaMallocHost",
             "cudaFree", "cudaFreeHost", "cudaMemcpy"):
    getattr(cuda, name).restype = c.c_int
cuda.cudaSetDevice.argtypes = [c.c_int]
cuda.cudaMalloc.argtypes = [c.POINTER(P), Z]
cuda.cudaMallocHost.argtypes = [c.POINTER(P), Z]
cuda.cudaFree.argtypes = [P]
cuda.cudaFreeHost.argtypes = [P]
cuda.cudaMemcpy.argtypes = [P, P, Z, c.c_int]
lib.engram_last_error.restype = c.c_char_p
lib.engram_abi_version.restype = c.c_uint32
lib.engram_create.argtypes = [c.c_uint32, c.c_uint16, c.c_char_p]
lib.engram_create.restype = P
lib.engram_register.argtypes = [P, c.c_int, c.POINTER(U), c.POINTER(U), c.c_uint32]
lib.engram_metadata.argtypes = [P, P, Z]
lib.engram_metadata.restype = c.c_int64
lib.engram_connect.argtypes = [P, P, Z, c.POINTER(U), c.POINTER(U), c.c_uint32, c.c_uint32]
lib.engram_fetch_row264.argtypes = [P, U, c.c_uint32, Z, Z, Z, c.c_uint32,
                                   c.c_uint32, c.c_uint32, U, U, U, U]
lib.engram_destroy.argtypes = [P]

def ck(rc, where):
    if rc:
        raise RuntimeError(f"{where}: CUDA status {rc}")

def ok(rc, where):
    if rc < 0:
        raise RuntimeError(f"{where}: {lib.engram_last_error().decode()}")
    return rc

if lib.engram_abi_version() != 3:
    raise RuntimeError("expected ABI 3")
owner = reader = None
table = P()
arena = P()
ids_dev = P()
stats_dev = P()
count, rows, width = 32, 64, 264
ids = (c.c_int64 * count)(*(i * 7 % rows for i in range(count)))
try:
    ck(cuda.cudaSetDevice(0), "select owner GPU")
    ck(cuda.cudaMallocHost(c.byref(table), rows * width), "allocate owner")
    table_data = (c.c_uint8 * (rows * width)).from_address(table.value)
    for i in range(rows * width):
        table_data[i] = (i * 13 + 17) % 251
    owner = lib.engram_create(0, a.owner_port, b"rebuild-owner")
    if not owner:
        raise RuntimeError(lib.engram_last_error().decode())
    addresses, lengths = (U * 1)(table.value), (U * 1)(rows * width)
    ok(lib.engram_register(owner, 1, addresses, lengths, 1), "register owner")
    md_size = ok(lib.engram_metadata(owner, None, 0), "metadata size")
    metadata = c.create_string_buffer(md_size)
    ok(lib.engram_metadata(owner, metadata, md_size), "metadata")

    ck(cuda.cudaSetDevice(1), "select reader GPU")
    for ptr, size in ((arena, 16384), (ids_dev, c.sizeof(ids)), (stats_dev, 64)):
        ck(cuda.cudaMalloc(c.byref(ptr), size), "allocate reader")
    ck(cuda.cudaMemcpy(ids_dev, c.cast(ids, P), c.sizeof(ids), 1), "copy IDs")
    reader = lib.engram_create(1, a.reader_port, b"rebuild-reader")
    if not reader:
        raise RuntimeError(lib.engram_last_error().decode())
    ok(lib.engram_register(reader, 0, (U * 1)(arena.value), (U * 1)(16384), 1),
       "register reader")
    ok(lib.engram_connect(reader, metadata, md_size, addresses, lengths, 1, 0), "connect")
    ok(lib.engram_fetch_row264(reader, ids_dev.value, count, 0, 0, 12288,
                               0, 0, 32, stats_dev.value, 0, 0, rows), "fetch")
    ck(cuda.cudaDeviceSynchronize(), "finish fetch")
    data = (c.c_uint8 * (count * width))()
    stats = (U * 8)()
    ck(cuda.cudaMemcpy(c.cast(data, P), arena, c.sizeof(data), 2), "read output")
    ck(cuda.cudaMemcpy(c.cast(stats, P), stats_dev, c.sizeof(stats), 2), "read status")
    expected = bytes(table_data[row * width + col] for row in ids for col in range(width))
    if stats[0] or stats[4] != count or bytes(data) != expected:
        raise RuntimeError(f"validation failed: error={stats[0]} completions={stats[4]}")
    print(f"PASS rebuilt ABI-3: {count} packed rows, {stats[4]} completions")
finally:
    if reader:
        ck(cuda.cudaSetDevice(1), "select reader for cleanup")
        ok(lib.engram_destroy(reader), "destroy reader")
    for ptr in (stats_dev, ids_dev, arena):
        if ptr.value:
            ck(cuda.cudaFree(ptr), "free reader")
    if owner:
        ck(cuda.cudaSetDevice(0), "select owner for cleanup")
        ok(lib.engram_destroy(owner), "destroy owner")
    if table.value:
        ck(cuda.cudaFreeHost(table), "free owner")
