"""Synthetic two-node ABI-3 smoke. Coordinate only through a private directory.

The owner writes a metadata file and waits for a stop file. Copy that file to
the reader with a trusted channel; start the reader, wait for it to finish, then
create the stop file on the owner. Synchronization and shutdown are mandatory:
metadata exposes remote memory keys and this test has no lease server.
"""
import argparse
import base64
import ctypes as c
import json
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("role", choices=("owner", "reader"))
p.add_argument("--library", required=True)
p.add_argument("--port", type=int, required=True)
p.add_argument("--metadata", type=Path, required=True)
p.add_argument("--stop", type=Path)
p.add_argument("--timeout", type=float, default=120)
a = p.parse_args()
cuda = c.CDLL("libcudart.so.12")
lib = c.CDLL(a.library, mode=c.RTLD_GLOBAL)
P, U, Z = c.c_void_p, c.c_uint64, c.c_size_t
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
ck(cuda.cudaSetDevice(0), "select GPU")
context = None
table, arena, ids_dev, stats_dev = P(), P(), P(), P()
count, rows, width = 32, 64, 264
ids = (c.c_int64 * count)(*(i * 7 % rows for i in range(count)))
try:
    if a.role == "owner":
        if a.stop is None or a.stop.exists() or a.metadata.exists():
            raise ValueError("owner requires fresh metadata and stop paths")
        ck(cuda.cudaMallocHost(c.byref(table), rows * width), "allocate owner")
        values = (c.c_uint8 * (rows * width)).from_address(table.value)
        for i in range(rows * width):
            values[i] = (i * 13 + 17) % 251
        context = lib.engram_create(0, a.port, b"rebuilt-cross-owner")
        if not context:
            raise RuntimeError(lib.engram_last_error().decode())
        addresses, lengths = (U * 1)(table.value), (U * 1)(rows * width)
        ok(lib.engram_register(context, 1, addresses, lengths, 1), "register owner")
        size = ok(lib.engram_metadata(context, None, 0), "metadata size")
        metadata = c.create_string_buffer(size)
        ok(lib.engram_metadata(context, metadata, size), "metadata")
        with a.metadata.open("x") as f:
            json.dump({"md": base64.b64encode(metadata.raw).decode(),
                       "address": table.value, "length": rows * width}, f)
        a.metadata.chmod(0o600)
        print("OWNER_READY", flush=True)
        deadline = time.monotonic() + a.timeout
        while not a.stop.exists():
            if time.monotonic() > deadline:
                # A live reader may still access this MR. Retain it by keeping
                # this owner alive until its operator signals a safe stop.
                print("WAITING_FOR_SAFE_STOP", flush=True)
                deadline = float("inf")
            time.sleep(0.1)
    else:
        info = json.loads(a.metadata.read_text())
        blob = base64.b64decode(info["md"], validate=True)
        md = c.create_string_buffer(blob)
        for ptr, size in ((arena, 16384), (ids_dev, c.sizeof(ids)), (stats_dev, 64)):
            ck(cuda.cudaMalloc(c.byref(ptr), size), "allocate reader")
        ck(cuda.cudaMemcpy(ids_dev, c.cast(ids, P), c.sizeof(ids), 1), "copy IDs")
        context = lib.engram_create(0, a.port, b"rebuilt-cross-reader")
        if not context:
            raise RuntimeError(lib.engram_last_error().decode())
        ok(lib.engram_register(context, 0, (U * 1)(arena.value), (U * 1)(16384), 1),
           "register reader")
        ok(lib.engram_connect(context, md, len(blob), (U * 1)(info["address"]),
                              (U * 1)(info["length"]), 1, 0), "connect")
        ok(lib.engram_fetch_row264(context, ids_dev.value, count, 0, 0, 12288,
                                   0, 0, 32, stats_dev.value, 0, 0, rows), "fetch")
        ck(cuda.cudaDeviceSynchronize(), "finish fetch")
        data, stats = (c.c_uint8 * (count * width))(), (U * 8)()
        ck(cuda.cudaMemcpy(c.cast(data, P), arena, c.sizeof(data), 2), "read output")
        ck(cuda.cudaMemcpy(c.cast(stats, P), stats_dev, c.sizeof(stats), 2), "read status")
        expected = bytes(((row * width + col) * 13 + 17) % 251
                         for row in ids for col in range(width))
        if stats[0] or stats[4] != count or bytes(data) != expected:
            raise RuntimeError(f"validation failed: error={stats[0]} completions={stats[4]}")
        print(f"PASS cross-node rebuilt ABI-3: {count} packed rows, {stats[4]} completions")
finally:
    if context:
        ok(lib.engram_destroy(context), "destroy context")
    for ptr in (stats_dev, ids_dev, arena):
        if ptr.value:
            ck(cuda.cudaFree(ptr), "free reader")
    if table.value:
        ck(cuda.cudaFreeHost(table), "free owner")
