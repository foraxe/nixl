"""Read-only check of the pinned NCCL GDAKI registration prerequisite."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=7)
args = parser.parse_args()
cuda = C.CDLL("libcuda.so.1")
init = cuda.cuInit(0)
value, driver = C.c_int(-1), C.c_int(-1)
query = cuda.cuDeviceGetAttribute(C.byref(value), 124, args.gpu) if init == 0 else init
cuda.cuDriverGetVersion(C.byref(driver))
raw_uuid = C.create_string_buffer(16)
uuid_status = cuda.cuDeviceGetUuid(C.byref(raw_uuid), args.gpu) if init == 0 else init
peer = Path("/sys/module/nvidia_peermem").exists()
blocked = query == 0 and value.value == 0 and not peer
result = dict(gpu=args.gpu, gpu_uuid=str(uuid.UUID(bytes=raw_uuid.raw)) if uuid_status == 0 else None,
              cuInit=init, attribute=124, attribute_name="CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED",
              query_status=query, value=value.value, driver_api=driver.value,
              driver_libraries=sorted({x.split()[-1] for x in open("/proc/self/maps") if "libcuda.so" in x}),
              kernel=os.uname().release, nvidia_peermem=peer,
              nv_peer_mem=Path("/sys/module/nv_peer_mem").exists(),
              verdict="BLOCKED" if blocked else "CHECK_REMAINING_REQUIREMENTS")
print(json.dumps(result, indent=2))
raise SystemExit(3 if blocked else 0)
