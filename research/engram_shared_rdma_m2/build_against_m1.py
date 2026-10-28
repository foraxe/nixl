#!/usr/bin/env python3
"""Build only the research shim against a retained, matching NIXL build."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


ap = argparse.ArgumentParser()
ap.add_argument("--build", type=Path, required=True)
args = ap.parse_args()
build = args.build.resolve()
commands = json.loads((build / "compile_commands.json").read_text())
old = next(x for x in commands if x["file"].endswith("shared_engram_m1.cu"))
original = shlex.split(old["command"])
flags = []
i = 0
while i < len(original):
    value = original[i]
    if value in ("-o", "-MT", "-MF"):
        i += 2
    elif value in ("-MD", "-c") or value.endswith("shared_engram_m1.cu"):
        i += 1
    else:
        flags.append(value)
        i += 1
output = build / "research/engram_shared_rdma_m2/libengram_native.so"
output.parent.mkdir(parents=True, exist_ok=True)
source = Path(__file__).with_name("engram_native.cu").resolve()
command = flags + ["-Xcompiler=-fPIC", "-shared", "--cudart=shared", str(source),
                   "-Lsrc/core", "-lnixl", "-Lsrc/infra", "-lnixl_build",
                   "-Xlinker", "-rpath", "-Xlinker",
                   "$ORIGIN:$ORIGIN/../../src/core:$ORIGIN/../../src/infra",
                   "-L/opt/mellanox/doca/lib/x86_64-linux-gnu", "-ldoca_gpunetio",
                   "-o", str(output)]
with (output.parent / ".build.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    (output.parent / "build-command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, cwd=build, check=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("source changed during build; binary is unqualified")
    identity = {"source_sha256": source_hash,
                "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "core_sha256": hashlib.sha256((build / "src/core/libnixl.so").read_bytes()).hexdigest()}
    (output.parent / "build-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
print(output)
