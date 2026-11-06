# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run two local GPU ranks through mlx5_0; no NVLink/LSA transfer fallback."""
import os
from pathlib import Path
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
TOOLS = Path("/workspace/gpunetio/kernel-copy-tools")
PREFIX = TOOLS / "gdaki-compare/install"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 run_local.py UNIQUE_RUN_TAG")
    env = dict(
        os.environ,
        RUN_TAG=sys.argv[1],
        OWNER_ADDR="127.0.0.1",
        NCCL_ROOT="/workspace/gpunetio/nccl-core-workload/build",
        NIXL_SHIM=os.environ.get(
            "NIXL_SHIM", str(TOOLS / "gdaki-compare/engram_native_nixl35.so")
        ),
        BENCH_BINARY=os.environ.get(
            "BENCH_BINARY", str(TOOLS / "gdaki-compare/gin_nixl_bench_gdaki")
        ),
        NCCL_NET="IB",
        NCCL_IB_HCA="=mlx5_0",
        NCCL_IB_GID_INDEX="3",
        NCCL_LSA_TEAM_SIZE="1",
        NCCL_P2P_DISABLE="1",
        NIXL_ENABLE_GPUNETIO_DEVICE_API="1",
        GPUNETIO_DEVICE_API_RDMA="mlx5_0",
        GPUNETIO_DEVICE_API_OOB_INTERFACE="lo",
        GPUNETIO_DEVICE_API_GID_INDEX="3",
    )
    lib = PREFIX / "lib/aarch64-linux-gnu"
    env["NIXL_PLUGIN_DIR"] = os.environ.get("NIXL_PLUGIN_DIR", str(lib / "plugins"))
    env["LD_LIBRARY_PATH"] = ":".join(
        map(
            str,
            [
                lib / "plugins",
                lib,
                TOOLS / "doca35/root/opt/mellanox/doca/lib/aarch64-linux-gnu",
                TOOLS / "doca35/root/usr/lib/aarch64-linux-gnu",
                "/usr/local/cuda/lib64",
                env.get("LD_LIBRARY_PATH", ""),
            ],
        )
    )
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if occupied:
        raise SystemExit("GPUs occupied; refusing to run: " + occupied)
    ready = threading.Event()
    children, threads = [], []

    def output(proc, rank):
        for line in proc.stdout:
            print(f"rank{rank}: {line}", end="", flush=True)
            if rank == 0 and "BOOTSTRAP_READY" in line:
                ready.set()

    try:
        for rank, role in enumerate(("owner", "reader")):
            rank_env = dict(
                env, CUDA_VISIBLE_DEVICES=str(rank), NIXL_PORT=str(29581 + rank)
            )
            cpu_set = "0-7" if rank == 0 else "8-15"
            proc = subprocess.Popen(
                [
                    "taskset",
                    "-c",
                    cpu_set,
                    "bash",
                    str(HERE / "run_existing_rank.sh"),
                    role,
                ],
                env=rank_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            children.append(proc)
            thread = threading.Thread(target=output, args=(proc, rank), daemon=True)
            thread.start()
            threads.append(thread)
            if rank == 0 and not ready.wait(30):
                raise RuntimeError("owner did not reach bootstrap in 30 seconds")
        statuses = [p.wait(timeout=360) for p in reversed(children)]
        for thread in threads:
            thread.join(timeout=5)
        if any(statuses):
            raise SystemExit("rank failure: " + str(statuses))
    finally:
        # Only terminate process groups created by this invocation, never foreign jobs.
        import signal

        for proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    main()
