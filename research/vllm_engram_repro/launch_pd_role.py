"""Launch one explicit vLLM P or D role for a matched Engram experiment."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("prefill", "decode"), required=True)
    p.add_argument("--mode", choices=("native", "remote"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument(
        "--devices", required=True, help="CUDA-visible GPU UUIDs or ordinals"
    )
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    p.add_argument("--served-model-name", default="engram-model")
    p.add_argument("--moe-backend", help="optional vLLM MoE backend")
    p.add_argument("--ready", type=Path)
    p.add_argument("--library", type=Path)
    p.add_argument("--reader-port-base", type=int)
    p.add_argument("--rdma-device")
    p.add_argument(
        "--extra", action="append", default=[], help="one additional CLI argument"
    )
    args = p.parse_args()
    if args.mode == "remote" and not all(
        (args.ready, args.library, args.reader_port_base, args.rdma_device)
    ):
        p.error(
            "remote mode requires --ready, --library, --reader-port-base, --rdma-device"
        )
    if len(args.devices.split(",")) != args.tp:
        p.error("--devices must provide exactly --tp GPUs")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("ENGRAM_SHARED_", "ENGRAM_RESEARCH_")):
            env.pop(key)
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    engram = {"cpu_offload": True, "dp_shared_memory": False}
    if args.mode == "remote":
        engram["rdma_provider"] = "vllm_rdma_provider"
        env.update(
            ENGRAM_SHARED_READY=str(args.ready),
            ENGRAM_SHARED_LIBRARY=str(args.library),
            ENGRAM_SHARED_MODEL=str(args.model),
            ENGRAM_SHARED_READER_PORT_BASE=str(args.reader_port_base),
            GPUNETIO_DEVICE_API_RDMA=args.rdma_device,
            GPUNETIO_DEVICE_API_GID_INDEX="3",
            GPUNETIO_DEVICE_API_OOB_INTERFACE="eth0",
        )
    kv_config = {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer" if args.role == "prefill" else "kv_consumer",
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model),
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "--language-model-only",
        "--shutdown-timeout",
        "60",
        "--load-format",
        "safetensors",
        "--safetensors-load-strategy",
        "lazy",
        "--engram-config",
        json.dumps(engram),
        "--kv-transfer-config",
        json.dumps(kv_config),
    ]
    if args.moe_backend:
        command.extend(
            ("--kernel-config", json.dumps({"moe_backend": args.moe_backend}))
        )
    command.extend(args.extra)
    print(
        json.dumps(
            {
                "event": "launch",
                "role": args.role,
                "mode": args.mode,
                "tp": args.tp,
                "max_model_len": args.max_model_len,
            }
        ),
        flush=True,
    )
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
