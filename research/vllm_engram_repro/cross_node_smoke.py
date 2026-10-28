"""Two-node READ smoke for the vLLM bridge using two tiny synthetic tables."""

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from engram_transport import Context
from owner_control import ControlError, OwnerServer

ROWS = 64
LAYERS = (1, 14)


def model_files(path):
    path.mkdir(parents=True, exist_ok=True)
    config = json.dumps({"engram_num_embeddings": [ROWS, ROWS]}).encode()
    index = b'{}'
    (path / "config.json").write_bytes(config)
    (path / "model.safetensors.index.json").write_bytes(index)
    return hashlib.sha256(config).hexdigest(), hashlib.sha256(index).hexdigest()


def table_data(layer):
    row = torch.arange(ROWS)[:, None]
    col = torch.arange(256)[None, :]
    block = torch.arange(8)[None, :]
    values = ((row % 7) + layer + col % 4).to(torch.float8_e4m3fn)
    scales = (127 + (row + block) % 3).to(torch.uint8)
    table = torch.empty((ROWS, 264), dtype=torch.uint8, pin_memory=True)
    table[:, :256].copy_(values.view(torch.uint8))
    table[:, 256:].copy_(scales)
    return table


def owner(args):
    config_hash, index_hash = model_files(args.model)
    torch.cuda.set_device(0)
    tables = [table_data(layer) for layer in LAYERS]
    context = Context(args.library, 0, args.qp_port)
    context.register(tables, host=True)
    component_layout = {
        "weight": {"offset": 0, "bytes_per_row": 256, "dtype": "F8_E4M3"},
        "scale": {"offset": 256, "bytes_per_row": 8, "dtype": "F8_E8M0"},
    }
    metadata = {
        "checkpoint_id": "vllm-scaleout-synthetic-smoke-v1",
        "config_sha256": config_hash,
        "index_sha256": index_hash,
        "storage_layout": "row264",
        "lengths": list(context.lengths),
        "addresses": list(context.addresses),
        "native_md": base64.b64encode(context.metadata()).decode(),
        "owner_gpu": 0,
        "table_specs": [
            {"layer": layer, "component_layout": component_layout} for layer in LAYERS
        ],
    }
    control = OwnerServer(metadata, host="0.0.0.0", port=args.control_port).start()
    args.ready.write_text(json.dumps({"host": args.host, "port": control.address[1],
        "generation": control.generation, "checkpoint_id": metadata["checkpoint_id"]}))
    print("OWNER_READY", flush=True)
    while not args.stop.exists():
        time.sleep(0.25)
    try:
        control.close()
    except ControlError:
        print("OWNER_HAS_LIVE_LEASES", flush=True)
        raise
    context.close()
    print("OWNER_DRAINED", flush=True)


def reader(args):
    model_files(args.model)
    os.environ["ENGRAM_SHARED_READY"] = str(args.ready)
    os.environ["ENGRAM_SHARED_LIBRARY"] = str(args.library)
    os.environ["ENGRAM_SHARED_MODEL"] = str(args.model)
    os.environ["ENGRAM_SHARED_READER_PORT_BASE"] = str(args.qp_port)
    os.environ["ENGRAM_SHARED_MAX_QUERIES"] = "1024"
    os.environ.pop("ENGRAM_SHARED_PREFETCH", None)
    os.environ.pop("ENGRAM_SHARED_READ_PLAN", None)
    os.environ.pop("ENGRAM_SHARED_READ_LANES", None)
    from vllm_rdma_provider import VllmRemoteReader

    reader = VllmRemoteReader()
    indices = torch.tensor([[2, 10, 11], [3, 20, 50], [5, -1, 15]], device="cuda")
    selected = indices[:, 1:]
    valid = (selected >= 10) & (selected < 50)
    cols = torch.arange(256, device="cuda")
    blocks = torch.arange(8, device="cuda")
    for layer in LAYERS:
        table = SimpleNamespace(dim=256, block_size=32, head_start=1,
            part_n_hash_cols=2, rdma_layer_id=layer, vocab_start_idx=10,
            vocab_end_idx=50, num_embeddings=ROWS)
        raw = ((selected.abs()[..., None] % 7) + layer + cols % 4).to(torch.float8_e4m3fn)
        scales = (127 + (selected.abs()[..., None] + blocks) % 3).to(torch.uint8)
        for dtype, width in ((torch.bfloat16, 256), (torch.uint8, 264)):
            output = torch.empty((3, 2, width), dtype=dtype, device="cuda")
            reader.lookup(table, indices, output)
            if dtype == torch.bfloat16:
                expected = torch.where(valid[..., None], raw.float() *
                    (2.0 ** (scales[..., cols // 32].int() - 127)), 0).to(dtype)
                torch.testing.assert_close(output, expected)
            else:
                torch.testing.assert_close(output[..., :256],
                    torch.where(valid[..., None], raw.view(torch.uint8), 0))
                torch.testing.assert_close(output[..., 256:],
                    torch.where(valid[..., None], scales, 0))
    torch.cuda.synchronize()
    reader.runtime.close()
    print("PASS: cross-node RDMA, two tables, BF16/packed, invalid rows, lease released")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=("owner", "reader"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--qp-port", type=int, required=True)
    p.add_argument("--host")
    p.add_argument("--control-port", type=int)
    p.add_argument("--stop", type=Path)
    args = p.parse_args()
    if args.role == "owner":
        if not args.host or not args.control_port or not args.stop or args.ready.exists() or args.stop.exists():
            p.error("owner requires host, ports, and fresh ready/stop paths")
        owner(args)
    else:
        reader(args)
