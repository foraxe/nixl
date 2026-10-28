"""Validate RDMA rows against bounded direct reads of the real checkpoint."""

import argparse
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from real_checkpoint_owner import inspect_checkpoint
from vllm_rdma_provider import VllmRemoteReader


def read_checkpoint_row(manifest, layer, row):
    parts = []
    for kind, width in (("weight", 256), ("scale", 8)):
        spec = next(
            s for s in manifest["specs"] if s["layer"] == layer and s["kind"] == kind
        )
        shard = manifest["shards"][spec["shard"]]
        offset = shard["data_start"] + spec["data_offsets"][0] + row * width
        with open(shard["path"], "rb", buffering=0) as file:
            part = os.pread(file.fileno(), width, offset)
        if len(part) != width:
            raise RuntimeError(f"short checkpoint row for layer {layer}")
        parts.append(torch.frombuffer(bytearray(part), dtype=torch.uint8))
    return torch.cat(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ready", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--export-reference", action="store_true")
    parser.add_argument("--with-ucx", action="store_true")
    parser.add_argument("--ucx-after-reader", action="store_true")
    parser.add_argument("--with-kv-engine", action="store_true")
    parser.add_argument("--kv-host")
    args = parser.parse_args()
    manifest = inspect_checkpoint(args.model)
    if args.export_reference:
        rows = {}
        for layer, count in zip((1, 14), manifest["rows"]):
            rows[str(layer)] = [
                base64.b64encode(
                    bytes(read_checkpoint_row(manifest, layer, row))
                ).decode()
                for row in (0, 1, count // 2, count - 1)
            ]
        args.reference.write_text(
            json.dumps({"identity": manifest["checkpoint_identity"], "rows": rows})
        )
        print("OWNER_REFERENCE_EXPORTED")
        return
    if not all((args.ready, args.library, args.port)):
        parser.error("reader requires --ready, --library, and --port")
    reference = json.loads(args.reference.read_text())
    if reference["identity"] != manifest["checkpoint_identity"]:
        raise ValueError("checkpoint metadata identity differs from source reference")
    if args.with_ucx and not args.ucx_after_reader:
        from nixl._api import nixl_agent

        ucx = nixl_agent("vllm-engram-combined-smoke")
        if "UCX" not in ucx.backends:
            raise RuntimeError("UCX backend not instantiated")
    os.environ.update(
        ENGRAM_SHARED_READY=str(args.ready),
        ENGRAM_SHARED_LIBRARY=str(args.library),
        ENGRAM_SHARED_MODEL=str(args.model),
        ENGRAM_SHARED_READER_PORT_BASE=str(args.port),
        ENGRAM_SHARED_MAX_QUERIES="1024",
    )
    for name in (
        "ENGRAM_SHARED_PREFETCH",
        "ENGRAM_SHARED_READ_PLAN",
        "ENGRAM_SHARED_READ_LANES",
    ):
        os.environ.pop(name, None)
    reader = VllmRemoteReader()
    try:
        if args.with_kv_engine:
            if not args.kv_host:
                parser.error("--with-kv-engine requires --kv-host")
            from mooncake.engine import TransferEngine

            kv_engine = TransferEngine()
            if kv_engine.initialize(
                args.kv_host, "P2PHANDSHAKE", "rdma", "mlx5_bond_0"
            ):
                raise RuntimeError("KV transfer engine initialization failed")
        if args.with_ucx and args.ucx_after_reader:
            from nixl._api import nixl_agent

            ucx = nixl_agent("vllm-engram-combined-smoke")
            if "UCX" not in ucx.backends:
                raise RuntimeError("UCX backend not instantiated")
        for layer, rows in zip((1, 14), manifest["rows"]):
            table = SimpleNamespace(
                dim=256,
                block_size=32,
                head_start=0,
                part_n_hash_cols=2,
                rdma_layer_id=layer,
                vocab_start_idx=0,
                vocab_end_idx=rows,
                num_embeddings=rows,
            )
            examples = torch.zeros((6, 264), dtype=torch.uint8)
            for index, encoded in enumerate(reference["rows"][str(layer)]):
                data = base64.b64decode(encoded, validate=True)
                if len(data) != 264:
                    raise ValueError("invalid source reference row")
                examples[index].copy_(
                    torch.frombuffer(bytearray(data), dtype=torch.uint8)
                )
            choices = torch.tensor(
                [0, 1, rows // 2, rows - 1, -1, rows], dtype=torch.int32
            )
            for count in (6, 514):
                # All 514 positions are valid so this crosses the 512-WQE
                # window, rather than merely passing 514 input positions.
                selected = torch.arange(count) % (6 if count == 6 else 4)
                ids = choices[selected].reshape(-1, 2).cuda()
                out = torch.empty(
                    (count // 2, 2, 264), dtype=torch.uint8, device="cuda"
                )
                reader.lookup(table, ids, out)
                expected = examples[selected].reshape_as(out)
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                stats = reader.runtime.stats.cpu().tolist()
                assert stats[0] == 0
                if count == 514:
                    assert stats[4] == 514, stats
        print("PASS: owner rows, scales and 514-query boundary match byte-for-byte")
    finally:
        reader.runtime.close()


if __name__ == "__main__":
    main()
