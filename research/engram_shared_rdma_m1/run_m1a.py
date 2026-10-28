#!/usr/bin/env python3
"""Single-node M1 smoke: one pinned DRAM owner, two concurrent GPU readers, reconnect."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def wait_file(path: Path, proc: subprocess.Popen, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if proc.poll() is not None:
            raise RuntimeError(f"process exited before {path.name}: status={proc.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(0.01)


def launch(binary: Path, args: list[str], log: Path, env: dict[str, str]) -> subprocess.Popen:
    stream = log.open("wb")
    proc = subprocess.Popen([str(binary), *args], stdout=stream, stderr=subprocess.STDOUT, env=env)
    proc._log_stream = stream
    return proc


def finish(proc: subprocess.Popen, name: str, timeout: float = 90.0) -> None:
    status = proc.wait(timeout=timeout)
    proc._log_stream.close()
    if status:
        raise RuntimeError(f"{name} failed with status {status}; inspect its retained log")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--owner-gpu", type=int, default=0)
    ap.add_argument("--reader-gpus", type=int, nargs=2, default=[1, 2])
    ap.add_argument("--owner-port", type=int, default=56131)
    ap.add_argument("--reader-ports", type=int, nargs=3, default=[56132, 56133, 56134])
    ap.add_argument("--long-loops", type=int, default=100)
    ap.add_argument("--long-pause-ms", type=int, default=20,
                    help="pace long reader batches so reconnect overlaps its active run")
    args = ap.parse_args()
    binary = args.binary.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    coord = out / "coord"
    coord.mkdir()
    base_env = os.environ.copy()
    owner = launch(binary, ["owner", str(coord), str(args.owner_gpu), str(args.owner_port), "owner"],
                   out / "owner.log", base_env)
    children: list[subprocess.Popen] = [owner]
    try:
        wait_file(coord / "owner.ready", owner)
        gpu_a, gpu_b = args.reader_gpus
        p_long = launch(binary, ["reader", str(coord), str(gpu_b), str(args.reader_ports[1]),
                                 "long", "91", "32", str(args.long_loops), str(args.long_pause_ms)],
                        out / "reader-long.log", base_env)
        children.append(p_long)
        wait_file(coord / "reader.long.started", p_long)
        wait_file(coord / "reader.long.progress", p_long)
        p_short = launch(binary, ["reader", str(coord), str(gpu_a), str(args.reader_ports[0]),
                                  "short", "17", "32", "1", "0"],
                         out / "reader-short.log", base_env)
        children.append(p_short)
        wait_file(coord / "reader.short.started", p_short)
        finish(p_short, "reader-short")
        if p_long.poll() is not None:
            raise RuntimeError("long reader exited before reconnect test; increase --long-loops")
        p_reconnect = launch(binary, ["reader", str(coord), str(gpu_a), str(args.reader_ports[2]),
                                      "reconnect", "203", "32", "1", "0"],
                             out / "reader-reconnect.log", base_env)
        children.append(p_reconnect)
        wait_file(coord / "reader.reconnect.started", p_reconnect)
        if p_long.poll() is not None:
            raise RuntimeError("long reader exited before reconnect attached")
        finish(p_reconnect, "reader-reconnect")
        finish(p_long, "reader-long")
        (coord / "stop-owner").write_text("all readers drained\n")
        finish(owner, "owner")
        print("M1A_PASS owner=one pinned Host DRAM allocation readers=2 concurrent=1 reconnect=1")
        print(f"evidence={out}")
        return 0
    except Exception as exc:
        for proc in children:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in children:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if hasattr(proc, "_log_stream") and not proc._log_stream.closed:
                proc._log_stream.close()
        print(f"M1A_FAIL {exc}", file=sys.stderr)
        print(f"retained_evidence={out}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
