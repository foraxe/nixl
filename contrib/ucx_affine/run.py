#!/usr/bin/env python3
"""Run a paired two-endpoint UCX affine replay from a private endpoint config."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


def remote(endpoint, args):
    return endpoint["shell_prefix"] + args


def preflight(endpoint, minimum_free_mib):
    out = subprocess.check_output(remote(endpoint, [
        "nvidia-smi", "-i", str(endpoint["gpu"]),
        "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ]), text=True)
    _, free, util = [int(part.strip()) for part in out.strip().split(",")]
    if free < minimum_free_mib or util > 5:
        raise RuntimeError(f"GPU not idle: free={free} MiB, utilization={util}%")
    return out


def command(config, role, arm):
    endpoint = config[role]
    env = dict(endpoint["env"])
    env["CUDA_VISIBLE_DEVICES"] = str(endpoint["gpu"])
    env["NIXL_UCX_SGL_ENABLE"] = "1"
    env["NIXL_LOG_LEVEL"] = "ERROR"
    if role == "sender":
        env["NIXL_PLUGIN_DIR"] = endpoint[f"{arm}_plugin_dir"]
        if arm == "affine":
            env["PD_REPLAY_UCX_AFFINE"] = "1"
    else:
        env["NIXL_PLUGIN_DIR"] = endpoint["plugin_dir"]
    args = ["env"] + [f"{key}={value}" for key, value in env.items()]
    args += ["timeout", "--signal=TERM", "--kill-after=5s", "90s", "taskset", "-c",
             str(endpoint["cpu"]), endpoint["executable"], role, "UCX",
             config["receiver_ip"], str(config["control_port"]), str(endpoint["oob_port"]),
             config["device"], config["interface"], str(config["gid_index"]),
             str(config["iterations"]), str(config["warmup"])]
    return remote(endpoint, args)


def run_one(config, root, arm):
    root.mkdir()
    cores = {}
    plugins = {}
    for role in ("sender", "receiver"):
        endpoint = config[role]
        (root / f"{role}-preflight.csv").write_text(
            preflight(endpoint, config.get("min_free_mib", 4096)))
        files = [endpoint["executable"], *endpoint["identity_files"]]
        plugin = (endpoint[f"{arm}_plugin_dir"] if role == "sender"
                  else endpoint["plugin_dir"])
        files.append(plugin + "/libplugin_UCX.so")
        hashes = subprocess.check_output(remote(endpoint, ["sha256sum", *files]), text=True)
        (root / f"{role}-hashes.txt").write_text(hashes)
        core = [line.split()[0] for line in hashes.splitlines()
                if line.split()[-1].endswith("/libnixl.so")]
        if len(core) != 1:
            raise RuntimeError(f"{role} identity_files must include libnixl.so")
        cores[role] = core[0]
        plugins[role] = plugin + "/libplugin_UCX.so"
    if cores["sender"] != cores["receiver"]:
        raise RuntimeError("sender and receiver use different NIXL cores")
    with (root / "receiver.log").open("w") as log:
        receiver = subprocess.Popen(command(config, "receiver", arm), stdout=log,
                                    stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while "\nREADY\n" not in "\n" + (root / "receiver.log").read_text():
                if receiver.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("receiver did not become ready")
                time.sleep(0.2)
            with (root / "sender.log").open("w") as sender_log:
                sender = subprocess.run(command(config, "sender", arm), stdout=sender_log,
                                        stderr=subprocess.STDOUT, timeout=100, check=False)
            if sender.returncode:
                raise RuntimeError(f"sender exited {sender.returncode}")
            if receiver.wait(timeout=100):
                raise RuntimeError("receiver failed")
        finally:
            if receiver.poll() is None:
                receiver.terminate()
                receiver.wait(timeout=5)
    if any("REPLAY_PASS" not in (root / f"{role}.log").read_text()
           for role in ("sender", "receiver")):
        raise RuntimeError("GPU validation did not pass")
    for role in ("sender", "receiver"):
        if plugins[role] not in (root / f"{role}.log").read_text():
            raise RuntimeError(f"{role} did not map the expected UCX plugin")
    rows = [[float(value) for value in line.split()[2:]]
            for line in (root / "sender.log").read_text().splitlines()
            if line.startswith("SAMPLE ")]
    if len(rows) != config["iterations"]:
        raise RuntimeError(f"expected {config['iterations']} samples, got {len(rows)}")
    sorted_request = sorted(row[4] for row in rows)
    return {"count": len(rows), "request_p50_us": statistics.median(sorted_request),
            "request_p99_us": sorted_request[math.ceil(0.99 * len(rows)) - 1],
            "plan_p50_us": statistics.median(row[0] for row in rows),
            "prepare_p50_us": statistics.median(row[1] for row in rows),
            "post_p50_us": statistics.median(row[2] for row in rows),
            "post_done_p50_us": statistics.median(row[3] for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = []
    for pair in range(config["pairs"]):
        order = ("baseline", "affine") if pair % 2 == 0 else ("affine", "baseline")
        for arm in order:
            label = f"pair{pair}-{arm}"
            result = run_one(config, args.output / label, arm)
            results.append({"pair": pair, "arm": arm, **result})
            (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(label, result, flush=True)


if __name__ == "__main__":
    main()
