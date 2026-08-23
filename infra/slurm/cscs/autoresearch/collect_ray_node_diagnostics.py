# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect one host-memory and process snapshot from every live Ray node."""

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


CGROUP_FILES = (
    "memory.current",
    "memory.max",
    "memory.peak",
    "memory.events",
    "memory.events.local",
    "memory.swap.current",
    "memory.swap.max",
)


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


@ray.remote(num_cpus=0)
def _snapshot() -> dict:
    cgroup = Path("/sys/fs/cgroup")
    process_listing = subprocess.run(
        [
            "ps",
            "-eo",
            "pid,ppid,stat,pcpu,pmem,rss,vsz,etime,wchan:32,comm,args",
            "--sort=-rss",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp": time.time(),
        "loadavg": _read(Path("/proc/loadavg")),
        "meminfo": _read(Path("/proc/meminfo")),
        "self_cgroup": _read(Path("/proc/self/cgroup")),
        "cgroup": {name: _read(cgroup / name) for name in CGROUP_FILES},
        "processes": process_listing.stdout,
        "ps_stderr": process_listing.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ray.init(address="auto", logging_level="ERROR")
    live_nodes = [node for node in ray.nodes() if node.get("Alive", False)]
    future_to_node = {}
    for node in live_nodes:
        node_id = node["NodeID"]
        future = _snapshot.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
        ).remote()
        future_to_node[future] = node_id

    snapshots = []
    ready, pending = ray.wait(
        list(future_to_node),
        num_returns=len(future_to_node),
        timeout=120,
    )
    for future in ready:
        node_id = future_to_node[future]
        try:
            snapshot = ray.get(future)
        except Exception as error:  # diagnostics must survive partial cluster failure
            snapshot = {"node_id": node_id, "error": repr(error)}
        else:
            snapshot["node_id"] = node_id
        snapshots.append(snapshot)
    for future in pending:
        snapshots.append(
            {
                "node_id": future_to_node[future],
                "error": "node snapshot did not finish within 120 seconds",
            }
        )

    payload = {
        "captured_at": time.time(),
        "ray_node_count": len(live_nodes),
        "snapshots": sorted(
            snapshots,
            key=lambda item: (item.get("hostname", ""), item["node_id"]),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
