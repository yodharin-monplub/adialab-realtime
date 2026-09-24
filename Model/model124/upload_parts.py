"""Resumable chunked scp upload (the vast host drops connections every ~2-3 min).

python upload_parts.py LOCAL_FILE REMOTE_DIR HOST PORT KEY [--part-mb 25] [--workers 4]
Splits into parts, uploads with a thread pool, verifies remote sizes, retries,
then concatenates on the remote side and verifies the total size.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import time


def ssh(args, host, port, key, cmd):
    return subprocess.run(["ssh", "-i", key, "-p", str(port), "-o", "BatchMode=yes",
                           "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=30",
                           f"root@{host}", cmd], capture_output=True, text=True, timeout=120)


def remote_size(host, port, key, path):
    r = ssh(None, host, port, key, f"stat -c %s {path} 2>/dev/null || echo 0")
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("local"); ap.add_argument("remote_dir"); ap.add_argument("host")
    ap.add_argument("port", type=int); ap.add_argument("key")
    ap.add_argument("--part-mb", type=int, default=25); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    local = Path(a.local)
    parts_dir = local.parent / "parts"
    parts_dir.mkdir(exist_ok=True)
    size = local.stat().st_size
    part_bytes = a.part_mb * 1024 * 1024
    parts = []
    with local.open("rb") as f:
        i = 0
        while True:
            chunk = f.read(part_bytes)
            if not chunk:
                break
            p = parts_dir / f"{local.name}.{i:03d}"
            if not p.exists() or p.stat().st_size != len(chunk):
                p.write_bytes(chunk)
            parts.append((p, len(chunk)))
            i += 1
    print(f"{len(parts)} parts of {a.part_mb} MB from {size / 1e6:.0f} MB", flush=True)
    ssh(None, a.host, a.port, a.key, f"mkdir -p {a.remote_dir}/parts")

    def upload(item):
        p, n = item
        remote = f"{a.remote_dir}/parts/{p.name}"
        for attempt in range(8):
            if remote_size(a.host, a.port, a.key, remote) == n:
                return p.name, attempt
            subprocess.run(["scp", "-i", a.key, "-P", str(a.port), "-o", "BatchMode=yes",
                            "-o", "StrictHostKeyChecking=accept-new", str(p), f"root@{a.host}:{remote}"],
                           capture_output=True, text=True, timeout=600)
            time.sleep(2)
        raise RuntimeError(f"failed {p.name}")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for name, attempts in pool.map(upload, parts):
            done += 1
            if done % 4 == 0 or done == len(parts):
                print(f"{done}/{len(parts)} parts ({time.time() - t0:.0f}s, last attempts {attempts})", flush=True)
    names = " ".join(f"parts/{p.name}" for p, _ in parts)
    ssh(None, a.host, a.port, a.key, f"cd {a.remote_dir} && cat {names} > {local.name}")
    total = remote_size(a.host, a.port, a.key, f"{a.remote_dir}/{local.name}")
    print(f"remote {local.name}: {total} bytes (expected {size}) -> {'OK' if total == size else 'MISMATCH'}", flush=True)
    if total == size:
        ssh(None, a.host, a.port, a.key, f"rm -rf {a.remote_dir}/parts")


if __name__ == "__main__":
    main()
