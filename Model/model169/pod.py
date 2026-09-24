"""Model169: rent / inspect / stop a RunPod GPU pod for the neural student (user's instruction 2026-09-23:
"if deploy neural network, deploy in a pod").

The pod is created with our own SSH public key (Other/runpod_key.pub) so files can be copied with plain
scp/ssh - nothing is downloaded or installed locally.  Always stop the pod when the run is finished:
    python models/model169/pod.py list          what is running and what it costs
    python models/model169/pod.py create        create one GPU pod (asks for nothing, prints ssh details)
    python models/model169/pod.py ssh POD_ID    print the ssh/scp command lines
    python models/model169/pod.py stop POD_ID   TERMINATE the pod (billing stops)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
API = "https://rest.runpod.io/v1"
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4080"]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def key():
    kv = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines() if "=" in l and not l.startswith("#"))
    return "".join(kv["RUNPOD_API_KEY"].split()).strip('"')


def call(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json",
                                        "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(r, timeout=120) as response:
            text = response.read().decode()
            return json.loads(text) if text.strip() else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:400]}")


def show(pod):
    ssh = [p for p in (pod.get("portMappings") or {}).items()] if isinstance(pod.get("portMappings"), dict) else []
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "lastStatusChange", "machineId",
                                              "costPerHr", "publicIp", "portMappings", "gpu", "machine")}, indent=2, default=str)[:1200])
    return ssh


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "list"
    if what == "list":
        pods = call("/pods")
        pods = pods if isinstance(pods, list) else pods.get("pods", [])
        if not pods:
            print("no pods")
        for p in pods:
            show(p)
        return
    if what == "create":
        pub = (ROOT / "Other/runpod_key.pub").read_text().strip()
        body = {"name": "structural-break-gru", "imageName": IMAGE, "cloudType": "COMMUNITY",
                "gpuTypeIds": GPUS, "gpuTypePriority": "availability", "gpuCount": 1,
                "containerDiskInGb": 40, "volumeInGb": 40, "volumeMountPath": "/workspace",
                "ports": ["22/tcp"], "supportPublicIp": True, "minDownloadMbps": 100, "minUploadMbps": 50,
                "env": {"PUBLIC_KEY": pub}}
        pod = call("/pods", "POST", body)
        pid = pod.get("id")
        print("created", pid, "cost/hr", pod.get("costPerHr"), flush=True)
        for _ in range(60):
            time.sleep(10)
            p = call(f"/pods/{pid}")
            if p.get("publicIp") and p.get("portMappings"):
                show(p); return
            print("waiting for ip...", p.get("desiredStatus"), p.get("lastStatusChange"), flush=True)
        print("pod did not report an IP yet; run `list` later")
        return
    pid = sys.argv[2]
    if what == "ssh":
        p = call(f"/pods/{pid}")
        ip, ports = p.get("publicIp"), p.get("portMappings") or {}
        port = ports.get("22")
        keyfile = ROOT / "Other/runpod_key"
        print(f'ssh -i "{keyfile}" -p {port} root@{ip}')
        print(f'scp -i "{keyfile}" -P {port} <file> root@{ip}:/workspace/')
        return
    if what == "stop":
        print(call(f"/pods/{pid}", "DELETE"))
        print("terminated", pid)
        return
    raise SystemExit("unknown command")


if __name__ == "__main__":
    main()
