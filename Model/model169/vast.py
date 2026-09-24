"""Model169: rent a vast.ai GPU for the neural student (user's instruction: deploy the network in a pod).

Uses the key in .env (Other/.env is the master copy; the root .env is a stale copy - keep them in sync).
Files are copied with plain scp using the locally generated key Other/runpod_key, so nothing is installed.
ALWAYS destroy the instance when the run is done - vast bills by the second.
  python models/model169/vast.py keys                 show / upload our ssh public key to the account
  python models/model169/vast.py offers [MAXDPH]      cheapest matching offers
  python models/model169/vast.py create OFFER_ID      create the instance
  python models/model169/vast.py list                 running instances (id, status, ssh, $/h)
  python models/model169/vast.py ssh ID               print ssh / scp command lines
  python models/model169/vast.py destroy ID           DESTROY the instance (billing stops)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
API = "https://console.vast.ai/api/v0"
IMAGE = "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"


def key():
    env = ROOT / ".env"
    env = env if env.exists() else ROOT / "Other/.env"   # the root copy sometimes disappears (OneDrive)
    kv = dict(l.strip().split("=", 1) for l in env.read_text().splitlines() if "=" in l and not l.startswith("#"))
    return "".join(kv["VAST_AI_API_KEY"].split()).strip('"')


def call(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json",
                                        "Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(r, timeout=180) as response:
            text = response.read().decode()
            return json.loads(text) if text.strip() else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:500]}")


def offers(max_dph=0.5):
    q = {"verified": {"eq": True}, "rentable": {"eq": True}, "num_gpus": {"eq": 1},
         "dph_total": {"lte": max_dph}, "inet_down": {"gte": 300}, "inet_up": {"gte": 100},
         "disk_space": {"gte": 30}, "reliability2": {"gte": 0.97}, "cuda_max_good": {"gte": 12.1},
         "gpu_name": {"in": ["RTX 4090", "RTX 3090", "RTX 4080", "RTX 3090 Ti", "RTX A5000", "RTX 4070 Ti"]},
         "type": "on-demand", "order": [["dph_total", "asc"]], "limit": 12}
    import urllib.parse
    d = call("/search/asks?q=" + urllib.parse.quote(json.dumps(q)))
    rows = d.get("offers", d if isinstance(d, list) else [])
    for o in rows:
        print(f"  id {o['id']:>10}  {o.get('gpu_name'):<24} {o.get('num_gpus')}x  ${o.get('dph_total'):.3f}/h  "
              f"down {o.get('inet_down'):.0f} up {o.get('inet_up'):.0f} Mbps  rel {o.get('reliability2'):.3f}  "
              f"{o.get('geolocation')}")
    return rows


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "list"
    if what == "keys":
        pub = (ROOT / "Other/runpod_key.pub").read_text().strip()
        have = call("/ssh/")
        keys = have if isinstance(have, list) else have.get("results", have.get("ssh_keys", []))
        if any(pub.split()[1] in json.dumps(k) for k in keys):
            print("our key is already on the account")
        else:
            print(call("/ssh/", "POST", {"ssh_key": pub}))
        return
    if what == "offers":
        offers(float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
        return
    if what == "create":
        offer = int(sys.argv[2])
        body = {"client_id": "me", "image": IMAGE, "disk": 40, "runtype": "ssh",
                "label": "structural-break-neural", "onstart": "touch /workspace/ready"}
        print(call(f"/asks/{offer}/", "PUT", body))
        return
    if what == "list":
        d = call("/instances/")
        for i in d.get("instances", []):
            print(f"  id {i['id']}  {i.get('actual_status')}  {i.get('gpu_name')}  ${i.get('dph_total'):.3f}/h  "
                  f"ssh {i.get('ssh_host')}:{i.get('ssh_port')}  label {i.get('label')}")
        if not d.get("instances"):
            print("no instances")
        return
    ident = sys.argv[2]
    if what == "ssh":
        d = call("/instances/")
        for i in d.get("instances", []):
            if str(i["id"]) == ident:
                k = ROOT / "Other/runpod_key"
                print(f'ssh -i "{k}" -o StrictHostKeyChecking=no -p {i["ssh_port"]} root@{i["ssh_host"]}')
                print(f'scp -i "{k}" -o StrictHostKeyChecking=no -P {i["ssh_port"]} <file> root@{i["ssh_host"]}:/workspace/')
        return
    if what == "destroy":
        print(call(f"/instances/{ident}/", "DELETE", {}))
        print("destroyed", ident)
        return
    raise SystemExit("unknown command")


if __name__ == "__main__":
    main()
