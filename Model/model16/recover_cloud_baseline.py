"""Recover the user's last successful Model 6 deployment, read-only remotely."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import re

import requests
from crunch.api import Client, auth

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "models/model16/upstream_cloud_submission9"


def main():
    key = os.environ.get("CRUNCH_API_KEY")
    if not key:
        key = re.search(r"Crunch_Api_key\s*=\s*(\S+)", (ROOT/"AGENTS.md").read_text()).group(1)
    client = Client.from_env(auth=auth.ApiKeyAuth(key), show_progress=False)
    project = client.competitions.get("structural-break-real-time").projects.get(project_identifier="minimum-mantis")
    # The clone endpoint requires a push token. The documented file-download
    # endpoint accepts the user's API key and needs no new credential.
    files = project.submissions.get(9).files.list()
    endpoint = f"/v3/competitions/{project.competition.id}/projects/{project.user_id}/{project.name}/submissions/9/files/download"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    def fetch(item):
        name = item.name
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or (relative.name != ".gitignore" and relative.suffix not in {".py", ".txt", ".json", ".npz", ".joblib", ".md", ".cpp", ".so"}):
            raise ValueError(f"Unexpected upstream file: {name}")
        destination = OUTPUT/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                response = client.api.get(endpoint, params={"fileName":name}, timeout=(30,120))
                response.raise_for_status()
            except requests.RequestException:
                raise RuntimeError(f"Download failed: {name}") from None
            temporary = destination.with_suffix(destination.suffix+".download")
            temporary.write_bytes(response.content)
            if temporary.stat().st_size != item.size:
                raise RuntimeError(f"Downloaded size mismatch: {name}")
            temporary.replace(destination)
        return dict(path=name, bytes=destination.stat().st_size,
                    sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, files))
    runs = project.runs.list(submission_number=9)
    history = [{key:r._attrs.get(key) for key in ("id", "duration", "cpu", "memory", "forceFirstTrain", "determinismCheckEnabled", "success")} for r in runs]
    (OUTPUT/"RECOVERY_MANIFEST.json").write_text(json.dumps(dict(submission_number=9, files=records, cloud_runs=history), indent=2)+"\n")
    print(json.dumps(dict(files=len(records), bytes=sum(r["bytes"] for r in records), cloud_runs=history), indent=2))


if __name__ == "__main__":
    main()
