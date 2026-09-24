"""Download only the official training inputs; credentials never enter logs."""
from pathlib import Path
import os
import re

import requests
from crunch.api import Client, auth


def main():
    root = Path(__file__).resolve().parents[1]
    key = os.environ.get("CRUNCH_API_KEY")
    if not key:
        match = re.search(r"Crunch_Api_key\s*=\s*(\S+)", (root / "AGENTS.md").read_text())
        if not match:
            raise SystemExit("Set CRUNCH_API_KEY to download the training data.")
        key = match.group(1)
    client = Client.from_env(auth=auth.ApiKeyAuth(key), show_progress=False)
    competition = client.competitions.get("structural-break-real-time")
    release = competition.rounds.last.phases.get_submission().get_data_release()
    directory = root / "train"
    directory.mkdir(exist_ok=True)
    for name in ("X_train.parquet", "y_train_index.parquet"):
        item = release.data_files[name]
        path = directory / name
        if path.exists() and path.stat().st_size == item.size:
            print(f"Already present: {name}", flush=True)
            continue
        if path.exists():
            raise RuntimeError(f"Existing file has unexpected size: {path}")
        temporary = path.with_suffix(".download")
        try:
            with requests.get(item.url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            if temporary.stat().st_size != item.size:
                raise RuntimeError(f"Size mismatch downloading {name}")
            temporary.replace(path)
        except requests.RequestException:
            # A requests exception may contain a signed URL; don't expose it.
            raise RuntimeError(f"Training download failed for {name}; rerun to retry") from None
        print(f"Downloaded {name}: {item.size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
