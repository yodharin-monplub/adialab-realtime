"""Print status of a cloud run (read-only): python models/run_status.py RUN_ID"""
import logging
from pathlib import Path
import sys

from crunch.api import Client, auth
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main(run_id: int):
    logging.disable(logging.CRITICAL)
    env = dotenv_values(ROOT / ".env")
    key = env.get("CRUNCH_API_KEY_KRITTANUT") or env.get("CRUNCH_API_KEY")
    client = Client.from_env(auth=auth.ApiKeyAuth(key), show_progress=False)
    project = client.competitions.get("structural-break-real-time").projects.get(
        project_identifier="sympathetic-trout")
    for run in project.runs.list():
        a = run._attrs
        if a.get("id") == run_id:
            keys = ("id", "status", "success", "exitCode", "exitReason", "createdAt", "startedAt", "endedAt")
            print({k: a.get(k) for k in keys})
            return
    print("run not found")


if __name__ == "__main__":
    main(int(sys.argv[1]))
