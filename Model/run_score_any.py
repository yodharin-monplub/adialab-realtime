"""Fetch a cloud run's status and prediction scores from EITHER account (read-only).

  python models/run_score_any.py RUN_ID [krittanut|yodharin]

krittanut -> project sympathetic-trout (default), yodharin -> project minimum-mantis.
Same score decoding as models/run_score.py; added 2026-09-25 because submissions now
live on both accounts (see the crunch-accounts memory).
"""
import json
import logging
import sys
from pathlib import Path

from crunch.api import Client, auth
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_score import score_record  # noqa: E402  (same decoding, one implementation)

ACCOUNTS = {"krittanut": ("CRUNCH_API_KEY_KRITTANUT", "sympathetic-trout"),
            "yodharin": ("CRUNCH_API_KEY_YODHARIN", "minimum-mantis")}


def main(run_id: int, account: str = "krittanut"):
    logging.disable(logging.CRITICAL)
    keyname, project_name = ACCOUNTS[account]
    env = dotenv_values(ROOT / "Other/.env") if (ROOT / "Other/.env").exists() else dotenv_values(ROOT / ".env")
    client = Client.from_env(auth=auth.ApiKeyAuth(env[keyname]), show_progress=False)
    project = client.competitions.get("structural-break-real-time").projects.get(
        project_identifier=project_name)
    run = project.runs.get(run_id)
    result = {"account": account, "project": project_name}
    result.update({k: run._attrs.get(k) for k in ("id", "status", "success", "duration", "createdAt")})
    prediction = run._attrs.get("prediction")
    if prediction:
        result["prediction_id"] = prediction.get("id")
        result["scores"] = [score_record(s) for s in project.predictions.get(prediction["id"]).scores.list()]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "krittanut")
