"""Fetch a completed cloud run's prediction scores (read-only).

  python models/run_score.py RUN_ID
"""
import json
import logging
import math
from pathlib import Path
import sys

from crunch.api import Client, auth
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def score_record(score):
    attrs = score._attrs.get("metric")
    if not isinstance(attrs, dict):
        metric = score.metric
        attrs = {} if metric is None else metric._attrs
    value, multiplier = score._attrs.get("value"), attrs.get("multiplier")
    result = {"metric": attrs.get("name"), "displayed_value": value, "multiplier": multiplier}
    if (isinstance(value, (int, float)) and isinstance(multiplier, (int, float))
            and math.isfinite(value) and math.isfinite(multiplier) and multiplier > 0):
        result["unscaled_value"] = value / multiplier
    return result


def main(run_id: int):
    logging.disable(logging.CRITICAL)
    env = dotenv_values(ROOT / ".env")
    key = env.get("CRUNCH_API_KEY_KRITTANUT") or env.get("CRUNCH_API_KEY")
    client = Client.from_env(auth=auth.ApiKeyAuth(key), show_progress=False)
    project = client.competitions.get("structural-break-real-time").projects.get(
        project_identifier="sympathetic-trout")
    run = project.runs.get(run_id)
    result = {k: run._attrs.get(k) for k in ("id", "status", "success", "duration", "exitCode", "exitReason")}
    prediction = run._attrs.get("prediction")
    if prediction:
        result["prediction_id"] = prediction.get("id")
        result["scores"] = [score_record(s) for s in project.predictions.get(prediction["id"]).scores.list()]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]))
