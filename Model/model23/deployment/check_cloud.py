"""Read-only cloud monitor, rate limited to one check per ten minutes."""
from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import math
import re
import time
import sys

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT/'models/model23/deployment_validation/cloud_monitor.json'

def score_record(score):
    # SDK Score.metric passes metadata as Metric's target argument in this version.
    attrs = score._attrs.get('metric')
    if not isinstance(attrs, dict):
        metric = score.metric
        attrs = {} if metric is None else metric._attrs
    value, multiplier = score._attrs.get('value'), attrs.get('multiplier')
    result = dict(metric=attrs.get('name'), displayed_value=value, multiplier=multiplier)
    if (isinstance(value,(int,float)) and isinstance(multiplier,(int,float))
            and math.isfinite(value) and math.isfinite(multiplier) and multiplier > 0):
        result['unscaled_value'] = value/multiplier
        if result['metric'] == 'ts-auc':
            result['gain_vs_model20'] = result['unscaled_value']-.6234481054281362
    return result

def main():
    now = time.time()
    if STATE.exists():
        previous = json.loads(STATE.read_text())
        remaining = 600 - (now-previous['checked_at_unix'])
        if remaining > 0:
            print(json.dumps(dict(check_due=False, next_check_seconds=int(remaining)+1,
                last_verified_status=previous['status'], run_id=114854)))
            return
    from crunch.api import Client, auth
    logging.disable(logging.CRITICAL)
    key = re.search(r'Crunch_Api_key\s*=\s*(\S+)',(ROOT/'AGENTS.md').read_text()).group(1)
    client = Client.from_env(auth=auth.ApiKeyAuth(key),show_progress=False)
    project = client.competitions.get('structural-break-real-time').projects.get(project_identifier='minimum-mantis')
    assert project.user_id == 13209 and project.competition.id == 23
    assert project.submissions.get(13).id == 80033
    run = project.runs.get(114854)
    result = {k:run._attrs.get(k) for k in ('status','success','duration','forceFirstTrain')}
    result.update(run_id=114854, submission_id=80033, checked_at_unix=time.time(),
        checked_at_utc=datetime.now(timezone.utc).isoformat())
    prediction = run._attrs.get('prediction')
    if result['status']=='COMPLETED' and prediction:
        result['prediction_id']=prediction['id']
        result['scores']=[score_record(s)
            for s in project.predictions.get(prediction['id']).scores.list()]
    temporary=STATE.with_suffix('.partial.json')
    temporary.write_text(json.dumps(result,indent=2)+'\n')
    temporary.replace(STATE)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('Monitor check failed:',type(exc).__name__,getattr(getattr(exc,'response',None),'status_code',None))
        sys.exit(1)
