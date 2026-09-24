"""Fetch the successful Model20 cloud runtime via read-only Crunch API calls."""
from pathlib import Path
import json
import logging
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from crunch.api import Client, auth
from models.model11.pipeline import write_json


def main():
    logging.disable(logging.CRITICAL)
    key = re.search(r'Crunch_Api_key\s*=\s*(\S+)', (ROOT/'AGENTS.md').read_text()).group(1)
    client = Client.from_env(auth=auth.ApiKeyAuth(key), show_progress=False)
    project = client.competitions.get('structural-break-real-time').projects.get(project_identifier='minimum-mantis')
    if project.user_id != 13209 or project.competition.id != 23 or project.submissions.get(12).id != 79379:
        raise ValueError('Unexpected reference project/submission')
    run = next(r for r in project.runs.list(submission_number=12) if r.id == 114669)
    fields = ('id', 'status', 'success', 'duration', 'cpu', 'memory', 'forceFirstTrain', 'deterministic')
    result = {name:run._attrs.get(name) for name in fields}
    if result['status'] != 'COMPLETED' or result['success'] is not True:
        raise ValueError('Reference run was not successful')
    result.update(model=20, submission_number=12, submission_id=79379,
        source='Crunch project minimum-mantis read-only API', prediction_id=run._attrs['prediction']['id'])
    write_json(ROOT/'models/model23/deployment_validation/model20_cloud_reference.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('Reference lookup failed:', type(exc).__name__,
            getattr(getattr(exc, 'response', None), 'status_code', None))
        raise SystemExit(1) from None
