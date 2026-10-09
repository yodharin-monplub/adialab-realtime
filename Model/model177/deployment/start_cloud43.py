"""Start exactly one inference-only cloud run for the uploaded Submission43."""
from pathlib import Path
import json
import logging

from crunch.api import Client, auth
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'Model/model177/deployment_validation'
RECEIPT = OUTPUT / 'cloud_run_receipt43.json'


def main():
    if RECEIPT.exists():
        print(RECEIPT.read_text())
        return
    push = json.loads((OUTPUT / 'push_receipt43.json').read_text())
    if not (push['remote_acceptance_verified'] and push['platform_valid'] and
            push['libraries_verified'] and push['uploaded']):
        raise ValueError('Cloud submission is not remotely accepted and valid')
    settings = json.loads((OUTPUT / 'cloud_run_settings43.json').read_text())
    if settings['submissionId'] != push['submission_id']:
        raise ValueError('Cloud settings refer to a different submission')
    if settings['forceFirstTrain'] is not False or settings['trainFrequency'] != 0:
        raise ValueError('Cloud training must be disabled')
    estimate = json.loads((OUTPUT / 'validation_submission43.json').read_text())
    if estimate['estimated_cloud_hours'] > 10:
        raise ValueError('Cloud cost exceeds local safety gate')
    env = dotenv_values(ROOT / 'Other/.env')
    key = env.get('CRUNCH_API_KEY_YODHARIN')  # account owning minimum-mantis (user 13209)
    if not key:
        raise ValueError('Missing CRUNCH_API_KEY in .env')
    logging.disable(logging.CRITICAL)
    client = Client.from_env(auth=auth.ApiKeyAuth(key), show_progress=False)
    project = client.competitions.get('structural-break-real-time').projects.get(
        project_identifier='minimum-mantis')
    if project.user_id != 13209 or project.competition.id != 23:
        raise ValueError('Cloud credential points to a different project')
    submitted = project.submissions.get(push['submission_number'])
    if submitted._attrs.get('id') != push['submission_id'] or not submitted._attrs.get('valid'):
        raise ValueError('Uploaded candidate is not the expected valid submission')
    existing = [run for run in project.runs.list() if
                (run._attrs.get('submission') or {}).get('id') == push['submission_id']]
    if existing:
        record = dict(run_id=existing[0]._attrs['id'],
                      status=existing[0]._attrs.get('status'),
                      submission_id=push['submission_id'],
                      created_by_this_script=False)
        RECEIPT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))
        return
    route = (f'/v3/competitions/{project.competition.id}/projects/'
             f'{project.user_id}/{project.name}/runs')
    response = client.api.post(route, json=settings)
    response.raise_for_status()
    result = response.json()
    if (result.get('submission') or {}).get('id') != push['submission_id']:
        raise ValueError('Cloud run response refers to a different submission')
    record = dict(run_id=result['id'], status=result.get('status'),
                  submission_id=push['submission_id'],
                  forceFirstTrain=result.get('forceFirstTrain'),
                  trainFrequency=result.get('trainFrequency'),
                  created_by_this_script=True)
    with RECEIPT.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
