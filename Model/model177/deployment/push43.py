"""Upload the sealed Submission43 bundle (Model177 lattice gate on Submission42) - ONLY after the user's explicit approval; never creates a cloud run.
Usage: python Model/model177/deployment/push43.py [--push]"""
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import os

from crunch.api import Client, auth
from crunch.command.push import list_code_files, list_model_files, push
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / 'submissions/submission43'
OUTPUT = ROOT / 'Model/model177/deployment_validation'
RECEIPT = OUTPUT / 'push_receipt43.json'
MESSAGE = ('Submission 43: Submission 42 + float32 lattice gate (Model177): series whose z-scored history sits on a '
           'quantisation lattice score 0.01x until the online values leave the lattice, then 1.0; '
           'pretrained inference, training code included, 12 workers')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push', action='store_true', help='Actually upload; otherwise validate only')
    args = parser.parse_args()
    if RECEIPT.exists():
        raise RuntimeError('Submission43 upload receipt already exists; inspect before retrying')
    env = dotenv_values(ROOT / 'Other/.env')
    key = env.get('CRUNCH_API_KEY_YODHARIN')  # account owning minimum-mantis (user 13209), user's instruction 2026-09-23
    if not key:
        raise ValueError('Missing CRUNCH_API_KEY in .env')
    package = json.loads((OUTPUT / 'package_submission43.json').read_text())
    archive = ROOT / package['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != package['sha256']:
        raise ValueError('Sealed archive changed')
    manifest = json.loads((BUNDLE / 'bundle_manifest.json').read_text())
    for name, record in manifest['files'].items():
        data = (BUNDLE / name).read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'Changed sealed submission file: {name}')
        if key.encode() in data:
            raise ValueError(f'Credential in submission file: {name}')
    os.chdir(BUNDLE)
    code = list(list_code_files(str(BUNDLE), 'resources'))
    models = list(list_model_files(str(BUNDLE), 'resources'))
    actual = {name.replace('\\', '/') for _, name in code} | {'resources/' + name.replace('\\', '/') for _, name in models}
    if actual != set(manifest['files']) | {'bundle_manifest.json'}:
        raise ValueError('Cloud upload inventory changed')
    if any(name.startswith('resources/') for _, name in code):
        raise ValueError('Model resources unexpectedly uploaded as code')
    print(f'Validated {len(code)} code files and {len(models)} model files', flush=True)
    if not args.push:
        return
    client = Client.from_env(auth=auth.ApiKeyAuth(key), show_progress=False)
    project = client.competitions.get('structural-break-real-time').projects.get(
        project_identifier='minimum-mantis')
    if project.user_id != 13209 or project.competition.id != 23:
        raise ValueError('Cloud credential points to the wrong competition project')
    if any(item._attrs.get('message') == MESSAGE for item in project.submissions.list()):
        raise RuntimeError('Matching remote submission already exists')
    route = (f'/v3/competitions/{project.competition.id}/projects/'
             f'{project.user_id}/{project.name}/tokens')
    response = client.api.post(route, json={})
    response.raise_for_status()
    token = response.json()
    if token.get('type') != 'TEMPORARY' or not token.get('plain'):
        raise RuntimeError('Temporary project upload token was not issued')
    if token.get('project') and (token['project'].get('name') != project.name or
                                 token['project'].get('userId') != project.user_id):
        raise ValueError('Upload token was issued for another project')

    def create_submission(competition_identifier, user_identifier, project_identifier,
                          message, main_file_path, model_directory_path, type,
                          code_files, model_files):
        response = client.api.post(
            f'/v4/competitions/{competition_identifier}/projects/'
            f'{user_identifier}/{project_identifier}/submissions',
            json=dict(message=message, mainFilePath=main_file_path,
                      modelDirectoryPath=model_directory_path, type=type,
                      codeFiles=code_files, modelFiles=model_files, pushToken=token['plain']))
        response.raise_for_status()
        result = response.json()
        receipt = dict(submission_number=result['number'], submission_id=result['id'],
                       project=project.name, message=MESSAGE, code_files=len(code_files),
                       model_files=len(model_files), uploaded=True, cloud_run_started=False,
                       bundle_manifest_sha256=hashlib.sha256(
                           (BUNDLE / 'bundle_manifest.json').read_bytes()).hexdigest())
        with RECEIPT.open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        return result

    with patch.object(Client, 'from_project', return_value=(client, project)), \
         patch.object(client.api, 'create_submission', side_effect=create_submission):
        submission = push(message=MESSAGE, main_file_path='main.py',
                          model_directory_relative_path='resources',
                          include_installed_packages_version=False, no_afterword=True, dry=False)
    fresh = project.submissions.get(submission.number)
    if fresh._attrs['id'] != submission._attrs['id']:
        raise ValueError('Remote submission ID changed during upload')
    receipt = json.loads(RECEIPT.read_text())
    receipt['remote_acceptance_verified'] = True
    receipt['platform_valid'] = fresh._attrs.get('valid')
    receipt['libraries_verified'] = fresh._attrs.get('librariesVerified')
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    settings_path = OUTPUT / 'cloud_run_settings43.json'
    settings = json.loads(settings_path.read_text())
    settings['submissionId'] = receipt['submission_id']
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
