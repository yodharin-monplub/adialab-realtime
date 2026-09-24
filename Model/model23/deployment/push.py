"""Push the sealed Model 23 bundle only; never create a cloud run."""
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import os
import re

from crunch.api import Client, auth
from crunch.command.push import list_code_files, list_model_files, push

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "submissions/submission23"
RECEIPT = ROOT / "models/model23/deployment_validation/push_receipt.json"
MESSAGE = "Submission 23: paired aligned25/aligned60 XGBoost; pretrained inference only; 12 workers"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Actually upload; otherwise validate locally only")
    args = parser.parse_args()
    if RECEIPT.exists():
        raise RuntimeError("An upload receipt already exists; inspect it before attempting another push")
    key = os.environ.get("CRUNCH_API_KEY") or re.search(
        r"Crunch_Api_key\s*=\s*(\S+)", (ROOT / "AGENTS.md").read_text()).group(1)
    package = json.loads((RECEIPT.parent / "package.json").read_text())
    archive = ROOT / package["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == package["sha256"]
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    expected = manifest["files"]
    for name, spec in expected.items():
        data = (BUNDLE / name).read_bytes()
        assert len(data) == spec["bytes"]
        assert hashlib.sha256(data).hexdigest() == spec["sha256"], name
        assert key.encode() not in data, f"Credential in {name}"
    os.chdir(BUNDLE)  # SDK ignore rules are relative to the submission root.
    code = list(list_code_files(str(BUNDLE), "resources"))
    models = list(list_model_files(str(BUNDLE), "resources"))
    actual = {name for _, name in code} | {"resources/" + name for _, name in models}
    assert actual == set(expected) | {"bundle_manifest.json"}, "Upload inventory changed"
    assert not any(name.startswith("resources/") for _, name in code)
    print(f"Verified {len(code)} code files and {len(models)} model files", flush=True)
    if not args.push:
        return
    client = Client.from_env(auth=auth.ApiKeyAuth(key), show_progress=False)
    project = client.competitions.get("structural-break-real-time").projects.get(project_identifier="minimum-mantis")
    assert project.user_id == 13209 and project.competition.id == 23
    assert not any(s._attrs.get("message") == MESSAGE for s in project.submissions.list()), "Matching remote submission already exists"
    route = f"/v3/competitions/{project.competition.id}/projects/{project.user_id}/{project.name}/tokens"
    response = client.api.post(route, json={})
    response.raise_for_status()
    token = response.json()
    assert token["type"] == "TEMPORARY" and token.get("plain")
    # Some server versions omit the optional project object; the creation URL
    # already scopes this temporary token to the verified competition/project.
    if token.get("project"):
        assert token["project"]["name"] == project.name and token["project"]["userId"] == project.user_id

    def create_submission(competition_identifier, user_identifier, project_identifier,
                          message, main_file_path, model_directory_path, type, code_files, model_files):
        response = client.api.post(
            f"/v4/competitions/{competition_identifier}/projects/{user_identifier}/{project_identifier}/submissions",
            json=dict(message=message, mainFilePath=main_file_path,
                      modelDirectoryPath=model_directory_path, type=type,
                      codeFiles=code_files, modelFiles=model_files, pushToken=token["plain"]))
        response.raise_for_status()
        result = response.json()
        receipt = dict(submission_number=result["number"], submission_id=result["id"],
                       project=project.name, message=message, code_files=len(code_files),
                       model_files=len(model_files), uploaded=True, cloud_run_started=False,
                       bundle_manifest_sha256=hashlib.sha256((BUNDLE / "bundle_manifest.json").read_bytes()).hexdigest(),
                       temporary_token_expires=token.get("validUntil"))
        # Persist acceptance before SDK cleanup or later read-back checks.
        with RECEIPT.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        return result

    with patch.object(Client, "from_project", return_value=(client, project)), \
         patch.object(client.api, "create_submission", side_effect=create_submission):
        submission = push(message=MESSAGE, main_file_path="main.py", model_directory_relative_path="resources",
                          include_installed_packages_version=False, no_afterword=True, dry=False)
    fresh = project.submissions.get(submission.number)
    assert fresh._attrs["id"] == submission._attrs["id"]
    receipt = json.loads(RECEIPT.read_text())
    receipt["remote_acceptance_verified"] = True
    receipt["platform_valid"] = fresh._attrs.get("valid")
    receipt["libraries_verified"] = fresh._attrs.get("librariesVerified")
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    settings_path = ROOT / "models/model23/deployment_validation/cloud_run_settings.json"
    settings = json.loads(settings_path.read_text())
    settings["submissionId"] = receipt["submission_id"]
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # API errors can contain credential-bearing URLs or response bodies.
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        print(f"Push stopped: {type(exc).__name__}; HTTP status {status}. Check the receipt before retrying.", flush=True)
        raise SystemExit(1) from None

