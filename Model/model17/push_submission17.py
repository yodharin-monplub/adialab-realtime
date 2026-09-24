"""Upload the sealed Model 17 using a prompted token; never start a cloud run."""
from pathlib import Path
from unittest.mock import patch
import getpass
import hashlib
import json
import os
import re

from crunch.api import Client, auth
from crunch.command.push import list_code_files, list_model_files, push

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "submissions/submission17"
OUTPUT = ROOT / "models/model17/deployment_validation"
RECEIPT = OUTPUT / "push_receipt.json"
MESSAGE = "Submission 17: pretrained 85/15 Model16/Model6 timeout-safe blend; 12 inference workers"


def main():
    if RECEIPT.exists():
        raise RuntimeError("A push receipt already exists; inspect before retrying")
    token = getpass.getpass("Crunch project upload token: ").strip()
    if not token:
        raise ValueError("Empty token")
    key = os.environ.get("CRUNCH_API_KEY") or re.search(
        r"Crunch_Api_key\s*=\s*(\S+)", (ROOT / "AGENTS.md").read_text()).group(1)
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    package = json.loads((OUTPUT / "package.json").read_text())
    archive = ROOT / package["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == package["sha256"]
    expected = manifest["files"]
    for name, spec in expected.items():
        data = (BUNDLE / name).read_bytes()
        assert len(data) == spec["bytes"] and hashlib.sha256(data).hexdigest() == spec["sha256"], name
        assert key.encode() not in data and token.encode() not in data, "Credential detected in bundle"
    os.chdir(BUNDLE)
    code = list(list_code_files(str(BUNDLE), "resources"))
    models = list(list_model_files(str(BUNDLE), "resources"))
    assert {name for _, name in code} | {"resources/" + name for _, name in models} == set(expected) | {"bundle_manifest.json"}
    assert len(code) == package["code_files"] and len(models) == package["model_files"]
    print(f"Verified {len(code)} source files and {len(models)} model files", flush=True)
    client = Client.from_env(auth=auth.ApiKeyAuth(key), show_progress=False)
    project = client.competitions.get("structural-break-real-time").projects.get(project_identifier="minimum-mantis")
    assert project.user_id == 13209 and project.competition.id == 23
    assert not any(s._attrs.get("message") == MESSAGE for s in project.submissions.list()), "Matching submission already exists remotely"

    def create_submission(competition_identifier, user_identifier, project_identifier,
                          message, main_file_path, model_directory_path, type, code_files, model_files):
        response = client.api.post(
            f"/v4/competitions/{competition_identifier}/projects/{user_identifier}/{project_identifier}/submissions",
            json=dict(message=message, mainFilePath=main_file_path, modelDirectoryPath=model_directory_path,
                      type=type, codeFiles=code_files, modelFiles=model_files, pushToken=token))
        response.raise_for_status()
        result = response.json()
        receipt = dict(submission_number=result["number"], submission_id=result["id"], project=project.name,
                       message=message, code_files=len(code_files), model_files=len(model_files),
                       uploaded=True, cloud_run_started=False,
                       bundle_manifest_sha256=hashlib.sha256((BUNDLE / "bundle_manifest.json").read_bytes()).hexdigest())
        with RECEIPT.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        print(f"Accepted as platform submission {result['number']}, ID {result['id']}", flush=True)
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
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        print(f"Push stopped: {type(error).__name__}; HTTP status {status}. Inspect the receipt before retrying.", flush=True)
        raise SystemExit(1) from None
