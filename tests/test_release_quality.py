import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_api_audit_script_passes_with_warnings_as_errors():
    # Keep project warnings fatal while tolerating PyTorch's optional NumPy
    # bridge warning in a valid torch-only installation.
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "PYTHONWARNINGS": "error,ignore:Failed to initialize NumPy"}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_public_api.py"), "--json"],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout)["ok"]


def test_release_metadata_and_required_files_are_present():
    for name in ("LICENSE", "README.md", "CHANGELOG.md", "pyproject.toml", "api/public-api-v1.json"):
        assert (ROOT / name).is_file()
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'tensorfem = ["public_api_manifest.json"]' in pyproject
    assert 'tensorfem = "tensorfem.cli:main"' in pyproject


def test_full_offline_release_smoke_is_opt_in():
    if os.environ.get("TENSORFEM_RELEASE_SMOKE") != "1":
        return
    subprocess.run([sys.executable, str(ROOT / "scripts/release_smoke.py"), "--build"], check=True, cwd=ROOT)
