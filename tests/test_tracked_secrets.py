"""The credential gate must catch synthetic leaks without repeating the secret."""

import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "secret_gate", ROOT / "scripts/check_tracked_secrets.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_recognized_tokens_and_credential_files_fail_without_value_disclosure(tmp_path):
    def encode(data):
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=")

    jwt = b".".join([encode({"alg": "HS256"}), encode({"sub": "synthetic-test"}), b"x" * 32])
    token = "gh" + "p_" + "x" * 36
    (tmp_path / "leak.py").write_text(token)
    (tmp_path / ".tok").write_bytes(jwt)
    password = "synthetic-password"
    (tmp_path / "auth.py").write_text("payload = " + repr(dict(password=password)))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_tracked_secrets.py"), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert token not in result.stdout + result.stderr
    assert jwt.decode() not in result.stdout + result.stderr
    assert password not in result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert {item["rule"] for item in report["findings"]} == {
        "credential_file",
        "jwt",
        "github_token",
        "credential_literal",
    }


def test_templates_are_allowed_and_tree_is_clean(tmp_path):
    (tmp_path / ".env.example").write_text("TENSOR_API_TOKEN=replace-me\n")
    (tmp_path / "readme.md").write_text(
        "Use an environment variable, never a committed credential."
    )
    assert gate.scan_paths(tmp_path, [".env.example", "readme.md"]) == []
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_tracked_secrets.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
