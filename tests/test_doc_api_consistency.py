"""Guard: README/examples must stay consistent with the shipped API.

The README advertises an install-and-verify workflow, a minimal example and
pointers into ``examples/``. Those are user-facing contracts, so they are
checked mechanically instead of by review:

* every ``from tensorfem... import ...``/``tensorfem.attr`` chain in README
  code fences and in every ``examples/*.py`` resolves in the current tree;
* every keyword argument used at such a call site exists in the real signature;
* every relative path and ``python examples/...`` command quoted by README
  exists on disk;
* the CLI commands the README tells users to run actually run.

The audit implementation lives in ``scripts/audit_doc_api_consistency.py`` so
it can also be run by hand (``--out`` JSON report for evidence).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_doc_api_consistency.py"


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    out = tmp_path_factory.mktemp("doc-audit") / "audit.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--examples", "--out", str(out)],
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=900,
    )
    payload = json.loads(out.read_text()) if out.exists() else None
    return proc, payload


def test_readme_and_examples_resolve_against_shipped_api(audit):
    proc, payload = audit
    assert payload is not None, proc.stderr
    errors = payload["summary"]["errors"]
    assert not errors, "documentation/code drift:\n" + "\n".join(
        f"[{e['kind']}] {e['source']}:{e['where']} {e['claim']} :: {e['detail']}"
        for e in errors
    )
    assert proc.returncode == 0
    # The audit has to stay non-trivial: a broken extractor would silently pass.
    assert payload["summary"]["total"] >= 150


def _cli(argv, timeout=900):
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-c",
         "import sys;from tensorfem.cli import main;sys.argv=['tensorfem',*sys.argv[1:]];main()",
         *argv],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_documented_capabilities_command_runs():
    """`tensorfem capabilities` is one of the README's install-verification commands."""
    capabilities = _cli(["capabilities"], timeout=300)
    assert capabilities.returncode == 0, capabilities.stderr[-2000:]
    payload = json.loads(capabilities.stdout)
    assert payload["stable"] and payload["experimental"]


@pytest.mark.slow
def test_documented_verify_command_runs():
    """`tensorfem verify --output ...` is the README's release gate; ~100 s."""
    verify = _cli(["verify", "--output", "/dev/null"])
    assert verify.returncode == 0, verify.stdout[-2000:] + verify.stderr[-2000:]
    assert '"passed": true' in verify.stdout