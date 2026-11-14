"""Anchor certificates must name evidence that is committed, not just numbers.

Why this test exists
--------------------
``benchmarks/verified/<anchor>/gates_verified.json`` is a machine-checkable
certificate: every gate carries ``actual``, ``operator`` and ``threshold``, so a
reader recomputes PASS/FAIL without re-running the solver.  The *numbers* are
therefore self-verifying -- but the figure the numbers were read from is not:
until this test existed the whole verified registry shipped **zero** committed
images.  Two failure modes followed and both were live:

* an anchor declared ``fig_convergence``/``fig_field`` that was never committed
  (``fem_heat_conduction_l2``), so its certificate pointed at evidence that does
  not exist -- the auditor grades that ``E_ARTIFACT_MISSING``;
* the two refinement-ladder anchors wrote their charts into the run directory
  and declared nothing (``W_UNDECLARED_ARTIFACTS``), so the certificate's
  evidence chain ended at a file only the original runner ever saw.

A reader of a verified anchor should be able to open the chart behind the gate
verdicts.  These tests pin that contract from the shipped tree only: they read
JSON, check ``git ls-files`` and inspect the PNG header -- they never run a
solver, so they cost milliseconds on every commit.

The auto-discovery test is the anti-regression half: a figure committed into an
anchor directory but *not* named in its certificate fails, so evidence cannot
drift in without being declared.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFIED = REPO / "benchmarks" / "verified"

#: Anchors whose verdict is read off a figure (mesh ladder / field map) and
#: whose certificate must therefore carry the committed image.
FIGURE_ANCHORS = (
    "fem_pinched_cylinder",
    "fem_shell_scordelis_lo",
    "fem_heat_conduction_l2",
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
#: A matplotlib chart of these figures is tens of kB; anything this small is a
#: placeholder (or a truncated write) and must not pass as evidence.
MIN_FIGURE_BYTES = 5_000


def _payload(anchor: str) -> dict:
    path = VERIFIED / anchor / "gates_verified.json"
    assert path.is_file(), f"{anchor}: no certificate at {path.relative_to(REPO)}"
    return json.loads(path.read_text())


def _declared_artifacts(anchor: str) -> list[str]:
    artifacts = _payload(anchor).get("artifacts")
    assert isinstance(artifacts, dict) and artifacts, (
        f"{anchor}: certificate declares no artifacts, so its evidence is unnameable"
    )
    declared: list[str] = []
    for value in artifacts.values():
        if isinstance(value, str):
            declared.append(value)
        elif isinstance(value, list):
            declared.extend(item for item in value if isinstance(item, str))
    return declared


def _tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "benchmarks/verified"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    return {item for item in out.decode().split("\0") if item}


@pytest.mark.parametrize("anchor", FIGURE_ANCHORS)
def test_certificate_declares_figure_evidence(anchor: str) -> None:
    figures = [name for name in _declared_artifacts(anchor) if name.endswith(".png")]
    assert figures, f"{anchor}: certificate declares no figure, nothing backs the verdict visually"
    payload = _payload(anchor)
    assert payload.get("schema") == "tensorfem.verified-gates.v1", anchor


@pytest.mark.parametrize("anchor", FIGURE_ANCHORS)
def test_declared_evidence_exists_and_is_committed(anchor: str) -> None:
    tracked = _tracked_files()
    directory = VERIFIED / anchor
    for name in _declared_artifacts(anchor):
        rel = (directory / name).relative_to(REPO).as_posix()
        assert (directory / name).is_file(), f"{anchor}: declared artifact {name} is not on disk"
        assert rel in tracked, f"{anchor}: declared artifact {name} is not committed (git add it)"


@pytest.mark.parametrize("anchor", FIGURE_ANCHORS)
def test_declared_figure_is_a_non_trivial_png(anchor: str) -> None:
    directory = VERIFIED / anchor
    for name in _declared_artifacts(anchor):
        if not name.endswith(".png"):
            continue
        blob = (directory / name).read_bytes()
        assert blob.startswith(PNG_MAGIC), f"{anchor}: {name} is not a PNG"
        assert len(blob) > MIN_FIGURE_BYTES, (
            f"{anchor}: {name} is {len(blob)} B -- a placeholder, not a chart"
        )


def test_every_committed_figure_is_declared() -> None:
    """Auto-discovery: a figure in an anchor directory must be named by it."""
    tracked = _tracked_files()
    for png in sorted(p for p in tracked if p.endswith(".png")):
        path = Path(png)
        anchor = path.parent.name
        if anchor not in {d.name for d in VERIFIED.iterdir() if d.is_dir()}:
            continue  # figures outside an anchor (docs/, papers/) are not certificates
        declared = set(_declared_artifacts(anchor)) if (VERIFIED / anchor / "gates_verified.json").is_file() else set()
        assert path.name in declared, (
            f"{png} is committed but {anchor}'s certificate does not name it; "
            "declare it in the artifacts map (or do not commit it)"
        )


@pytest.mark.parametrize("anchor", FIGURE_ANCHORS)
def test_auditor_grades_the_artifacts_clean(anchor: str) -> None:
    """The shipped evidence must satisfy the auditor's own artifact checks."""
    import importlib.util
    import sys

    tool_path = REPO / "tools" / "verified_anchor_audit.py"
    spec = importlib.util.spec_from_file_location("verified_anchor_audit_evidence", tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    audit = module.audit_registry()
    records = audit["records"] if isinstance(audit, dict) else audit.records

    def _field(record, name):
        return record[name] if isinstance(record, dict) else getattr(record, name)

    record = next((r for r in records if _field(r, "anchor") == anchor), None)
    assert record is not None, f"{anchor}: not discovered by the certificate audit"
    defects = list(_field(record, "defects"))
    warnings = list(_field(record, "warnings"))
    artifact_defects = [code for code in defects if code.startswith("E_ARTIFACT")]
    assert artifact_defects == [], f"{anchor}: artifact defects {artifact_defects}"
    assert "W_UNDECLARED_ARTIFACTS" not in warnings, (
        f"{anchor}: evidence is on disk but the certificate names none of it"
    )