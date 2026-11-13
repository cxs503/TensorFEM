"""Fail-closed gate over the *renderable figure* policy of anchor certificates.

``benchmarks/verified/*/gates_verified.json`` used to declare binary figures
under ``artifacts``.  The delivery merge gate strips binaries, so those anchors
went red once per merge cycle (git history: ``d094673b`` restored 32 figures,
``c4070685`` restored them again; on 2026-11-13 ``git ls-files '*.png'`` counted
489 repo-wide and 0 under ``benchmarks/verified/``).  The policy implemented by
``tools/certificate_renderables.py`` keeps the evidence claim while making it
keepable: an absent figure moves to ``figures_rendered_on_demand`` with a
committed producer script that names it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "certificate_renderables.py"


def _load():
    spec = importlib.util.spec_from_file_location("certificate_renderables", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    assert TOOL.is_file()
    return _load()


@pytest.fixture(scope="module")
def audit(tool):
    return tool.audit_registry()


def test_policy_is_clean_on_the_live_registry(audit):
    """No unregistered violation: an absent declared figure must be migrated or booked as debt."""
    assert audit["violations"] == [], json.dumps(audit["violations"][:5], indent=1)


def test_renderable_entries_carry_a_committed_producer(tool):
    tracked = tool.tracked_files()
    entries = 0
    for path in sorted(tool.VERIFIED_DIR.glob("*/gates_verified.json")):
        payload = json.loads(path.read_text())
        for entry in payload.get(tool.RENDERABLE_KEY) or []:
            entries += 1
            assert entry["rendered_by"] in tracked, entry
            assert (REPO_ROOT / entry["rendered_by"]).parent == path.parent, entry
            assert tool.producer_for(entry["path"], path.parent, tracked) == entry["rendered_by"]
    assert entries >= 30, "the migration is expected to cover the figure-only anchors"


def test_migration_never_hides_a_figure_that_exists(tool, tmp_path):
    """Existence wins: a present figure must not be laundered into the renderable list."""
    directory = tool.VERIFIED_DIR / "fem_sphere_pressure_membrane"
    payload = json.loads((directory / "gates_verified.json").read_text())
    payload["artifacts"]["existing_figure"] = "README.md"
    payload[tool.RENDERABLE_KEY] = list(payload.get(tool.RENDERABLE_KEY) or []) + [
        {"path": "README.md", "rendered_by": "benchmarks/verified/fem_sphere_pressure_membrane/build_report.py"}
    ]
    target = tmp_path / "fem_sphere_pressure_membrane"
    target.mkdir()
    (target / "gates_verified.json").write_text(json.dumps(payload))
    record = tool.audit_certificate(target / "gates_verified.json", tool.tracked_files())
    codes = {v["code"] for v in record["violations"]}
    assert "E_RENDERABLE_BUT_PRESENT" in codes


def test_figure_without_producer_stays_declared(tool, tmp_path):
    """No producer means no renderable claim: the figure stays a hard defect.

    Historically this was asserted on ``fem_winkler_plate_k_sweep`` (which had no
    producer and sat in the debt registry). It now has one
    (``tools/anchor_figure_rebuild.py``) and is certified, so the property is
    asserted on a synthetic anchor where the condition is actually true.
    """
    target = tmp_path / "fem_synthetic_no_producer"
    target.mkdir()
    payload = {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": target.name,
        "gates": [{"id": "g", "quantity": "synthetic", "value": 0.0,
                   "threshold": 1.0, "operator": "<=", "passed": True}],
        "passed": True,
        "artifacts": {"profile_png": "synthetic_profile.png"},
    }
    (target / "gates_verified.json").write_text(json.dumps(payload))
    record = tool.audit_certificate(target / "gates_verified.json", tool.tracked_files())
    assert {v["code"] for v in record["violations"]} == {"E_FIGURE_DECLARED_BUT_ABSENT"}
    assert all(v["anchor"] == target.name for v in record["violations"])


def test_rebuilt_anchor_left_the_debt_registry(tool):
    debt = json.loads((tool.VERIFIED_DIR / "anchor_certificate_debt.json").read_text())["anchors"]
    for anchor in ("fem_inverse_identification", "fem_sent_j_integral",
                   "fem_winkler_plate_k_sweep"):
        assert anchor not in debt, f"{anchor} is repaired but still listed as debt"
