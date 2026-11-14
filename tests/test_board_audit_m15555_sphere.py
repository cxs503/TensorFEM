"""Guards for the m15555 board audit of CASE-SPHERE-PRESSURE (mission 15405).

The audit used to be prose.  Its own evidence inventory had gone stale twice
(``result.json`` 43850 B vs 43980 B committed, ``run_verification.py`` 31267 B vs
the 1613 B thin wrapper that replaced the monolith), so the audit is now a
generated artefact (``scripts/audit_m15555_sphere_board.py``) with a drift gate
and a *measured* reproduction claim.

These tests defend the substance, not the file format:

* every number the audit prints about the case is re-derived here from
  ``result.json`` / ``gates_verified.json`` with the test's own arithmetic --
  the generator is not trusted to check itself;
* the evidence carriers' bytes/sha256 recorded in ``verify_output.json`` still
  match the files on disk, and the stale hand-typed numbers are gone for good;
* ``--check`` reports the audit in sync (doc + json + figure + reproduction);
* the "the run reproduces" claim is re-measured here, independently of the
  generator, with the predeclared 1e-9 relative criterion;
* the committed figure is real, lives under ``docs/`` (the only path class the
  push gate carries binaries from) and matches a fresh render up to PNG noise.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO_ROOT / "benchmarks" / "verified" / "fem_sphere_pressure_membrane"
RESULT_JSON = ANCHOR_DIR / "result.json"
GATES_JSON = ANCHOR_DIR / "gates_verified.json"
AUDIT_DIR = REPO_ROOT / "docs" / "board_audit" / "m15555"
AUDIT_DOC = AUDIT_DIR / "AUDIT_m15405_CASE_SPHERE_PRESSURE.md"
VERIFY_JSON = AUDIT_DIR / "verify_output.json"
FIGURE = AUDIT_DIR / "fig_m15555_sphere_audit.png"
GENERATOR = REPO_ROOT / "scripts" / "audit_m15555_sphere_board.py"

#: Predeclared reproduction criterion (see the generator's docstring).
REPRO_REL_TOL = 1e-9
TOLERANCE = 0.03


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT_JSON.read_text())


@pytest.fixture(scope="module")
def certificate() -> dict:
    return json.loads(GATES_JSON.read_text())


@pytest.fixture(scope="module")
def verify_output() -> dict:
    assert VERIFY_JSON.is_file(), "m15555 audit must ship its machine-extracted numbers"
    return json.loads(VERIFY_JSON.read_text())


@pytest.fixture(scope="module")
def audit_text() -> str:
    return AUDIT_DOC.read_text()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(a, b, path="", tol=REPRO_REL_TOL, drift=None, leaves=None):
    """Independent leaf comparison (own implementation, not the generator's)."""
    if drift is None:
        drift, leaves = [], [0]
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                drift.append(f"{path}/{key}: key set differs")
                continue
            _compare(a[key], b[key], f"{path}/{key}", tol, drift, leaves)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            drift.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (u, v) in enumerate(zip(a, b)):
                _compare(u, v, f"{path}[{i}]", tol, drift, leaves)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            drift.append(f"{path}: {a} != {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        leaves[0] += 1
        if a != b:
            rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
            if rel > tol:
                drift.append(f"{path}: rel {rel:.3e}")
    elif a != b:
        drift.append(f"{path}: {a!r} != {b!r}")
    return drift, leaves[0]


# --------------------------------------------------------------------------- #
# 1. the audit is in sync with the evidence
# --------------------------------------------------------------------------- #
def test_audit_generator_check_is_clean():
    """`--check` re-derives everything (incl. a real re-run) and must pass."""
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_evidence_carriers_unchanged(verify_output, result, certificate):
    """Bytes + sha256 quoted by the audit still describe the committed evidence."""
    recorded = {row["path"]: row for row in verify_output["inventory"]["evidence"]}
    on_disk = {
        "benchmarks/verified/fem_sphere_pressure_membrane/result.json": RESULT_JSON,
        "benchmarks/verified/fem_sphere_pressure_membrane/gates_verified.json": GATES_JSON,
    }
    assert set(recorded) == set(on_disk)
    for rel, path in on_disk.items():
        assert recorded[rel]["bytes"] == path.stat().st_size, f"{rel}: byte count drifted"
        assert recorded[rel]["sha256"] == _sha256(path), f"{rel}: sha256 drifted"
        assert recorded[rel]["git_tracked"] is True
    assert result["case"] == "CASE-SPHERE-PRESSURE"
    assert certificate["anchor"] == "fem_sphere_pressure_membrane"


def test_stale_hand_typed_numbers_are_gone(audit_text, verify_output):
    """The two numbers that rotted must not come back as hand-typed prose."""
    result_bytes = RESULT_JSON.stat().st_size
    assert f"{result_bytes} B" in audit_text
    # the two rotted numbers may survive only as *history* inside the finding that
    # records them, never as a current reading anywhere else
    for stale in ("43850", "31267"):
        holders = [line for line in audit_text.splitlines() if stale in line]
        assert holders, f"{stale} should still be documented as the stale reading"
        for line in holders:
            assert "上一版" in line or "重构前" in line, (
                f"stale artefact size {stale} is quoted outside the F4 history line: {line}"
            )
    # producers are quoted by path only, so editing one cannot invalidate the audit
    roles = {row["path"] for row in verify_output["inventory"]["producers"]}
    assert "benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py" in roles
    assert not any("bytes" in row for row in verify_output["inventory"]["producers"])


# --------------------------------------------------------------------------- #
# 2. the audit's case numbers are re-derived here, independently
# --------------------------------------------------------------------------- #
def test_membrane_stress_numbers_rederived(result, certificate, verify_output):
    ladder = result["convergence"]["ladder"]["exact_meridian"]
    row8 = next(row for row in ladder if row["facets"] == 8)
    exact = row8["sigma_ref_pa"]
    # own arithmetic straight from the raw Gauss-point arrays
    worst = max(abs(sigma / exact - 1.0) for sigma in row8["sigma_theta_pa"])
    assert worst == pytest.approx(row8["sigma_theta_max_rel_error"], rel=1e-12)
    assert worst < TOLERANCE
    reported = verify_output["case_numbers"]["site_facets_8"]
    assert reported["sigma_theta_rel_error"] == pytest.approx(worst, rel=1e-12)
    assert reported["sigma_theta_max_pa"] == pytest.approx(max(row8["sigma_theta_pa"]), rel=1e-15)

    gates = {g["id"]: g for g in certificate["gates"]}
    g1 = gates["G1.sigma_theta_vs_pR_2t"]
    assert g1["value"] == pytest.approx(worst, rel=1e-12)
    assert g1["threshold"] == TOLERANCE and g1["passed"] is True

    # every gating gate recomputes from its own value/operator/threshold
    operators = {
        "<=": lambda v, t: v <= t,
        "<": lambda v, t: v < t,
        ">=": lambda v, t: v >= t,
        ">": lambda v, t: v > t,
    }
    for gate in certificate["gates"]:
        assert gate["operator"] in operators, gate["id"]
        expected = operators[gate["operator"]](gate["value"], gate["threshold"])
        assert gate["passed"] is bool(expected), gate["id"]
    assert verify_output["certificate"]["n_passed"] == len(certificate["gates"])
    assert verify_output["certificate"]["passed"] is True


def test_controls_and_falsified_prediction_are_intact(certificate, verify_output):
    gates = {g["id"]: g for g in certificate["gates"]}
    for gate_id in ("G9.chord_control_fails", "G10.flat_facet_shell4_control_fails"):
        assert gates[gate_id]["passed"] is True
        assert gates[gate_id]["operator"] == ">="  # negative control must MISS
        assert gates[gate_id]["value"] > gates[gate_id]["threshold"]

    predictions = certificate["predictions"]
    falsified = [p for p in predictions if p["outcome"] == "falsified"]
    assert [p["id"] for p in falsified] == ["P1.reference_band_reaches_t_over_R_0.05"]
    assert falsified[0]["predicted_threshold"] == 0.05
    band = verify_output["case_numbers"]["reference_band"]
    assert band["t_over_R_band_within_tol"] == [0.001, 0.02]
    assert band["prediction_outcome"] == "falsified"
    # the falsified prediction is not a gate: the gating count must not include it
    assert len(predictions) == 1
    assert all(not g["id"].startswith("P1") for g in certificate["gates"])


# --------------------------------------------------------------------------- #
# 3. "the run reproduces" is re-measured here, not recalled
# --------------------------------------------------------------------------- #
def test_reproduction_independent_of_generator():
    """Re-execute the case and require no drift beyond the predeclared 1e-9."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tensorfem import sphere_pressure_case as sp  # noqa: PLC0415

    committed = json.loads(RESULT_JSON.read_text())
    with tempfile.TemporaryDirectory(prefix="m15555_test_") as tmp:
        fresh = sp.run_case(out_dir=tmp)

    volatile = ("figures", "environment", "flat_facet_control")
    for record in (committed, fresh):
        record.pop("figures", None)
        record.pop("environment", None)
        control = record.get("flat_facet_control") or {}
        control.pop("wall_clock_s", None)

    drift_control, leaves_control = _compare(committed["flat_facet_control"],
                                             fresh["flat_facet_control"])
    assert not drift_control, f"flat-facet control drifted beyond 1e-9: {drift_control[:5]}"

    drift, leaves = _compare(
        {k: v for k, v in committed.items() if k not in volatile},
        {k: v for k, v in fresh.items() if k not in volatile},
    )
    assert leaves > 1000, "the comparison must actually walk the solve record"
    assert not drift, f"production path drifted beyond 1e-9: {drift[:5]}"
    assert leaves + leaves_control > 1000


def test_reproduction_verdict_is_recorded(verify_output):
    repro = verify_output["reproduction"]
    assert repro["reproduced"] is True
    assert repro["drift_paths"] == []
    assert repro["rel_tolerance"] == REPRO_REL_TOL
    assert repro["compared_leaves"] > 1000
    # the noise bound is reported as a one-sided margin, never as a measured value
    # that a 1-ulp coin could flip
    assert isinstance(repro["noise_margin_decades_geq"], int)
    assert repro["noise_margin_decades_geq"] >= 1
    assert repro["control_reruns"] >= 2
    assert "flat_facet_control/wall_clock_s" in repro["excluded_paths"]
    assert set(repro["excluded_paths"]) == set(repro["excluded_reason"])


# --------------------------------------------------------------------------- #
# 4. the figure is real committed evidence
# --------------------------------------------------------------------------- #
def test_figure_is_committed_under_docs():
    assert FIGURE.is_file() and FIGURE.stat().st_size > 20_000
    rel = FIGURE.relative_to(REPO_ROOT).as_posix()
    assert rel.startswith("docs/"), "only docs/ figures survive the push gate"
    assert FIGURE.name in AUDIT_DOC.read_text(), "the audit doc must embed its figure"


def test_figure_matches_a_fresh_render(tmp_path):
    """PNG metadata may differ across library versions; the pixels may not."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcdefaults()  # an earlier solve may have left global style state behind
    import matplotlib.image as mpimg
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import audit_m15555_sphere_board as gen  # noqa: PLC0415

    fresh = tmp_path / Path(FIGURE.name).name
    gen.render_figure(str(fresh))
    committed = mpimg.imread(FIGURE)
    rendered = mpimg.imread(fresh)
    assert committed.shape == rendered.shape
    assert np.allclose(committed, rendered, atol=1e-6)