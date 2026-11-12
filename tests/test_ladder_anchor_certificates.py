"""Certificate invariants for the anchors closed by ``tools/certify_ladder_anchor.py``.

These are *invariant* assertions, not "the script ran" smoke tests:

* every gate in ``gates_verified.json`` is recomputable -- its declared verdict
  must equal ``operator(value, threshold)``, so a hand-edit that flips a verdict
  turns red without re-running any solver;
* every gate value must still be reproducible from the anchor's own
  ``result.json`` (the tool re-derives each quantity independently), so a
  refreshed artifact cannot drift away from its certificate;
* the certified anchors must be absent from the shrink-only debt registry --
  otherwise a later ``--write`` could silently re-admit them as debt;
* the folded-in reference sensitivity (fem_plate order on two different
  references) and the recorded anomaly (fem_kerr_plate free-plate finest pair)
  are pinned, so a future "cleanup" of these numbers has to be deliberate.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = REPO_ROOT / "benchmarks" / "verified"
DEBT_JSON = VERIFIED_DIR / "anchor_certificate_debt.json"
GATE_SCHEMA = "tensorfem.verified-gates.v1"

sys.path.insert(0, str(REPO_ROOT / "tools"))

import certify_ladder_anchor as certify  # noqa: E402

OPERATORS = {
    "<=": lambda v, t: v <= t,
    "<": lambda v, t: v < t,
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
    "==": lambda v, t: math.isclose(v, t, rel_tol=1e-12, abs_tol=0.0),
}

CERTIFIED = ("fem_beam", "fem_plate", "fem_kerr_plate")


def _envelope(anchor: str) -> dict:
    path = VERIFIED_DIR / anchor / "gates_verified.json"
    assert path.is_file(), f"{anchor} lost its certificate envelope"
    return json.loads(path.read_text())


@pytest.mark.parametrize("anchor", CERTIFIED)
def test_envelope_schema_and_identity(anchor: str) -> None:
    payload = _envelope(anchor)
    assert payload["schema"] == GATE_SCHEMA
    assert payload["anchor"] == anchor, "audit rejects a mismatched anchor name"
    assert isinstance(payload["gates"], list) and payload["gates"], "no gates = no certificate"
    assert isinstance(payload["passed"], bool)


@pytest.mark.parametrize("anchor", CERTIFIED)
def test_every_gate_verdict_is_recomputable(anchor: str) -> None:
    payload = _envelope(anchor)
    for gate in payload["gates"]:
        value = gate["value"]
        threshold = gate["threshold"]
        operator = gate["operator"]
        assert operator in OPERATORS, f"{anchor}/{gate['id']}: unknown operator {operator!r}"
        assert isinstance(value, (int, float)) and isinstance(threshold, (int, float))
        assert math.isfinite(value) and math.isfinite(threshold)
        expected = bool(OPERATORS[operator](value, threshold))
        assert gate["passed"] is expected, (
            f"{anchor}/{gate['id']}: declares passed={gate['passed']} but "
            f"{value!r} {operator} {threshold!r} is {expected}"
        )


@pytest.mark.parametrize("anchor", CERTIFIED)
def test_overall_flag_matches_gates(anchor: str) -> None:
    payload = _envelope(anchor)
    failed = [g["id"] for g in payload["gates"] if not g["passed"]]
    assert payload["failed_gates"] == failed
    assert payload["passed"] is (not failed)


@pytest.mark.parametrize("anchor", CERTIFIED)
def test_declared_artifacts_exist_and_are_tracked(anchor: str) -> None:
    payload = _envelope(anchor)
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict) and artifacts
    paths = []
    for item in artifacts.values():
        paths.extend([item] if isinstance(item, str) else item)
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    for rel in paths:
        assert (REPO_ROOT / rel).is_file(), f"{anchor}: declared artifact {rel} is missing"
        assert rel in tracked, f"{anchor}: declared artifact {rel} is not tracked by git"


@pytest.mark.parametrize("anchor", CERTIFIED)
def test_envelope_is_not_stale(anchor: str) -> None:
    """Re-deriving every gate from result.json must reproduce the shipped envelope."""
    spec = certify.ANCHORS[anchor]
    fresh = certify.certify(anchor, spec, write=False)  # raises SystemExit when stale
    assert fresh["passed"] is True


@pytest.mark.parametrize("anchor", CERTIFIED)
def test_certified_anchor_is_not_registered_as_debt(anchor: str) -> None:
    debt = json.loads(DEBT_JSON.read_text())["anchors"]
    assert anchor not in debt, (
        f"{anchor} is certified yet still listed in the shrink-only debt registry; "
        "run tools/verified_anchor_audit.py --write"
    )


def test_debt_registry_cannot_grow_silently() -> None:
    """The audit must agree with the registry, and the registry must stay <= 24 entries."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "verified_anchor_audit.py"), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    debt = json.loads(DEBT_JSON.read_text())["anchors"]
    assert len(debt) <= 24, (
        f"debt registry grew to {len(debt)} entries; certification is shrink-only "
        "(was 24 after fem_beam/fem_plate/fem_kerr_plate were closed)"
    )


def test_fem_plate_order_depends_on_the_reference() -> None:
    """Pinned finding: 1.9440 is the *series* fit; the 0.00406 table gives 2.03.

    Both exceed the 1.5 gate, so the verdict is robust, but the stored scalar is
    only reproducible against one of the two references shipped in the artifact.
    """
    result = json.loads((VERIFIED_DIR / "fem_plate" / "result.json").read_text())
    levels = result["levels"]

    def fit(metric: str) -> float:
        return certify._order_by_least_squares(levels, xkey="n_x", ykey=metric, lo=2, hi=24)

    series = fit("error_vs_navier_series")
    timoshenko = fit("error_vs_timoshenko_00406")
    assert series == pytest.approx(result["observed_order"], rel=1e-9), (
        "the stored observed_order is the convergence rate against the Navier series"
    )
    assert timoshenko == pytest.approx(2.0315, abs=5e-4)
    assert abs(series - timoshenko) > 0.05, "the reference sensitivity is ~4.5%, not zero"
    assert series >= 1.5 and timoshenko >= 1.5, "the gate verdict is reference-independent"


def test_fem_kerr_plate_free_plate_finest_pair_is_anomalous() -> None:
    """Pinned finding: the free-plate branch's last pair drops the error by ~72x.

    That is what inflates the artifact's last-two-level ``convergence_order`` to
    6.17 for a 2nd-order element, which is why the certificate gates the
    least-squares order and records the anomaly instead of hiding it.
    """
    result = json.loads((VERIFIED_DIR / "fem_kerr_plate" / "result.json").read_text())
    branch = result["results"]["free_plate"]
    errs = [mesh["rel_err"] for mesh in branch["meshes"]]
    finest_ratio = errs[-1] / errs[-2]
    assert finest_ratio < 0.05, "the free-plate finest-pair jump is ~1/72"
    assert branch["convergence_order"] > 5.0, "stored last-pair order is the inflated 6.17"

    envelope = _envelope("fem_kerr_plate")
    gated = {gate["id"]: gate for gate in envelope["gates"]}
    assert gated["kerr_convergence_order_lsq"]["value"] == pytest.approx(2.0042, abs=5e-4)
    assert any("6.17" in text for text in envelope["limitations"]), (
        "the anomaly must stay documented in the certificate"
    )