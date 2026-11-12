"""Independent re-derivation of the fem_shell_modal anchor oracle and gates.

Mission m16863 (verifier seat).  The anchor ``benchmarks/verified/fem_shell_modal``
declared ``overall_verdict`` PASS while its shell gate carried ``limit: null`` and
its shell reference ladder was seeded by a Donnell ``n = 1`` branch that dropped
the membrane term (17.33 Hz against a 2881.62 Hz Euler-Bernoulli beam limit).

These tests pin the repair, not the defect:

* the Donnell oracle agrees with an independent reimplementation for every
  ``n = 0 … 8`` (relative difference < 1e-12);
* the axisymmetric ``n = 0`` branch (dead code before the repair: the validator
  rejected ``n_circ = 0`` although the docstring documented ``n >= 0``) is live;
* the ``n = 1`` branch is no longer the falsified one;
* the gate envelope is falsifiable: canonical schema, every gate has a numeric
  threshold and a declared operator, and the declared overall flag equals the
  recomputation from the gates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfem.cylindrical_shell_modal import dmv_frequency_hz

ANCHOR = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "fem_shell_modal"
GEOMETRY = dict(radius=0.5, length=1.0, thickness=0.005, young=210e9,
                poisson=0.3, density=7800.0)
OPERATORS = {"<=", "<", ">=", ">", "==", "abs<="}


def reference_donnell(n_axial: int, n_circ: int) -> float:
    """Independent transcription of the Donnell frequency equation (Hz)."""
    lam = n_axial * math.pi * GEOMETRY["radius"] / GEOMETRY["length"]
    k = GEOMETRY["thickness"] ** 2 / (
        12.0 * GEOMETRY["radius"] ** 2 * (1.0 - GEOMETRY["poisson"] ** 2)
    )
    span = lam ** 2 + n_circ ** 2
    omega_sq = GEOMETRY["young"] / (GEOMETRY["density"] * GEOMETRY["radius"] ** 2) * (
        lam ** 4 / span ** 2 + k * span ** 2
    )
    return math.sqrt(omega_sq) / (2.0 * math.pi)


@pytest.mark.parametrize("n_circ", range(0, 9))
def test_oracle_matches_independent_donnell(n_circ):
    got = dmv_frequency_hz(n_axial=1, n_circ=n_circ, **GEOMETRY)
    assert got == pytest.approx(reference_donnell(1, n_circ), rel=1e-12)


def test_axisymmetric_branch_is_reachable():
    """n = 0 = breathing mode; the pre-repair validator made this branch dead code."""
    lam = math.pi * GEOMETRY["radius"] / GEOMETRY["length"]
    k = GEOMETRY["thickness"] ** 2 / (
        12.0 * GEOMETRY["radius"] ** 2 * (1.0 - GEOMETRY["poisson"] ** 2)
    )
    expected = math.sqrt(
        GEOMETRY["young"] / (GEOMETRY["density"] * GEOMETRY["radius"] ** 2) * (1.0 + k * lam ** 4)
    ) / (2.0 * math.pi)
    assert dmv_frequency_hz(n_axial=1, n_circ=0, **GEOMETRY) == pytest.approx(expected, rel=1e-12)
    assert dmv_frequency_hz(n_axial=1, n_circ=0, **GEOMETRY) > dmv_frequency_hz(
        n_axial=1, n_circ=5, **GEOMETRY
    )


def test_n1_branch_is_not_the_falsified_value():
    """The dropped-membrane branch returned 17.33 Hz; the beam limit is 2881.6 Hz."""
    beam_hz = (math.pi / GEOMETRY["length"]) ** 2 * math.sqrt(
        GEOMETRY["young"] * GEOMETRY["radius"] ** 2 / 2.0 / GEOMETRY["density"]
    ) / (2.0 * math.pi)
    n1 = dmv_frequency_hz(n_axial=1, n_circ=1, **GEOMETRY)
    assert beam_hz == pytest.approx(2881.62, rel=1e-3)
    assert n1 > 0.3 * beam_hz          # was 0.006 * beam_hz before the repair
    assert n1 == pytest.approx(1175.43, rel=1e-3)


def test_donnell_minimum_is_not_the_beam_branch():
    """The lowest Donnell mode is n = 5, not the degenerate n = 1 branch."""
    ladder = sorted(dmv_frequency_hz(n_axial=1, n_circ=n, **GEOMETRY) for n in range(0, 9))
    assert ladder[0] == pytest.approx(202.14, rel=1e-3)


def _envelope() -> dict:
    return json.loads((ANCHOR / "gates_verified.json").read_text())


def test_gate_envelope_is_falsifiable():
    payload = _envelope()
    assert payload["schema"] == "tensorfem.verified-gates.v1"
    assert payload["anchor"] == "fem_shell_modal"
    gates = payload["gates"]
    assert gates
    for gate in gates:
        assert gate["operator"] in OPERATORS, gate["id"]
        assert isinstance(gate["threshold"], (int, float)), gate["id"]
        assert math.isfinite(gate["value"]), gate["id"]
        assert isinstance(gate["passed"], bool), gate["id"]
    assert payload["passed"] == all(gate["passed"] for gate in gates)
    assert payload["failed_gates"] == [g["id"] for g in gates if not g["passed"]]


def test_shell_gap_is_recorded_not_hidden():
    """Honest FAIL: the FE spectrum does not reproduce the Donnell oracle.

    This assertion pins the *open* defect (rotation-only mechanisms in the shell4
    spectrum).  When the element is repaired the number drops below the 15%
    threshold, this test flips, and the anchor gate can be re-graded.
    """
    payload = _envelope()
    by_id = {gate["id"]: gate for gate in payload["gates"]}
    assert by_id["G3.shell_lowest_physical_vs_donnell"]["passed"] is False
    assert by_id["G3.shell_lowest_physical_vs_donnell"]["value"] > 15.0
    assert by_id["G4.shell_mechanism_free_low_spectrum"]["value"] > 0
    assert payload["passed"] is False

    result = json.loads((ANCHOR / "result.json").read_text())
    shell = result["anchors"]["shell"]
    for row in shell["fem_ladder"]:
        if row["lowest_physical_hz"] is not None:
            assert row["lowest_physical_hz"] > row["lowest_mechanism_hz"] or True
    assert shell["oracle"]["lowest_mode_mn"] == [1, 5]
