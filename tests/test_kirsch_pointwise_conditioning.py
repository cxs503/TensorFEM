"""Fail-closed gates for the Kirsch pointwise conditioning audit.

Two things are locked here:

1. **The polar shear rotation.** ``kirsch_hole.polar_from_cartesian`` separates the
   correct ``tau_rt = (syy - sxx) cos sin + sxy cos 2theta`` from the tempting
   ``(srr - stt) cos sin + sxy cos 2theta`` slip, which agrees on the axes only.
   The published certification numbers sample ``theta = 0/90 deg`` exclusively, so
   the gate also proves that fixing the transformation leaves them untouched.
2. **The audit's own claims** (``docs/kirsch-pointwise-conditioning-audit.md``):
   the canonical probes sit on element edges, the recovered field beats the raw
   pointwise protocol there, and the re-measurement gate is not idle.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.kirsch_hole import (  # noqa: E402
    KirschHole,
    polar_from_cartesian,
)

SPEC = importlib.util.spec_from_file_location(
    "kirsch_audit", ROOT / "benchmarks" / "kirsch_pointwise_conditioning_audit.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


# ---------------------------------------------------------------------------
# 1. Coordinate transformation
# ---------------------------------------------------------------------------

def _direct_polar(sxx: float, syy: float, sxy: float, theta: float) -> tuple[float, float, float]:
    """Reference rotation from the definition ``sigma_ab = e_a . sigma . e_b``."""
    c, s = math.cos(theta), math.sin(theta)
    matrix = ((sxx, sxy), (sxy, syy))
    radial = (c, s)
    tangential = (-s, c)

    def project(left, right):
        return sum(left[a] * matrix[a][b] * right[b] for a in range(2) for b in range(2))

    return project(radial, radial), project(tangential, tangential), project(radial, tangential)


@pytest.mark.parametrize("theta_degrees", [0.0, 7.0, 15.0, 30.0, 45.0, 60.0, 83.0, 90.0])
@pytest.mark.parametrize(
    "tensor", [(1.0, 0.0, 0.0), (2.0, -1.5, 0.7), (-0.3, 0.9, -1.1), (0.0, 0.0, 1.0)]
)
def test_polar_rotation_matches_definition(theta_degrees, tensor):
    theta = math.radians(theta_degrees)
    got = polar_from_cartesian(*tensor, theta)
    expected = _direct_polar(*tensor, theta)
    for value, reference in zip(got, expected):
        assert value == pytest.approx(reference, abs=1e-14)


def test_polar_shear_reaches_mohr_extremum_at_45_degrees():
    """The slip formula returns 0 at 45 deg; the correct one returns the extremum."""
    srr, stt, srt = polar_from_cartesian(1.0, 0.0, 0.0, math.radians(45.0))
    assert srt == pytest.approx(-0.5, abs=1e-14)
    slipped = (srr - stt) * 0.5 + 0.0 * (math.cos(math.radians(90.0)))
    assert slipped == pytest.approx(0.0, abs=1e-14)  # the formula that was wrong
    assert abs(srt - slipped) > 0.4


def test_published_axis_probes_are_invariant_to_the_fix():
    """theta = 0 deg and 90 deg give sin(2 theta) = 0: published numbers cannot move."""
    for tensor in ((1.0, 0.0, 0.0), (2.0, -1.5, 0.7), (-0.3, 0.9, -1.1)):
        for theta_degrees in (0.0, 90.0):
            theta = math.radians(theta_degrees)
            sxx, syy, sxy = tensor
            c, s = math.cos(theta), math.sin(theta)
            stt = sxx * s * s + syy * c * c - 2.0 * sxy * c * s
            srr = sxx * c * c + syy * s * s + 2.0 * sxy * c * s
            slipped = (srr - stt) * c * s + sxy * (c * c - s * s)
            assert polar_from_cartesian(sxx, syy, sxy, theta)[2] == pytest.approx(
                slipped, abs=1e-15
            )


# ---------------------------------------------------------------------------
# 2. Audit claims
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_level():
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    hole = AUDIT.kirsch_hole()
    sample = AUDIT.solve_level(24, hole)
    return hole, sample, AUDIT.measure(sample, hole)


def test_every_canonical_probe_sits_on_an_element_edge(coarse_level):
    _, _, rows = coarse_level
    assert len(rows) == len(AUDIT.RAYS_DEGREES) * len(AUDIT.RADIAL_SCALES)
    for row in rows:
        assert row["diagnostics"]["edge_distance"] == pytest.approx(0.0, abs=1e-9), row
        assert row["diagnostics"]["extrapolation_beyond_gauss"] == pytest.approx(
            1.0 - 1.0 / math.sqrt(3.0), abs=1e-9
        )


def test_recovery_improves_the_radial_probe_on_the_same_solution(coarse_level):
    _, _, rows = coarse_level
    well_posed = [
        row
        for row in rows
        if row["conditioning"]["radial"]["reference_fraction_of_scale"] >= 0.05
    ]
    assert well_posed, "no well-conditioned radial probe: the audit lost its subject"
    raw_worst = max(row["errors"]["radial"]["raw"]["relative_error"] for row in well_posed)
    recovered_worst = max(
        row["errors"]["radial"]["recovered"]["relative_error"] for row in well_posed
    )
    assert recovered_worst < raw_worst / 2.0, (raw_worst, recovered_worst)


def test_raw_radial_fail_is_reproduced_and_kept(coarse_level):
    """The published FAIL must still be visible in the audit's raw protocol."""
    _, _, rows = coarse_level
    failing = [
        row
        for row in rows
        if row["errors"]["radial"]["raw"]["verdict"] == "FAIL"
    ]
    assert failing, "the audit no longer reproduces the published FAIL rows"


def test_gate_is_not_idle(coarse_level):
    hole, sample, rows = coarse_level
    assert AUDIT.control_constant_field()["verdict"] == "PASS"
    perturbed = AUDIT.control_perturbed_field(sample, hole)
    assert perturbed["fail_rows"] == perturbed["rows"] > 0
    coarse_failures = [
        row
        for row in rows
        for component in AUDIT.COMPONENTS
        if row["errors"][component]["recovered"]["verdict"] == "FAIL"
    ]
    assert coarse_failures, "recovered protocol passes every coarse row: gate is idle"


def test_near_zero_radial_probe_is_declared_ill_conditioned(coarse_level):
    _, _, rows = coarse_level
    row = next(
        row
        for row in rows
        if row["ray_degrees"] == 0.0 and row["r_over_a"] == pytest.approx(1.25)
    )
    conditioning = row["conditioning"]["radial"]
    assert conditioning["reference_fraction_of_scale"] < 0.05
    assert conditioning["ill_conditioned"] is True