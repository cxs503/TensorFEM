"""Fail-closed gates for the strong-form audit of the closed-form references.

The tree states every accuracy claim relative to an analytical solution, so the
analytical solutions themselves need a check that does not go through the finite
element code.  ``tensorfem.reference_strong_form_audit`` supplies it by
substituting the closed forms into the governing equations numerically.

These tests pin three things:

1. the references satisfy equilibrium, Beltrami-Michell, strain compatibility
   and their own boundary/statics conditions to round-off;
2. the gates are **not vacuous**: physically realistic corruptions of the same
   references (a dropped ``3 a**4 / r**4`` term, a constant instead of a
   parabolic end shear) are rejected, and an artificial ``eps``-sized
   perturbation is detected down to ``eps = 1e-6``;
3. the residual really is a property of the field and not of the difference
   step: the 4th-order stencil shows the 4th-order decay.
"""
import json
import math

import numpy as np
import pytest
import torch

from tensorfem.airy_cantilever import AiryCantilever
from tensorfem.kirsch_hole import KirschHole
from tensorfem.reference_strong_form_audit import (
    INFORMATIONAL,
    PASS_TOLERANCE,
    STENCIL_ORDER,
    audit_airy_cantilever,
    audit_kirsch_hole,
    audit_lame_cylinder,
    beltrami_residual,
    compatibility_residual,
    divergence_residual,
    judge_suite,
    kirsch_displacement,
    kirsch_stress,
    lame_stress,
    perturbation_control,
    polar_samples,
    run_audit_suite,
    stencil_control,
)

D = torch.float64


# --------------------------------------------------------------------------- #
# 1. the references are strong-form solutions
# --------------------------------------------------------------------------- #
def test_kirsch_reference_is_a_strong_form_solution():
    report = audit_kirsch_hole()
    assert report.passed(PASS_TOLERANCE), report.to_dict()
    # the hole surface is traction free *exactly*, not approximately
    assert report.metrics["boundary_hole_radial"] == 0.0
    assert report.metrics["boundary_hole_shear"] == 0.0
    assert report.metrics["equilibrium"] < 1e-6
    assert report.metrics["beltrami"] < 1e-9
    assert report.metrics["compatibility"] < 1e-6


def test_kirsch_remote_field_decays_at_the_expected_order():
    """``sigma - T`` must fall like ``1/r**2``; a wrong exponent would show here."""
    report = audit_kirsch_hole()
    assert report.metrics["far_field_order"] == pytest.approx(2.0, abs=0.05)
    assert report.metrics["far_field_40a"] < 5e-3


@pytest.mark.parametrize("plane", ["strain", "stress"])
def test_lame_reference_is_a_strong_form_solution(plane):
    report = audit_lame_cylinder(plane=plane)
    assert report.passed(PASS_TOLERANCE), report.to_dict()
    # both rim tractions reproduce the applied pressures without residual
    assert report.metrics["boundary_inner_pressure"] == 0.0
    assert report.metrics["boundary_outer_pressure"] == 0.0
    assert report.metrics["polar_equilibrium"] < 1e-6


def test_lame_stress_is_independent_of_the_plane_assumption():
    """The documented claim: a traction-driven plane problem has one stress field.

    Only the axial stress and the radial displacement may change between plane
    stress and plane strain.
    """
    strain = audit_lame_cylinder(plane="strain")
    stress = audit_lame_cylinder(plane="stress")
    for key in ("equilibrium", "beltrami", "polar_equilibrium",
                "boundary_inner_pressure", "boundary_outer_pressure"):
        assert strain.metrics[key] == pytest.approx(stress.metrics[key], abs=1e-12)
    # plane strain carries sigma_z = nu (sigma_r + sigma_theta); plane stress none
    assert strain.metrics["axial_identity"] < 1e-12
    assert stress.metrics["axial_identity"] == 0.0


def test_airy_cantilever_reference_is_a_strong_form_solution():
    report = audit_airy_cantilever()
    assert report.passed(PASS_TOLERANCE), report.to_dict()
    # traction-free long edges and the exact end statics
    assert report.metrics["boundary_free_edge_top"] == 0.0
    assert report.metrics["boundary_free_edge_bottom"] == 0.0
    assert report.metrics["statics_end_shear"] < 1e-10
    assert report.metrics["statics_root_moment"] < 1e-10


def test_airy_cantilever_root_rotation_is_reported_not_hidden():
    """The reference is pinned at a point, so its 'fixed' root still rotates.

    The audit reports that rotation as a fraction of the tip deflection instead
    of letting it pass as a modelling detail: it is a real, if sub-percent,
    systematic offset in every displacement comparison against this reference.
    """
    model = AiryCantilever()
    report = audit_airy_cantilever(model)
    rotation = report.metrics["root_rotation"]
    assert 0.0 < rotation < 0.02
    # it is the rigid rotation (1+nu) P c**2 / (E I) acting over the span
    radius = model.half_height
    expected = ((1.0 + model.poisson) * abs(model.end_shear) * radius**2
                / (model.young_modulus * model.second_moment))
    assert rotation == pytest.approx(expected * model.length / abs(model.tip_deflection()),
                                    rel=2e-2)


# --------------------------------------------------------------------------- #
# 2. the gates are not vacuous
# --------------------------------------------------------------------------- #
def _corrupt_kirsch_hoop(hole: KirschHole):
    """Drop the ``3 (a/r)**4`` term of the hoop stress: a single-coefficient typo."""
    def field(points: torch.Tensor) -> torch.Tensor:
        x = points[:, 0].numpy()
        y = points[:, 1].numpy()
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        a, t = hole.radius, hole.far_field
        c2 = np.cos(2.0 * theta)
        srr = t / 2.0 * (1.0 - (a / r) ** 2) + t / 2.0 * (
            1.0 - 4.0 * (a / r) ** 2 + 3.0 * (a / r) ** 4) * c2
        stt = t / 2.0 * (1.0 + (a / r) ** 2) - t / 2.0 * (1.0) * c2
        srt = -t / 2.0 * (1.0 + 2.0 * (a / r) ** 2 - 3.0 * (a / r) ** 4) * np.sin(2.0 * theta)
        c, s = np.cos(theta), np.sin(theta)
        sxx = srr * c * c + stt * s * s - 2.0 * srt * c * s
        syy = srr * s * s + stt * c * c + 2.0 * srt * c * s
        sxy = (srr - stt) * c * s + srt * (c * c - s * s)
        return torch.as_tensor(np.stack((sxx, syy, sxy), axis=1), dtype=D)

    return field


def test_a_one_coefficient_typo_in_kirsch_is_rejected():
    hole = KirschHole()
    points = polar_samples(1.12 * hole.radius, 4.0 * hole.radius, 121, seed=0)
    clean = divergence_residual(kirsch_stress(hole), points, hole.radius)
    corrupted = divergence_residual(_corrupt_kirsch_hoop(hole), points, hole.radius)
    assert clean < PASS_TOLERANCE
    assert corrupted > 1e-2, corrupted
    # ... and the defect is visible through the Beltrami identity as well
    assert beltrami_residual(_corrupt_kirsch_hoop(hole), points, hole.radius) > 1e-3


def test_a_constant_end_shear_instead_of_the_parabolic_one_is_rejected():
    """Replacing ``c**2 - y**2`` by ``c**2`` breaks radial equilibrium."""
    model = AiryCantilever()
    c, inertia, load = model.half_height, model.second_moment, model.end_shear

    def wrong(points: torch.Tensor) -> torch.Tensor:
        x, y = points[:, 0], points[:, 1]
        sigma_x = -load * (model.length - x) * y / inertia
        sigma_y = torch.zeros_like(sigma_x)
        tau_xy = load * c**2 / (2.0 * inertia) * torch.ones_like(sigma_x)
        return torch.stack((sigma_x, sigma_y, tau_xy), dim=1)

    h = 1e-2 * c
    grid = torch.stack(
        (
            0.05 * model.length + 0.9 * model.length
            * torch.linspace(0.0, 1.0, 33, dtype=D),
            0.5 * c * torch.linspace(-1.0, 1.0, 33, dtype=D),
        ),
        dim=1,
    )
    clean = divergence_residual(lambda p: model.stress(p), grid, c, h=h)
    corrupted = divergence_residual(wrong, grid, c, h=h)
    assert clean < PASS_TOLERANCE
    assert corrupted > 1e-2, corrupted


def test_a_wrong_displacement_amplitude_is_rejected_by_the_compatibility_gate():
    """The compatibility gate must catch a displacement that is not the stress integral."""
    model = AiryCantilever()

    def stretched(points: torch.Tensor) -> torch.Tensor:
        return 1.001 * model.displacement(points)

    generator = torch.Generator().manual_seed(11)
    x = 0.05 * model.length + 0.9 * model.length * torch.rand(121, generator=generator, dtype=D)
    y = 0.9 * model.half_height * (2.0 * torch.rand(121, generator=generator, dtype=D) - 1.0)
    points = torch.stack((x, y), dim=1)
    clean = compatibility_residual(lambda p: model.displacement(p), lambda p: model.stress(p),
                                   model.young_modulus, model.poisson, points, model.half_height)
    corrupted = compatibility_residual(stretched, lambda p: model.stress(p),
                                       model.young_modulus, model.poisson, points,
                                       model.half_height)
    assert clean < PASS_TOLERANCE
    assert corrupted > 5e-4, corrupted


def test_perturbation_control_detects_a_millionth_level_defect():
    hole = KirschHole()
    points = polar_samples(1.12, 4.0, 121, seed=0)
    table = perturbation_control(kirsch_stress(hole), points, hole.radius)
    baseline = table["baseline"]
    for amplitude in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        assert table[f"{amplitude:.0e}"] == pytest.approx(amplitude, rel=0.05)
    # one order of magnitude above the clean baseline is still detected
    assert table["1e-06"] > 10.0 * baseline
    assert table["1e-06"] == pytest.approx(1e-6, rel=0.05)


# --------------------------------------------------------------------------- #
# 3. the residual is a property of the field, not of the difference step
# --------------------------------------------------------------------------- #
def test_stencil_shows_the_fourth_order_decay():
    hole = KirschHole()
    points = polar_samples(1.12, 4.0, 121, seed=0)
    control = stencil_control(kirsch_stress(hole), points, hole.radius)
    residuals = [row["residual"] for row in control["table"]]
    assert residuals == sorted(residuals, reverse=True)
    assert control["order"] >= STENCIL_ORDER - 0.5
    assert min(residuals) < 1e-9


# --------------------------------------------------------------------------- #
# 4. the suite contract
# --------------------------------------------------------------------------- #
def test_suite_passes_and_is_json_serialisable():
    result = run_audit_suite(samples=81)
    assert result["verdict"]["passed"] is True
    assert result["stencil_order"] >= STENCIL_ORDER - 0.5
    assert result["detection_floor"] <= 1e-6
    round_tripped = json.loads(json.dumps(result))
    assert round_tripped["verdict"]["passed"] is True
    for report in round_tripped["reports"]:
        gated = {k: v for k, v in report["metrics"].items() if k not in INFORMATIONAL}
        assert max(abs(value) for value in gated.values()) <= PASS_TOLERANCE


def test_judge_flags_a_report_that_breaks_its_gate():
    good = audit_airy_cantilever()
    assert judge_suite([good])["passed"] is True
    broken = type(good)("broken", {"equilibrium": 1e-3, "root_rotation": 0.01})
    verdict = judge_suite([good, broken])
    assert verdict["passed"] is False
    assert verdict["references"]["broken"]["worst_metric"] == "equilibrium"


def test_lame_cartesian_adapter_matches_the_polar_closed_form():
    """The adapter that lets the polar field be audited in Cartesian coordinates."""
    stress = lame_stress(1.0, 2.0, 5.0, 1.0, young=200e9, poisson=0.3, plane="strain")
    point = torch.tensor([[0.0, 1.5]], dtype=D)
    sxx, syy, sxy = stress(point)[0]
    # on the y axis sigma_xx is the hoop stress and sigma_yy the radial one
    assert abs(sxy) < 1e-12
    assert syy < sxx  # internal pressure: hoop > radial
    assert math.isfinite(float(sxx))


def test_kirsch_displacement_adapter_is_finite_off_axis():
    hole = KirschHole()
    points = polar_samples(1.12, 2.0, 7, seed=3)
    displacement = kirsch_displacement(hole)
    values = displacement(points)
    assert values.shape == (7, 2)
    assert bool(torch.all(torch.isfinite(values)))