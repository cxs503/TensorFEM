"""Pressurised-cylinder membrane state for the flat Shell4 facet family."""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.shell_benchmarks import scordelis_lo_roof
from tensorfem.shell_membrane_benchmark import (
    pressurised_cylinder,
    pressurised_cylinder_reference,
)
from tensorfem.shell4 import shell4_stiffness

CONFIGS = (("shell4", "sri"), ("shell4", "mitc"), ("shell4_mitc", "sri"), ("shell4_mitc", "mitc"))


def test_reference_state():
    exact = pressurised_cylinder_reference(100.0, 1.0, 1000.0, 1.0, 0.3, 100.0)
    assert exact["hoop_stress"] == pytest.approx(100.0)
    assert exact["radial_displacement"] == pytest.approx(10.0)
    assert exact["end_axial_displacement"] == pytest.approx(-1.5)


@pytest.mark.parametrize("formulation,plate_shear", CONFIGS)
def test_membrane_state_is_within_three_percent(formulation, plate_shear):
    result = pressurised_cylinder(
        nt=16, nx=2, formulation=formulation, plate_shear=plate_shear
    )
    assert abs(result.radial_error) < 0.03
    # two independent gauge-invariant measures agree (least-squares fit + arc length)
    assert result.perimeter_radial_displacement == pytest.approx(
        result.mean_radial_displacement, rel=1e-9
    )
    assert abs(result.perimeter_error) < 0.03
    assert abs(result.axial_error) < 0.03
    assert result.reaction_residual < 1e-9
    # the facet reproduces the uniform membrane state exactly: no facet-scale ripple
    assert result.relative_ripple < 1e-9


def test_membrane_state_is_independent_of_both_formulation_knobs():
    """The membrane state carries no in-plane shear and no plate field.

    Both knobs are therefore null operators on it, which is what dissolves the
    'in-plane mechanism affects membrane problems' open item.
    """
    reference = pressurised_cylinder(nt=12, nx=2)
    for formulation, plate_shear in CONFIGS:
        other = pressurised_cylinder(
            nt=12, nx=2, formulation=formulation, plate_shear=plate_shear
        )
        assert torch.allclose(other.displacement, reference.displacement, atol=1e-8)


def test_convergence_is_monotone():
    errors = [
        abs(pressurised_cylinder(nt=nt, nx=2).radial_error) for nt in (8, 16, 32)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] < 0.03


def test_scordelis_lo_formulation_keyword_is_backwards_compatible():
    default = scordelis_lo_roof(8, 8)
    explicit = scordelis_lo_roof(8, 8, formulation="shell4", plate_shear="sri")
    assert torch.equal(default.displacement, explicit.displacement)


def test_scordelis_lo_plate_shear_rule_controls_the_bending_answer():
    bad = scordelis_lo_roof(8, 8, plate_shear="sri").probe_displacement
    good = scordelis_lo_roof(8, 8, plate_shear="mitc").probe_displacement
    reference = -0.3024
    assert abs(bad / reference - 1.0) > 0.30
    assert abs(good / reference - 1.0) < abs(bad / reference - 1.0)


def test_planar_float32_facet_of_large_radius_is_accepted():
    radius = 100.0
    angles = [0.0, 2.0 * math.pi / 8]
    ring = torch.tensor(
        [[0.0, radius * math.cos(a), radius * math.sin(a)] for a in angles],
        dtype=torch.float32,
    )
    axial = ring + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    xyz = torch.stack((ring[0], axial[0], axial[1], ring[1]))
    assert xyz.dtype == torch.float32
    assert torch.isfinite(shell4_stiffness(xyz, 200.0e9, 0.3, 0.01)).all()
