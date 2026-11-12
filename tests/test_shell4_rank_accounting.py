"""Gauge-aware rank accounting for the flat Shell4 facet family.

These tests pin the *interpretation* of the zero-energy census.  The merged
census counts everything outside the six rigid-body modes as spurious, which
reports one false positive per facet: a constant section rotation about the
facet normal with zero displacement carries no strain and is invisible to the
difference-only drilling stabilisation.  That direction is a gauge -- it cannot
be communicated between facets, because adjacent facets do not share a normal.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.shell4 import shell4_stiffness
from tensorfem.shell4_mitc import shell4_mitc_stiffness
from tensorfem.shell4_rank import (
    drilling_gauge_mode,
    element_rank_census,
    facet_rank_accounting,
    rigid_body_modes,
)

YOUNG, POISSON, THICKNESS = 200.0e9, 0.3, 0.01
CONFIGS = (("shell4", "sri"), ("shell4", "mitc"), ("shell4_mitc", "sri"), ("shell4_mitc", "mitc"))


def _square() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )


@pytest.fixture(scope="module")
def square() -> torch.Tensor:
    return _square()


def test_gauge_is_exactly_zero_energy(square):
    gauge = drilling_gauge_mode(square)
    for builder in (shell4_stiffness, shell4_mitc_stiffness):
        for plate_shear in ("sri", "mitc"):
            k = builder(square, YOUNG, POISSON, THICKNESS, plate_shear=plate_shear)
            residual = float(torch.linalg.vector_norm(k @ gauge)) / float(k.abs().max())
            assert residual < 1e-12, (builder.__name__, plate_shear, residual)


def test_gauge_carries_a_normal_section_rotation(square):
    # The raw (0, n) field is projected off the six rigid-body modes, so the
    # gauge is not component-wise zero outside r_z; what defines it is that it
    # is orthogonal to every rigid motion and still exactly zero-energy.
    gauge = drilling_gauge_mode(square)
    rigid = rigid_body_modes(square)
    assert float(torch.linalg.vector_norm(rigid.T @ gauge)) < 1e-12
    assert float(torch.linalg.vector_norm(gauge[5::6])) > 0.5  # r_z, the normal
    assert abs(float(torch.linalg.vector_norm(gauge)) - 1.0) < 1e-12


def test_gauge_is_orthogonal_to_the_rigid_body_modes(square):
    gauge = drilling_gauge_mode(square)
    rigid = rigid_body_modes(square)
    assert float(torch.linalg.vector_norm(rigid.T @ gauge)) < 1e-12


@pytest.mark.parametrize("formulation,plate_shear", CONFIGS)
def test_accounting_matches_the_legacy_census_plus_the_gauge(square, formulation, plate_shear):
    accounting = facet_rank_accounting(
        square, YOUNG, POISSON, THICKNESS,
        formulation=formulation, plate_shear=plate_shear,
    )
    legacy = element_rank_census(square, YOUNG, POISSON, THICKNESS, plate_shear=plate_shear)
    assert accounting.zero_modes == legacy.zero_modes
    assert accounting.gauge_modes == 1
    assert accounting.gauge_residual < 1e-12
    assert accounting.rigid_modes == 6
    # the accounting removes exactly the gauge from the legacy mechanism count
    assert accounting.spurious_modes == legacy.spurious_modes - 1


def test_hourglass_population_is_the_transverse_shear_rule(square):
    """Two genuine mechanisms live in the transverse-shear block, none elsewhere."""
    counts = {}
    for formulation, plate_shear in CONFIGS:
        counts[(formulation, plate_shear)] = facet_rank_accounting(
            square, YOUNG, POISSON, THICKNESS,
            formulation=formulation, plate_shear=plate_shear,
        ).spurious_modes
    assert counts[("shell4", "sri")] == 2
    assert counts[("shell4", "mitc")] == 0
    assert counts[("shell4_mitc", "sri")] == 2
    assert counts[("shell4_mitc", "mitc")] == 0


def test_surviving_mechanisms_of_the_one_point_rule_are_transverse(square):
    accounting = facet_rank_accounting(
        square, YOUNG, POISSON, THICKNESS, formulation="shell4", plate_shear="sri"
    )
    assert len(accounting.mechanisms) == 2
    for mechanism in accounting.mechanisms:
        # genuine hourglasses move the transverse displacement w
        assert float(torch.linalg.vector_norm(mechanism[2::6])) > 1e-3


def test_mitc_variant_legacy_mechanism_is_the_gauge(square):
    """The 'in-plane mechanism' of the merged report is the gauge, not a defect."""
    accounting = facet_rank_accounting(
        square, YOUNG, POISSON, THICKNESS, formulation="shell4_mitc", plate_shear="mitc"
    )
    assert accounting.spurious_modes == 0
    gauge = drilling_gauge_mode(square)
    legacy = element_rank_census(square, YOUNG, POISSON, THICKNESS, plate_shear="mitc")
    assert legacy.spurious_modes == 1
    legacy_overlaps = [abs(float(gauge @ m)) for m in legacy.mechanisms]
    assert max(legacy_overlaps) > 1.0 - 1e-8  # ... but one legacy mechanism is


def test_curved_facet_admits_the_same_gauge():
    """The gauge is a frame-local statement: it does not need a flat model."""
    theta = math.radians(20.0)
    normal = torch.tensor([0.0, math.sin(theta), math.cos(theta)], dtype=torch.float64)
    e1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    e2 = torch.linalg.cross(normal, e1)
    origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    xyz = torch.stack(
        (origin, origin + e1, origin + e1 + e2, origin + e2)
    )
    accounting = facet_rank_accounting(
        xyz, YOUNG, POISSON, THICKNESS, formulation="shell4", plate_shear="mitc"
    )
    assert accounting.gauge_modes == 1
    assert accounting.spurious_modes == 0
    assert accounting.gauge_residual < 1e-12


def test_planar_float32_facet_is_not_rejected():
    """A planar O(100) facet must not trip the coplanarity guard in float32."""
    radius, nt = 100.0, 8
    angles = torch.linspace(0.0, 2.0 * math.pi, nt + 1, dtype=torch.float32)[:4]
    lower = torch.stack(
        (torch.zeros(4), radius * torch.cos(angles), radius * torch.sin(angles)), -1
    )
    upper = lower + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    xyz = torch.stack((lower[0], upper[0], upper[1], lower[1]))
    assert xyz.dtype == torch.float32
    k = shell4_stiffness(xyz, 200.0e9, 0.3, 0.01)
    assert torch.isfinite(k).all()