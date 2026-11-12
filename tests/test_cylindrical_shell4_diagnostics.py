"""Regression tests for :mod:`tensorfem.cylindrical_shell4_diagnostics`.

These tests pin the *diagnosis* of the cylindrical Shell4 element, not an
acceptance verdict.  They exist because the verified anchors
(``benchmarks/verified/fem_pinched_cylinder``, ``fem_shell_scordelis_lo``)
reported a "known limitation" (non-monotone convergence) that is in fact a
rank deficiency of the element: membrane and transverse shear are both
integrated at one point, leaving 5 spurious mechanisms per element.  The
element can therefore match a single benchmark value while its refinement
behaviour is arbitrary.

Fix status: **open**.  When the element is made rank sufficient (e.g. MITC4
assumed shear + 2x2 membrane), ``SPURIOUS_MODES`` below must be lowered to 0
and the ladder test must be tightened to require monotone convergence.
"""
import pytest

from tensorfem.cylindrical_shell4_diagnostics import (
    RIGID_MODES,
    consistent_integration_nullspace_dimension,
    element_nullspace_dimension,
    pinched_cylinder_ladder,
    rigid_body_patch,
)

# Documented defect magnitudes (2026-11 audit, branch agent/m16095).
SPURIOUS_MODES = 5


def test_element_is_rank_deficient():
    """24 DOFs, 6 rigid modes -> any extra zero eigenvalue is a mechanism."""
    nullspace = element_nullspace_dimension()
    assert nullspace >= RIGID_MODES
    assert nullspace - RIGID_MODES == SPURIOUS_MODES, (
        f"cylindrical_shell4 has {nullspace - RIGID_MODES} spurious mechanisms "
        "(expected 5 while the one-point membrane/shear integration is in place)")


def test_consistent_integration_removes_the_mechanisms():
    """Root cause check: 2x2 membrane+shear integration leaves only rigid modes."""
    assert consistent_integration_nullspace_dimension() <= RIGID_MODES + 2


def test_rigid_modes_carry_negligible_energy():
    """Axial translation must be a pure null mode; in-plane modes are facet-limited."""
    energy = rigid_body_patch()
    assert energy["Tx"] < 1e-18
    # Facet approximation: O(h^2), small but non-zero for in-plane rigid motion.
    assert energy["Ty"] < 1e-4
    assert energy["Tz"] < 1e-4
    # Rotation about the axis is represented exactly by the element (no facet error).
    assert abs(energy["Rx"]) < 1e-12


def test_pinched_cylinder_ladder_is_not_monotone():
    """Coarse-mesh ladder currently oscillates: accuracy at 16x16 is accidental."""
    ladder = pinched_cylinder_ladder((2, 4, 6, 8))
    errors = [row["error_pct"] for row in ladder]
    assert errors[0] > 50.0                      # 2x2 far off: sanity
    assert errors[1] < errors[2] > errors[3], (
        "expected a non-monotone dip-and-rise sequence at coarse meshes, "
        f"got {errors}")


@pytest.mark.parametrize("n", [4, 8])
def test_ladder_values_are_reproducible(n):
    """Guard against silent changes of the element numerics."""
    known = {4: 2.857305750705985, 8: 3.3502146285166345}
    row = pinched_cylinder_ladder((n,))[0]
    assert row["error_pct"] == pytest.approx(known[n], rel=1e-9)