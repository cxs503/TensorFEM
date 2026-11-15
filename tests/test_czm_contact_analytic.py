"""Analytic verification of the CZM contact solve (spherical ice edge).

Closes the verification debt flagged in
``docs/AUDIT-GAP-PP-1-adversarial-verification.md`` (section 4): the physical
correctness of the CZM contact solver itself (``czm_contact_patch``) had never
been checked against an independent reference.

Reference (derived independently here, not imported from the benchmark script)
-------------------------------------------------------------------------------
Spherical ice edge: relief ``s(r) = amp * (r/R)^2`` about the edge centre.
Opening ``g(r) = delta0 - s(r)``; undamaged compressive branch of the bilinear
cohesive law = linear penalty ``p = k * g`` for ``g > 0``, capped by the
crushing ceiling.  For ``delta0 < amp`` with the ceiling inactive the contact
is a circle of radius ``a = R * sqrt(delta0/amp)`` carrying the parabolic
Winkler-foundation pressure

    p(r) = p0 * (1 - r^2 / a^2),   p0 = k * delta0,
    F = pi * k * R^2 * delta0^2 / (2 * amp),   mean = p0 / 2.

(Paraboloid indenter on a Winkler foundation -- the Hertz analogue with a
linear foundation; standard contact-mechanics result.)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.ice_panel_patch_contact import (
    czm_contact_patch,
    default_law,
    patch_from_contact,
    solve_rect_panel,
)

PANEL = (0.8, 0.7)
CENTRE = (0.4, 0.35)
K = 4.0e9        # Pa/m, default_law stiffness
CEILING = 2.0e6  # Pa
R, AMP, HALF = 1.0, 0.02, 0.10
DELTA0 = 1.5e-4  # p0 = 0.6 MPa < ceiling; a = 0.0866 m < box half 0.10 m


def _analytic(k: float, R: float, amp: float, delta0: float) -> dict:
    p0 = k * delta0
    a = R * math.sqrt(delta0 / amp)
    return {
        "p0": p0,
        "a": a,
        "area": math.pi * a**2,
        "F": math.pi * k * R**2 * delta0**2 / (2.0 * amp),
        "mean": p0 / 2.0,
    }


def _solve(grid: int = 97, delta0: float = DELTA0, **kw) -> object:
    ref = _analytic(K, R, AMP, delta0)
    edge = {"kind": "spherical", "radius": R, "amplitude": AMP, "box_half": HALF}
    return czm_contact_patch(
        law=default_law(),
        ceiling=CEILING,
        edge=edge,
        total_force=ref["F"],
        panel=PANEL,
        centre=CENTRE,
        grid=grid,
        **kw,
    )


def _ref_profile(patch, p0: float, a: float) -> np.ndarray:
    dX, dY = np.meshgrid(
        patch.x.numpy() - CENTRE[0], patch.y.numpy() - CENTRE[1], indexing="ij"
    )
    r2 = dX**2 + dY**2
    return np.where(r2 < a**2, p0 * (1.0 - r2 / a**2), 0.0)


def test_spherical_edge_peak_pressure_matches_analytic():
    """p0 = k * delta0: the bisection must recover the analytic approach."""
    ref = _analytic(K, R, AMP, DELTA0)
    patch = _solve()
    assert patch.peak_pressure == pytest.approx(ref["p0"], rel=1e-4)
    assert patch.approach == pytest.approx(DELTA0, rel=1e-4)


def test_spherical_edge_contact_area_matches_analytic():
    """Contact circle radius a = R * sqrt(delta0/amp).

    The contact mask is a grid-point test (``g > 0``), so the area carries an
    O(h) boundary staircase; at 97x97 the measured deviation is -0.22 % and it
    halves per grid doubling (see the grid-convergence test).
    """
    ref = _analytic(K, R, AMP, DELTA0)
    patch = _solve()
    assert patch.contact_area == pytest.approx(ref["area"], rel=5e-3)


def test_spherical_edge_peak_over_mean_is_two():
    """Parabolic distribution: peak/mean = 2 exactly (analytic)."""
    patch = _solve()
    mean = patch.total_force / patch.contact_area
    assert patch.peak_pressure / mean == pytest.approx(2.0, rel=5e-3)


def test_spherical_edge_pressure_profile_matches_parabola():
    """p(r) = p0 (1 - r^2/a^2) pointwise on the contact grid (interior)."""
    ref = _analytic(K, R, AMP, DELTA0)
    patch = _solve()
    p_ref = _ref_profile(patch, ref["p0"], ref["a"])
    p_num = patch.pressure.numpy()
    inside = p_ref > 0.05 * ref["p0"]
    rel = np.abs(p_num - p_ref) / p_ref
    assert rel[inside].max() < 5e-3
    # outside the contact circle the pressure must be zero
    outside = p_ref == 0.0
    assert p_num[outside].max() == 0.0


def test_spherical_edge_grid_convergence():
    """L2 error of the pressure field must decrease with grid refinement."""
    ref = _analytic(K, R, AMP, DELTA0)
    l2 = []
    for grid in (49, 97, 193):
        patch = _solve(grid=grid)
        p_ref = _ref_profile(patch, ref["p0"], ref["a"])
        l2.append(
            math.sqrt(float(((patch.pressure.numpy() - p_ref) ** 2).sum()))
            / math.sqrt(float((p_ref**2).sum()))
        )
    assert l2[0] > l2[1] > l2[2]
    assert l2[2] < 1e-5  # converged level at 193x193


def test_capped_case_flat_top_and_force():
    """Ceiling active (p0 = 4 MPa > 2 MPa): flat top at the ceiling, force exact.

    R = 0.4, amp = 0.02, delta0 = 1e-3: a = 0.0894 m (inside the box),
    capped radius rc = R sqrt((delta0 - ceiling/k)/amp) = 0.0632 m.
    F = pi*ceiling*rc^2 + 2*pi*k*[delta0 (a^2-rc^2)/2 - amp (a^4-rc^4)/(4 R^2)].
    """
    R2, delta0, ceiling = 0.4, 1.0e-3, 2.0e6
    a = R2 * math.sqrt(delta0 / AMP)
    rc = R2 * math.sqrt((delta0 - ceiling / K) / AMP)
    F = (
        math.pi * ceiling * rc**2
        + 2.0 * math.pi * K * (delta0 * (a**2 - rc**2) / 2.0
                          - AMP * (a**4 - rc**4) / (4.0 * R2**2))
    )
    edge = {"kind": "spherical", "radius": R2, "amplitude": AMP, "box_half": HALF}
    patch = czm_contact_patch(
        law=default_law(),
        ceiling=ceiling,
        edge=edge,
        total_force=F,
        panel=PANEL,
        centre=CENTRE,
        grid=97,
    )
    assert patch.peak_pressure == pytest.approx(ceiling, rel=1e-12)
    assert patch.total_force == pytest.approx(F, rel=1e-6)
    dX, dY = np.meshgrid(
        patch.x.numpy() - CENTRE[0], patch.y.numpy() - CENTRE[1], indexing="ij"
    )
    r = np.sqrt(dX**2 + dY**2)
    flat = r < 0.9 * rc
    assert (patch.pressure.numpy()[flat] == ceiling).all()


def test_czm_patch_end_to_end_force_conserving_and_reproducible():
    """CZM contact -> projection -> plate: force conserving, reproducible."""
    patch = _solve()
    # patch_from_contact returns (nodes, elements, load, info)
    nodes, elements, load, info = patch_from_contact(patch, 32, 28, *PANEL)
    assert info["force_error_rel"] < 1e-9
    sol1 = solve_rect_panel(32, 28, *PANEL, load=load, **{
        "young": 206.0e9, "poisson": 0.3, "thickness": 0.022,
    })
    patch2 = _solve()
    nodes2, elements2, load2, info2 = patch_from_contact(patch2, 32, 28, *PANEL)
    sol2 = solve_rect_panel(32, 28, *PANEL, load=load2, **{
        "young": 206.0e9, "poisson": 0.3, "thickness": 0.022,
    })
    s1 = sol1.stress_at(*CENTRE)
    s2 = sol2.stress_at(*CENTRE)
    assert s1["sigma_Mx_MPa"] == pytest.approx(s2["sigma_Mx_MPa"], rel=1e-12)
    assert s1["sigma_Mx_MPa"] > 0.0
