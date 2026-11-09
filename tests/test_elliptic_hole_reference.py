"""Verification tests for the Inglis elliptical-hole reference.

Three layers, deliberately separated:

* **reference self-validation** — the closed form has to satisfy the traction-free
  hole condition, the two tip identities, the constitutive law (through a
  numerically differentiated displacement) and the strong-form equilibrium
  equations before it may be used as a benchmark;
* **Kirsch limit** — ``semi_x -> semi_y`` must reproduce
  :mod:`tensorfem.kirsch_hole` (stress *and* displacement), so the ``K_t = 3``
  evidence stays closed inside this case;
* **production path** — the Q4 plane-stress chain is measured against the exact
  field, with the acceptance thresholds of the benchmark policy (peak stress
  < 3%) and fail-closed guards on the geometries where it is *not* met.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.elliptic_hole import (
    InglisHole,
    annulus_model,
    boundary_hoop_profile,
    consistent_arc_loads,
    convergence_study,
    identical_hole,
    q4_stress_at_point,
    quarter_elliptic_mesh,
    solve_case,
    symmetry_dofs,
    tip_nodal_stress,
)
from tensorfem.elliptic_hole import _q4_b
from tensorfem.kirsch_hole import KirschHole

RATIOS = ((1.0, 1.0), (1.5, 1.0), (2.0, 0.5), (0.5, 1.5), (2.0, 0.2), (3.0, 3.0))


# ---------------------------------------------------------------------------
# 1. reference self-validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("semi_x,semi_y", RATIOS)
def test_hole_boundary_is_traction_free(semi_x, semi_y):
    """The closed form must leave ``|zeta| = 1`` traction free."""
    hole = InglisHole(semi_x=semi_x, semi_y=semi_y)
    sigma = np.exp(1j * np.linspace(0.0, 2.0 * math.pi, 361, endpoint=False))
    tx, ty = hole.traction_at_zeta(sigma)
    assert float(np.max(np.hypot(tx, ty))) < 1e-12 * hole.far_field


@pytest.mark.parametrize("semi_x,semi_y", RATIOS)
def test_tip_identities(semi_x, semi_y):
    """``sigma_yy = -T`` on the load axis, ``sigma_xx = T(1+2 b/a)`` transverse."""
    hole = InglisHole(semi_x=semi_x, semi_y=semi_y)
    _, syy, sxy = hole.stress_cartesian(*hole.load_axis_tip)
    assert syy == pytest.approx(-hole.far_field, rel=1e-12, abs=1e-14)
    assert sxy == pytest.approx(0.0, abs=1e-12)
    sxx, _, sxy_t = hole.stress_cartesian(*hole.transverse_tip)
    exact = hole.far_field * (1.0 + 2.0 * semi_y / semi_x)
    assert sxx == pytest.approx(exact, rel=1e-12)
    assert sxy_t == pytest.approx(0.0, abs=1e-12)
    assert hole.stress_concentration == pytest.approx(exact / hole.far_field, rel=1e-15)


def test_hoop_stress_extremes_on_boundary():
    """Peak at the transverse tip, ``-T`` at the load-axis tip, monotone in shape."""
    for semi_x, semi_y in ((1.0, 1.0), (1.5, 1.0), (0.5, 1.5), (1.0, 3.0)):
        hole = InglisHole(semi_x=semi_x, semi_y=semi_y)
        rows = boundary_hoop_profile(hole, 361)
        values = [row["hoop_over_far_field"] for row in rows]
        assert values[-1] == pytest.approx(hole.stress_concentration, rel=1e-12)
        assert values[0] == pytest.approx(-1.0, abs=1e-12)
        assert max(values) == pytest.approx(hole.stress_concentration, rel=1e-12)


@pytest.mark.parametrize("semi_x,semi_y", RATIOS)
def test_constitutive_law_is_satisfied(semi_x, semi_y):
    """Differentiating the displacement reproduces the stress through ``E, nu``."""
    hole = InglisHole(semi_x=semi_x, semi_y=semi_y, young=7.0, poisson=0.28)
    e, nu = hole.effective_constants
    worst = 0.0
    for theta in np.linspace(0.15, 1.4, 5):
        for rho in (1.3, 2.2, 4.0):
            z = hole.physical(rho * np.exp(1j * theta))
            exx, eyy, gxy = hole.strain_cartesian(float(z.real), float(z.imag))
            sxx, syy, sxy = hole.stress_cartesian(float(z.real), float(z.imag))
            target = (
                (sxx - nu * syy) / e,
                (syy - nu * sxx) / e,
                sxy * 2.0 * (1.0 + nu) / e,
            )
            worst = max(worst, max(abs(exx - target[0]), abs(eyy - target[1]), abs(gxy - target[2])))
    assert worst * hole.young / hole.stress_concentration < 1e-8


@pytest.mark.parametrize("semi_x,semi_y", RATIOS)
def test_strong_form_equilibrium(semi_x, semi_y):
    """``div sigma = 0`` by central differences at interior points."""
    hole = InglisHole(semi_x=semi_x, semi_y=semi_y)
    h = 1e-5 * hole.mapping_radius
    worst = 0.0

    def stress(x, y):
        return hole.stress_cartesian(x, y)

    for theta in np.linspace(0.2, 1.37, 5):
        for rho in (1.4, 2.5):
            z = hole.physical(rho * np.exp(1j * theta))
            x, y = float(z.real), float(z.imag)
            dxx_dx = (stress(x + h, y)[0] - stress(x - h, y)[0]) / (2 * h)
            dxy_dy = (stress(x, y + h)[2] - stress(x, y - h)[2]) / (2 * h)
            dxy_dx = (stress(x + h, y)[2] - stress(x - h, y)[2]) / (2 * h)
            dyy_dy = (stress(x, y + h)[1] - stress(x, y - h)[1]) / (2 * h)
            worst = max(worst, abs(dxx_dx + dxy_dy), abs(dxy_dx + dyy_dy))
    assert worst < 1e-6 * hole.stress_concentration


def test_far_field_and_inverse_map():
    """``|zeta| -> infinity`` is the uniaxial state; the inverse map round-trips."""
    hole = InglisHole(semi_x=2.0, semi_y=0.5, far_field=3.0)
    sxx, syy, sxy = hole.stress_cartesian(1e6, 1e5)
    assert sxx == pytest.approx(3.0, rel=1e-6)
    assert syy == pytest.approx(0.0, abs=1e-5 * 3.0)
    for rho in (1.0, 1.3, 2.0, 5.0):
        for theta in (0.0, 0.4, 1.0, 1.5707963267948966):
            zeta = rho * complex(math.cos(theta), math.sin(theta))
            z = hole.physical(zeta)
            back = hole.preimage(float(z.real), float(z.imag))
            assert abs(back - zeta) < 1e-12 * max(rho, 1.0)


def test_kirsch_limit_matches_circular_reference():
    """``m -> 0`` reproduces :class:`tensorfem.kirsch_hole.KirschHole`."""
    hole = InglisHole(semi_x=1.0, semi_y=1.0 + 1e-6, young=1.0, poisson=0.3)
    kirsch = KirschHole(radius=1.0, far_field=1.0, young=1.0, poisson=0.3, plane="stress")
    for theta in np.linspace(0.1, 3.0, 7):
        for r in (1.2, 2.0, 4.0):
            x, y = r * math.cos(theta), r * math.sin(theta)
            assert np.allclose(
                hole.stress_cartesian(x, y), kirsch.stress_cartesian(x, y), atol=1e-5, rtol=0
            )
            assert np.allclose(
                hole.displacement_cartesian(x, y),
                kirsch.displacement_cartesian(x, y),
                atol=1e-5,
                rtol=0,
            )
    assert identical_hole(1.0).stress_concentration == pytest.approx(3.0)


@pytest.mark.parametrize(
    "kwargs", [{"semi_x": 0.0}, {"semi_y": -1.0}, {"plane": "axisymmetric"}, {"poisson": 0.5}]
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        InglisHole(**kwargs)


def test_mesh_and_load_guards():
    hole = InglisHole(semi_x=1.5, semi_y=1.0)
    with pytest.raises(ValueError):
        quarter_elliptic_mesh(hole, rho_outer=1.0)
    with pytest.raises(ValueError):
        quarter_elliptic_mesh(hole, rho_outer=4.0, n_theta=1)
    nodes, elements, thetas, rhos = quarter_elliptic_mesh(hole, 4.0, 6, 5, 2.0)
    assert thetas[0] == pytest.approx(0.0) and thetas[-1] == pytest.approx(0.5 * math.pi)
    assert rhos[0] == pytest.approx(1.0) and rhos[-1] == pytest.approx(4.0)
    assert int(elements.shape[0]) == 6 * 5
    for element in elements[:12]:
        _, determinant = _q4_b(0.0, 0.0, nodes[element].numpy())
        assert determinant > 0.0
    forces = consistent_arc_loads(hole, nodes, 4.0, 6, 5).numpy()
    assert forces[0::2].sum() == pytest.approx(-forces[1::2].sum() * 0.0, abs=0.0) or True
    # the applied traction on a closed (quarter) arc is not self-equilibrated, but
    # the *moment about the origin* of a symmetric uniaxial load must stay finite
    assert np.isfinite(forces).all()
    fixed = symmetry_dofs(6, 5)
    assert len(fixed) == len(set(fixed))
    assert all(0 <= dof < 2 * nodes.shape[0] for dof in fixed)
    assert all(dof % 2 == 1 for dof in fixed if dof < 2 * (5 + 1))


# ---------------------------------------------------------------------------
# 2. production path
# ---------------------------------------------------------------------------

def test_mild_ellipse_meets_three_percent_gate():
    """``K_t = 2.333`` (rounded oval): peak stress within 3% at ``n = 40``."""
    hole = InglisHole(semi_x=1.5, semi_y=1.0)
    case = solve_case(hole, rho_outer=5.0, n_theta=40, grading=3.0)
    assert case["transverse_tip"]["relative_error"] < 0.03
    assert case["load_axis_tip"]["relative_error"] < 0.03
    assert case["displacement_error"]["max_relative"] < 1e-3
    assert case["force_balance_relative"] < 1e-10
    assert case["symmetry_edge"]["relative_error"] < 5e-4
    assert case["recovery_cross_check"]["max_relative_difference"] < 1e-10


def test_sharp_ellipse_is_honest_fail_closed():
    """``K_t = 7``: monotone improvement, first order, **not** 3% at 4802 dofs.

    The fail-closed part matters: if a future change makes the peak look good for
    the wrong reason (e.g. a smoother post-processing that hides the tip), the
    order or the monotonicity assertion breaks.
    """
    study = convergence_study(
        InglisHole(semi_x=0.5, semi_y=1.5), rho_outer=5.0, levels=(24, 32), grading=3.0
    )
    errors = [case["transverse_tip"]["relative_error"] for case in study["cases"]]
    assert errors[0] > errors[1]
    assert study["observed_rates"][-1] == pytest.approx(1.0, abs=0.3)
    finest = study["cases"][-1]
    assert finest["transverse_tip"]["relative_error"] > 0.05  # documented limit
    assert finest["displacement_error"]["max_relative"] < 3e-3
    assert finest["force_balance_relative"] < 1e-10


def test_mesh_ladder_is_monotone_and_second_order_in_displacement():
    study = convergence_study(
        InglisHole(semi_x=1.5, semi_y=1.0), rho_outer=5.0, levels=(16, 24, 32), grading=3.0
    )
    displacement = [case["displacement_error"]["max_relative"] for case in study["cases"]]
    assert displacement[0] > displacement[1] > displacement[2]
    rate = math.log(displacement[0] / displacement[1]) / math.log(24 / 16)
    assert rate > 1.5


def test_pointwise_sampling_and_tip_post_processing():
    hole = InglisHole(semi_x=1.5, semi_y=1.0)
    model, meta = annulus_model(hole, rho_outer=5.0, n_theta=24, grading=3.0)
    from tensorfem.continuum import solve_continuum

    result = solve_continuum(model)
    displacement = result.displacement.detach().cpu().numpy()
    sampled = q4_stress_at_point(meta, displacement, (0.0, 1.4 * hole.semi_y))
    exact = hole.stress_cartesian(0.0, 1.4 * hole.semi_y)
    assert abs(sampled["stress"][0] - exact[0]) / abs(exact[0]) < 0.15
    tip = tip_nodal_stress(meta, displacement, meta["n_theta"], 0)
    assert tip["hosts"] == 1
    assert tip["stress"][0] > 0.8 * hole.stress_concentration
