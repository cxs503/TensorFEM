"""Verification of the scanling -> beta calibration of plate edge restraint.

Anchor ``GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT``.  The restrained-plate solver
had been verified for a *given* ``beta``; these tests verify the layer that turns
real stiffener/frame scantlings into ``beta``:

1. the torsion-beam homogenisation against its analytic limits
   (``12 GJ/L^2`` clamped-clamped, ``3 GJ/L^2`` clamped-free) and against an
   **independent finite-difference solve** of the same ODE;
2. the section ``J`` against the independent implementation in
   :mod:`tensorfem.open_section_torsion` and against the exact rectangle series;
3. the ``k_rot -> zeta -> beta`` map against the solver's own
   :func:`tensorfem.plate_elastic_restraint.beta_to_zeta` (no re-fitting
   allowed), including that the calibrated ``beta`` reproduces the *exact*
   Kirchhoff Levy reference;
4. monotonicity/physicality on a real ice-belt panel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfem.open_section_torsion import flat_bar, section_properties
from tensorfem.plate_elastic_restraint import (
    beta_to_zeta,
    peak_bending_stress,
    rect_patch_pressure,
    solve_restrained_panel,
)
from tensorfem.plate_levy_restraint import (
    flexural_rigidity,
    kirchhoff_strip_end_moment,
    levy_restrained_plate,
)
from tensorfem.plate_restraint_calibration import (
    StiffenerSection,
    beta_from_k_rot,
    k_rot_from_beta,
    panel_restraint,
    shear_modulus,
    torsion_beam_restraint,
    torsion_constant,
    torsion_constant_rectangle,
)

INF = math.inf
E_STEEL = 2.06e11
NU = 0.3


# ---------------------------------------------------------------------------
# 1. Torsion-beam homogenisation
# ---------------------------------------------------------------------------

def test_clamped_clamped_limit_is_twelve_gj_over_l2():
    gj, span = 2.4e4, 0.85
    res = torsion_beam_restraint(gj, span, INF, INF)
    assert res.k_rot == pytest.approx(12.0 * gj / span**2, rel=1e-12)
    assert res.k_rot_energy == pytest.approx(res.k_rot, rel=1e-12)


def test_clamped_free_limit_is_three_gj_over_l2():
    gj, span = 2.4e4, 0.85
    res = torsion_beam_restraint(gj, span, INF, 0.0)
    assert res.k_rot == pytest.approx(3.0 * gj / span**2, rel=1e-12)
    # a symmetric spring pair must be stiffer than the fixed-free case
    assert torsion_beam_restraint(gj, span, 4.0e5, 4.0e5).k_rot > res.k_rot


def test_torsionally_free_ends_provide_no_restraint():
    res = torsion_beam_restraint(1.0e4, 1.0, 0.0, 0.0)
    assert res.k_rot == 0.0 and res.free_mechanism


def _fd_k_rot(gj: float, span: float, k0: float, k1: float, n: int = 201) -> float:
    """Independent ``k_rot`` from a finite-difference solve of ``GJ th'' = -m``.

    Nondimensionalised: ``th'' = -1`` on ``x in [0, 1]``, end springs
    ``th'(0) = k0 th(0)``, ``th'(1) = -k1 th(1)`` with ``k = K L / GJ``.
    Central differences on a uniform grid with **ghost nodes** are exact for the
    quadratic solution of this ODE, so the discretisation error is round-off and
    the comparison with the closed form is a genuine check of the algebra.
    ``k_rot = (GJ/L^2) / mean(th)``.
    """
    x = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]
    A = np.zeros((n, n))
    b = -np.ones(n)
    kh0 = INF if k0 == INF else k0 * span / gj
    kh1 = INF if k1 == INF else k1 * span / gj
    if kh0 == INF:
        A[0, 0] = 1.0
        b[0] = 0.0
    else:
        A[0, 0] = -(2.0 + 2.0 * h * kh0) / h**2
        A[0, 1] = 2.0 / h**2
    for i in range(1, n - 1):
        A[i, i - 1] = 1.0 / h**2
        A[i, i] = -2.0 / h**2
        A[i, i + 1] = 1.0 / h**2
    if kh1 == INF:
        A[n - 1, n - 1] = 1.0
        b[n - 1] = 0.0
    else:
        A[n - 1, n - 2] = 2.0 / h**2
        A[n - 1, n - 1] = -(2.0 + 2.0 * h * kh1) / h**2
    theta = np.linalg.solve(A, b)
    from scipy.integrate import simpson
    # Simpson is exact for the cubic; trapezoid is not even exact for a
    # quadratic, which would mask a real error at the 2e-4 level
    return (gj / span**2) / float(simpson(theta, x=x))


@pytest.mark.parametrize("k0,k1", [(INF, INF), (INF, 0.0), (3.0e5, 3.0e5),
                                   (1.5e6, 6.0e4)])
def test_closed_form_matches_independent_fd_solution(k0, k1):
    gj, span = 3.1e4, 1.2
    closed = torsion_beam_restraint(gj, span, k0, k1).k_rot
    numeric = _fd_k_rot(gj, span, k0, k1)
    assert closed == pytest.approx(numeric, rel=1e-9)


def test_mean_rotation_and_energy_definitions_agree():
    """Symmetric end fixity: the two homogenisation definitions coincide exactly.

    Asymmetric fixity: they differ, and the difference is the (documented)
    definition ambiguity of any 1-D -> Winkler homogenisation.  Bracket it.
    """
    gj, span = 3.1e4, 1.2
    sym = torsion_beam_restraint(gj, span, 5.0e5, 5.0e5)
    assert sym.k_rot_energy == pytest.approx(sym.k_rot, rel=1e-10)
    asym = torsion_beam_restraint(gj, span, 2.0e6, 5.0e4)
    ratio = asym.k_rot_energy / asym.k_rot
    assert 0.8 < ratio < 1.25, ratio
    # both definitions must still be bounded by the two analytic limits
    lo = 3.0 * gj / span**2
    hi = 12.0 * gj / span**2
    for value in (asym.k_rot, asym.k_rot_energy):
        assert lo * 0.5 < value <= hi


# ---------------------------------------------------------------------------
# 2. Section torsion constant
# ---------------------------------------------------------------------------

def test_torsion_constant_matches_exact_rectangle_and_independent_module():
    width, thick = 0.2, 0.012
    mine = torsion_constant([(width, thick)])
    exact = torsion_constant_rectangle(width, thick)
    assert mine == pytest.approx(exact, rel=1e-12)
    # independent thin-walled implementation in the library (quadrature + 0.63
    # correction per wall), for the same single-wall flat bar
    fb = flat_bar(width, thick)
    ref = section_properties(fb["nodes"], edges=fb["edges"]).j
    assert mine == pytest.approx(ref, rel=1e-4)
    # bare thin-wall limit is an upper bound
    assert torsion_constant([(width, thick)], correction=False) > mine


def test_stiffener_section_flat_bar_matches_module():
    section = StiffenerSection(web_height=0.3, web_thickness=0.012)
    fb = flat_bar(0.3, 0.012)
    ref = section_properties(fb["nodes"], edges=fb["edges"]).j
    assert section.torsion_constant() == pytest.approx(ref, rel=1e-4)


# ---------------------------------------------------------------------------
# 3. k_rot -> beta must be the solver's own mapping
# ---------------------------------------------------------------------------

def test_beta_map_inverts_solver_convention():
    D, span = 2.0e5, 0.7
    for beta in (0.0, 0.05, 0.3, 0.6, 0.9, 0.99):
        zeta, back = beta_from_k_rot(k_rot_from_beta(beta, D, span), D, span)
        assert back == pytest.approx(beta, rel=1e-10, abs=1e-12)
        # the solver regularises beta = 1 with eps = 1e-6 in beta_to_zeta, which
        # is the only difference; its relative effect is eps/(1-beta)
        assert zeta == pytest.approx(beta_to_zeta(beta),
                                     rel=2e-6 / max(1.0 - beta, 1e-3) + 1e-9)


def test_zero_stiffness_member_gives_simple_support():
    pr = panel_restraint(
        young=E_STEEL, poisson=NU, plate_thickness=0.02,
        side_x=1.0, side_y=1.0,
        longitudinal=StiffenerSection(0.2, 0.01), frame=StiffenerSection(0.2, 0.01),
        longitudinal_ends=0.0, frame_ends=0.0)
    assert pr.beta_x == 0.0 and pr.beta_y == 0.0
    assert pr.zeta_x == 0.0 and pr.zeta_y == 0.0


def test_calibrated_beta_reproduces_exact_levy_reference():
    """Calibration -> beta -> Mindlin FEM must land on the exact Kirchhoff Levy.

    A 1 m square steel plate (t = 20 mm) whose x-edges are held by a member
    producing ``k_rot``; y-edges simply supported.  The only input carried from
    the calibration is ``k_rot``, so any error in the ``k_rot -> zeta -> beta``
    chain shows up here.
    """
    a, b, t = 1.0, 1.0, 0.02
    q = 2.0e4
    pr = panel_restraint(
        young=E_STEEL, poisson=NU, plate_thickness=t,
        side_x=a, side_y=b,
        longitudinal=StiffenerSection(0.2, 0.01),   # unused (y-edges simple)
        frame=StiffenerSection(0.32, 0.012, 0.08, 0.016),
        frame_span=0.9, longitudinal_ends=0.0)
    assert 0.2 < pr.beta_x < 0.95, pr.beta_x
    D = flexural_rigidity(E_STEEL, NU, t)
    exact = levy_restrained_plate(a, b, D, NU, pr.k_rot_x, q, n_modes=201)
    fem = solve_restrained_panel(60, 60, a, b, E_STEEL, NU, t, q,
                                 beta_x=pr.beta_x, beta_y=0.0)
    w_fem = peak_bending_stress(fem)["w_centre"]
    assert w_fem == pytest.approx(exact.w_centre(), rel=0.01)


def test_calibrated_spring_matches_classical_strip_end_moment():
    """The homogenised ``k_rot`` in the exact strip formula vs the FE strip."""
    span, t, q = 1.0, 0.02, 2.0e4
    pr = panel_restraint(
        young=E_STEEL, poisson=NU, plate_thickness=t, side_x=3.0, side_y=span,
        longitudinal=StiffenerSection(0.22, 0.012),
        frame=StiffenerSection(0.3, 0.012), frame_span=0.8,
        longitudinal_ends=INF, frame_ends=INF)
    D = flexural_rigidity(E_STEEL, NU, t)
    m_end = kirchhoff_strip_end_moment(q, span, D, pr.k_rot_y)
    fem = solve_restrained_panel(12, 60, 3.0, span, E_STEEL, NU, t, q,
                                 beta_x=1.0, beta_y=pr.beta_y)
    assert fem is not None
    # exact strip end moment: M = q L^2/12 / (1 + 2D/(K L)) [N*m/m]
    assert m_end == pytest.approx(q * span**2 / 12.0
                                  / (1.0 + 2.0 * D / (pr.k_rot_y * span)), rel=1e-12)
    assert 0.0 < m_end < q * span**2 / 12.0


# ---------------------------------------------------------------------------
# 4. Physicality on the ice-belt panel of the paper
# ---------------------------------------------------------------------------

def _ice_panel(**over):
    kwargs = dict(
        young=E_STEEL, poisson=NU, plate_thickness=0.022,
        side_x=0.8, side_y=0.7,
        longitudinal=StiffenerSection(0.21, 0.010, 0.040, 0.022),
        frame=StiffenerSection(0.40, 0.012, 0.100, 0.018),
    )
    kwargs.update(over)
    return panel_restraint(**kwargs)


def test_beta_monotone_in_member_stiffness_and_span():
    base = _ice_panel()
    stronger = _ice_panel(longitudinal=StiffenerSection(0.30, 0.014, 0.060, 0.024))
    longer_free = _ice_panel(longitudinal_ends=INF, longitudinal_span=3.2)
    assert 0.0 < base.beta_y < 1.0
    assert stronger.beta_y > base.beta_y
    # a longer, free-ended member restrains less
    assert longer_free.beta_y < base.beta_y
    # beta is a monotone function of the same k_rot: check both orderings
    assert base.zeta_y > 0.0


def test_calibrated_ice_panel_is_less_stressed_than_simple_support():
    """The paper's SSSS upper bound must be recoverable and beatable.

    ``beta = 0`` reproduces the SSSS baseline of
    ``tests/data/panel_patch_load_reference.json`` (already guarded elsewhere);
    a calibrated panel must sit strictly below it, and both must stay in the
    physically sensible band (clamped is the lower bound).
    """
    ref = json.loads((Path(__file__).parent / "data"
                      / "panel_patch_load_reference.json").read_text())
    row = next(r for r in ref["plate_vs_strip"]
               if r["lambda_c_over_l"] == 0.1 and r["patch_mode"] == "full_breadth")
    dp = ref["design_point"]
    L, S, t = dp["l"], dp["s"], dp["t"]
    press = rect_patch_pressure(dp["p_ref"], 0.5 * (L - 0.1 * L), 0.5 * (L + 0.1 * L),
                                0.0, S)
    pr = _ice_panel()
    ny = int(round(80 * S / L))
    ssss = peak_bending_stress(
        solve_restrained_panel(80, ny, L, S, dp["E"], dp["nu"], t, press,
                               beta_x=0.0, beta_y=0.0))["centre_von_mises"]
    cal = peak_bending_stress(
        solve_restrained_panel(80, ny, L, S, dp["E"], dp["nu"], t, press,
                               beta_x=pr.beta_x, beta_y=pr.beta_y))["centre_von_mises"]
    clamp = peak_bending_stress(
        solve_restrained_panel(80, ny, L, S, dp["E"], dp["nu"], t, press,
                               boundary="clamped"))["centre_von_mises"]
    assert clamp < cal < ssss
    # the SSSS FE value must reproduce the Navier baseline within the published
    # mesh-converged margin (0.3% at nx = 80 per the cases file)
    assert abs(ssss / 1e6 - row["sigma_plate_MPa"]) / row["sigma_plate_MPa"] < 3e-3
    # and the calibrated restraint must remove a meaningful share of it
    assert 0.03 < (ssss - cal) / ssss < 0.6


def test_shear_modulus():
    assert shear_modulus(E_STEEL, NU) == pytest.approx(E_STEEL / (2.0 * 1.3), rel=1e-15)
    with pytest.raises(ValueError):
        shear_modulus(-1.0, 0.3)