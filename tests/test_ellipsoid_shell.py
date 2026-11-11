"""Gates for the CASE-NAFEMS-LE11 spheroidal-shell case
(:mod:`tensorfem.ellipsoid_shell` over :mod:`tensorfem.axisym_shell`).

What is qualified here:

* G1 geometry: nodes lie on the ellipse, the osculating radius reproduces the
  pole / equator / sphere closed forms, and the grading key is monotone;
* G2 reference: the closed-form membrane forces satisfy cap statics identically
  and the independent Rayleigh-Ritz membrane solve reproduces them and the
  spherical ``w = pR^2(1-nu)/(2Et)`` limit;
* G3 the case gate itself (5%): point values at the pole and the equator plus
  the equatorial normal displacement, on the reporting mesh (128 elements);
* G4 the sphere limit of the FE model is exact to round-off (the route is not
  merely "close" on a constant-curvature surface);
* G5 a fail-closed record of where the route is *not* usable: on a strongly
  oblate spheroid the coarse meshes (<= 16 elements) are still >10% off and the
  equatorial hoop error plateaus near 2.6% (discrete boundary layer), so the
  case must not be quoted below the reporting mesh.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.ellipsoid_shell import (ellipsoid_pressure, membrane_forces,
                                       membrane_shape, meridian_curvature_radius,
                                       spheroid_element_radii, spheroid_nodes,
                                       spheroid_theta)

A, B, T, E, NU, P = 1.0, 0.5, 0.01, 1.0, 0.3, 1.0


@pytest.fixture(scope="module")
def fine():
    return ellipsoid_pressure(a=A, b=B, thickness=T, young=E, poisson=NU, pressure=P,
                              facets=128)


# --------------------------------------------------------------------------- #
# G1 geometry
# --------------------------------------------------------------------------- #
def test_G1_nodes_lie_on_the_ellipse_and_span_pole_to_equator():
    for facets in (4, 7, 32):
        nodes = spheroid_nodes(A, B, facets)
        assert len(nodes) == facets + 1
        assert nodes[0] == pytest.approx((0.0, B), abs=1e-12)
        assert nodes[-1] == pytest.approx((A, 0.0), abs=1e-12)
        for r, z in nodes:
            assert r * r / A ** 2 + z * z / B ** 2 == pytest.approx(1.0, abs=1e-12)
        assert all(nodes[i][0] < nodes[i + 1][0] for i in range(facets))
        assert all(nodes[i][1] > nodes[i + 1][1] for i in range(facets))


def test_G1_curvature_radius_closed_forms():
    # pole: a^2/b, equator: b^2/a, sphere: a
    assert meridian_curvature_radius(A, B, 0.0) == pytest.approx(A * A / B)
    assert meridian_curvature_radius(A, B, 0.5 * math.pi) == pytest.approx(B * B / A)
    assert meridian_curvature_radius(2.0, 2.0, 0.37) == pytest.approx(2.0)
    # every element radius is finite and positive over the meridian
    radii = spheroid_element_radii(A, B, 16)
    assert len(radii) == 16
    assert all(r > 0.0 and math.isfinite(r) for r in radii)
    # the spheroid curvature is monotone pole -> equator for b < a
    assert all(radii[i] > radii[i + 1] for i in range(len(radii) - 1))


def test_G1_grading_key_is_monotone_and_validated():
    th = spheroid_theta(A, B, 8, 1.0)
    assert th[-1] == pytest.approx(0.5 * math.pi)
    assert all(b > a for a, b in zip(th, th[1:]))
    assert spheroid_theta(A, B, 8, 0.5)[1] > th[1]        # clusters near the equator
    assert spheroid_theta(A, B, 8, 2.0)[1] < th[1]        # clusters near the pole
    with pytest.raises(ValueError):
        spheroid_theta(A, B, 8, 0.0)


# --------------------------------------------------------------------------- #
# G2 reference values
# --------------------------------------------------------------------------- #
def test_G2_membrane_forces_satisfy_cap_statics():
    # 2 pi r N_phi sin(phi) = p pi r^2  at every eccentric angle
    for i in range(1, 40):
        th = 0.5 * math.pi * i / 40
        n_phi, _ = membrane_forces(A, B, P, th)
        ds = math.hypot(A * math.cos(th), B * math.sin(th))
        r, sn = A * math.sin(th), B * math.sin(th) / ds
        assert 2.0 * math.pi * r * n_phi * sn == pytest.approx(math.pi * r * r * P,
                                                               rel=1e-12)


def test_G2_membrane_forces_closed_forms():
    # pole: both forces p a^2/(2b); equator: p a/2 and p a (1 - a^2/(2 b^2))
    n_pole_phi, n_pole_th = membrane_forces(A, B, P, 0.0)
    assert n_pole_phi == pytest.approx(P * A * A / (2.0 * B))
    assert n_pole_th == pytest.approx(P * A * A / (2.0 * B))
    n_eq_phi, n_eq_th = membrane_forces(A, B, P, 0.5 * math.pi)
    assert n_eq_phi == pytest.approx(P * A / 2.0)
    assert n_eq_th == pytest.approx(P * A * (1.0 - A * A / (2.0 * B * B)))
    # oblate b/a = 0.5: the hoop force changes sign (compression at the equator)
    assert n_pole_th > 0.0 > n_eq_th
    # sphere limit: p a / 2 everywhere
    for i in range(10):
        th = 0.5 * math.pi * i / 9
        n_phi, n_th = membrane_forces(2.0, 2.0, 3.0, th)
        assert n_phi == pytest.approx(3.0 * 2.0 / 2.0)
        assert n_th == pytest.approx(3.0 * 2.0 / 2.0)


def test_G2_rayleigh_ritz_reference_reproduces_forces_and_sphere_limit():
    ref = membrane_shape(A, B, T, E, NU, P)
    assert ref["N_selfcheck_phi"] < 1e-8
    assert ref["N_selfcheck_theta"] < 1e-8
    assert abs(ref["u_equator"]) < 1e-8                     # symmetry at the equator
    sph = membrane_shape(1.0, 1.0, T, E, NU, P)
    exact = P * 1.0 ** 2 * (1.0 - NU) / (2.0 * E * T)
    assert sph["w_equator"] == pytest.approx(exact, rel=1e-10)
    # oblate: the equator moves inward (hoop compression), the pole outward
    assert ref["w_equator"] < 0.0 < ref["w_pole"]
    # the equatorial hoop strain of membrane theory fixes w_eq at the equator
    _, n_th = membrane_forces(A, B, P, 0.5 * math.pi)
    n_phi, _ = membrane_forces(A, B, P, 0.5 * math.pi)
    eps_theta = (n_th - NU * n_phi) / (E * T)
    assert ref["w_equator"] == pytest.approx(A * eps_theta, rel=1e-8)


# --------------------------------------------------------------------------- #
# G3 the case gate
# --------------------------------------------------------------------------- #
def test_G3_case_gate_five_percent(fine):
    assert fine["facets"] == 128
    assert fine["N_pole_err"] < 0.05
    assert fine["N_theta_pole_err"] < 0.05
    assert fine["N_phi_equator_err"] < 0.05
    assert fine["N_theta_equator_err"] < 0.05
    assert fine["w_equator_err"] < 0.05
    assert fine["err_phi_max_scaled"] < 0.05
    assert fine["err_theta_max_scaled"] < 0.05
    # the reported numbers themselves (p a^2/(2b), p a/2, p a(1-a^2/2b^2))
    assert fine["N_pole_ref"] == pytest.approx(1.0, rel=1e-3)
    assert fine["N_phi_equator_ref"] == pytest.approx(0.5, rel=1e-3)
    assert fine["N_theta_equator_ref"] == pytest.approx(-1.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# G4 sphere limit of the FE route
# --------------------------------------------------------------------------- #
def test_G4_finite_element_sphere_is_exact():
    out = ellipsoid_pressure(a=1.0, b=1.0, thickness=T, young=E, poisson=NU, pressure=P,
                             facets=16)
    assert out["err_phi_max_scaled"] < 1e-6
    assert out["err_theta_max_scaled"] < 1e-6
    exact = P * (1.0 - NU) / (2.0 * E * T)
    assert out["w_equator_fe"] == pytest.approx(exact, rel=1e-6)
    assert out["N_pole_ref"] == pytest.approx(P / 2.0)


# --------------------------------------------------------------------------- #
# G5 fail-closed record of the limits
# --------------------------------------------------------------------------- #
def test_G5_coarse_mesh_stays_out_of_tolerance():
    coarse = ellipsoid_pressure(a=A, b=B, thickness=T, young=E, poisson=NU, pressure=P,
                                facets=8)
    assert coarse["err_phi_max_scaled"] > 0.10       # measured 0.40
    assert coarse["N_theta_equator_err"] > 0.10      # measured 0.37


def test_G5_error_decreases_with_refinement_and_plateaus_at_the_equator():
    errors, plateau = [], []
    for facets in (32, 64, 128):
        out = ellipsoid_pressure(a=A, b=B, thickness=T, young=E, poisson=NU,
                                 pressure=P, facets=facets)
        errors.append(out["err_phi_max_scaled"])
        plateau.append(out["N_theta_equator_err"])
    assert all(b < a for a, b in zip(errors, errors[1:]))
    # first-order convergence: each doubling at least halves the field-max error
    assert errors[1] < 0.65 * errors[0]
    assert errors[2] < 0.65 * errors[1]
    # but the equatorial hoop error plateaus near 2.6% (discrete boundary layer)
    assert 0.02 < plateau[-1] < 0.035
    assert abs(plateau[-1] - plateau[-2]) < 0.01


def test_G5_equator_grading_helps_and_pole_grading_does_not():
    base = ellipsoid_pressure(a=A, b=B, thickness=T, young=E, poisson=NU, pressure=P,
                              facets=64)
    equator_refined = ellipsoid_pressure(a=A, b=B, thickness=T, young=E, poisson=NU,
                                         pressure=P, facets=64, grading=0.5)
    pole_refined = ellipsoid_pressure(a=A, b=B, thickness=T, young=E, poisson=NU,
                                      pressure=P, facets=64, grading=1.5)
    assert equator_refined["err_phi_max_scaled"] < base["err_phi_max_scaled"]
    assert pole_refined["err_phi_max_scaled"] > base["err_phi_max_scaled"]


def test_G5_prolate_spheroid_is_easier_than_oblate():
    pro = ellipsoid_pressure(a=0.5, b=1.0, thickness=T, young=E, poisson=NU, pressure=P,
                             facets=128)
    obl = ellipsoid_pressure(a=1.0, b=0.5, thickness=T, young=E, poisson=NU, pressure=P,
                             facets=128)
    assert pro["err_phi_max_scaled"] < 0.01
    assert pro["err_phi_max_scaled"] < obl["err_phi_max_scaled"]
    assert obl["N_theta_equator_ref"] < 0.0 < pro["N_theta_equator_ref"]