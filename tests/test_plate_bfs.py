"""Tests for :mod:`tensorfem.plate_bfs` (BFS C1 plate element + Kirchhoff oracles).

The suite pins the properties the P19 verification ladder relies on:

* the independently written bending matrix agrees with
  :func:`tensorfem.plate_buckling_hermite.bfs_plate_element_matrices`;
* the element reproduces rigid-body motion and a constant-curvature field
  exactly (patch tests, i.e. the element is consistent);
* the consistent mass and load carry the exact element mass ``rho t A`` and
  load resultants ``q A``;
* the assembled simply supported and clamped plates converge to the Navier
  series and the Timoshenko table value (the two plate rungs of the ladder);
* every analytic oracle used by the ladder is itself pinned.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem import plate_bfs as pb
from tensorfem.plate_buckling_hermite import bfs_plate_element_matrices

E, NU, T, RHO = 210.0e9, 0.3, 0.01, 7850.0
Q = 1.0e4
A = 1.0


# ---------------------------------------------------------------------------
# Element level
# ---------------------------------------------------------------------------

def test_element_bending_matches_reference_implementation():
    """Two independent derivations of the same element must agree."""
    ours = pb.bfs_element_matrices(0.13, 0.21, E, NU, T)["bending"]
    reference, _ = bfs_plate_element_matrices(0.13, 0.21, E, NU, T)
    assert np.abs(ours - reference).max() <= 1e-10 * np.abs(reference).max()


def test_element_is_symmetric_and_positive_semidefinite():
    bending = pb.bfs_element_matrices(0.2, 0.15, E, NU, T)["bending"]
    assert np.allclose(bending, bending.T, rtol=0.0, atol=1e-6)
    eigenvalues = np.linalg.eigvalsh(bending)
    assert eigenvalues.min() >= -1e-6 * eigenvalues.max()


def test_element_annihilates_rigid_body_motion():
    """Constant and linear transverse fields carry no bending energy."""
    length_x, length_y = 0.2, 0.15
    bending = pb.bfs_element_matrices(length_x, length_y, E, NU, T)["bending"]
    for grad_x, grad_y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, -0.5)):
        dofs = np.zeros(16)
        for node, (di, dj) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            x, y = di * length_x, dj * length_y
            dofs[4 * node + 0] = 1.0 + grad_x * x + grad_y * y
            dofs[4 * node + 1] = grad_x
            dofs[4 * node + 2] = grad_y
            dofs[4 * node + 3] = 0.0
        assert np.abs(bending @ dofs).max() <= 1e-6 * np.abs(bending).max()


def test_element_reproduces_constant_curvature_energy():
    """``w = x^2 / 2`` has ``w,xx = 1`` so ``u^T K u = D * area`` exactly."""
    length_x, length_y = 0.1, 0.25
    bending = pb.bfs_element_matrices(length_x, length_y, E, NU, T)["bending"]
    dofs = np.zeros(16)
    for node, (di, dj) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        x = di * length_x
        dofs[4 * node + 0] = 0.5 * x**2
        dofs[4 * node + 1] = x
    rigidity = pb.plate_rigidity(E, NU, T)
    assert dofs @ bending @ dofs == pytest.approx(rigidity * length_x * length_y, rel=1e-10)


def test_element_mass_and_load_resultants():
    element = pb.bfs_element_matrices(0.1, 0.2, E, NU, T, density=RHO, surface_load=Q)
    mass, load = element["mass"], element["load"]
    assert mass is not None
    # a rigid translation (w = 1, all slopes zero) must carry exactly rho t A of mass
    translation = np.zeros(16)
    translation[0::4] = 1.0
    assert translation @ mass @ translation == pytest.approx(RHO * T * 0.1 * 0.2, rel=1e-12)
    assert np.allclose(mass, mass.T, rtol=0.0, atol=1e-12)
    assert load.sum() == pytest.approx(Q * 0.1 * 0.2, rel=1e-12)
    # consistent load of a uniform pressure: the w-dof at a corner carries
    # q*lx*ly/4 and the w_,x dof carries the analytic integral
    # q * (lx^2 / 12) * (ly / 2) because the slope shape is length-scaled.
    length_x, length_y = 0.1, 0.2
    assert load[0] == pytest.approx(Q * length_x * length_y / 4.0, rel=1e-12)
    assert load[1] == pytest.approx(Q * length_x**2 / 12.0 * length_y / 2.0, rel=1e-12)
    assert load[2] == pytest.approx(Q * length_y**2 / 12.0 * length_x / 2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Mesh level (the L2 plate rungs)
# ---------------------------------------------------------------------------

def _centre(n_x: int, boundary: str) -> tuple[float, dict[str, np.ndarray]]:
    plate = pb.assemble_bfs_plate(
        n_x, n_x, A, A, E, NU, T, density=None, surface_load=Q
    )
    fixed = pb.bfs_boundary_dofs(n_x, n_x, boundary)
    u = pb.solve_bfs_static(plate, fixed)
    point = np.array([[A / 2.0, A / 2.0]])
    moments = pb.bfs_moments(plate, u, point, E, NU, T)
    return float(pb.bfs_evaluate(plate, u, point)["w"][0]), moments


def test_simply_supported_plate_matches_navier_series():
    navier = pb.ssss_navier_centre(A, A, Q, pb.plate_rigidity(E, NU, T), terms=11)
    w8, _ = _centre(8, "ssss")
    w16, _ = _centre(16, "ssss")
    assert abs(w8 - navier) / navier < 2e-3
    assert abs(w16 - navier) / navier < 1e-3
    assert abs(w16 - navier) < abs(w8 - navier)  # monotone convergence


def test_simply_supported_plate_centre_moment():
    moments_ref = pb.ssss_navier_moments(A / 2, A / 2, A, A, Q, NU, terms=301)
    _, moments = _centre(16, "ssss")
    assert abs(moments["M_x"][0] - moments_ref[0]) / moments_ref[0] < 2e-2
    assert abs(moments["M_y"][0] - moments_ref[1]) / moments_ref[1] < 2e-2
    assert moments_ref[0] / (Q * A**2) == pytest.approx(pb.TIMOSHENKO_SSSS_MOMENT, rel=2e-3)


def test_clamped_plate_matches_timoshenko_table():
    rigidity = pb.plate_rigidity(E, NU, T)
    w, moments = _centre(20, "cccc")
    alpha = w * rigidity / (Q * A**4)
    assert alpha == pytest.approx(pb.TIMOSHENKO_CCCC_ALPHA, rel=2e-2)
    assert moments["M_x"][0] / (Q * A**2) == pytest.approx(
        pb.TIMOSHENKO_CCCC_MOMENT, rel=2e-2
    )


def test_plate_modes_match_navier_eigenvalues():
    n_x = 10
    plate = pb.assemble_bfs_plate(n_x, n_x, A, A, E, NU, T, density=RHO)
    omega = pb.solve_bfs_modes(plate, pb.bfs_boundary_dofs(n_x, n_x, "ssss"), modes=4)
    exact = pb.ssss_plate_angular_frequencies(A, A, E, NU, T, RHO, 3, 3)
    for computed, (_, _, reference) in zip(omega, exact[:4]):
        assert abs(computed - reference) / reference < 5e-3
    # the (1,1) mode is degenerate-free for a square plate: single lowest mode
    assert exact[0][2] < exact[1][2]


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

def test_beam_oracles():
    EI, length, load = 1000.0, 10.0, 1.0
    assert pb.simply_supported_uniform_load_deflection(
        length / 2.0, length, load, EI
    ) == pytest.approx(5.0 * load * length**4 / (384.0 * EI), rel=1e-14)
    assert pb.simply_supported_uniform_load_deflection(0.0, length, load, EI) == 0.0
    assert pb.cantilever_tip_load_deflection(10.0, 0.1, 1000.0) == pytest.approx(
        1000.0 * 0.1 / (3.0 * 1000.0), rel=1e-14
    )
    omega = pb.ss_beam_angular_frequencies(4.0, 1.0e5, 20.0, modes=3)
    assert omega[0] == pytest.approx((math.pi / 4.0) ** 2 * math.sqrt(1.0e5 / 20.0))
    assert omega[1] / omega[0] == pytest.approx(4.0)


def test_time_saving_series_truncation_is_documented():
    """The mandated 11x11 Navier truncation is already within 1e-4 on w_c."""
    rigidity = pb.plate_rigidity(E, NU, T)
    coarse = pb.ssss_navier_centre(A, A, Q, rigidity, terms=11)
    fine = pb.ssss_navier_centre(A, A, Q, rigidity, terms=301)
    assert abs(coarse - fine) / fine < 1e-4
    # moments converge more slowly: the 11x11 truncation is only ~1e-3 accurate
    coarse_m = pb.ssss_navier_moments(A / 2, A / 2, A, A, Q, NU, terms=11)
    fine_m = pb.ssss_navier_moments(A / 2, A / 2, A, A, Q, NU, terms=301)
    assert 1e-4 < abs(coarse_m[0] - fine_m[0]) / fine_m[0] < 1e-2


def test_boundary_dofs_are_subsets_and_counts():
    ssss = pb.bfs_boundary_dofs(4, 4, "ssss")
    cccc = pb.bfs_boundary_dofs(4, 4, "cccc")
    assert set(ssss).issubset(set(cccc))
    # hard SS removes w plus the tangential slope: two dofs on an edge node,
    # three on a corner (both tangential slopes).  Perimeter nodes = 4*4 = 16.
    assert len(ssss) == 2 * 16 + 4
    assert len(cccc) == 3 * 16
    with pytest.raises(ValueError):
        pb.bfs_boundary_dofs(4, 4, "free")