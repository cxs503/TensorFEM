"""Skew invariance of the Q4 Mindlin assumed-natural shear field ("mitc").

Before 2026-11 the ``mitc`` branch of :func:`tensorfem.plate.q4_mindlin_stiffness`
sampled the *physical* transverse shear rows ``gamma_xz``/``gamma_yz`` at the four
mid-sides.  That is only equivalent to the Dvorkin-Bathe assumed field for
rectangles: on a parallelogram the exact shear-free constant-curvature state
picks up a parasitic shear energy that grows with the skew (24x the bending
energy at 30 deg of skew), i.e. the "locking-free" element locks on skew meshes
and the Morley rhombic-plate benchmark cannot be reproduced with it.  The
covariant sampling implemented now is exact for any parallelogram.

The tests below are invariants, not smoke tests:

* the exact constant-curvature state carries no shear energy to round-off, at
  every skew angle (the property the old sampling violated);
* the element keeps exactly the three rigid-body mechanisms (no spurious mode),
  which is what the ``mitc`` treatment exists for;
* rectangles are unchanged (the two samplings coincide there), so nothing in
  the existing rectangular validation ladder moves.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.plate import q4_mindlin_stiffness

E, NU, T = 10.92, 0.3, 0.1
SKEW_ANGLES = [90.0, 75.0, 60.0, 45.0, 30.0, 20.0]


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)
    yield


def parallelogram(angle_deg: float, side: float = 1.0) -> torch.Tensor:
    a = math.radians(angle_deg)
    return torch.tensor(
        [[0.0, 0.0],
         [side, 0.0],
         [side + side * math.cos(a), side * math.sin(a)],
         [side * math.cos(a), side * math.sin(a)]],
        dtype=torch.float64,
    )


def curvature_state(xy: torch.Tensor, kx: float = 1.0, kxy: float = 0.7,
                    ky: float = 1.3) -> torch.Tensor:
    """Nodal values of the exact constant-curvature state (shear free)."""
    x, y = xy[:, 0], xy[:, 1]
    w = -(kx * x ** 2 + kxy * x * y + ky * y ** 2) / 2.0
    theta_x = kx * x + kxy * y / 2.0
    theta_y = kxy * x / 2.0 + ky * y
    return torch.stack((w, theta_x, theta_y), dim=1).reshape(-1)


@pytest.mark.parametrize("angle", SKEW_ANGLES)
def test_mitc_is_shear_free_on_skew_parallelograms(angle):
    xy = parallelogram(angle)
    u = curvature_state(xy)
    bending = q4_mindlin_stiffness(
        xy, E, NU, T, shear_correction=1e-14, shear_treatment="mitc",
    )
    ke = q4_mindlin_stiffness(xy, E, NU, T, shear_treatment="mitc")
    e_bend = float(u @ bending @ u)
    e_total = float(u @ ke @ u)
    assert e_bend > 0.0
    assert abs(e_total - e_bend) / e_bend < 1e-10


@pytest.mark.parametrize("angle", SKEW_ANGLES)
def test_mitc_rank_is_exactly_the_rigid_body_modes(angle):
    xy = parallelogram(angle)
    ke = q4_mindlin_stiffness(xy, E, NU, T, shear_treatment="mitc")
    eigenvalues = torch.linalg.eigvalsh(ke)
    zero = int((eigenvalues < 1e-9 * float(eigenvalues.max())).sum())
    assert zero == 3


def test_mitc_and_sri_coincide_on_a_rectangle_bending_state():
    """On a rectangle both shear samplings are the same field."""
    xy = parallelogram(90.0)
    u = curvature_state(xy)
    mitc = q4_mindlin_stiffness(xy, E, NU, T, shear_treatment="mitc")
    sri = q4_mindlin_stiffness(xy, E, NU, T, shear_treatment="sri")
    bending = q4_mindlin_stiffness(
        xy, E, NU, T, shear_correction=1e-14, shear_treatment="mitc",
    )
    # shear-free state: both energies equal the bending energy
    e_bend = float(u @ bending @ u)
    assert abs(float(u @ mitc @ u) - e_bend) / e_bend < 1e-10
    assert abs(float(u @ sri @ u) - e_bend) / e_bend < 1e-10


def test_rigid_body_modes_carry_no_energy_on_skew_elements():
    """Zero strain needs theta constant *and* w linear: w,x = -theta_x,
    w,y = -theta_y (a rotation, not a bare constant rotation vector)."""
    xy = parallelogram(30.0)
    x, y = xy[:, 0], xy[:, 1]
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    modes = [
        torch.stack((one, zero, zero), dim=1).reshape(-1),      # translation
        torch.stack((-x, one, zero), dim=1).reshape(-1),        # rotation about y
        torch.stack((-y, zero, one), dim=1).reshape(-1),        # rotation about x
    ]
    for treatment in ("sri", "mitc"):
        ke = q4_mindlin_stiffness(xy, E, NU, T, shear_treatment=treatment)
        for vector in modes:
            energy = float(vector @ ke @ vector)
            scale = float(torch.linalg.norm(vector) ** 2) * float(ke.abs().max())
            assert abs(energy) < 1e-10 * scale
