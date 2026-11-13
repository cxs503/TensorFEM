"""Regression gate for CASE-CANTILEVER-TIP.

The case is the calculation-pool anchor ``CASE-CANTILEVER-TIP``: the tip
deflection of a slender cantilever beam under a transverse end load, against
Euler--Bernoulli beam theory (delta = P*L^3/(3*E*I)) and the Timoshenko
bending-plus-shear correction (delta = P*L^3/(3*E*I) + P*L/(kappa*A*G)).
Tolerance: <3% relative error vs the Euler--Bernoulli analytic reference.

These tests re-derive the analytic reference from first principles and re-run
the two cheap model levels (1-D cubic-Hermite Euler--Bernoulli frame elements
and a 3-D HEX20 solid at a coarse mesh), so a regression in either the frame
static solver or the solid static path is caught without paying for the full
convergence/sensitivity sweep (which lives in
``benchmarks/cantilever_tip_deflection.py``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensorfem.frame2d import FrameModel, solve_frame_static
from tensorfem.hex20 import Hex20Model, solve_hex20, structured_hex20_mesh

# --------------------------------------------------------------------------- #
# Problem definition (mirrors benchmarks/cantilever_tip_deflection.py)
# --------------------------------------------------------------------------- #
L = 1.0          # beam length [m]
b = 0.01         # width [m]
h = 0.01         # height [m]
E = 200e9        # Young's modulus [Pa]
nu = 0.3         # Poisson's ratio
P = 5.0          # tip load [N]

A = b * h
I = b * h**3 / 12.0
G = E / (2.0 * (1.0 + nu))
kappa = 5.0 / 6.0  # shear correction factor (rectangular section)

# Analytic references
delta_eb = P * L**3 / (3.0 * E * I)          # Euler-Bernoulli
delta_shear = P * L / (kappa * A * G)         # shear contribution
delta_timoshenko = delta_eb + delta_shear     # Timoshenko

DTYPE = torch.float64
TOLERANCE = 0.03  # 3%


def _eb_frame_level(n_elements):
    """Cantilever with cubic-Hermite Euler-Bernoulli frame elements."""
    nodes = torch.stack([
        torch.linspace(0.0, L, n_elements + 1, dtype=DTYPE),
        torch.zeros(n_elements + 1, dtype=DTYPE)
    ], dim=1)
    elements = torch.stack([torch.arange(n_elements), torch.arange(1, n_elements + 1)], dim=1)
    forces = torch.zeros(3 * (n_elements + 1), dtype=DTYPE)
    forces[3 * n_elements + 1] = -P
    model = FrameModel(
        nodes=nodes, elements=elements,
        young_modulus=torch.tensor([E], dtype=DTYPE),
        area=torch.tensor([A], dtype=DTYPE),
        inertia=torch.tensor([I], dtype=DTYPE),
        forces=forces, fixed_dofs=torch.tensor([0, 1, 2])
    )
    result = solve_frame_static(model)
    return -float(result.displacement[3 * n_elements + 1])


def _hex20_level(n_axial, n_cross=1):
    """Cantilever as 3-D HEX20 solid (coarse h-refinement)."""
    nodes, cells = structured_hex20_mesh(
        L, b, h, n_axial, n_cross, n_cross, dtype=DTYPE
    )
    right = torch.where(nodes[:, 0] == L)[0]
    forces = torch.zeros(3 * len(nodes), dtype=DTYPE)
    forces[3 * right + 2] = -P / len(right)
    left = torch.where(nodes[:, 0] == 0.0)[0]
    fixed = torch.stack([3 * left, 3 * left + 1, 3 * left + 2], dim=1).reshape(-1)
    model = Hex20Model(
        nodes=nodes, elements=cells,
        young_modulus=torch.tensor(E, dtype=DTYPE),
        poisson_ratio=torch.tensor(nu, dtype=DTYPE),
        forces=forces, fixed_dofs=fixed
    )
    result = solve_hex20(model)
    return -float(result.displacement[3 * right + 2].mean())


# --------------------------------------------------------------------------- #
# Reference self-consistency
# --------------------------------------------------------------------------- #
def test_analytic_reference_self_consistent():
    """Euler-Bernoulli and Timoshenko references must be internally consistent."""
    # EB: delta = P*L^3/(3*E*I) evaluated independently
    assert delta_eb == pytest.approx(P * L**3 / (3.0 * E * I), rel=1e-15)
    # Timoshenko = EB + shear term
    assert delta_timoshenko == pytest.approx(delta_eb + delta_shear, rel=1e-15)
    # Shear correction factor for rectangular section
    assert kappa == pytest.approx(5.0 / 6.0)


def test_geometry_declared_slenderness():
    """L/h = 100 (slender), so shear contribution is small but non-negligible."""
    assert L / h == pytest.approx(100.0)
    assert A == pytest.approx(b * h)
    assert I == pytest.approx(b * h**3 / 12.0)
    # Shear contribution is a small fraction of the EB deflection
    assert delta_shear / delta_eb < 0.05


# --------------------------------------------------------------------------- #
# Frame element (Euler-Bernoulli) gate
# --------------------------------------------------------------------------- #
def test_frame_tip_deflection_within_tolerance():
    """Cubic-Hermite EB frame element must reproduce the analytic tip deflection."""
    tip = _eb_frame_level(8)
    rel_err = abs(tip - delta_eb) / delta_eb
    assert rel_err < TOLERANCE, f"frame tip error {rel_err*100:.2e}% > 3%"


def test_frame_is_exact_for_cubic_eb_solution():
    """The analytic EB tip deflection is a cubic polynomial in x, which the
    cubic-Hermite frame element reproduces exactly (up to roundoff)."""
    tip = _eb_frame_level(4)
    rel_err = abs(tip - delta_eb) / delta_eb
    assert rel_err < 1e-9, f"frame tip error {rel_err*100:.2e}% not machine-exact"


# --------------------------------------------------------------------------- #
# HEX20 solid gate
# --------------------------------------------------------------------------- #
def test_hex20_tip_deflection_within_tolerance():
    """Coarse 3-D solid check (nx=8): exercises the solid static path end to end."""
    tip = _hex20_level(8)
    rel_err = abs(tip - delta_eb) / delta_eb
    assert rel_err < TOLERANCE, f"hex20 tip error {rel_err*100:.2f}% > 3%"


def test_hex20_convergence_order_is_positive():
    """Refined solid meshes must be at least as accurate as the coarse one."""
    coarse = _hex20_level(8)
    fine = _hex20_level(32)
    err_c = abs(coarse - delta_eb) / delta_eb
    err_f = abs(fine - delta_eb) / delta_eb
    assert err_f <= err_c + 1e-15


# --------------------------------------------------------------------------- #
# Cross-model consistency
# --------------------------------------------------------------------------- #
def test_hex20_within_tolerance_of_both_references():
    """At L/h=100 the shear contribution is ~0.008%, so the EB and Timoshenko
    references are nearly identical.  The 3-D solid (which models shear
    deformation) must sit within the 3% tolerance of both references."""
    tip = _hex20_level(32)
    assert abs(tip - delta_eb) / delta_eb < TOLERANCE
    assert abs(tip - delta_timoshenko) / delta_timoshenko < TOLERANCE
