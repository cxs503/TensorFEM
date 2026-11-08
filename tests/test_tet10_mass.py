"""Verification of the TET10 consistent / lumped mass matrix.

Why this file exists
--------------------
``docs/tet10-integration-accuracy.md`` established that the degree-three Keast
rule shipped with ``tet10_stiffness`` cannot integrate ``N_i N_j`` and left
``tet10_mass`` as an explicit debt ("must switch rules").  The production
:func:`tensorfem.quadratic_solid.tet10_consistent_mass` closes that debt with a
collapsed Gauss rule.  This file pins the closure against independent references:

1. the exact barycentric shape-function integrals,
   ``int N_i dV = -V/20`` (vertex) and ``+V/5`` (mid-edge),
2. an independent NumPy 24^3 Duffy reference (``tet10_integration_audit``),
3. the shipped five-point Keast rule, which must stay rejected (fail-closed),
4. a physical anchor: the fixed-free rod axial frequencies, whose exact values are
   ``(2n-1) pi / (2L) sqrt(M_c / rho)``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfem.quadratic_solid import (
    tet10_consistent_mass,
    tet10_lumped_mass,
    tet10_mass_quadrature,
    tet10_quadrature,
    tet10_volume,
)
from tensorfem.solid_mass import assemble_solid_mass, mass_properties
from tensorfem.solid_mass_audit import axial_rod_case, confined_modulus, rod_axial_angular_frequency
from tensorfem.tet10_integration_audit import (
    duffy_rule,
    element_mass,
    relative_frobenius,
    tet10_geometry,
)

D = torch.float64


def _element(kind="regular", curvature=0.0):
    return torch.tensor(tet10_geometry(kind, curvature), dtype=D).reshape(1, 10, 3)


def _reference(kind="regular", curvature=0.0, rho=1.0):
    return element_mass(tet10_geometry(kind, curvature), rho, *duffy_rule(24))


def test_straight_sided_mass_matches_analytic_barycentric_integrals():
    """Row sums are `int N_i dV`: `-rho V/20` at a vertex, `+rho V/5` at a mid-edge node."""
    rho = 7850.0
    x = _element()
    volume = 1.0 / 6.0
    block = tet10_consistent_mass(x, rho)[0][0::3, 0::3]
    rows = block.sum(-1)
    assert torch.allclose(rows[:4], torch.full((4,), -rho * volume / 20.0, dtype=D), rtol=1e-13)
    assert torch.allclose(rows[4:], torch.full((6,), rho * volume / 5.0, dtype=D), rtol=1e-13)
    # total mass: 4 * (-1/20) + 6 * (1/5) = 1
    assert math.isclose(float(rows.sum()), rho * volume, rel_tol=1e-13)
    # the scalar block is reproduced by the independent 24^3 Duffy reference
    assert relative_frobenius(block.numpy(), _reference(rho=rho)) < 1e-12


def test_default_rule_is_exact_for_the_mass_integrand():
    """Order 4 collapsed Gauss reproduces the reference on straight, distorted and curved tets."""
    for kind, curvature in (("regular", 0.0), ("distorted", 0.0), ("distorted", 0.12)):
        block = tet10_consistent_mass(_element(kind, curvature), 1.0)[0][0::3, 0::3]
        assert relative_frobenius(block.numpy(), _reference(kind, curvature)) < 1e-12
        # mass conservation: total = rho * element volume
        volume = float(tet10_volume(_element(kind, curvature)))
        assert math.isclose(float(block.sum()), volume, rel_tol=1e-13)


def test_monomial_exactness_of_the_collapsed_rule():
    """Order 4 is degree-5 exact; order 6 is degree-6 exact.

    The guard is honest about the order accounting: the collapse weight ``u^2 v``
    costs two degrees, so ``order`` points per direction are exact only for total
    degree ``<= 2*order - 3``.  The TET10 mass integrand nevertheless uses at most
    degree two per collapsed variable, which is why order 4 is already exact for it
    (pinned by ``test_default_rule_is_exact_for_the_mass_integrand``).
    """
    from math import factorial

    def exact(a, b, c):
        return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3)

    points, weights = tet10_mass_quadrature(4)
    assert math.isclose(float(weights.sum()), 1.0 / 6.0, rel_tol=1e-14)
    high_points, high_weights = tet10_mass_quadrature(6)
    worst = worst_high = 0.0
    for a in range(7):
        for b in range(7 - a):
            for c in range(7 - a - b):
                value = float((high_weights * high_points[:, 0] ** a * high_points[:, 1] ** b
                               * high_points[:, 2] ** c).sum())
                worst_high = max(worst_high, abs(value - exact(a, b, c)))
                if a + b + c > 5:
                    continue
                value = float((weights * points[:, 0] ** a * points[:, 1] ** b
                               * points[:, 2] ** c).sum())
                worst = max(worst, abs(value - exact(a, b, c)))
    assert worst < 1e-14
    assert worst_high < 1e-15


def test_shipped_keast_rule_still_fails_the_mass_matrix():
    """Fail-closed: the documented degree-three debt is not silently forgotten.

    With the Keast five-point rule the consistent mass loses 19.68 % in the
    Frobenius norm, 70.8 % of every vertex diagonal and 19.8 % of every mid-edge
    diagonal.  If somebody switches the production default back to Keast, this
    test fails first.
    """
    reference = _reference()
    block = tet10_consistent_mass(_element(), 1.0, rule=tet10_quadrature(dtype=D))[0][0::3, 0::3]
    assert relative_frobenius(block.numpy(), reference) > 0.15
    deviation = (torch.diagonal(block) - torch.tensor(np.diag(reference))) / torch.tensor(np.diag(reference))
    assert float(deviation[:4].min()) < -0.70 and float(deviation[4:].max()) > -0.25
    # mass conservation alone cannot detect the error: it still sums to rho V
    assert math.isclose(float(block.sum()), 1.0 / 6.0, rel_tol=1e-12)


def test_consistent_mass_is_symmetric_positive_definite():
    """A single element has no null mode: ``M`` must be SPD (rigid modes are K-null, not M-null)."""
    mass = tet10_consistent_mass(_element(), 2700.0)[0]
    assert torch.allclose(mass, mass.T, rtol=0.0, atol=1e-9 * float(mass.abs().max()))
    values = torch.linalg.eigvalsh((mass + mass.T) / 2.0)
    assert float(values.min()) > 0.0


def test_hrz_lumped_mass_is_positive_and_mass_conserving():
    rho = 1000.0
    lumped = tet10_lumped_mass(_element(), rho)[0]
    diagonal = torch.diagonal(lumped)
    assert float(diagonal.min()) > 0.0  # row-sum lumping would give -rho V/20 < 0
    # each scalar nodal mass is repeated for ux, uy, uz
    assert math.isclose(float(diagonal.sum()) / 3.0, rho * (1.0 / 6.0), rel_tol=1e-13)


def test_assembly_and_mass_properties_for_two_element_rod():
    """Assembly in ``solid_mass`` accepts TET10 and recovers rho V exactly."""
    nodes, elements = _rod_mesh(4)
    mass = assemble_solid_mass(nodes, elements, 7850.0, element_type="tet10")
    total = float(torch.ones(3 * len(nodes), dtype=D) @ mass @ torch.ones(3 * len(nodes), dtype=D)) / 3.0
    assert math.isclose(total, 7850.0 * float(tet10_volume(nodes[elements]).sum()), rel_tol=1e-12)
    props = mass_properties(nodes, elements, 7850.0, element_type="tet10")
    assert props.mass_error < 1e-12


def _rod_mesh(elements: int):
    from tensorfem.solid3d import hex_to_tet_mesh, structured_hex_mesh
    from tensorfem.quadratic_solid import tet4_to_tet10

    nodes, hexes = structured_hex_mesh(10.0, 0.2, 0.2, elements, 1, 1, dtype=D)
    nodes, cells = tet4_to_tet10(nodes, hex_to_tet_mesh(nodes, hexes)[1])
    return nodes, cells


def test_tet10_rod_axial_frequencies_converge_to_the_exact_rod():
    """Physical anchor: fixed-free rod, ``omega_n = (2n-1) pi / (2L) sqrt(M_c/rho)``."""
    length, young, poisson, density = 10.0, 210.0e9, 0.3, 7850.0
    exact = [rod_axial_angular_frequency(n, length, confined_modulus(young, poisson), density)
             for n in (1, 2)]
    cases = [axial_rod_case(n, element_type="tet10", modes=2) for n in (4, 8, 16)]
    errors = [[abs(case.frequencies[i] / exact[i] - 1.0) for i in range(2)] for case in cases]
    assert errors[-1][0] < 0.01, errors
    assert errors[-1][1] < 0.02, errors
    # monotone improvement of the first mode on the two finest meshes
    assert errors[-1][0] < errors[-2][0] < errors[-3][0]
    # TET10 must beat TET4 on the same 4-element mesh for mode 1
    tet4 = axial_rod_case(4, element_type="tet4", modes=2)
    assert abs(tet4.frequencies[0] - exact[0]) > errors[0][0] * exact[0]
