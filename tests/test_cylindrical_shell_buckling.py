"""Tests for the flat-facet cylindrical Shell4 eigenbuckling assembly.

Two layers are covered:

* exact invariants of the new assembly (facet planarity, generator-aligned local
  axis, shared theta seam, membrane hoop energy against the chord/arc area
  ratio, Donnell reference algebra);
* a *fail-closed regression* on the physics debt: the flat-facet cylinder is
  over-stiff for axial compression and its critical mode stops being
  axisymmetric once the circumferential mesh is resolved, so the classical
  3 % agreement is NOT reached.  These assertions exist so the gap cannot be
  closed by editing a tolerance; they must be replaced only by a mesh family
  that really converges (see docs/cylindrical-shell-geometric-stiffness-audit.md).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.cylindrical_shell_buckling import (
    classical_axial_buckling_resultant,
    cylindrical_shell_axial_buckling,
    cylindrical_shell_mesh,
    donell_axisymmetric_load,
)
from tensorfem.shell4 import shell4_local_frame, shell4_stiffness

CASE = dict(radius=1.0, length=1.2, young=200e9, poisson=0.3, thickness=0.01)


def test_mesh_geometry_invariants():
    R, L, nz, nt = 1.0, 1.2, 4, 6
    nodes, elements = cylindrical_shell_mesh(R, L, nz, nt)
    assert nodes.shape == ((nz + 1) * nt, 3), "theta seam must be shared, not duplicated"
    assert elements.shape == (nz * nt, 4)
    # unique node set (no duplicated seam nodes at theta = 0 / 2*pi)
    assert torch.unique(nodes, dim=0).shape[0] == nodes.shape[0]
    for el in elements:
        xyz = nodes[el]
        assert torch.allclose(
            torch.linalg.vector_norm(xyz[:, 1:], dim=1),
            torch.full((4,), R, dtype=nodes.dtype),
            atol=1e-12,
        )
        # planar facet (ruled cylinder patch)
        nrm = torch.linalg.cross(xyz[1] - xyz[0], xyz[3] - xyz[0], dim=0)
        nrm = nrm / torch.linalg.vector_norm(nrm)
        assert abs(float((xyz[2] - xyz[0]) @ nrm)) < 1e-12
        # first local axis is a generator, so [Nx,0,0] is an axial compression
        _, basis = shell4_local_frame(xyz)
        assert abs(abs(float(basis[0, 0])) - 1.0) < 1e-12


def test_mesh_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        cylindrical_shell_mesh(-1.0, 1.0, 4, 6)
    with pytest.raises(ValueError):
        cylindrical_shell_mesh(1.0, 1.0, 1, 6)
    with pytest.raises(ValueError):
        cylindrical_shell_mesh(1.0, 1.0, 4, 1.5)


def test_donnell_reference_algebra():
    finite = donell_axisymmetric_load(**CASE)
    classical = classical_axial_buckling_resultant(
        young=CASE["young"],
        poisson=CASE["poisson"],
        thickness=CASE["thickness"],
        radius=CASE["radius"],
    )
    # The discrete half-wave restriction can only sit at or above the asymptote.
    assert finite >= classical
    assert finite / classical < 1.001
    # Textbook value for the asymptote.
    E, t, R, nu = CASE["young"], CASE["thickness"], CASE["radius"], CASE["poisson"]
    assert classical == pytest.approx(E * t**2 / (R * math.sqrt(3 * (1 - nu**2))), rel=1e-14)
    # A longer cylinder admits the asymptotic wavelength, so it must come closer.
    longer = donell_axisymmetric_load(length=2.4, young=E, poisson=nu, thickness=t, radius=R)
    assert longer <= finite * (1 + 1e-9)
    assert longer / classical - 1.0 < 1e-3


def test_hoop_membrane_energy_matches_facet_area_ratio():
    """Uniform radial shrink: the faceted ring must reproduce the chord/arc ratio."""
    R, L = CASE["radius"], CASE["length"]
    E, nu, t = CASE["young"], CASE["poisson"], CASE["thickness"]
    for nt in (8, 16, 32):
        nodes, elements = cylindrical_shell_mesh(R, L, 2, nt)
        ndof = 6 * len(nodes)
        K = torch.zeros((ndof, ndof), dtype=nodes.dtype)
        for el in elements:
            ids = torch.stack(tuple(6 * el + k for k in range(6)), 1).reshape(-1)
            K[ids[:, None], ids] += shell4_stiffness(nodes[el], E, nu, t)
        radial = nodes.clone()
        radial[:, 0] = 0.0
        radial = radial / torch.linalg.vector_norm(radial, dim=1, keepdim=True)
        u = torch.zeros(ndof, dtype=nodes.dtype)
        for c in range(3):
            u[c::6] = -radial[:, c]
        fem = 0.5 * float(u @ K @ u)
        continuum = 0.5 * E * t / R**2 * (2 * math.pi * R * L)
        facet_area = nt * math.sin(math.pi / nt) / math.pi
        # The prescribed shrink blocks axial contraction, so the membrane energy
        # carries the plane-stress 1/(1-nu**2) factor on top of the chord/arc
        # area ratio of the faceted ring.
        assert abs(fem / continuum / facet_area * (1 - nu**2) - 1.0) < 5e-3


def test_buckling_load_positive_and_finite():
    res = cylindrical_shell_axial_buckling(ntheta=16, nz=6, **CASE)
    value = float(res.load_factors[0])
    assert math.isfinite(value) and value > 0


def test_axial_refinement_reduces_over_stiffness():
    """Fixed ntheta: refining along the generator must move toward theory."""
    finite = donell_axisymmetric_load(**CASE)
    coarse = float(cylindrical_shell_axial_buckling(ntheta=8, nz=4, **CASE).load_factors[0])
    fine = float(cylindrical_shell_axial_buckling(ntheta=8, nz=8, **CASE).load_factors[0])
    assert coarse > fine > finite


@pytest.mark.parametrize("nz,ntheta", [(6, 8), (10, 8)])
def test_over_stiffness_debt_is_locked(nz, ntheta):
    """Fail-closed: coarse flat-facet cylinders are >30 % above Donnell theory."""
    finite = donell_axisymmetric_load(**CASE)
    value = float(cylindrical_shell_axial_buckling(ntheta=ntheta, nz=nz, **CASE).load_factors[0])
    assert (value - finite) / finite > 0.30


def test_critical_mode_loses_axisymmetry_on_a_resolved_ring():
    """The measured 17 % under-prediction is carried by a non-axisymmetric mode."""
    nz, nt = 20, 32
    res = cylindrical_shell_axial_buckling(ntheta=nt, nz=nz, **CASE)
    nodes = res.nodes
    mode = res.modes[:, 0].reshape(-1, 6)
    radial = nodes.clone()
    radial[:, 0] = 0.0
    radial = radial / torch.linalg.vector_norm(radial, dim=1, keepdim=True)
    amp = ((mode[:, :3] * radial).sum(1)).abs().reshape(nz + 1, nt)[1:-1]
    spread = (amp.max(1).values - amp.min(1).values) / amp.max(1).values.clamp_min(1e-30)
    assert float(spread.max()) > 1e-3, "expected a circumferential mode at ntheta=32"
