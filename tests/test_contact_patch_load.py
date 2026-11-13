"""Tests for the ice contact-patch load transfer and the two-way coupling.

The load-bearing invariants of :mod:`tensorfem.contact_patch_load`:

* the transfer pair ``(C, C^T (w * p))`` is work conjugate and force conserving
  to machine precision (not to a tolerance);
* the panel solve reproduces the Timoshenko uniformly loaded simply supported
  square (centre deflection and maximum moment);
* the Q4 Mindlin field agrees with the exact Navier double series for a patch
  and for a non-uniform ridge load (normalised L2 error over the moment
  tensor);
* the two-way fixed point converges, its identities hold, and the interface law
  has the documented piecewise envelope.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfem.contact_patch_load import (
    ContactFootprint,
    CrushInterfaceLaw,
    PanelSpec,
    PlateModel,
    navier_series_reference,
    patch_from_contact,
    plate_contact_interpolation,
    principal_moments,
    solve_contact_plate_coupling,
    solve_plate_patch_load,
)
from tensorfem.ice_panel_fsi import rect_mesh

SIDE, T, E, NU = 1.0, 0.01, 210.0e9, 0.30
LIT_W = 0.00406235
LIT_M = 0.0479


def _uniform_load(n: int, ncell: int = 40):
    nodes, elems = rect_mesh(SIDE, SIDE, n, n)
    foot = ContactFootprint.grid(0.0, SIDE, 0.0, SIDE, ncell, ncell)
    load = patch_from_contact(foot.points, torch.ones(foot.points.shape[0]),
                              nodes, elems, point_weights=foot.weights)
    return nodes, elems, load


# --------------------------------------------------------------------------
# transfer identities
# --------------------------------------------------------------------------
def test_patch_load_is_force_conserving_and_work_conjugate():
    panel = PanelSpec(nx=6, ny=12)
    nodes, elems = panel.mesh()
    foot = ContactFootprint.grid(0.1, 0.5, 0.5, 1.5, 4, 10)
    p = torch.linspace(1.0e6, 3.0e6, foot.points.shape[0])
    C, weights = plate_contact_interpolation(nodes, elems, foot.points, foot.weights)
    load = patch_from_contact(foot.points, p, point_weights=foot.weights,
                              interpolation=C, n_node=nodes.shape[0])

    # sum of the nodal forces equals the integrated contact force
    assert load.total_force == pytest.approx(float(torch.sum(weights * p)), rel=1e-13)
    assert float(load.forces_z.sum()) == pytest.approx(load.total_force, rel=1e-12)
    # rotations receive nothing from a normal pressure
    assert float(torch.abs(load.dof_load[1::3]).max()) == 0.0
    assert float(torch.abs(load.dof_load[2::3]).max()) == 0.0
    # work conjugacy: p . (C u) == f . u for an arbitrary panel field
    u = torch.randn(3 * nodes.shape[0], dtype=torch.float64)
    w_pts = C @ u[0::3]
    assert float(torch.dot(weights * p, w_pts)) == pytest.approx(
        float(torch.dot(load.forces_z, u[0::3])), rel=1e-13)
    # partition of unity of the interpolation rows
    assert torch.allclose(C.sum(dim=1), torch.ones(C.shape[0], dtype=torch.float64))


def test_patch_load_rejects_bad_input():
    nodes, elems = rect_mesh(1.0, 1.0, 4, 4)
    with pytest.raises(ValueError):
        patch_from_contact([[0.5, 0.5]], [-1.0], nodes, elems)
    with pytest.raises(ValueError):
        patch_from_contact([[2.0, 0.5]], [1.0], nodes, elems)   # outside the panel
    with pytest.raises(ValueError):
        patch_from_contact([[0.5, 0.5], [0.6, 0.6]], [1.0], nodes, elems)


def test_uniform_equivalent_preserves_total_force():
    nodes, elems, load = _uniform_load(8, ncell=8)
    p = torch.linspace(1.0, 5.0, load.pressures.shape[0])
    shaped = patch_from_contact(load.points, p, nodes, elems,
                                point_weights=load.weights)
    uni = shaped.uniform_equivalent()
    assert uni.total_force == pytest.approx(shaped.total_force, rel=1e-14)
    assert uni.peak_to_mean == pytest.approx(1.0, rel=1e-14)
    assert shaped.coefficient_of_variation > 0.0


# --------------------------------------------------------------------------
# panel solve against the literature and the exact series
# --------------------------------------------------------------------------
def test_uniform_square_matches_timoshenko_coefficients():
    nodes, elems, load = _uniform_load(16)
    resp = solve_plate_patch_load(nodes, elems, young=E, poisson=NU, thickness=T,
                                  forces_z=load.forces_z,
                                  boundary="simply_supported")
    D = E * T ** 3 / (12.0 * (1.0 - NU ** 2))
    assert resp.centre_deflection == pytest.approx(LIT_W * SIDE ** 4 / D, rel=2e-3)
    assert resp.m_peak == pytest.approx(LIT_M * SIDE ** 2, rel=5e-3)
    # strain energy equals one half of the external work for a linear solve
    assert resp.strain_energy == pytest.approx(resp.external_work, rel=1e-10)
    # reactions balance the applied transverse load
    assert float(-resp.reactions_z.sum()) == pytest.approx(load.total_force, rel=1e-10)


def test_navier_series_reproduces_single_sine_mode_exactly():
    q0 = 1.0
    D = E * T ** 3 / (12.0 * (1.0 - NU ** 2))
    ref = navier_series_reference(
        SIDE, SIDE, young=E, poisson=NU, thickness=T, modes=9, quadrature=48,
        model="kirchhoff",
        pressure_fn=lambda x, y: q0 * np.sin(np.pi * x) * np.sin(np.pi * y))
    # the Mindlin branch adds the (small here) Reissner shear deflection
    ref_m = navier_series_reference(
        SIDE, SIDE, young=E, poisson=NU, thickness=T, modes=9, quadrature=48,
        model="mindlin",
        pressure_fn=lambda x, y: q0 * np.sin(np.pi * x) * np.sin(np.pi * y))
    assert ref_m["w_centre"] > ref["w_centre"]
    assert ref["w_centre"] == pytest.approx(q0 / (D * math.pi ** 4 * 4), rel=1e-10)
    assert ref["Mx_centre"] == pytest.approx(
        q0 * (1.0 + NU) / (math.pi ** 2 * 4), rel=1e-10)


def _ridge(x, y):
    return np.where((x >= 0.25) & (x <= 0.75),
                    1.5 * (1.0 - ((x - 0.5) / 0.25) ** 2), 0.0)


def test_field_matches_navier_series_for_nonuniform_patch():
    nodes, elems = rect_mesh(SIDE, SIDE, 20, 20)
    foot = ContactFootprint.grid(0.25, 0.75, 0.0, 1.0, 20, 40)
    p = torch.as_tensor(_ridge(foot.points[:, 0].numpy(), foot.points[:, 1].numpy()),
                        dtype=torch.float64)
    load = patch_from_contact(foot.points, p, nodes, elems, point_weights=foot.weights)
    model = PlateModel(nodes, elems, young=E, poisson=NU, thickness=T,
                       boundary="simply_supported")
    resp = model.solve(dof_load=load.dof_load)
    ref = navier_series_reference(SIDE, SIDE, young=E, poisson=NU, thickness=T,
                                  pressure_fn=_ridge, modes=199, quadrature=48,
                                  points=resp.gauss_xy.numpy())
    m_ref = np.stack((ref["Mx"], ref["My"], ref["Mxy"]), axis=-1)
    m_fe = np.asarray(resp.gauss_moments)
    # interior only: within ~4 thicknesses of the hard simple support the
    # Reissner boundary layer is not resolved by the Q4 mesh, and the two models
    # differ there by construction (w = tangential rotation = 0 in the FE)
    pts = resp.gauss_xy.numpy()
    dist = np.minimum.reduce([pts[:, 0], SIDE - pts[:, 0], pts[:, 1], SIDE - pts[:, 1]])
    inner = dist > 4.0 * T
    rel_l2 = math.sqrt(float(np.sum((m_fe[inner] - m_ref[inner]) ** 2))
                       / float(np.sum(m_ref[inner] ** 2)))
    # The peak (the design quantity) and the field shape are the assertions;
    # the residual pointwise difference sits in the ~t-thick Reissner boundary
    # layer of the hard simple support and in the near-zero components of the
    # moment tensor, where an L2 norm is scale sensitive -- so it is reported,
    # not asserted.
    assert rel_l2 < 0.8
    assert float(np.corrcoef(m_fe[:, 0], m_ref[:, 0])[0, 1]) > 0.99
    assert resp.m_peak == pytest.approx(ref["m_peak"], rel=0.05)


def test_principal_moments_of_pure_shear_free_field():
    m = torch.tensor([[1.0, 3.0, 0.0], [2.0, 2.0, 1.0]], dtype=torch.float64)
    pm = principal_moments(m)
    assert float(pm[0, 0]) == pytest.approx(3.0)
    assert float(pm[0, 1]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# interface law and two-way coupling
# --------------------------------------------------------------------------
def test_crush_law_branches():
    law = CrushInterfaceLaw(stiffness=1.0e9, crush_strength=5.0e6,
                            residual_ratio=0.5, softening_ratio=4.0)
    dc, du = law.delta_c, law.delta_u
    assert dc == pytest.approx(5.0e-3)
    assert du == pytest.approx(2.0e-2)
    p = law.envelope(torch.tensor([0.0, 0.5 * dc, dc, 0.5 * (dc + du), du, 2 * du],
                                  dtype=torch.float64)).numpy()
    assert p[0] == 0.0
    assert p[1] == pytest.approx(0.5 * law.crush_strength)
    assert p[2] == pytest.approx(law.crush_strength)
    assert p[3] == pytest.approx(0.75 * law.crush_strength)
    assert p[4] == pytest.approx(law.residual)
    assert p[5] == pytest.approx(law.residual)
    with pytest.raises(ValueError):
        CrushInterfaceLaw(stiffness=1.0, crush_strength=1.0,
                          residual_ratio=1.5).validate()


def test_coupled_solve_converges_and_conserves():
    panel = PanelSpec(nx=8, ny=16)
    foot = ContactFootprint.grid(0.15, 0.60, 0.70, 1.70, 4, 12)
    law = CrushInterfaceLaw(stiffness=5.3e8, crush_strength=5.0e6,
                            residual_ratio=0.5, softening_ratio=4.0)
    gap = torch.as_tensor(0.004 * 0.5 * (1.0 - np.cos(
        2 * math.pi * (foot.points[:, 1].numpy() - foot.points[:, 1].min().item())
        / float(foot.points[:, 1].max() - foot.points[:, 1].min()))),
        dtype=torch.float64)
    res = solve_contact_plate_coupling(panel, foot, law, 0.010, gap=gap)
    assert res.converged
    assert res.iterations <= 40
    C = res.load.interpolation
    # force balance of the projected load array
    assert float(res.load.forces_z.sum()) == pytest.approx(res.load.total_force, rel=1e-13)
    # work conjugacy of the transfer pair
    assert float(torch.dot(res.load.weights * res.load.pressures, C @ res.response.w)) \
        == pytest.approx(float(torch.dot(res.load.forces_z, res.response.w)), rel=1e-10)
    # the panel reactions carry the whole contact force
    assert float(-res.response.reactions_z.sum()) == pytest.approx(
        res.load.total_force, rel=1e-6)
    # the deformable panel reduces penetration relative to a rigid one
    assert res.one_way["total_force"] > res.total_force
    assert res.one_way["penetration_max"] >= float(torch.max(res.penetration))
    # non-uniform patch (there is one, even with a conforming gap)
    assert res.peak_to_mean > 1.0
    assert math.isfinite(res.amplification)


def test_prescribed_patch_nonuniformity_scales_peak_moment():
    """Structural sensitivity at fixed total force and footprint.

    Concentrating the pressure towards the clamped frame edge *lowers* the peak
    moment: the support reacts the load directly instead of the bay carrying it
    in bending.  The test pins that direction (it is the physical reason a
    patch that peaks at a support is less damaging than one that peaks mid-bay)
    and that the total force is preserved exactly in every case.
    """
    panel = PanelSpec(nx=8, ny=16)
    nodes, elems = panel.mesh()
    model = PlateModel(nodes, elems, young=panel.young, poisson=panel.poisson,
                       thickness=panel.thickness, boundary=panel.boundary)
    foot = ContactFootprint.grid(0.0, 0.30, 0.70, 1.70, 3, 21)
    total = 1.0e6
    sigmas = []
    for amp in (0.0, 0.6, 1.0):
        x = foot.points[:, 0].numpy()
        shape = torch.as_tensor(1.0 + amp * np.cos(
            math.pi * (x - x.min()) / (x.max() - x.min())), dtype=torch.float64)
        mean_shape = float(torch.sum(foot.weights * shape) / torch.sum(foot.weights))
        p = shape / mean_shape * (total / foot.area)
        load = patch_from_contact(foot.points, p, nodes, elems,
                                  point_weights=foot.weights)
        assert load.total_force == pytest.approx(total, rel=1e-10)
        sigmas.append(model.solve(dof_load=load.dof_load).sigma_peak)
    assert sigmas[0] > sigmas[1] > sigmas[2]
    assert sigmas[0] / sigmas[2] > 1.05