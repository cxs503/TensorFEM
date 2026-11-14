"""Regression + independent cross-check tests for ``patch_from_contact``.

These guard the two defects fixed in mission m20624 (see
``docs/VERIFY-patch-from-contact-units.md``):

* the end-to-end chain (contact -> projected load -> plate solve) re-applied the
  consistent nodal *forces* [N] as *pressures* [Pa], scaling the applied load by
  one element area (deflections off by ~144x at nx=12);
* the projection used global inverse-distance weighting, which inflated the
  patch force by +24 % ... +666 % instead of reproducing the contact force.

The checks below assert physics, not just execution: an exact uniform case, the
projected force, and the peak bending stress against the independent Kirchhoff
Navier series / the independent finite-element path in
``tensorfem.ice_panel_patch_contact``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import tensorfem.patch_from_contact as P
import tensorfem.ice_panel_patch_contact as IC
from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation

SIDE = 1.0
E = 2.06e11
NU = 0.3
T = 0.01


def _uniform_contact(nx: int, pressure: float) -> P.ContactPressureData:
    nodes, _ = P.structured_square_mesh(nx, SIDE, dtype=torch.float64)
    xn = nodes.numpy()
    return P.ContactPressureData(
        node_positions=xn.copy(),
        normal_pressures=np.full(xn.shape[0], pressure),
        contact_area=SIDE ** 2,
    )


def test_uniform_plate_end_to_end_equals_direct_solve():
    """A uniform contact field must reproduce a direct uniform-pressure solve."""
    nx, k, p0 = 12, 1.0e7, 1.0e6
    direct = solve_plate_on_winkler_foundation(nx, SIDE, E, NU, T, k, p0)
    w_direct = float(np.max(np.abs(direct.displacement[0::3].numpy())))

    result, patch = P.contact_pressure_to_plate_load(
        _uniform_contact(nx, p0), nx, SIDE, E, NU, T, k,
    )
    w_chain = float(np.max(np.abs(result.displacement[0::3].numpy())))
    assert w_chain == pytest.approx(w_direct, rel=1e-8)
    assert patch.metadata["load_balance_rel"] == pytest.approx(0.0, abs=1e-10)


def test_legacy_force_as_pressure_defect_is_gone():
    """The load vector entries are forces [N]; re-using them as [Pa] is wrong.

    Documents the fixed defect: the legacy chain reproduced a deflection equal to
    ``element_area x`` the correct one.
    """
    nx, k, p0 = 12, 1.0e7, 1.0e6
    direct = solve_plate_on_winkler_foundation(nx, SIDE, E, NU, T, k, p0)
    w_direct = float(np.max(np.abs(direct.displacement[0::3].numpy())))
    nodes, elements = P.structured_square_mesh(nx, SIDE, dtype=torch.float64)
    nodes_np, elements_np = nodes.numpy(), elements.numpy()
    patch = P.patch_from_contact(_uniform_contact(nx, p0), nodes_np, elements_np)

    legacy_fn = P._build_pressure_callable_from_nodes(nodes_np, patch.load_vector[0::3])
    legacy = solve_plate_on_winkler_foundation(nx, SIDE, E, NU, T, k, legacy_fn)
    w_legacy = float(np.max(np.abs(legacy.displacement[0::3].numpy())))

    ratio = w_legacy / w_direct
    element_area = (SIDE / nx) ** 2
    assert ratio == pytest.approx(element_area, rel=0.02)
    # the fixed chain carries the nodal pressure field [Pa], not the forces
    assert patch.nodal_pressures is not None
    assert np.allclose(patch.nodal_pressures, p0, rtol=1e-12)


@pytest.mark.parametrize("name,peak,sigma", [
    ("mild", 5.0e6, 0.15),
    ("moderate", 8.0e6, 0.10),
    ("sharp", 15.0e6, 0.05),
])
def test_projection_conserves_the_contact_force(name, peak, sigma):
    """Projected patch force must equal the analytical Gaussian force (<2 %)."""
    nx = 12
    pos, pres = P._gaussian_contact_grid(0.5, 0.5, sigma, sigma, peak, SIDE, 41)
    data = P.ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * sigma * sigma,
    )
    nodes, elements = P.structured_square_mesh(nx, SIDE, dtype=torch.float64)
    res = P.patch_from_contact(data, nodes.numpy(), elements.numpy())
    analytic = peak * 2.0 * math.pi * sigma * sigma
    assert res.metadata["projection_method"] == "exact_contact_grid_bilinear"
    assert res.total_force == pytest.approx(analytic, rel=2e-2)


def test_projection_conserves_force_for_coarse_and_fine_contact_data():
    """Contact-data refinement must keep the projected force within a few percent.

    The residual is dominated by the plate-mesh smearing of the patch tail, not
    by the contact resolution, so it is not required to be monotone.
    """
    peak, sigma = 15.0e6, 0.05
    nodes, elements = P.structured_square_mesh(12, SIDE, dtype=torch.float64)
    analytic = peak * 2.0 * math.pi * sigma * sigma
    errors = []
    for n_per_axis in (21, 41, 81):
        pos, pres = P._gaussian_contact_grid(0.5, 0.5, sigma, sigma, peak, SIDE, n_per_axis)
        data = P.ContactPressureData(pos, pres, 2.0 * math.pi * sigma * sigma)
        res = P.patch_from_contact(data, nodes.numpy(), elements.numpy())
        errors.append(abs(res.total_force / analytic - 1.0))
    assert max(errors) < 5.0e-2
    assert min(errors) < 2.0e-2


def test_scan_reports_peak_stress_and_force_balance():
    scan = P.scan_contact_configurations(
        plate_nx=8, plate_side=SIDE, young=E, poisson=NU,
        thickness=T, winkler_k=1.0e7, n_contact_per_axis=21,
    )
    cfgs = scan["configurations"]
    assert len(cfgs) >= 3
    for cfg in cfgs:
        for key in ("peak_stress_projected", "peak_stress_uniform",
                    "delta_nonuniform_over_uniform_pct", "projection_error_pct"):
            assert key in cfg
        assert cfg["peak_stress_projected"]["sigma_x_MPa"] > 0.0
        assert cfg["peak_stress_uniform"]["sigma_x_MPa"] > 0.0
        for val in cfg["delta_nonuniform_over_uniform_pct"].values():
            assert math.isfinite(val)
        # the chain is force-consistent to machine precision
        assert abs(cfg["load_balance_rel"]) < 1e-10
        # and the contact data discretisation error stays a fraction of a percent
        assert abs(cfg["projection_error_pct"]["sigma_x"]) < 3.0
    summary = scan["summary"]
    assert summary["all_configurations_valid"] is True
    assert summary["max_abs_load_balance_rel"] < 1e-10
    assert summary["n_configurations"] >= 3


# ---------------------------------------------------------------------------
# Independent cross-check: Kirchhoff Navier series + the ice_panel_patch_contact FEM
# ---------------------------------------------------------------------------
def _navier_sigma(contact_patch, x: float, y: float) -> float:
    mom = IC.navier_rect_moments(
        contact_patch, SIDE, SIDE, young=E, poisson=NU, thickness=T, x=x, y=y,
    )
    return 6.0 * abs(mom["moment_x"]) / T ** 2 / 1e6


def test_uniform_patch_peak_stress_matches_navier_series():
    """Uniform 0.4x0.4 patch, 40 kN, free (k=0) SSSS square plate, nx=16."""
    half, force, nx = 0.2, 4.0e4, 16
    p0 = force / (4.0 * half * half)
    xs = np.linspace(0.5 - half, 0.5 + half, 9)
    ys = np.linspace(0.5 - half, 0.5 + half, 9)
    XX, YY = np.meshgrid(xs, ys)
    data = P.ContactPressureData(
        node_positions=np.column_stack([XX.ravel(), YY.ravel()]),
        normal_pressures=np.full(XX.size, p0),
        contact_area=4.0 * half * half,
    )
    sol, _ = P.contact_pressure_to_plate_load(data, nx, SIDE, E, NU, T, 0.0)
    stress = P.peak_stresses(sol)

    patch = IC.uniform_patch_pressure(
        total_force=force, centre=(0.5, 0.5), half_x=half, half_y=half,
        panel=(SIDE, SIDE), grid=33,
    )
    ref = _navier_sigma(patch, *stress["sigma_x_at_m"])
    # A uniform patch has a *discontinuous* edge pressure; the projected field is
    # the plate-mesh bilinear interpolant, so the edge is smeared over one element
    # -> O(h) error (about +7 % at nx=16, see the convergence table in
    # docs/VERIFY-patch-from-contact-units.md).  Smooth fields are not affected
    # (Gaussian patch: <1 %, next test).
    assert stress["sigma_x_MPa"] == pytest.approx(ref, rel=1.0e-1)


def test_gaussian_patch_peak_stress_matches_navier_series():
    """Non-uniform Gaussian patch: the *projected* load must match the series."""
    peak, sigma_p, nx, grid = 2.0e6, 0.08, 16, 33
    pos, pres = P._gaussian_contact_grid(0.5, 0.5, sigma_p, sigma_p, peak, SIDE, grid)
    data = P.ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * sigma_p ** 2,
    )
    sol, patch = P.contact_pressure_to_plate_load(data, nx, SIDE, E, NU, T, 0.0)
    stress = P.peak_stresses(sol)

    xg = np.unique(np.round(pos[:, 0], 12))
    yg = np.unique(np.round(pos[:, 1], 12))
    contact_patch = IC.ContactPatch(
        x=torch.as_tensor(xg, dtype=torch.float64),
        y=torch.as_tensor(yg, dtype=torch.float64),
        pressure=torch.as_tensor(pres.reshape(yg.size, xg.size).T, dtype=torch.float64),
        total_force=float(patch.total_force),
        contact_area=float((xg[-1] - xg[0]) * (yg[-1] - yg[0])),
        peak_pressure=peak,
        centroid=(0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])),
        approach=0.0,
        opening=torch.zeros((xg.size, yg.size), dtype=torch.float64),
        residual=0.0,
        config={"kind": "verification"},
    )
    ref = _navier_sigma(contact_patch, *stress["sigma_x_at_m"])
    assert stress["sigma_x_MPa"] == pytest.approx(ref, rel=2e-2)


def test_two_independent_fem_paths_agree_for_the_same_patch():
    """This module vs ice_panel_patch_contact (exact conservative projection)."""
    peak, sigma_p, nx, grid = 2.0e6, 0.08, 16, 33
    pos, pres = P._gaussian_contact_grid(0.5, 0.5, sigma_p, sigma_p, peak, SIDE, grid)
    data = P.ContactPressureData(pos, pres, 2.0 * math.pi * sigma_p ** 2)
    sol, _ = P.contact_pressure_to_plate_load(data, nx, SIDE, E, NU, T, 0.0)
    mine = P.peak_stresses(sol)["sigma_x_MPa"]

    xg = np.unique(np.round(pos[:, 0], 12))
    yg = np.unique(np.round(pos[:, 1], 12))
    contact_patch = IC.ContactPatch(
        x=torch.as_tensor(xg, dtype=torch.float64),
        y=torch.as_tensor(yg, dtype=torch.float64),
        pressure=torch.as_tensor(pres.reshape(yg.size, xg.size).T, dtype=torch.float64),
        total_force=float(np.trapezoid(np.trapezoid(pres.reshape(yg.size, xg.size).T, yg, axis=1), xg)),
        contact_area=float((xg[-1] - xg[0]) * (yg[-1] - yg[0])),
        peak_pressure=peak,
        centroid=(0.5, 0.5),
        approach=0.0,
        opening=torch.zeros((xg.size, yg.size), dtype=torch.float64),
        residual=0.0,
        config={"kind": "verification"},
    )
    _, _, f, info = IC.patch_from_contact(contact_patch, nx, nx, SIDE, SIDE)
    assert info["force_error_rel"] < 5e-2
    sol_ic = IC.solve_rect_panel(
        nx, nx, SIDE, SIDE, young=E, poisson=NU, thickness=T, load=f,
    )
    theirs = sol_ic.peak_stress()["sigma_Mx_MPa"]
    # two independent projection paths + two independent assembly paths:
    # agree to a few percent (this module's edge smearing vs exact splitting)
    assert mine == pytest.approx(theirs, rel=1.0e-1)