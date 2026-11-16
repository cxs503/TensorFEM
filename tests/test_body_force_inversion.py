"""Acceptance and honesty tests for :mod:`tensorfem.body_force_inversion`.

The suite is ordered from the forward model outwards, because an inversion is
only worth its forward operator:

1. ``test_stiffness_*`` / ``test_patch_*`` -- the Q4 plane-stress operator
   reproduces constant-strain states to machine precision (patch test with
   *inhomogeneous* Dirichlet data, which is where a naive patch test silently
   fails: the exact field is not zero on the clamped edge).
2. ``test_manufactured_*`` -- with an analytic body force derived from a
   trigonometric field, the nodal displacement error converges at the expected
   O(h^2), so the operator is consistent, not merely exact on polynomials.
3. ``test_inversion_*`` -- the anchor acceptance case: <= 10 % of the nodes
   observed, relative L2 field error < 5 %, <= 50 adjoint iterations.
4. ``test_negative_*`` -- the cases that must *fail* and say so: noisy data, a
   truth outside the inversion basis, and the full nodal unknown.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from tensorfem.body_force_inversion import (
    InversionOptions,
    ModalBasis,
    ObservationSet,
    PlaneStressMesh,
    PlaneStressModel,
    run_body_force_inversion,
)

TRUTH_ORDER2 = [1.0, 0.5, -0.4, 0.3, 0.6, -0.7, 0.25, 0.4]
CASE_MESH = PlaneStressMesh(Lx=1.0, Ly=0.25, nx=20, ny=10)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _boundary_dofs(mesh: PlaneStressMesh) -> torch.Tensor:
    """All DOFs on the mesh boundary (x = 0, x = Lx, y = 0, y = Ly)."""
    ids = []
    for iy in range(mesh.ny + 1):
        for ix in range(mesh.nx + 1):
            if ix in (0, mesh.nx) or iy in (0, mesh.ny):
                n = mesh.node_index(ix, iy)
                ids.extend((2 * n, 2 * n + 1))
    return torch.tensor(sorted(ids), dtype=torch.long)


def _solve_dirichlet(
    model: PlaneStressModel, u_exact: torch.Tensor, load: torch.Tensor | None = None
) -> torch.Tensor:
    """Solve with the exact field imposed on the *whole* boundary, interior free.

    Imposing only the clamped edge and applying the exact field's own nodal
    residual as the load would be circular: the residual at the free DOFs is an
    identity in the assembled ``K``, so it can never fail.  With every boundary
    DOF prescribed, the interior equilibrium is *not* imposed by the setup and
    the patch test has content.
    """
    mesh = model.mesh
    K = model.stiffness()
    fixed = _boundary_dofs(mesh)
    free = torch.ones(mesh.n_dof, dtype=torch.bool)
    free[fixed] = False
    rhs = torch.zeros(mesh.n_dof, dtype=torch.float64) if load is None else load.clone()
    rhs[free] = rhs[free] - K[free][:, fixed] @ u_exact[fixed]
    u = torch.zeros(mesh.n_dof, dtype=torch.float64)
    u[fixed] = u_exact[fixed]
    u[free] = torch.linalg.solve(K[free][:, free], rhs[free])
    return u


def _uniaxial(coords: torch.Tensor, eps: float, nu: float) -> torch.Tensor:
    u = torch.zeros(2 * coords.shape[0], dtype=torch.float64)
    u[0::2] = eps * coords[:, 0]
    u[1::2] = -nu * eps * coords[:, 1]
    return u


def _pure_shear(coords: torch.Tensor, eps: float, nu: float) -> torch.Tensor:
    u = torch.zeros(2 * coords.shape[0], dtype=torch.float64)
    u[0::2] = eps * coords[:, 1]
    return u


def _pure_bending(coords: torch.Tensor, eps: float, nu: float) -> torch.Tensor:
    """Airy pure bending; exact for Q4 only where ``u_y`` stays bilinear (nu = 0)."""
    u = torch.zeros(2 * coords.shape[0], dtype=torch.float64)
    u[0::2] = -eps * coords[:, 0] * coords[:, 1]
    u[1::2] = 0.5 * eps * (coords[:, 0] ** 2 + nu * coords[:, 1] ** 2)
    return u


# --------------------------------------------------------------------------- #
# 1. forward model: stiffness and patch tests
# --------------------------------------------------------------------------- #
def test_stiffness_is_symmetric_and_has_rigid_translation_kernel():
    model = PlaneStressModel(mesh=PlaneStressMesh(1.0, 0.25, 6, 3))
    K = model.stiffness()
    scale = float(K.abs().max())
    assert float((K - K.T).abs().max()) / scale < 1.0e-14

    mesh = model.mesh
    for comp in (0, 1):
        t = torch.zeros(mesh.n_dof, dtype=torch.float64)
        t[comp::2] = 1.0
        assert float((K @ t).abs().max()) / scale < 1.0e-12


@pytest.mark.parametrize("nu", [0.0, 0.3, 0.49])
def test_patch_tests_reproduce_constant_strain_exactly(nu: float):
    mesh = PlaneStressMesh(1.0, 0.25, 8, 4)
    model = PlaneStressModel(mesh=mesh, nu=nu)
    coords = mesh.coords()
    for field in (_uniaxial, _pure_shear):
        u_exact = field(coords, 1.0e-3, nu)
        u = _solve_dirichlet(model, u_exact)
        rel = float((u - u_exact).norm() / u_exact.norm())
        assert rel < 1.0e-11, f"{field.__name__} at nu={nu}: rel err {rel:.2e}"


def test_patch_test_linear_bending_state_exact_when_representable():
    # The Airy pure-bending field has u_y quadratic in y; only nu = 0 keeps it
    # bilinear, so this is the one case a Q4 patch must reproduce exactly.
    mesh = PlaneStressMesh(1.0, 0.25, 8, 4)
    model = PlaneStressModel(mesh=mesh, nu=0.0)
    u_exact = _pure_bending(mesh.coords(), 1.0e-3, 0.0)
    u = _solve_dirichlet(model, u_exact)
    assert float((u - u_exact).norm() / u_exact.norm()) < 1.0e-11


# --------------------------------------------------------------------------- #
# 2. manufactured body force -> O(h^2) consistency
# --------------------------------------------------------------------------- #
def _manufactured_case(nx: int, ny: int, nu: float = 0.3, E: float = 210.0e9):
    """``u_x = A sin(pi x/Lx) sin(pi y/Ly)``, ``u_y = 0``, body force ``-div sigma``."""
    Lx, Ly, amp = 1.0, 0.25, 1.0e-4
    mesh = PlaneStressMesh(Lx, Ly, nx, ny)
    model = PlaneStressModel(mesh=mesh, E=E, nu=nu)
    c = E / (1.0 - nu * nu)
    g = c * (1.0 - nu) / 2.0
    kx, ky = math.pi / Lx, math.pi / Ly

    def bx(x, y):
        return amp * (c * kx * kx + g * ky * ky) * torch.sin(kx * x) * torch.sin(ky * y)

    def by(x, y):
        return -amp * (g + c * nu) * kx * ky * torch.cos(kx * x) * torch.cos(ky * y)

    load = model.load_matrix([(bx, 0), (by, 1)]).sum(dim=1)
    coords = mesh.coords()
    u_exact = torch.zeros(mesh.n_dof, dtype=torch.float64)
    u_exact[0::2] = amp * torch.sin(kx * coords[:, 0]) * torch.sin(ky * coords[:, 1])
    u = _solve_dirichlet(model, u_exact, load)
    interior = torch.ones(mesh.n_dof, dtype=torch.bool)
    interior[_boundary_dofs(mesh)] = False
    return float((u - u_exact)[interior].norm() / u_exact[interior].norm())


def test_manufactured_body_force_converges_at_second_order():
    e_coarse = _manufactured_case(16, 4)
    e_fine = _manufactured_case(32, 8)
    rate = math.log(e_coarse / e_fine) / math.log(2.0)
    assert e_fine < e_coarse
    assert rate > 1.6, f"observed rate {rate:.2f} (coarse {e_coarse:.3e}, fine {e_fine:.3e})"


# --------------------------------------------------------------------------- #
# 3. the anchor acceptance case
# --------------------------------------------------------------------------- #
def test_sparse_observation_inversion_meets_anchor_acceptance():
    res = run_body_force_inversion(
        mesh=CASE_MESH,
        truth_coeffs=TRUTH_ORDER2,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-12),
    )
    # reported fraction of observed nodes, not the requested one
    assert res["observations"]["fraction"] <= 0.10 + 1.0e-12
    assert res["acceptance"]["fraction_ok"] is True
    # field recovery: the anchor asks for <= 5 % relative L2 error
    assert res["field_rel_l2"] < 0.05, f"field error {res['field_rel_l2']:.3e}"
    assert res["field_rel_l2"] < 1.0e-3, "noiseless in-span case should be near exact"
    # adjoint iteration budget
    assert res["iterations"] <= 50
    assert res["adjoint_iteration_budget"] == 50
    assert res["observations"]["n_observation"] == 2 * res["observations"]["n_node_sampled"]
    # the truth is inside the basis, so nothing may be flagged as out of span
    assert res["out_of_span"] is False
    # and the coefficients themselves must come back
    rec = torch.tensor(res["coefficients"]["recovered"])
    truth = torch.tensor(res["coefficients"]["truth"])
    assert float((rec - truth).norm() / truth.norm()) < 1.0e-4


def test_iteration_budget_is_hard():
    res = run_body_force_inversion(
        mesh=CASE_MESH,
        truth_coeffs=TRUTH_ORDER2,
        order=2,
        fraction=0.10,
        options=InversionOptions(max_adjoint=3, alpha_rel=1.0e-12),
    )
    assert res["iterations"] <= 3
    assert res["stop_reason"] == "budget"
    assert res["adjoint_solves"] == res["iterations"]
    assert len(res["convergence"]) == res["iterations"]


def test_convergence_curve_is_monotone_and_archived_in_result():
    res = run_body_force_inversion(
        mesh=CASE_MESH,
        truth_coeffs=TRUTH_ORDER2,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-12),
    )
    resid = [entry["relative_data_residual"] for entry in res["convergence"]]
    assert len(resid) >= 5
    assert all(b <= a + 1.0e-14 for a, b in zip(resid, resid[1:])), resid
    # the field error must also fall, terminating below the 5 % band
    field = [entry["field_rel_l2"] for entry in res["convergence"]]
    assert field[-1] < field[0]
    assert field[-1] < 0.05


def test_result_is_json_serialisable():
    res = run_body_force_inversion(
        mesh=PlaneStressMesh(1.0, 0.25, 8, 4),
        truth_coeffs=TRUTH_ORDER2,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-12),
    )
    payload = json.dumps(res, allow_nan=False)
    assert json.loads(payload)["schema"] == "tensorfem.body-force-inversion/1"


def test_damping_is_unit_invariant():
    """The same problem in Pa and in a scaled unit system must give the same field.

    This is the regression test for the dimensionful-damping defect: with a raw
    weight on ``||c||^2`` the recovered field changed with the unit system
    (E = 210 GPa drove ``c`` to zero for any alpha above ~1e-30).
    """
    kwargs = dict(truth_coeffs=TRUTH_ORDER2, order=2, fraction=0.10)
    opts = dict(alpha_rel=1.0e-12)
    res_pa = run_body_force_inversion(
        mesh=CASE_MESH, options=InversionOptions(**opts), E=210.0e9, **kwargs
    )
    res_kpa = run_body_force_inversion(
        mesh=CASE_MESH, options=InversionOptions(**opts), E=210.0e6, **kwargs
    )
    rec_pa = torch.tensor(res_pa["coefficients"]["recovered"])
    rec_kpa = torch.tensor(res_kpa["coefficients"]["recovered"])
    truth = torch.tensor(TRUTH_ORDER2)
    assert float((rec_pa - truth).norm() / truth.norm()) < 1.0e-3
    assert float((rec_pa - rec_kpa).norm() / truth.norm()) < 1.0e-9


# --------------------------------------------------------------------------- #
# 4. negative controls: the cases that must fail honestly
# --------------------------------------------------------------------------- #
def test_noise_degrades_accuracy_and_is_reported_not_hidden():
    clean = run_body_force_inversion(
        mesh=CASE_MESH,
        truth_coeffs=TRUTH_ORDER2,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-12),
    )
    noisy = run_body_force_inversion(
        mesh=CASE_MESH,
        truth_coeffs=TRUTH_ORDER2,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-12, noise=0.01),
    )
    assert noisy["field_rel_l2"] > clean["field_rel_l2"]
    # 1 % sensor noise does NOT meet the 5 % band at this conditioning; the
    # result must say so instead of returning a confident number.
    assert noisy["field_rel_l2"] > 0.05
    assert noisy["acceptance"]["field_within_5pct"] is False
    assert noisy["observations"]["noise_relative"] == pytest.approx(0.01)


def test_truth_outside_basis_is_flagged_as_model_error():
    def extra(x, y):
        return 0.6 * torch.sin(5 * math.pi * x / 1.0) * torch.sin(3 * math.pi * y / 0.25)

    res = run_body_force_inversion(
        mesh=CASE_MESH,
        truth_coeffs=TRUTH_ORDER2,
        truth_extra=extra,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-12),
    )
    assert res["out_of_span"] is True
    assert res["field_rel_l2"] > 0.05, "out-of-span content cannot be recovered"
    assert res["acceptance"]["field_within_5pct"] is False


def test_full_nodal_inversion_is_rank_deficient_and_says_so():
    mesh = PlaneStressMesh(1.0, 0.25, 8, 4)
    basis = ModalBasis(mesh=mesh, order=2)
    truth = list(basis.nodal_field(torch.tensor(TRUTH_ORDER2), 0)) + list(
        basis.nodal_field(torch.tensor(TRUTH_ORDER2), 1)
    )
    res = run_body_force_inversion(
        mesh=mesh,
        truth_coeffs=truth,
        mode="nodal",
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-8),
    )
    assert res["null_space_dim"] is not None and res["null_space_dim"] > 0
    assert res["mode"] == "nodal"
    # minimum-norm solution is not the truth: the honest verdict is a failure
    assert res["field_rel_l2"] > 0.05
    assert res["acceptance"]["field_within_5pct"] is False


def test_observation_fraction_cap_uses_floor():
    mesh = PlaneStressMesh(1.0, 0.25, 12, 6)  # 91 nodes
    obs = ObservationSet.sample(mesh, 0.10, seed=0)
    assert obs.node_ids.numel() == 9
    assert obs.fraction <= 0.10
    with pytest.raises(ValueError):
        ObservationSet.sample(mesh, 1.5)
    with pytest.raises(ValueError):
        ObservationSet.sample(mesh, 0.0)


def test_modal_inversion_requires_damping_and_matching_truth_size():
    with pytest.raises(ValueError):
        run_body_force_inversion(
            mesh=CASE_MESH,
            truth_coeffs=TRUTH_ORDER2,
            order=2,
            fraction=0.10,
            options=InversionOptions(alpha_rel=0.0),
        )
    with pytest.raises(ValueError):
        run_body_force_inversion(
            mesh=CASE_MESH,
            truth_coeffs=[1.0, 2.0],
            order=2,
            fraction=0.10,
        )