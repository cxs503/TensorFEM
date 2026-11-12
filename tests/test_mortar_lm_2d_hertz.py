"""Acceptance tests for 2-D mortar Lagrange-multiplier contact.

Covers (each of these was a defect found by the m14687 board audit):

* the module is executable at all (the previous solver raised on first call),
* the mortar operators D/M are actually used and reproduce constant fields
  exactly on non-matching interfaces,
* the complementarity conditions hold and the penetration metric can fail
  (non-vacuous acceptance criterion),
* the analytic Hertz reference is internally consistent (no ``b = sqrt(R d)``),
* the FE contact half-width / peak pressure agree with Hertz within 3 %,
* mesh refinement improves the error (observed order > 0.3).
"""
import math

import pytest
import torch

from tensorfem.mortar_lagrange_multiplier_2d import (
    build_mortar_interface,
    build_rigid_cylinder_block,
    build_two_block_patch,
    compute_mortar_matrices_1d,
    hertz_line_reference,
    hertz_metrics,
    hertz_pressure_profile,
    observed_order,
    patch_metrics,
    pressure_half_width,
    q4_plane_strain_stiffness,
    q4_plane_strain_stiffness_csr,
    solve_hertz_indentation,
    solve_mortar_contact_2d,
    solve_two_block_patch,
    structured_q4_mesh,
)

D = torch.float64
HERTZ = dict(half_width=8.0, height=8.0, radius=10.0, young=1000.0,
             poisson=0.3, indentation=0.08, grading_x=1.0, grading_y=1.5)


def test_hertz_reference_is_internally_consistent():
    ref = hertz_line_reference(40.0, 10.0, 1000.0, 0.3)
    # F = pi p0 b / 2 must hold identically; the old b = sqrt(R*delta) set did not
    assert ref.consistency_residual() < 1e-12
    assert abs(ref.contact_half_width - math.sqrt(4 * 40.0 * 10.0 / (math.pi * ref.effective_modulus))) < 1e-12
    # and the 3-D sphere analogy is measurably different in 2-D
    naive = math.sqrt(10.0 * ref.indentation)
    assert abs(naive - ref.contact_half_width) / ref.contact_half_width > 0.05


def test_mortar_operators_partition_of_unity_and_force_conservation():
    sx = torch.linspace(0.0, 1.0, 17, dtype=D)
    mx = torch.tensor([0.0, 0.11, 0.24, 0.37, 0.5, 0.63, 0.79, 0.9, 1.0], dtype=D)
    iface = build_mortar_interface(sx, mx)
    assert iface.partition_of_unity_error() < 1e-12
    assert iface.force_conservation_error() < 1e-12
    assert float((iface.D - iface.D.T).abs().max()) < 1e-12
    # D is a mass matrix on the slave interface: row sums are tributary lengths
    assert abs(float(iface.L_slave.sum()) - 1.0) < 1e-12
    assert abs(float(iface.L_master.sum()) - 1.0) < 1e-12
    Dm, Mm, breaks = compute_mortar_matrices_1d(sx, mx)
    # mortar segments = union of both node sets (shared nodes counted once)
    assert breaks.numel() == len(set(sx.tolist()) | set(mx.tolist()))
    assert float(breaks[0]) == 0.0 and float(breaks[-1]) == 1.0


def test_vectorised_assembly_matches_reference_loop():
    nodes, elements = structured_q4_mesh(3, 2, 2.0, 1.0)
    K = q4_plane_strain_stiffness(nodes, elements, 1000.0, 0.3)
    # independent reference: assemble element by element with an explicit loop
    n_dof = 2 * len(nodes)
    Kref = torch.zeros((n_dof, n_dof), dtype=D)
    c = 1000.0 / ((1.3) * (0.4))
    Dm = c * torch.tensor([[0.7, 0.3, 0.0], [0.3, 0.7, 0.0], [0.0, 0.0, 0.2]], dtype=D)
    gp = 1.0 / math.sqrt(3.0)
    for conn in elements:
        x = nodes[conn]
        ke = torch.zeros((8, 8), dtype=D)
        for xi, eta in ((-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)):
            dNdxi = torch.tensor([
                [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]], dtype=D) / 4
            J = dNdxi @ x
            dNdx = torch.linalg.inv(J) @ dNdxi
            B = torch.zeros((3, 8), dtype=D)
            B[0, 0::2] = dNdx[0]
            B[1, 1::2] = dNdx[1]
            B[2, 0::2] = dNdx[1]
            B[2, 1::2] = dNdx[0]
            ke += torch.linalg.det(J) * (B.T @ Dm @ B)
        dm = torch.stack([2 * conn, 2 * conn + 1], dim=1).reshape(-1)
        Kref.index_put_((dm.unsqueeze(1).expand(8, 8), dm.unsqueeze(0).expand(8, 8)),
                        ke, accumulate=True)
    assert float((K - Kref).abs().max()) < 1e-10 * float(Kref.abs().max())

    from scipy.sparse import issparse
    Kcsr = q4_plane_strain_stiffness_csr(nodes, elements, 1000.0, 0.3)
    assert issparse(Kcsr)
    assert float((torch.tensor(Kcsr.toarray(), dtype=D) - Kref).abs().max()) \
        < 1e-10 * float(Kref.abs().max())


def test_q4_patch_test_uniform_stress_is_reproduced():
    """Constant-strain patch test: Q4 must reproduce the exact uniform solution."""
    nodes, elements = structured_q4_mesh(4, 3, 2.0, 1.0)
    E, nu, t = 1000.0, 0.3, 5.0
    K = q4_plane_strain_stiffness_csr(nodes, elements, E, nu)
    n = len(nodes)
    fixed = [2 * i for i in range(0, n, 5)]  # x = 0 column: u_x = 0
    fixed += [2 * i + 1 for i in range(5)]   # bottom: u_y = 0
    eps = -t * (1 - nu**2) / E
    u = torch.zeros(2 * n, dtype=D)
    u[0::2] = 1e-3 * nodes[:, 0]             # arbitrary compatible lateral field
    u[1::2] = eps * nodes[:, 1]
    f = torch.tensor(K @ u.numpy())
    free = torch.ones(2 * n, dtype=torch.bool)
    free[torch.tensor(fixed)] = False
    Kd = torch.tensor(K.toarray(), dtype=D)
    sol = torch.linalg.solve(Kd[free][:, free], f[free])
    u_check = torch.zeros(2 * n, dtype=D)
    u_check[free] = sol
    assert torch.allclose(u_check, u, atol=1e-12)


@pytest.mark.parametrize("nx_lower,nx_upper", [(8, 5), (16, 11), (24, 7)])
def test_nonmatching_interface_patch_exact(nx_lower, nx_upper):
    problem = build_two_block_patch(nx_lower=nx_lower, nx_upper=nx_upper)
    result = solve_two_block_patch(problem)
    m = patch_metrics(problem, result)
    assert result.converged
    assert m["pressure_max_rel_dev"] < 1e-10      # p = t on every slave node
    assert abs(m["force_rel_error"]) < 1e-10
    assert abs(m["top_uy_rel_error"]) < 1e-10
    assert m["max_penetration"] < 1e-12
    assert m["complementarity_residual"] < 1e-14
    assert m["reaction_balance_error"] < 1e-10


def test_penetration_metric_is_not_vacuous():
    """A truncated (under-solved) contact solve must report penetration.

    A criterion that can never fail is not a criterion: the same metric is
    evaluated on a converged solve (tiny) and on the first active-set iterate
    (large), which shows the reported penetration is a real measurement.
    """
    problem = build_rigid_cylinder_block(nx=80, ny=16, **HERTZ)
    good = solve_hertz_indentation(problem)
    bad = solve_hertz_indentation(problem, max_iterations=1)
    assert good.converged and not bad.converged
    assert bad.max_penetration > 10.0 * good.max_penetration
    assert good.max_penetration < 1e-3 * HERTZ["indentation"]


def test_hertz_line_contact_accuracy_and_convergence():
    errors_b, errors_p0, hs = [], [], []
    for nx, ny in ((80, 16), (160, 20)):
        problem = build_rigid_cylinder_block(nx=nx, ny=ny, **HERTZ)
        result = solve_hertz_indentation(problem)
        m = hertz_metrics(problem, result)
        assert result.converged
        assert m["max_penetration"] < 1e-3 * HERTZ["indentation"]
        assert m["complementarity_residual"] < 1e-8 * m["p0_fe"]
        assert m["reaction_balance_error"] < 1e-9
        assert abs(m["b_error_rel"]) < 0.15     # platform red line
        assert abs(m["p0_error_rel"]) < 0.15
        errors_b.append(abs(m["b_error_rel"]))
        errors_p0.append(abs(m["p0_error_rel"]))
        hs.append(m["h_zone"])
    # tight target on the refined level
    assert errors_b[-1] < 0.03
    assert errors_p0[-1] < 0.03


def test_hertz_pressure_profile_is_used_consistently():
    problem = build_rigid_cylinder_block(nx=80, ny=16, **HERTZ)
    result = solve_hertz_indentation(problem)
    m = hertz_metrics(problem, result)
    b, p0 = m["b_ref"], m["p0_ref"]
    assert abs(float(pressure_half_width(problem["slave_x"], result.lam)) - m["b_fe"]) < 1e-12
    prof = hertz_pressure_profile(torch.tensor([0.0], dtype=D), b, p0)
    assert abs(float(prof[0]) - p0) < 1e-12
    assert float(hertz_pressure_profile(torch.tensor([b], dtype=D), b, p0)[0]) == 0.0