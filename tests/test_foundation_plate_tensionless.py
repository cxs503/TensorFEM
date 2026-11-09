"""Tests for the tensionless (one-sided) Winkler foundation on Mindlin plates.

The module under test is :mod:`tensorfem.foundation_plate_tensionless`; the
qualification evidence lives in
``benchmarks/verify_foundation_plate_tensionless.py`` and
``docs/foundation-plate-tensionless.md``.  The tests below are the fast subset:
they use small meshes but assert the same *kinds* of statements (element parity,
exactness, conservation, convexity/stationarity, independent solver agreement).
"""
import math

import numpy as np
import pytest
import torch

import tensorfem.foundation_plate_tensionless as ft
from tensorfem.plate import q4_mindlin_stiffness, structured_square_mesh
from tensorfem.plate_foundation_analytical import navier_ss_plate_uniform
from tensorfem.winkler_foundation import q4_winkler_foundation_stiffness

D = torch.float64
YOUNG, POISSON, THICK = 1.0e7, 0.30, 0.02
WINKLER = 1.0e6
ELL = ft.characteristic_length(YOUNG, POISSON, THICK, WINKLER)
RIGIDITY = ft.plate_rigidity(YOUNG, POISSON, THICK)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def square_mesh(n: int, side: float = 1.0):
    nodes, els = structured_square_mesh(n, side)
    return nodes.numpy(), els.numpy()


def graded_square(n: int, half_width: float, beta: float = 3.0):
    e = np.linspace(-1.0, 1.0, n + 1)
    x = half_width * np.sinh(beta * e) / math.sinh(beta)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    nodes = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    els = np.array([(j * (n + 1) + i, j * (n + 1) + i + 1,
                     j * (n + 1) + i + n + 2, j * (n + 1) + i + n + 1)
                    for j in range(n) for i in range(n)])
    return nodes, els


def element_areas(nodes, elements):
    xy = nodes[elements]
    return 0.5 * np.abs(
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
        + (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 3, 1] - xy[:, 0, 1])
        - (xy[:, 3, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
    )


def uniform_load(nodes, elements, pressure: float) -> np.ndarray:
    f = np.zeros(3 * nodes.shape[0])
    for e, a in zip(elements, element_areas(nodes, elements)):
        for nd in e:
            f[3 * nd] += pressure * a / 4.0
    return f


def hard_simply_supported(nodes, side):
    fixed = []
    tol = 1e-12 * max(side, 1.0)
    for nd, (x, y) in enumerate(nodes):
        if min(abs(x), abs(x - side)) < tol:
            fixed += [3 * nd, 3 * nd + 2]
        if min(abs(y), abs(y - side)) < tol:
            fixed += [3 * nd, 3 * nd + 1]
    return np.unique(np.asarray(fixed, dtype=int))


def centre_node(nodes, shift=(0.0, 0.0)):
    return int(np.argmin(((nodes - np.asarray(shift)) ** 2).sum(axis=1)))


def distorted_quads():
    rng = np.random.default_rng(3)
    base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return np.array([base + rng.uniform(-0.17, 0.17, base.shape) for _ in range(3)])


# ---------------------------------------------------------------------------
# element algebra
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("treatment", ["mitc", "sri"])
def test_batch_kernel_matches_library_element(treatment):
    xys = distorted_quads()
    batch = ft.q4_mindlin_batch(xys, 2.1e7, 0.3, 0.021, treatment)
    for e in range(xys.shape[0]):
        lib = q4_mindlin_stiffness(torch.tensor(xys[e], dtype=D), 2.1e7, 0.3, 0.021,
                                   shear_treatment=treatment).numpy()
        assert np.abs(batch[e] - lib).max() / np.abs(lib).max() < 1e-14


def test_consistent_foundation_matches_library_operator():
    xy = distorted_quads()[0]
    lib = q4_winkler_foundation_stiffness(torch.tensor(xy, dtype=D), 3.3e5).numpy()
    mine = ft.q4_foundation_consistent(xy, 3.3e5).numpy() if hasattr(
        ft.q4_foundation_consistent(xy, 3.3e5), "numpy") else \
        ft.q4_foundation_consistent(xy, 3.3e5).detach().numpy()
    batch, _, _ = ft.q4_foundation_batch(xy[None, :, :], 3.3e5, "consistent")
    assert np.abs(mine - lib).max() <= 1e-15 * np.abs(lib).max()
    assert np.abs(batch[0] - lib).max() <= 1e-14 * np.abs(lib).max()


@pytest.mark.parametrize("model", ["consistent", "lumped"])
def test_foundation_operator_conserves_the_resultant(model):
    """``1^T K_f 1`` on the transverse DOFs equals ``k A`` for both operators."""
    xy = distorted_quads()[1]
    k = 3.3e5
    ones = np.zeros(12)
    ones[[0, 3, 6, 9]] = 1.0
    kw, _, _ = ft.q4_foundation_batch(xy[None, :, :], k, model)
    area = float(element_areas(xy, np.array([[0, 1, 2, 3]]))[0])
    assert abs(float(ones @ kw[0] @ ones) - k * area) <= 1e-12 * k * area


def test_engaged_matrix_equals_full_foundation_operator():
    nodes, els = square_mesh(4, 1.0)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    full = ft.engaged_matrix(system, np.ones(system.n_gauss, dtype=bool))
    diff = abs((full - system.k_foundation_full)).max()
    assert diff <= 1e-10 * abs(system.k_foundation_full).max()


def test_gauss_load_vector_is_the_foundation_matvec():
    nodes, els = square_mesh(4, 1.0)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    u = np.random.default_rng(0).random(system.ndof)
    lhs = ft._gp_load_vector(system, system.foundation_k * system.w_at_gauss(u))
    rhs = system.k_foundation_full @ u
    assert np.abs(lhs - rhs).max() <= 1e-12 * np.abs(rhs).max()


def test_constant_deflection_is_a_rigid_mode_of_the_assembled_plate():
    nodes, els = square_mesh(3, 2.0 * ELL)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    u = np.zeros(system.ndof)
    u[system.w_dofs] = 1.0                # rigid translation: w = 1, theta = 0
    scale = float(np.abs(system.k_plate).max())
    assert np.abs(system.k_plate @ u).max() <= 1e-10 * scale
    # ... and the foundation operator sees it as a uniform pressure of k: the
    # resultant of the nodal reactions equals k * (plate area)
    reaction = (system.k_foundation_full @ u)[system.w_dofs]
    total = float(reaction.sum())
    target = WINKLER * float(element_areas(nodes, els).sum())
    assert abs(total - target) <= 1e-10 * target


# ---------------------------------------------------------------------------
# exactness: patch test and manufactured uniform-load field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, 8])
def test_one_sided_patch_test_is_exact(n):
    nodes, els = square_mesh(n, 1.0)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, 0.05, 5.0e5)
    slope = 0.008
    w_exact = slope * (0.5 - system.nodes[:, 0])          # zero on the x = 0.5 mesh line
    u_exact = np.zeros(system.ndof)
    u_exact[system.w_dofs] = w_exact
    u_exact[3 * np.arange(nodes.shape[0]) + 1] = slope     # gamma_xz = dw/dx + theta_x = 0
    p_exact = system.pressures(u_exact)
    loads = system.k_plate @ u_exact + ft._gp_load_vector(system, p_exact)
    out = ft.solve_plate_tensionless(system, loads, tol=1e-13)
    assert out["converged"]
    # half of the Gauss points are disengaged, yet the field is reproduced exactly
    assert 0 < int(out["engaged"].sum()) < system.n_gauss
    assert np.abs(out["u"] - u_exact).max() / np.abs(u_exact).max() <= 1e-12


def test_uniform_load_on_free_plate_has_exact_solution():
    """Free-edged plate + uniform load: exact solution ``w = q/k``, ``p = q``."""
    nodes, els = graded_square(12, 10.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    q0 = 1.0e3
    out = ft.solve_plate_tensionless(system, uniform_load(nodes, els, q0), tol=1e-13)
    assert out["engaged"].all()
    assert np.abs(out["w"] - q0 / WINKLER).max() / (q0 / WINKLER) <= 1e-12
    assert np.abs(out["pressure_gauss"] - q0).max() / q0 <= 1e-12


def test_simply_supported_plate_matches_navier_series():
    side = 6.0 * ELL
    q0 = 1.0e3
    nodes, els = square_mesh(24, side)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    out = ft.solve_plate_tensionless(system, uniform_load(nodes, els, q0),
                                     hard_simply_supported(nodes, side), tol=1e-12)
    exact = navier_ss_plate_uniform(q0, side, side, RIGIDITY, WINKLER, M=200, N=200)
    w_c = float(out["w"][centre_node(nodes, (side / 2, side / 2))])
    assert out["engaged"].all()
    assert abs(w_c - exact) / exact <= 5e-3


# ---------------------------------------------------------------------------
# convexity, stationarity, conservation
# ---------------------------------------------------------------------------
def test_gradient_is_the_gradient_of_the_potential():
    nodes, els = graded_square(8, 4.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    loads = np.zeros(system.ndof)
    loads[3 * centre_node(nodes)] = 1.0
    rng = np.random.default_rng(5)
    u = 1e-5 * rng.standard_normal(system.ndof)
    d = rng.standard_normal(system.ndof)
    eps = 1e-9
    fd = (ft.tensionless_energy(system, u + eps * d, loads)
          - ft.tensionless_energy(system, u - eps * d, loads)) / (2 * eps)
    grad = ft.tensionless_gradient(system, u, loads)
    assert abs(fd - float(grad @ d)) <= 1e-6 * max(abs(fd), 1e-30)


def test_solver_reaches_a_stationary_point_and_no_tension():
    nodes, els = graded_square(16, 8.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    loads = np.zeros(system.ndof)
    loads[3 * centre_node(nodes)] = 1.0
    out = ft.solve_plate_tensionless(system, loads, tol=1e-11, max_iter=60)
    assert out["converged"]
    assert out["residual"] <= 1e-10
    assert out["min_pressure"] >= -1e-12               # compression only
    p = out["pressure_gauss"]
    expected = system.foundation_k * np.maximum(out["w_gauss"], 0.0)
    assert np.abs(p - expected).max() <= 1e-12 * max(p.max(), 1e-30)


def test_total_reaction_equals_the_applied_load():
    nodes, els = graded_square(16, 8.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    loads = np.zeros(system.ndof)
    loads[3 * centre_node(nodes)] = 1.0
    out = ft.solve_plate_tensionless(system, loads, tol=1e-11, max_iter=60)
    assert abs(out["total_reaction"] - 1.0) <= 1e-10


def test_energy_ledger_closes():
    nodes, els = graded_square(16, 8.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    loads = np.zeros(system.ndof)
    loads[3 * centre_node(nodes)] = 1.0
    out = ft.solve_plate_tensionless(system, loads, tol=1e-11, max_iter=60)
    ledger = out["strain_energy"] + out["foundation_energy"] - 0.5 * out["external_work"]
    assert abs(ledger) <= 1e-10 * abs(out["external_work"])


def test_perturbations_increase_the_potential():
    """``Pi`` is convex, so every perturbation of the minimiser raises it."""
    nodes, els = graded_square(12, 6.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    loads = np.zeros(system.ndof)
    loads[3 * centre_node(nodes)] = 1.0
    out = ft.solve_plate_tensionless(system, loads, tol=1e-11, max_iter=60)
    rng = np.random.default_rng(11)
    eps = 1e-3 * float(np.abs(out["u"]).max())
    for _ in range(8):
        d = rng.standard_normal(system.ndof)
        d /= np.abs(d).max()
        assert ft.tensionless_energy(system, out["u"] + eps * d, loads) > out["energy"]


# ---------------------------------------------------------------------------
# one-sided behaviour
# ---------------------------------------------------------------------------
def test_lift_off_appears_under_a_central_point_load():
    nodes, els = graded_square(16, 8.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    loads = np.zeros(system.ndof)
    loads[3 * centre_node(nodes)] = 1.0
    out = ft.solve_plate_tensionless(system, loads, tol=1e-11, max_iter=60)
    fraction = float(system.gp_detJ[out["engaged"]].sum() / system.gp_detJ.sum())
    assert 0.0 < fraction < 1.0
    # the lifted-off region must be *outside* the loaded centre
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    assert out["w"][centre_node(nodes)] > 0.0
    assert out["w"][int(np.argmax(r))] < 0.0
    radius = ft.contact_radius(system, out["engaged"])
    assert 2.0 * ELL < radius < 3.5 * ELL


def test_contact_radius_is_mesh_converged():
    radii = []
    for n in (16, 24):
        nodes, els = graded_square(n, 8.0 * ELL, 3.0)
        system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
        loads = np.zeros(system.ndof)
        loads[3 * centre_node(nodes)] = 1.0
        out = ft.solve_plate_tensionless(system, loads, tol=1e-11, max_iter=60)
        radii.append(ft.contact_radius(system, out["engaged"]))
    assert abs(radii[0] - radii[1]) / radii[1] <= 0.02


def test_independent_lbfgs_solver_finds_the_same_solution():
    nodes, els = graded_square(12, 6.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    loads = np.zeros(system.ndof)
    loads[3 * centre_node(nodes)] = 1.0
    a = ft.solve_plate_tensionless(system, loads, tol=1e-11, max_iter=60)
    b = ft.solve_plate_tensionless_lbfgs(system, loads, max_iter=4000)
    assert abs(a["energy"] - b["energy"]) <= 1e-8 * abs(a["energy"])
    assert abs(ft.contact_radius(system, a["engaged"])
               - ft.contact_radius(system, system.w_at_gauss(b["u"]) > 0.0)) <= 1e-9


def test_initial_gap_disengages_the_foundation():
    """With ``g0`` above the deflection the plate behaves as if unsupported."""
    nodes, els = graded_square(12, 6.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    loads = np.zeros(system.ndof)
    loads[3 * centre_node(nodes)] = 1.0
    out = ft.solve_plate_tensionless(system, loads, gap=1.0, tol=1e-12, max_iter=30)
    assert not bool(out["engaged"].any())
    assert np.abs(out["pressure_gauss"]).max() == 0.0
    # an unsupported free plate responds as a rigid body to a point load
    assert np.abs(out["w"]).max() > 0.0


# ---------------------------------------------------------------------------
# penalty (rigid bed) alternative and limits
# ---------------------------------------------------------------------------
def test_penalty_penetration_follows_the_exact_law():
    nodes, els = graded_square(10, 6.0 * ELL, 2.5)
    q0 = 2.0e3
    for penalty in (1.0e6, 1.0e8):
        system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
        out = ft.solve_plate_unilateral_penalty(system, uniform_load(nodes, els, q0),
                                                penalty=penalty, max_iter=20)
        assert abs(out["penetration_max"] - q0 / penalty) <= 1e-9 * (q0 / penalty)
        # complementarity product R * penetration = q^2 / penalty exactly
        assert abs(out["complementarity_max"] - q0 * q0 / penalty) <= 1e-9 * (q0 * q0 / penalty)


def test_penalty_complementarity_product_decays_as_inverse_penalty():
    nodes, els = graded_square(10, 6.0 * ELL, 2.5)
    q0 = 2.0e3
    values = []
    for penalty in (1.0e5, 1.0e8):
        system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
        out = ft.solve_plate_unilateral_penalty(system, uniform_load(nodes, els, q0),
                                                penalty=penalty, max_iter=20)
        values.append(out["complementarity_max"])
    assert abs(values[0] / values[1] - 1.0e3) / 1.0e3 <= 0.05


def test_rigid_support_limit_is_recovered():
    """``k -> infinity`` on a free plate reproduces ``w k / q = 1`` and ``p = q``."""
    nodes, els = graded_square(10, 6.0 * ELL, 2.5)
    q0 = 2.0e3
    for wk in (1.0e6, 1.0e8):
        system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, wk)
        out = ft.solve_plate_tensionless(system, uniform_load(nodes, els, q0), tol=1e-12)
        assert abs(float(np.abs(out["w"]).max() * wk / q0) - 1.0) <= 1e-9
        assert abs(out["total_reaction"] / (q0 * system.gp_detJ.sum()) - 1.0) <= 1e-9


def test_no_foundation_limit_matches_the_plain_plate():
    side = 6.0 * ELL
    q0 = 1.0e3
    nodes, els = square_mesh(24, side)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, 1.0e-9)
    out = ft.solve_plate_tensionless(system, uniform_load(nodes, els, q0),
                                     hard_simply_supported(nodes, side), tol=1e-12)
    exact = navier_ss_plate_uniform(q0, side, side, RIGIDITY, 0.0, M=200, N=200)
    w_c = float(out["w"][centre_node(nodes, (side / 2, side / 2))])
    # the residual is the Mindlin first-order shear deflection of a Mindlin FE
    assert abs(w_c - exact) / exact <= 0.03


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------
def test_fixed_dofs_are_respected():
    nodes, els = graded_square(10, 6.0 * ELL, 2.5)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    loads = np.zeros(system.ndof)
    loads[3 * centre_node(nodes)] = 1.0
    fixed = np.array([0, 1, 2, system.ndof - 1])
    out = ft.solve_plate_tensionless(system, loads, fixed, tol=1e-12, max_iter=40)
    assert np.abs(out["u"][fixed]).max() == 0.0


def test_invalid_arguments_raise():
    nodes, els = square_mesh(2, 1.0)
    with pytest.raises(ValueError):
        ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER,
                                    shear_treatment="nope")
    with pytest.raises(ValueError):
        ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER,
                                    model="nope")
    with pytest.raises(ValueError):
        ft.q4_mindlin_batch(np.zeros((2, 4, 3)), YOUNG, POISSON, THICK)
    system = ft.plate_tensionless_system(nodes, els, YOUNG, POISSON, THICK, WINKLER)
    with pytest.raises(ValueError):
        ft.solve_plate_tensionless(system, np.zeros(system.ndof + 1))
    with pytest.raises(ValueError):
        ft.engaged_matrix(system, np.ones(3, dtype=bool))
    with pytest.raises(ValueError):
        ft.solve_plate_unilateral_penalty(system, np.zeros(system.ndof), penalty=0.0)


def test_characteristic_length_and_rigidity_are_consistent():
    assert abs(RIGIDITY - YOUNG * THICK**3 / (12 * (1 - POISSON**2))) <= 1e-18
    assert abs(ELL - (RIGIDITY / WINKLER) ** 0.25) <= 1e-15
    assert abs(ft.hetenyi_plate_centre_deflection(1.0, RIGIDITY, WINKLER)
               - 1.0 / (8.0 * math.sqrt(RIGIDITY * WINKLER))) <= 1e-18