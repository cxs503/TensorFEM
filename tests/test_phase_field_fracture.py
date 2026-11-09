"""Tests for :mod:`tensorfem.phase_field_fracture` (AT2 phase-field fracture).

The tests cover the discrete machinery (element stiffness against the production
plane-strain Q4, consistent tangent by finite differences, history irreversibility)
and the two cheapest physical anchors (K_I of a sharp edge crack vs the Tada
solution, monotone damage growth with the energy ledger).
"""

import math

import numpy as np
import pytest
import torch

from tensorfem.continuum import elasticity_matrix, q4_stiffness
from tensorfem.phase_field_fracture import (
    PhaseFieldProblem,
    PhaseFieldSolver,
    assemble,
    crack_line_elements,
    energies,
    nodes_on_segment,
    problem_strain_energy,
    run_load_path,
    sharp_crack_energy_release,
    structured_quad_mesh,
)

E, NU = 1000.0, 0.3


def test_q4_element_stiffness_matches_plane_strain_continuum():
    """A single undamaged element reproduces ``continuum.q4_stiffness``."""
    nodes = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2, 3]])
    problem = PhaseFieldProblem(nodes, elements, E, NU, Gc=1e-3, l=0.1)
    _, K, _ = assemble(problem, np.zeros(8), np.zeros(4), np.zeros((1, 4)))
    kuu = K[:8, :8].toarray()

    coords = torch.tensor(nodes, dtype=torch.float64).unsqueeze(0)
    dmat = elasticity_matrix(torch.tensor([E], dtype=torch.float64),
                             torch.tensor([NU], dtype=torch.float64), "strain")
    reference = q4_stiffness(coords, dmat, torch.tensor([1.0], dtype=torch.float64))[0][0].numpy()
    assert np.max(np.abs(reference - kuu)) / np.max(np.abs(reference)) < 1e-5


def test_elastic_residual_matches_continuum_assembly():
    """Tension/compression split must sum back to the linear elastic stress."""
    nodes, elements = structured_quad_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = PhaseFieldProblem(nodes, elements, E, NU, Gc=1e-3, l=0.1)
    rng = np.random.default_rng(3)
    u = 1e-3 * rng.standard_normal(2 * problem.n_nodes)
    R, _, _ = assemble(problem, u, np.zeros(problem.n_nodes), np.zeros((problem.n_elem, 4)))

    coords = torch.tensor(nodes, dtype=torch.float64)[torch.tensor(elements)]
    dmat = elasticity_matrix(torch.tensor([E], dtype=torch.float64),
                             torch.tensor([NU], dtype=torch.float64), "strain").expand(
        len(elements), 3, 3)
    ke, _ = q4_stiffness(coords, dmat, torch.tensor([1.0], dtype=torch.float64))
    reference = np.zeros(2 * problem.n_nodes)
    for e, conn in enumerate(elements):
        dofs = np.array([[2 * i, 2 * i + 1] for i in conn]).ravel()
        reference[dofs] += ke[e].numpy() @ u[dofs]
    assert np.linalg.norm(R[: 2 * problem.n_nodes] - reference) / np.linalg.norm(reference) < 1e-5


def test_consistent_tangent_matches_finite_differences():
    """The monolithic tangent must reproduce central differences of the residual."""
    nodes, elements = structured_quad_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = PhaseFieldProblem(nodes, elements, E, NU, Gc=1e-3, l=0.1)
    rng = np.random.default_rng(7)
    u = 2e-3 * rng.standard_normal(2 * problem.n_nodes)
    d = 0.15 * np.abs(rng.standard_normal(problem.n_nodes))
    history = np.zeros((problem.n_elem, 4))
    R, K, _ = assemble(problem, u, d, history)
    step = 1e-7
    err = 0.0
    scale = 0.0
    for dof in range(0, problem.n_dof, 13):
        delta = np.zeros(problem.n_dof)
        delta[dof] = step
        R2, _, _ = assemble(problem, u + delta[: 2 * problem.n_nodes],
                            d + delta[2 * problem.n_nodes:], history)
        fd = (R2 - R) / step
        ana = K @ delta / step
        err += np.linalg.norm(fd - ana) ** 2
        scale += np.linalg.norm(fd) ** 2
    assert math.sqrt(err / scale) < 1e-4


def test_void_element_crack_is_traction_free():
    """A fully void crack line must not transmit stress across its faces."""
    nodes, elements = structured_quad_mesh(0.0, 1.0, -0.5, 0.5, 20, 20)
    h = 1.0 / 20.0
    void = crack_line_elements(nodes, elements, (0.0, 0.0), (0.5, 0.0), tol=0.6 * h)
    problem = PhaseFieldProblem(nodes, elements, E, NU, Gc=1e-3, l=4 * h, void_elements=void)
    strain = problem.strain(np.zeros(2 * problem.n_nodes))
    assert float(torch.abs(strain).max()) == 0.0
    # the orphan nodes on the slit are pinned so the system stays solvable
    assert len(problem.orphan_nodes) > 0


def _sent_ki(nx=30, a_over_w=0.3, sigma=1.0):
    W = H = 1.0
    h = W / nx
    a = a_over_w * W

    def build(a_crack):
        # odd row count: y = 0 lies on an element centre line -> one-element-thick slit
        nodes, elements = structured_quad_mesh(0.0, W, -H, H, nx, 2 * nx + 1)
        void = crack_line_elements(nodes, elements, (0.0, 0.0), (a_crack, 0.0), tol=0.3 * h)
        return PhaseFieldProblem(nodes, elements, E, NU, Gc=1e-3, l=4 * h,
                                 void_elements=void), nodes

    problems = [build(a - h)[0], build(a)[0], build(a + h)[0]]
    nodes = build(a)[1]
    f = np.zeros(2 * problems[1].n_nodes)
    for y, sign in ((H, 1.0), (-H, -1.0)):
        edge = np.where(np.abs(nodes[:, 1] - y) < 1e-12)[0]
        edge = edge[np.argsort(nodes[edge, 0])]
        # consistent nodal loads for a uniform traction sigma
        for k in range(len(edge) - 1):
            length = nodes[edge[k + 1], 0] - nodes[edge[k], 0]
            f[2 * edge[k] + 1] += sign * sigma * length * 0.5
            f[2 * edge[k + 1] + 1] += sign * sigma * length * 0.5
    bl = int(np.argmin(nodes[:, 0] ** 2 + (nodes[:, 1] + H) ** 2))
    br = int(np.argmin((nodes[:, 0] - W) ** 2 + (nodes[:, 1] + H) ** 2))
    fixed = {2 * bl: 0.0, 2 * bl + 1: 0.0, 2 * br: 0.0}
    out = sharp_crack_energy_release(problems, f, fixed, h)
    G = out["G"]
    K = math.sqrt(G * E / (1.0 - NU ** 2))
    alpha = a / W
    factor = (1.12 - 0.231 * alpha + 10.55 * alpha ** 2
              - 21.72 * alpha ** 3 + 30.39 * alpha ** 4)
    return G, K, sigma * math.sqrt(math.pi * a) * factor


def test_sent_stress_intensity_factor_within_three_percent():
    """C1: K_I of a sharp edge crack against the Tada finite-width solution."""
    G, K_fem, K_ana = _sent_ki()
    assert 0.0 < G
    assert abs(K_fem - K_ana) / K_ana < 0.03


def test_sent_ki_mesh_convergence():
    coarser = _sent_ki(nx=20)[1:]
    finer = _sent_ki(nx=40)[1:]
    err_coarse = abs(coarser[0] - coarser[1]) / coarser[1]
    err_fine = abs(finer[0] - finer[1]) / finer[1]
    assert err_fine < err_coarse


def _propagation(l=0.06, h=0.03, delta=1.5e-4, n_step=12, a_over_w=0.3):
    W = H = 1.0
    nx = int(round(W / h))
    ny = int(round(2 * H / h))
    nodes, elements = structured_quad_mesh(0.0, W, -H, H, nx, ny)
    a = a_over_w * W
    notch = nodes_on_segment(nodes, (0.0, 0.0), (a, 0.0), tol=1e-9)
    problem = PhaseFieldProblem(nodes, elements, E, NU, Gc=1e-3, l=l, thickness=1.0)
    bl = int(np.argmin(nodes[:, 0] ** 2 + (nodes[:, 1] + H) ** 2))
    br = int(np.argmin((nodes[:, 0] - W) ** 2 + (nodes[:, 1] + H) ** 2))
    fixed = {2 * bl: 0.0, 2 * bl + 1: 0.0, 2 * br + 1: 0.0}
    solver = PhaseFieldSolver(problem, fixed=fixed)
    solver.fix_damage_nodes(notch, 1.0)
    top = np.where(np.abs(nodes[:, 1] - H) < 1e-12)[0]
    top = np.sort(top[np.argsort(nodes[top, 0])])
    control = [2 * int(n) + 1 for n in top]
    values = [delta * (k + 1) for k in range(n_step)]
    records = run_load_path(problem, solver, control, values, scheme="staggered",
                            max_iter=6, tol=1e-6)
    return problem, solver, records, a, h


def test_history_variable_is_irreversible():
    """C6: the damage field never decreases along a monotone load path."""
    problem, solver, records, a, h = _propagation()
    assert sum(r["irreversibility_violations"] for r in records) == 0
    assert records[-1]["d_max"] >= records[0]["d_max"]


def test_energy_ledger_balances():
    """C5: external work = degraded elastic energy + fracture energy (AT2)."""
    problem, solver, records, a, h = _propagation()
    last = records[-1]
    residual = abs(last["work"] - last["total"]) / last["work"]
    assert residual < 0.02, residual


def test_damage_grows_towards_the_notch_tip():
    problem, solver, records, a, h = _propagation()
    free = np.ones(problem.n_nodes, dtype=bool)
    free[nodes_on_segment(problem.nodes, (0.0, 0.0), (a, 0.0), tol=1e-9)] = False
    assert solver.d[free].max() > 0.0