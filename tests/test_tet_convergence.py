"""Evidence tests for the TET4 / TET10 h-convergence study module."""

import pytest
import torch

from tensorfem.convergence_verification import (
    ConvergenceLevel,
    evaluate_convergence_study,
    observed_order_from_errors,
)
from tensorfem.elasticity_convergence import ManufacturedElasticity, PureBendingOracle
from tensorfem.quadratic_solid import tet4_to_tet10
from tensorfem.solid3d import hex_to_tet_mesh, structured_hex_mesh
from tensorfem.tet_convergence import (
    duffy_tet_rule,
    tet4_manufactured_study,
    tet10_manufactured_study,
    tet_consistent_body_force,
    tet_dirichlet_study,
)

D = torch.float64


def manufactured_case(planar=True):
    return ManufacturedElasticity(
        length=2.0,
        width=0.4,
        height=0.6,
        young_modulus=210e9,
        poisson_ratio=0.3,
        amplitude=1e-4,
        planar=planar,
    )


def bending_case():
    return PureBendingOracle(
        length=2.0,
        width=0.4,
        height=0.6,
        young_modulus=210e9,
        poisson_ratio=0.3,
        curvature=1e-3,
    )


def small_tet_mesh(n=2, layers=3):
    nodes, hexes = structured_hex_mesh(2.0, 0.4, 0.6, 2 * n, n, layers, dtype=D)
    _, tets = hex_to_tet_mesh(nodes, hexes)
    return nodes, tets


# --- independent quadrature -------------------------------------------------


@pytest.mark.parametrize("order", [4, 6])
def test_duffy_rule_integrates_reference_moments(order):
    points, weights = duffy_tet_rule(order)
    assert float(weights.sum()) == pytest.approx(1.0 / 6.0, rel=1e-13)
    r, s, t = points.unbind(1)
    # int r dV = 1/24, int r^2 dV = 1/60, int r s dV = 1/120 on the unit tet.
    assert float((weights * r).sum()) == pytest.approx(1.0 / 24.0, rel=1e-12)
    assert float((weights * r * r).sum()) == pytest.approx(1.0 / 60.0, rel=1e-12)
    assert float((weights * r * s).sum()) == pytest.approx(1.0 / 120.0, rel=1e-12)
    # Partition of unity of the collapsed weights: every point is inside the tet.
    inside = (points >= 0.0).all(dim=1) & (points.sum(dim=1) <= 1.0)
    assert bool(inside.all())


def test_duffy_rule_rejects_degenerate_order():
    with pytest.raises(ValueError):
        duffy_tet_rule(1)


# --- conforming quadratic mesh ---------------------------------------------


def test_tet10_upgrade_shares_midside_nodes():
    nodes, tets = small_tet_mesh(n=2, layers=3)
    nodes10, cells = tet4_to_tet10(nodes, tets)
    assert cells.shape[1] == 10
    # The first four entries of every cell are the original vertices.
    assert torch.equal(cells[:, :4], tets)
    # Every midside node is shared: the number of generated nodes equals the
    # number of distinct grid edges, not six times the element count.
    edges = set()
    for cell in tets.tolist():
        for a, b in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)):
            edges.add(tuple(sorted((cell[a], cell[b]))))
    assert len(nodes10) == len(nodes) + len(edges)
    assert len(torch.unique(cells[:, 4:].reshape(-1))) == len(edges)
    # No duplicated coordinates between vertex and midside sets.
    distances = torch.cdist(nodes10, nodes10) + torch.eye(len(nodes10), dtype=D) * 1e6
    assert float(distances.min()) > 1e-6


# --- consistent body force --------------------------------------------------


def test_consistent_body_force_is_self_equilibrated():
    case = manufactured_case()
    nodes, tets = small_tet_mesh(n=2, layers=3)
    forces = tet_consistent_body_force(case, nodes, tets, quadratic=False, order=6)
    resultant = forces.reshape(-1, 3).sum(dim=0)
    scale = float(forces.abs().sum())
    assert scale > 0.0
    # The manufactured body force is -div(sigma) of a trigonometric field, so its
    # integral over the loaded mesh vanishes; the nodal loads must respect that.
    assert float(torch.linalg.norm(resultant)) < 1e-3 * scale


@pytest.mark.parametrize("quadratic", [False, True])
def test_consistent_body_force_reproduces_the_exact_first_moment(quadratic):
    """The load vector must integrate ``int x_i b_j dV`` exactly in the limit.

    Both families reproduce the coordinates linearly (partition of unity plus
    linear reproduction), so ``sum_a x_a (x) f_a`` equals the moment of the body
    force, which is what a consistent load vector is required to carry.  The
    moment is computed here with an independent tensor-product Gauss rule.
    """
    case = manufactured_case()
    nodes, tets = small_tet_mesh(n=2, layers=3)
    if quadratic:
        nodes_e, cells = tet4_to_tet10(nodes, tets)
        forces = tet_consistent_body_force(
            case, nodes_e, tets, connectivity=cells, quadratic=True, order=6
        )
    else:
        nodes_e, cells = nodes, tets
        forces = tet_consistent_body_force(case, nodes_e, tets, quadratic=False, order=6)
    assert forces.shape == (3 * len(nodes_e),)
    moment = (nodes_e[:, :, None] * forces.reshape(-1, 3)[:, None, :]).sum(dim=0)
    exact = _box_body_force_moment(case)
    assert torch.allclose(moment, exact, rtol=5e-3, atol=1e-9 * float(exact.abs().max()))


def _box_body_force_moment(case, gauss=8):
    """Exact integral of the body force over the box by a tensor-product rule."""
    import numpy as np

    x, w = np.polynomial.legendre.leggauss(gauss)
    x = torch.tensor(0.5 * (x + 1.0), dtype=D)
    w = torch.tensor(0.5 * w, dtype=D)
    points, weights = [], []
    for px, wx in zip(x, w):
        for py, wy in zip(x, w):
            for pz, wz in zip(x, w):
                points.append((px, py, pz))
                weights.append(wx * wy * wz)
    coords = torch.tensor(points, dtype=D) * torch.tensor(
        [case.length, case.width, case.height], dtype=D
    )
    weight = torch.tensor(weights, dtype=D) * case.length * case.width * case.height
    body = case.body_force(coords)
    return (coords[:, :, None] * body[:, None, :] * weight[:, None, None]).sum(dim=0)


# --- fail-closed guard ------------------------------------------------------


def test_study_rejects_mesh_without_interior_nodes():
    """A single layer makes every *vertex* a boundary node for TET4.

    The solution then equals the nodal interpolant and the L2 error is
    identically zero for any element type, so the study must be rejected
    instead of being reported as a perfect result.  (The TET10 upgrade is not
    affected: midside nodes on the hex main diagonals still lie in the volume.)
    """
    with pytest.raises(ValueError, match="no free DOF"):
        tet4_manufactured_study(manufactured_case(), [2], layers=1)


# --- manufactured-solution convergence --------------------------------------


@pytest.mark.parametrize("family", ["tet4", "tet10"])
def test_manufactured_error_decays_and_hits_analytic_energy(family):
    case = manufactured_case()
    runner = tet4_manufactured_study if family == "tet4" else tet10_manufactured_study
    records = runner(case, [2, 4, 8], layers=3)
    errors = [r["relative_l2_displacement_error"] for r in records]
    energies = [r["relative_strain_energy_error"] for r in records]
    assert all(e > 0.0 for e in errors)
    assert errors[1] < errors[0] / 2.0 and errors[2] < errors[1] / 2.0, errors
    assert energies[2] < energies[0] / 5.0, energies
    ratios = [a["mesh_size"] / b["mesh_size"] for a, b in zip(records, records[1:])]
    l2 = observed_order_from_errors(errors, ratios)
    energy = observed_order_from_errors(energies, ratios)
    # Theory is 2 for TET4 and 3 for TET10 in L2; the recorded evidence shows a
    # pre-asymptotic shortfall (TET4 1.16 -> 1.85, TET10 2.36 -> 2.63), so the
    # band only asserts that the measured rate approaches the declared order and
    # that the energy rate is at least first order for every family.
    lower, upper = (0.9, 2.3) if family == "tet4" else (2.0, 3.3)
    assert lower <= l2[-1] <= upper, (family, l2)
    assert energy[-1] >= 1.8, (family, energy)


def test_tet10_energy_error_is_much_smaller_than_tet4():
    case = manufactured_case()
    tet4 = tet4_manufactured_study(case, [2, 4], layers=3)
    tet10 = tet10_manufactured_study(case, [2, 4], layers=3)
    for coarse, fine in zip(tet4, tet10):
        # The quadratic space resolves the energy far better even though its
        # nodal displacement error is not always smaller on the coarsest mesh.
        assert fine["relative_strain_energy_error"] < coarse["relative_strain_energy_error"] / 5.0
        assert fine["n_free_dofs"] > coarse["n_free_dofs"]


def test_energy_study_reports_reference_comparison():
    case = manufactured_case()
    records = tet4_manufactured_study(case, [2, 4, 8], layers=3)
    levels = [ConvergenceLevel(mesh_size=r["mesh_size"], value=r["strain_energy"]) for r in records]
    report = evaluate_convergence_study(
        levels,
        qoi="strain_energy",
        reference_value=float(records[0]["exact_strain_energy"]),
        reference_tolerance=0.03,
    )
    assert report["status"] in ("passed", "failed")
    names = {check["name"] for check in report["checks"]}
    assert any("reference" in name for name in names)


# --- Dirichlet consistency --------------------------------------------------


def test_tet10_reproduces_quadratic_field_to_roundoff():
    records = tet_dirichlet_study(bending_case(), [2, 4], family="tet10", layers=3)
    for record in records:
        assert record["n_free_dofs"] > 0
        assert record["relative_l2_displacement_error"] < 1e-10
        assert record["relative_strain_energy_error"] < 1e-9


def test_tet4_dirichlet_displacement_is_nodally_exact_but_energy_is_not():
    """TET4 superconverges on the structured split while its energy does not.

    The measured L2 displacement error sits at roundoff (7.5e-17 at the
    coarsest mesh), so no order can be claimed from it; the constant-strain
    energy is still 259% low, decaying at about second order.  Both facts are
    asserted so the roundoff result cannot be quoted as energy accuracy.
    """
    records = tet_dirichlet_study(bending_case(), [2, 4], family="tet4", layers=3)
    assert all(r["relative_l2_displacement_error"] < 1e-12 for r in records)
    energies = [r["relative_strain_energy_error"] for r in records]
    assert energies[0] > 0.1
    assert energies[1] < energies[0] / 2.0
    tet10 = tet_dirichlet_study(bending_case(), [2, 4], family="tet10", layers=3)
    assert tet10[-1]["relative_strain_energy_error"] < energies[-1] / 1e6


def test_dirichlet_rejects_unknown_family():
    with pytest.raises(ValueError):
        tet_dirichlet_study(bending_case(), [2], family="tet20", layers=3)
