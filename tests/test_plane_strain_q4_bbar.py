"""Qualification tests for the locking-free plane-strain Q4 formulations.

Two layers:

* **element-level correctness** -- patch test on a distorted mesh, rigid-body
  energy, symmetry, null-space count (no hourglass modes) and the algebraic
  identity ``K_bbar == K_sri`` that holds for the 4-node quad;
* **the locking claim itself** -- against the closed-form Lame cylinder, with
  fail-closed gates so that neither the standard (locking) Q4 nor a regressed
  B-bar can be reported as acceptable.
"""
import math

import pytest
import torch

from tensorfem.continuum import (ContinuumModel, elasticity_matrix, q4_stiffness as standard_stiffness,
                                 rectangular_q4_mesh, solve_continuum)
from tensorfem.plane_strain_locking import (flatness, locking_sweep, observed_order,
                                            reference_self_check, refinement_study, solve_case)
from tensorfem.q4_volumetric import q4_bbar_stiffness, q4_sri_stiffness

LOCKING_FREE = ("bbar", "sri")
FINE = 0.4999


def _distorted_quad():
    return torch.tensor([[[0.0, 0.0], [2.3, 0.1], [2.6, 1.7], [-0.2, 1.1]]], dtype=torch.float64)


def _element_matrix(coordinates, poisson=0.3, formulation="standard"):
    dmat = elasticity_matrix(torch.tensor([210e3], dtype=torch.float64),
                             torch.tensor([float(poisson)], dtype=torch.float64), "strain")
    thickness = torch.ones(len(coordinates), dtype=torch.float64)
    if formulation == "standard":
        return standard_stiffness(coordinates, dmat, thickness)[0]
    if formulation == "bbar":
        return q4_bbar_stiffness(coordinates, dmat, thickness)[0]
    return q4_sri_stiffness(coordinates, dmat, thickness)[0]


@pytest.mark.parametrize("formulation", ("standard",) + LOCKING_FREE)
def test_element_stiffness_is_symmetric(formulation):
    stiffness = _element_matrix(_distorted_quad(), poisson=FINE, formulation=formulation)
    scale = float(stiffness.abs().max())
    assert float((stiffness - stiffness.transpose(1, 2)).abs().max()) / scale < 1e-12


@pytest.mark.parametrize("formulation", ("standard",) + LOCKING_FREE)
def test_rigid_body_motion_stores_no_energy(formulation):
    coordinates = _distorted_quad()
    stiffness = _element_matrix(coordinates, poisson=FINE, formulation=formulation)
    scale = float(stiffness.abs().max())
    x, y = coordinates[0, :, 0], coordinates[0, :, 1]
    ones, zeros = torch.ones(4, dtype=x.dtype), torch.zeros(4, dtype=x.dtype)
    modes = {
        "translation_x": torch.stack((ones, zeros), 1).reshape(-1),
        "translation_y": torch.stack((zeros, ones), 1).reshape(-1),
        "rotation": torch.stack((-y, x), 1).reshape(-1),
    }
    for name, mode in modes.items():
        energy = float(mode @ stiffness[0] @ mode)
        assert abs(energy) / (scale * float(mode @ mode)) < 1e-14, name


@pytest.mark.parametrize("formulation", LOCKING_FREE)
def test_no_spurious_hourglass_modes(formulation):
    """A single element must keep exactly three zero-energy modes."""
    eigenvalues = torch.linalg.eigvalsh(_element_matrix(_distorted_quad(), poisson=FINE,
                                                        formulation=formulation)[0])
    scale = float(eigenvalues.abs().max())
    assert int((eigenvalues.abs() < 1e-10 * scale).sum()) == 3


def test_bbar_and_sri_coincide_for_the_bilinear_quad():
    """Documented identity: ``K_bbar == K_sri`` for Q4 (single volumetric mode).

    Consequence for the report: the two formulations are *not* independent
    cross-checks in 2-D, they are the same operator; the independent evidence is
    the patch test plus the mesh-refinement order.
    """
    coordinates = _distorted_quad()
    bbar = _element_matrix(coordinates, poisson=FINE, formulation="bbar")
    sri = _element_matrix(coordinates, poisson=FINE, formulation="sri")
    assert float((bbar - sri).abs().max()) / float(bbar.abs().max()) < 1e-12


@pytest.mark.parametrize("formulation", LOCKING_FREE)
def test_locking_free_element_differs_from_standard_at_incompressibility(formulation):
    """The projection must actually change the operator in the incompressible limit."""
    coordinates = _distorted_quad()
    reference = _element_matrix(coordinates, poisson=FINE, formulation="standard")
    modified = _element_matrix(coordinates, poisson=FINE, formulation=formulation)
    # measured 0.24 (nu = 0.3) and 0.36 (nu = 0.4999) on this element; the gate is
    # deliberately below both but far above round-off
    assert float((modified - reference).abs().max()) / float(reference.abs().max()) > 0.2


@pytest.mark.parametrize("formulation", ("standard",) + LOCKING_FREE)
def test_constant_strain_patch_test(formulation):
    """A linear displacement field must be reproduced exactly on a distorted mesh."""
    nodes, cells = rectangular_q4_mesh(2.0, 1.0, 3, 3)
    # perturb interior nodes only: the boundary stays a straight-edged patch
    xmin, xmax = float(nodes[:, 0].min()), float(nodes[:, 0].max())
    ymin, ymax = float(nodes[:, 1].min()), float(nodes[:, 1].max())
    interior = ((nodes[:, 0] > xmin + 1e-9) & (nodes[:, 0] < xmax - 1e-9)
                & (nodes[:, 1] > ymin + 1e-9) & (nodes[:, 1] < ymax - 1e-9))
    nodes = nodes.clone()
    nodes[interior, 0] += 0.06 * torch.sin(3.0 * nodes[interior, 1])
    nodes[interior, 1] -= 0.05 * torch.cos(2.0 * nodes[interior, 0])

    gradient = torch.tensor([[1.0e-3, 4.0e-4], [-2.0e-4, -6.0e-4]], dtype=torch.float64)
    exact = nodes @ gradient.T                      # linear field u = G x
    boundary = ~interior
    fixed = torch.cat(((2 * boundary.nonzero()).reshape(-1),
                       (2 * boundary.nonzero() + 1).reshape(-1))).sort().values
    prescribed = exact.reshape(-1)[fixed]

    model = ContinuumModel(nodes, cells, torch.tensor(210.0e3, dtype=torch.float64),
                           torch.tensor(0.4999, dtype=torch.float64),
                           torch.tensor(1.0, dtype=torch.float64),
                           torch.zeros(2 * len(nodes), dtype=torch.float64), fixed,
                           plane="strain", prescribed_values=prescribed,
                           formulation=formulation)
    result = solve_continuum(model)
    displacement = result.displacement.reshape(-1, 2)
    scale = float(exact.abs().max())
    assert float((displacement - exact).abs().max()) / scale < 1e-10

    # The *nodal* field is reproduced exactly above.  The recovered centroid stress on
    # a distorted bilinear quad carries the classical O(distortion) "flexure" error --
    # that is a property of the bilinear map, not of the volumetric treatment -- so it
    # is only bounded here and required exact on the affine (parallelogram) mesh in
    # ``test_constant_strain_patch_test_stress_affine_mesh``.
    expected = _patch_stress(gradient)
    assert float((result.stress - expected).abs().max()) / float(expected.abs().max()) < 5e-3


def _patch_stress(gradient):
    strain = torch.tensor([gradient[0, 0], gradient[1, 1], gradient[0, 1] + gradient[1, 0]],
                          dtype=torch.float64)
    dmat = elasticity_matrix(torch.tensor([210.0e3], dtype=torch.float64),
                             torch.tensor([0.4999], dtype=torch.float64), "strain")
    return dmat[0] @ strain


@pytest.mark.parametrize("formulation", ("standard",) + LOCKING_FREE)
def test_constant_strain_patch_test_stress_affine_mesh(formulation):
    """On an affine (rectangular) mesh the constant-strain state must be exact."""
    nodes, cells = rectangular_q4_mesh(2.0, 1.0, 4, 3)
    gradient = torch.tensor([[1.0e-3, 4.0e-4], [-2.0e-4, -6.0e-4]], dtype=torch.float64)
    exact = nodes @ gradient.T
    xmin, xmax = float(nodes[:, 0].min()), float(nodes[:, 0].max())
    ymin, ymax = float(nodes[:, 1].min()), float(nodes[:, 1].max())
    interior = ((nodes[:, 0] > xmin + 1e-9) & (nodes[:, 0] < xmax - 1e-9)
                & (nodes[:, 1] > ymin + 1e-9) & (nodes[:, 1] < ymax - 1e-9))
    boundary = ~interior
    fixed = torch.cat(((2 * boundary.nonzero()).reshape(-1),
                       (2 * boundary.nonzero() + 1).reshape(-1))).sort().values
    model = ContinuumModel(nodes, cells, torch.tensor(210.0e3, dtype=torch.float64),
                           torch.tensor(0.4999, dtype=torch.float64),
                           torch.tensor(1.0, dtype=torch.float64),
                           torch.zeros(2 * len(nodes), dtype=torch.float64), fixed,
                           plane="strain", prescribed_values=exact.reshape(-1)[fixed],
                           formulation=formulation)
    result = solve_continuum(model)
    assert float((result.displacement.reshape(-1, 2) - exact).abs().max()) / float(
        exact.abs().max()) < 1e-10
    expected = _patch_stress(gradient)
    assert float((result.stress - expected).abs().max()) / float(expected.abs().max()) < 1e-10


def test_standard_path_is_unchanged_by_the_new_field():
    """``formulation`` must default to the old, fully integrated behaviour."""
    nodes, cells = rectangular_q4_mesh(1.0, 0.4, 4, 2)
    forces = torch.zeros(2 * len(nodes), dtype=torch.float64)
    forces[-1] = 1.0
    fixed = torch.tensor([0, 1, 2 * (len(nodes) - 1)], dtype=torch.long)
    base = ContinuumModel(nodes, cells, torch.tensor(70.0e3), torch.tensor(0.33),
                          torch.tensor(1.0), forces, fixed, plane="stress")
    explicit = ContinuumModel(nodes, cells, torch.tensor(70.0e3), torch.tensor(0.33),
                              torch.tensor(1.0), forces, fixed, plane="stress",
                              formulation="standard")
    assert torch.equal(solve_continuum(base).displacement, solve_continuum(explicit).displacement)


def test_formulation_validation():
    nodes, cells = rectangular_q4_mesh(1.0, 1.0, 1, 1)
    kwargs = dict(nodes=nodes, elements=cells, young_modulus=torch.tensor(1.0),
                  poisson_ratio=torch.tensor(0.3), thickness=torch.tensor(1.0),
                  forces=torch.zeros(2 * len(nodes)), fixed_dofs=torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        ContinuumModel(**kwargs, formulation="eas")
    cst_cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
    with pytest.raises(ValueError):
        ContinuumModel(nodes=nodes, elements=cst_cells, young_modulus=torch.tensor(1.0),
                       poisson_ratio=torch.tensor(0.3), thickness=torch.tensor(1.0),
                       forces=torch.zeros(2 * len(nodes)), fixed_dofs=torch.tensor([0, 1]),
                       element_type="cst", formulation="bbar")


@pytest.mark.parametrize("poisson", (0.3, FINE))
def test_reference_self_check_closes(poisson):
    residuals = reference_self_check(poisson)
    assert residuals["equilibrium"] < 1e-8
    assert residuals["traction"] < 1e-14
    assert residuals["constitutive"] < 1e-12


def test_reference_becomes_incompressible_in_the_limit():
    coarse = reference_self_check(0.3)["volumetric_strain_over_strain_scale"]
    limit = reference_self_check(FINE)["volumetric_strain_over_strain_scale"]
    assert 0.2 < coarse < 0.4
    assert limit < 1e-3


def test_energy_balance_of_the_pressure_load_path():
    record = solve_case(FINE, n_theta=16, n_radial=8, formulation="bbar")
    assert record["errors"]["energy_balance_relative"] < 1e-10


def test_locking_is_cured_flat_in_poisson():
    """The report's central claim, as a gate on the nu-sweep."""
    records = locking_sweep((0.3, 0.499, FINE), n_theta=16, n_radial=8)
    # the exact stress field is Poisson-ratio independent, so a locking-free
    # formulation must report the same radial-stress error at every nu
    assert flatness(records, "radial_stress_l2", "bbar") < 1.05
    assert flatness(records, "displacement_l2", "bbar") < 1.30
    # ... and the fully integrated Q4 must not
    assert flatness(records, "radial_stress_l2", "standard") > 50.0
    assert flatness(records, "displacement_l2", "standard") > 50.0


def test_bbar_meets_the_three_percent_gate_at_the_incompressible_limit():
    for n_theta, n_radial in ((8, 4), (16, 8), (32, 16)):
        record = solve_case(FINE, n_theta=n_theta, n_radial=n_radial, formulation="bbar")
        errors = record["errors"]
        assert errors["displacement_l2"] < 0.03, (n_theta, n_radial, errors["displacement_l2"])
        assert errors["radial_stress_l2"] < 0.03, (n_theta, n_radial, errors["radial_stress_l2"])
        assert errors["mean_stress_l2"] < 0.03


def test_standard_q4_stays_locked_fail_closed():
    """Nobody may report the fully integrated Q4 as passing the 3 % gate."""
    record = solve_case(FINE, n_theta=16, n_radial=8, formulation="standard")
    errors = record["errors"]
    assert errors["displacement_l2"] > 0.50
    assert errors["radial_stress_l2"] > 1.0
    assert errors["mean_stress_l2"] > 0.20
    # the recorded 2-D debt table of docs/near-incompressible-locking.md
    assert errors["displacement_l2"] == pytest.approx(0.619175, rel=1e-4)
    assert errors["radial_stress_l2"] == pytest.approx(1.552296, rel=1e-4)


def test_bbar_converges_at_the_expected_order():
    records = refinement_study(FINE, meshes=((8, 4), (16, 8), (32, 16)), formulations=("bbar",))
    assert 1.8 < observed_order(records, "displacement_l2", "bbar") < 2.2
    assert 1.8 < observed_order(records, "radial_stress_l2", "bbar") < 2.2
    errors = [r["errors"]["displacement_l2"] for r in sorted(records, key=lambda r: r["dofs"])]
    assert errors[0] > errors[1] > errors[2]