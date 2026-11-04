"""Gates for the plane-strain B-bar Q4 element (:mod:`tensorfem.plane_strain_bbar`).

Order of the assertions matters and mirrors ``tests/test_incompressible_locking.py``:
the element is first audited structurally (rank, patch test, equilibrium) and the
analytic oracle is verified against its own governing equations; only then are
the error numbers scored.  The locking of the fully integrated element is
asserted as a *failure* so that nobody can quote it as a pass.
"""
import math

import pytest
import torch

from tensorfem.continuum import ContinuumModel, elasticity_matrix, solve_continuum
from tensorfem.plane_strain_bbar import (bbar_element_eigenvalues, q4_bbar_stiffness,
                                         solve_continuum_bbar)
from tensorfem.plane_strain_locking import (lame_plane_strain_self_check, observed_order,
                                            refinement_study, solve_lame_case)

FAST = dict(n_theta=16, n_radial=8)


def _annulus_mesh(n_theta=16, n_radial=8, inner=0.5, outer=0.8):
    theta = torch.linspace(0.0, math.pi / 2.0, n_theta + 1, dtype=torch.float64)
    radius = torch.linspace(inner, outer, n_radial + 1, dtype=torch.float64)
    nodes = torch.stack([torch.stack((r * math.cos(t), r * math.sin(t)))
                         for t in theta for r in radius])
    cells = []
    stride = n_radial + 1
    for j in range(n_theta):
        for i in range(n_radial):
            n0 = j * stride + i
            cells.append((n0, n0 + stride, n0 + stride + 1, n0 + 1))
    return nodes, torch.tensor(cells, dtype=torch.long)


def test_reference_field_satisfies_its_own_governing_equations():
    for poisson in (0.3, 0.4999):
        check = lame_plane_strain_self_check(poisson)
        assert check["equilibrium"] < 1e-12
        assert check["traction"] < 1e-12
        assert check["constitutive"] < 1e-9
        assert check["axial"] < 1e-12
        # The in-plane Lame stress is independent of nu; the study relies on it.
        assert check["poisson_invariance"] < 1e-12


def test_bbar_element_has_three_rigid_modes_and_no_hourglass_mode():
    """A distorted element must keep exactly 8 - 3 = 5 deformation modes."""
    nodes = torch.tensor([[0.0, 0.0], [1.3, 0.1], [1.1, 0.9], [-0.2, 1.0]], dtype=torch.float64)
    for poisson in (0.3, 0.4999):
        eigenvalues = bbar_element_eigenvalues(nodes, poisson=poisson)
        normalised = eigenvalues / eigenvalues.max()
        zero_modes = int((normalised.abs() < 1e-10).sum())
        assert zero_modes == 3, normalised.tolist()
        assert int((normalised > 1e-10).sum()) == 5


def test_bbar_stiffness_is_symmetric_and_carries_rigid_translation():
    nodes = torch.tensor([[0.0, 0.0], [1.3, 0.1], [1.1, 0.9], [-0.2, 1.0]], dtype=torch.float64)
    coordinates = nodes.reshape(1, 4, 2)
    dmat = elasticity_matrix(coordinates.new_tensor([210.0e9]), coordinates.new_tensor([0.4999]),
                             "strain")
    stiffness, div_average = q4_bbar_stiffness(coordinates, dmat, coordinates.new_tensor([1.0]))
    assert torch.allclose(stiffness[0], stiffness[0].T, atol=1e-6 * float(stiffness.abs().max()))
    translation = torch.tensor([1.0, 0.0] * 4, dtype=torch.float64)
    assert float((stiffness[0] @ translation).abs().max()) < 1e-9 * float(stiffness.abs().max())
    # The mean dilatation of a rigid translation is zero by construction.
    assert float(div_average.abs().max()) < 1e-12 or float((div_average @ translation).abs().max()) < 1e-12


def test_bbar_passes_the_patch_test_on_the_production_solve_path():
    """Affine displacement prescribed on the whole boundary is reproduced inside."""
    length, height, nx, ny = 2.0, 1.0, 3, 2
    xs = torch.linspace(0.0, length, nx + 1, dtype=torch.float64)
    ys = torch.linspace(0.0, height, ny + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            cells.append((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1))
    cells = torch.tensor(cells, dtype=torch.long)
    gradient = torch.tensor([[1.0e-3, -4.0e-4], [2.0e-4, 7.0e-4]], dtype=torch.float64)
    exact = nodes @ gradient.T
    boundary = torch.zeros(2 * len(nodes), dtype=torch.bool)
    on_boundary = ((nodes[:, 0] <= 0.0) | (nodes[:, 0] >= length)
                   | (nodes[:, 1] <= 0.0) | (nodes[:, 1] >= height))
    boundary[0::2] = on_boundary
    boundary[1::2] = on_boundary
    fixed = torch.arange(2 * len(nodes))[boundary]
    model = ContinuumModel(nodes, cells, torch.tensor(210.0e9, dtype=torch.float64),
                           torch.tensor(0.4999, dtype=torch.float64),
                           torch.tensor(1.0, dtype=torch.float64),
                           torch.zeros(2 * len(nodes), dtype=torch.float64), fixed,
                           plane="strain", prescribed_values=exact.reshape(-1)[fixed])
    result = solve_continuum_bbar(model)
    assert float((result.displacement.reshape(-1, 2) - exact).abs().max()) < 1e-15
    reference = solve_continuum(model)
    assert float((result.displacement - reference.displacement).abs().max()) < 1e-15


def test_bbar_cures_locking_while_the_standard_element_fails_closed():
    bbar = solve_lame_case(0.4999, formulation="bbar", **FAST)
    standard = solve_lame_case(0.4999, formulation="standard", **FAST)
    assert 100.0 * bbar["errors"]["radial_stress_l2"] < 3.0, bbar["errors"]
    assert 100.0 * bbar["errors"]["displacement_l2"] < 3.0, bbar["errors"]
    # The unmitigated element must not be quotable as accurate.
    assert 100.0 * standard["errors"]["radial_stress_l2"] > 50.0, standard["errors"]
    assert 100.0 * standard["errors"]["displacement_l2"] > 50.0, standard["errors"]
    assert bbar["errors"]["support_equilibrium"] < 1e-8
    assert standard["errors"]["support_equilibrium"] < 1e-8


def test_bbar_stress_error_is_flat_in_poisson_ratio():
    """Locking shows up as error growth with nu; the cured element has none."""
    compressible = solve_lame_case(0.30, formulation="bbar", **FAST)
    incompressible = solve_lame_case(0.4999, formulation="bbar", **FAST)
    ratio = incompressible["errors"]["radial_stress_l2"] / compressible["errors"]["radial_stress_l2"]
    assert ratio < 1.10, ratio


def test_bbar_recovers_optimal_second_order_convergence():
    levels = ((24, 12), (32, 16), (40, 20))
    records = [solve_lame_case(0.4999, formulation="bbar", n_theta=n_theta, n_radial=n_radial)
               for n_theta, n_radial in levels]
    for key in ("displacement_l2", "radial_stress_l2"):
        order = observed_order(records[-2], records[-1], key)
        assert 1.8 < order < 2.2, (key, order)
    assert records[-1]["errors"]["radial_stress_l2"] < records[0]["errors"]["radial_stress_l2"]


def test_bbar_does_not_degrade_the_compressible_plane_stress_case():
    bbar = solve_lame_case(0.30, formulation="bbar", plane="stress", **FAST)
    standard = solve_lame_case(0.30, formulation="standard", plane="stress", **FAST)
    assert 100.0 * bbar["errors"]["radial_stress_l2"] < 1.0
    assert 100.0 * standard["errors"]["radial_stress_l2"] < 1.0