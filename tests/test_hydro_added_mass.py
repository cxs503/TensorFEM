"""Tests for the hydrodynamic added mass of a baffled panel.

The anchors used here are independent of the implementation:

* baffled circular piston radiation mass ``(8/3) rho a**3`` (Rayleigh);
* single-mode in-phase frequency law ``f_wet/f_dry = 1/sqrt(1+mu)``;
* exact integral ``int sin**2 = L/2`` for the structural generalised mass.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

from tensorfem.added_mass import (
    assemble_added_mass_consistent,
    assemble_added_mass_lumped,
)
from tensorfem.hydro_added_mass import (
    dipole_added_mass_matrix,
    disc_panels,
    equivalent_added_mass_per_area,
    modal_added_mass_ratio,
    panel_added_mass_ratio,
    piston_radiation_mass,
    sine_mode_values,
    wet_frequency_ratio,
)

ROOT = Path(__file__).parents[1]
PANEL = dict(lx=1.2, ly=0.8, thickness=0.008, rho_steel=7850.0, rho_water=1025.0)


def _panel_mesh(nx, ny, lx, ly):
    xs = torch.linspace(0.0, lx, nx + 1, dtype=torch.float64)
    ys = torch.linspace(0.0, ly, ny + 1, dtype=torch.float64)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elements.append((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1))
    return nodes, torch.tensor(elements, dtype=torch.long)


# ----------------------------------------------------------------------
# Anchor A1: baffled piston radiation mass
# ----------------------------------------------------------------------
def test_piston_dipole_sum_matches_rayleigh():
    a, rho = 0.5, 1025.0
    exact = piston_radiation_mass(a, rho)
    assert exact == pytest.approx(8.0 / 3.0 * rho * a ** 3)
    assert 0.0 < exact
    for n_cells, tol in ((32, 0.02), (64, 0.01)):
        centres, areas = disc_panels(a, n_cells)
        total = float(dipole_added_mass_matrix(centres, areas, rho).sum())
        assert total == pytest.approx(exact, rel=tol), n_cells


def test_disc_area_converges_and_operator_is_area_normalised():
    a, rho = 0.5, 1025.0
    exact = piston_radiation_mass(a, rho)
    errors, normalised = [], []
    for n_cells in (16, 32, 64):
        centres, areas = disc_panels(a, n_cells)
        total = float(dipole_added_mass_matrix(centres, areas, rho).sum())
        errors.append(abs(total - exact) / exact)
        normalised.append(abs(total / float(areas.sum())
                               - exact / (math.pi * a ** 2)) / (exact / (math.pi * a ** 2)))
    assert errors[0] > errors[-1]
    # Once the staircase area error is divided out the operator itself is
    # converged: 1.1 % at 16 cells, 0.03 % at 64 cells per side.
    assert normalised[0] < 0.02
    assert normalised[-1] < 1e-3
    assert normalised[0] > normalised[-1]


# ----------------------------------------------------------------------
# Operator structure
# ----------------------------------------------------------------------
def test_influence_matrix_symmetry_and_self_term():
    centres, areas = disc_panels(0.2, 8)
    rho = 1025.0
    matrix = dipole_added_mass_matrix(centres, areas, rho)
    assert torch.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0)
    diag = torch.diagonal(matrix)
    assert torch.allclose(diag, rho * areas * torch.sqrt(areas / math.pi))
    assert bool((diag > 0).all())


def test_far_field_pair_obeys_inverse_distance_law():
    rho = 1000.0
    centres = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float64)
    areas = torch.tensor([0.01, 0.04], dtype=torch.float64)
    matrix = dipole_added_mass_matrix(centres, areas, rho)
    expected = rho * float(areas[0] * areas[1]) / (2.0 * math.pi * 5.0)
    assert float(matrix[0, 1]) == pytest.approx(expected, rel=1e-12)


def test_mode_values_are_simply_supported_shape():
    centres = torch.tensor([[0.0, 0.0], [0.6, 0.4]], dtype=torch.float64)
    phi = sine_mode_values(centres, 1.2, 0.8)
    assert float(phi[0]) == pytest.approx(0.0, abs=1e-15)
    assert float(phi[1]) == pytest.approx(1.0, rel=1e-12)


def test_modal_ratio_is_quadratic_in_amplitude():
    centres, areas = disc_panels(0.2, 8)
    matrix = dipole_added_mass_matrix(centres, areas, 1000.0)
    phi = torch.linspace(0.0, 1.0, centres.shape[0], dtype=torch.float64)
    r1 = modal_added_mass_ratio(matrix, phi, 2.0)
    r2 = modal_added_mass_ratio(matrix, 3.0 * phi, 18.0)
    assert r1 == pytest.approx(r2, rel=1e-12)


# ----------------------------------------------------------------------
# Anchor A2/A3: panel ratio, single-mode law, FE assembly consistency
# ----------------------------------------------------------------------
def test_panel_ratio_is_mesh_converged():
    coarse = panel_added_mass_ratio(PANEL["lx"], PANEL["ly"], PANEL["thickness"],
                                    PANEL["rho_steel"], PANEL["rho_water"],
                                    nx=20, ny=14)
    fine = panel_added_mass_ratio(PANEL["lx"], PANEL["ly"], PANEL["thickness"],
                                  PANEL["rho_steel"], PANEL["rho_water"],
                                  nx=30, ny=20)
    assert fine["mu"] > 1.0
    assert fine["mu"] == pytest.approx(coarse["mu"], rel=1e-2)


def test_wet_frequency_ratio_follows_single_mode_law():
    assert wet_frequency_ratio(0.0) == 1.0
    for mu in (0.5, 3.0, 6.465):
        assert wet_frequency_ratio(mu) == pytest.approx(1.0 / math.sqrt(1.0 + mu))


def test_equivalent_uniform_added_mass_reproduces_modal_mass():
    """The consistent element assembly must integrate ``m_a * sin**2`` exactly."""
    lx, ly = PANEL["lx"], PANEL["ly"]
    result = panel_added_mass_ratio(lx, ly, PANEL["thickness"], PANEL["rho_steel"],
                                   PANEL["rho_water"], nx=30, ny=20)
    nodes, elements = _panel_mesh(24, 16, lx, ly)
    matrix = assemble_added_mass_consistent(nodes, elements,
                                           result["added_mass_per_area"])
    w = 3 * torch.arange(nodes.shape[0])
    m_ww = matrix[w[:, None], w]
    phi = sine_mode_values(nodes, lx, ly)
    generalised = float(phi @ (m_ww @ phi))
    exact = result["added_mass_per_area"] * lx * ly / 4.0
    assert generalised == pytest.approx(exact, rel=1e-2)
    # ... and it equals mu times the structural generalised mass (the identity
    # used in anchor A3).
    assert generalised == pytest.approx(result["mu"] * result["structural_generalised_mass"],
                                       rel=1e-2)


def test_lumped_total_equals_consistent_total_added_mass():
    lx, ly = PANEL["lx"], PANEL["ly"]
    nodes, elements = _panel_mesh(24, 16, lx, ly)
    m_a = 406.0
    consistent = assemble_added_mass_consistent(nodes, elements, m_a)
    lumped = assemble_added_mass_lumped(nodes, elements, m_a)
    area = lx * ly
    assert float(consistent.sum()) == pytest.approx(m_a * area, rel=1e-12)
    assert float(lumped.sum()) == pytest.approx(m_a * area, rel=1e-12)
    assert float(lumped.min()) == 0.0  # only w DOFs carry the mass


def test_fe_wet_frequency_ratio_matches_single_mode_law():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from wet_panel_added_mass import panel_frequency  # noqa: E402

    hydro = panel_added_mass_ratio(PANEL["lx"], PANEL["ly"], PANEL["thickness"],
                                  PANEL["rho_steel"], PANEL["rho_water"],
                                  nx=20, ny=14)
    dry = panel_frequency(6, 4, young=2.1e11, poisson=0.3,
                          thickness=PANEL["thickness"], rho_steel=PANEL["rho_steel"],
                          lx=PANEL["lx"], ly=PANEL["ly"])[0]
    wet = panel_frequency(6, 4, young=2.1e11, poisson=0.3,
                          thickness=PANEL["thickness"], rho_steel=PANEL["rho_steel"],
                          added_mass_per_area=hydro["added_mass_per_area"],
                          lx=PANEL["lx"], ly=PANEL["ly"])[0]
    assert wet / dry == pytest.approx(wet_frequency_ratio(hydro["mu"]), rel=3e-2)


# ----------------------------------------------------------------------
# Fail-closed inputs
# ----------------------------------------------------------------------
def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        piston_radiation_mass(0.0)
    with pytest.raises(ValueError):
        piston_radiation_mass(1.0, rho_water=-5.0)
    with pytest.raises(ValueError):
        disc_panels(-1.0, 8)
    centres, areas = disc_panels(0.2, 8)
    with pytest.raises(ValueError):
        dipole_added_mass_matrix(centres, areas[:-1], 1000.0)
    with pytest.raises(ValueError):
        dipole_added_mass_matrix(centres, -areas, 1000.0)
    with pytest.raises(ValueError):
        modal_added_mass_ratio(torch.eye(3), torch.ones(2), 1.0)
    with pytest.raises(ValueError):
        equivalent_added_mass_per_area(-0.1, 62.8)
    with pytest.raises(ValueError):
        wet_frequency_ratio(float("nan"))
    with pytest.raises(ValueError):
        panel_added_mass_ratio(1.2, 0.8, 0.0, 7850.0, 1025.0)
    with pytest.raises(ValueError):
        sine_mode_values(centres, 1.0, 1.0, m=0)