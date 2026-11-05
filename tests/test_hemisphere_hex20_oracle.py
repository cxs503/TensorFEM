"""Tests for the independent HEX20 continuum oracle of the 18-degree-hole hemisphere.

The point of this file is that the oracle is certified *before* it is used to
judge the shell element: exact Lamé elasticity, thin-strip bending at the
benchmark aspect ratio, global equilibrium, and the surface-stress bookkeeping.
The last test is fail-closed: the recorded shell4 surface-stress comparison must
stay below the 3% acceptance gate, so a future change cannot quietly declare the
shell stresses certified.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorfem.hemisphere_hex20_oracle import (MH_REFERENCE_DISPLACEMENT, hex20_parametric_centre,
                                               shell_solid_hex20_mesh,
                                               shell_surface_stress_table, von_mises)
from tensorfem.hemisphere_hex20_reference import (hemisphere_hex20_model, lame_ring_model,
                                                  thin_strip_bending_validation)

EVIDENCE = Path(__file__).resolve().parents[1] / "benchmarks/tutorials/hemisphere-hex20-oracle.json"


def test_lame_ring_is_exact_to_round_off():
    """The oracle must reproduce plane-strain Lamé elasticity on curved geometry."""
    record = lame_ring_model(4, 24, 2)
    assert record["max_relative_error"]["hoop"] < 1e-4
    assert record["max_relative_error"]["radial"] < 1e-4
    assert record["max_relative_error"]["axial"] < 1e-4
    assert record["equilibrium"]["free_residual_relative"] < 1e-8
    assert record["equilibrium"]["normalized_force_balance_error"] < 1e-8


def test_thin_strip_bending_budget():
    """Bending accuracy at the benchmark element-size/thickness ratio."""
    record = thin_strip_bending_validation(nx=16, ny=1, nz=2)
    assert record["relative_error"] < 0.015
    assert record["equilibrium"]["normalized_force_balance_error"] < 1e-5


def test_mesh_has_positive_jacobians_and_exact_surface_radius():
    """Every brick must be right-handed; surface nodes sit on the exact geometry."""
    nodes, elements = shell_solid_hex20_mesh(6, 8, 1)
    radius = torch.linalg.vector_norm(nodes, dim=1)
    assert float(radius.min()) == pytest.approx(10.0 - 0.02, rel=0, abs=1e-12)
    assert float(radius.max()) == pytest.approx(10.0 + 0.02, rel=0, abs=1e-12)
    centre = hex20_parametric_centre(nodes, elements)
    assert torch.all(torch.linalg.vector_norm(centre, dim=1) > 9.98)
    # corners + one node per mid-edge: no face-centre or interior nodes exist
    expected = ((6 + 1) * (8 + 1) * (1 + 1) + 6 * (8 + 1) * (1 + 1)
                + (6 + 1) * 8 * (1 + 1) + (6 + 1) * (8 + 1) * 1)
    assert len(nodes) == expected


def test_hemisphere_quarter_model_equilibrium_and_reference():
    """Coarse oracle run: solvable, balanced, and within 15% of the published value."""
    result, probe = hemisphere_hex20_model(16, 16, 1)
    equilibrium = result.equilibrium()
    assert equilibrium["free_residual_relative"] < 1e-6
    assert equilibrium["normalized_force_balance_error"] < 1e-5
    assert equilibrium["normalized_moment_balance_error"] < 1e-5
    displacement = float(result.displacement[3 * probe])
    assert 0.8 * MH_REFERENCE_DISPLACEMENT < displacement < 1.15 * MH_REFERENCE_DISPLACEMENT


def test_surface_stress_table_stations_are_exact():
    """Station angles must recover the mesh parametrisation exactly."""
    result, _ = hemisphere_hex20_model(4, 4, 1)
    table = shell_surface_stress_table(result, 4, 4, 1, outer=True)
    # the table is ordered azimuth-major: rows are theta, columns are phi
    phi = torch.rad2deg(table["polar_angle"]).reshape(4, 4)[0, :]
    theta = torch.rad2deg(table["azimuth"]).reshape(4, 4)[:, 0]
    expected_phi = [18.0 + 72.0 * (i + 0.5) / 4 for i in range(4)]
    expected_theta = [90.0 * (j + 0.5) / 4 for j in range(4)]
    assert phi.tolist() == pytest.approx(expected_phi, abs=2e-2)
    assert theta.tolist() == pytest.approx(expected_theta, abs=2e-2)
    assert float(table["von_mises"].min()) > 0.0
    # the outer free surface must be nearly traction free in its own normal
    normal = table["point"] / torch.linalg.vector_norm(table["point"], dim=1)[:, None]
    sxx, syy, szz, sxy, syz, szx = table["stress"].unbind(-1)
    sigma_nn = (sxx * normal[:, 0] ** 2 + syy * normal[:, 1] ** 2 + szz * normal[:, 2] ** 2
                + 2 * sxy * normal[:, 0] * normal[:, 1] + 2 * syz * normal[:, 1] * normal[:, 2]
                + 2 * szx * normal[:, 2] * normal[:, 0])
    # weakly satisfied on a 4x4 mesh: the surface stress is recovered, not imposed
    assert float(sigma_nn.abs().max()) < 0.10 * float(von_mises(table["stress"]).max())


def test_shell4_surface_stress_is_not_certified():
    """Fail-closed: the shell4 surface stress may not be reported as < 3% accurate."""
    # hemisphere_field_report() re-solves n=8..40 and is far too slow for a unit
    # test, so the fail-closed assertion is taken from the committed evidence and
    # from the flag the field module itself publishes.
    assert EVIDENCE.exists(), "run scripts/hemisphere_hex20_oracle_study.py first"
    if EVIDENCE.exists():
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        assert evidence["schema"] == "tensorfem.hemisphere-hex20-oracle.v1"
        assert evidence["stress_accuracy_qualified"] is False
        region = evidence["shell4"]["32"]["comparison_against_finest_oracle"]["regions"]["interior"]
        # the measured disagreement is several percent, so no region may claim 3%
        assert region["mean_relative_error"] > 0.03
        assert evidence["oracle_validation"]["lame_ring"]["max_relative_error"]["hoop"] < 1e-4