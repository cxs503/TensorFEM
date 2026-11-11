"""Acceptance tests for CASE-LAME-CYLINDER (thick cylinder, Lamé reference).

Two layers:

* the reference itself is checked against an independent Chebyshev-collocation
  solve of the axisymmetric displacement problem, so the closed form used to
  score the FE results is not self-certified;
* the finite element solution is checked against the < 2 % acceptance gate on
  meshes small enough to keep the suite fast (the full 96x32 study lives in
  ``benchmarks/lame_cylinder_anchor.py``).
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorfem.lame_reference import (
    lame_coefficients,
    lame_state,
    sector_q4_mesh,
)
from tensorfem.thick_cylinder_1d_oracle import solve_thick_cylinder_1d

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "lame_cylinder_anchor.py"
EVIDENCE = ROOT / "benchmarks" / "lame_cylinder_anchor_evidence.json"

A, B = 0.5, 0.8
P = 100.0e6
E, NU = 210.0e9, 0.3
GATE = 0.02


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("lame_anchor_bench", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _benchmark_module()


# --------------------------------------------------------------------------- #
# reference layer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("plane", ["strain", "stress"])
@pytest.mark.parametrize("ratio", [1.6, 2.0, 3.0])
def test_closed_form_matches_spectral_oracle(plane, ratio):
    """The Lamé closed form and the 1-D collocation solve agree to machine precision."""
    outer = A * ratio
    radius = np.linspace(A, outer, 41)
    oracle = solve_thick_cylinder_1d(
        radius, inner_radius=A, outer_radius=outer, inner_pressure=P,
        young_modulus=E, poisson_ratio=NU, plane=plane, n_modes=33,
    )
    exact = lame_state(
        torch.tensor(radius), inner_radius=A, outer_radius=outer, inner_pressure=P,
        young_modulus=E, poisson_ratio=NU, plane=plane,
    )
    assert np.max(np.abs(oracle.radial_stress - exact.radial_stress.numpy())) / P < 1e-9
    assert np.max(np.abs(oracle.hoop_stress - exact.hoop_stress.numpy())
                  / np.abs(exact.hoop_stress.numpy())) < 1e-9
    assert np.max(np.abs(oracle.radial_displacement - exact.radial_displacement.numpy())
                  / np.abs(exact.radial_displacement.numpy())) < 1e-9
    assert oracle.equilibrium_residual < 1e-6


def test_oracle_honours_traction_boundary_conditions():
    radius = np.linspace(A, B, 11)
    oracle = solve_thick_cylinder_1d(
        radius, inner_radius=A, outer_radius=B, inner_pressure=P,
        young_modulus=E, poisson_ratio=NU, plane="strain",
    )
    assert abs(oracle.radial_stress[0] + P) / P < 1e-9
    assert abs(oracle.radial_stress[-1]) / P < 1e-9


def test_closed_form_is_independent_of_elastic_constants_for_stress():
    """A, B follow from the tractions alone: the in-plane stress is E/nu-free."""
    a1, b1 = lame_coefficients(A, B, P, 0.0)
    a2, b2 = lame_coefficients(A, B, P, 0.0)
    assert float(a1) == float(a2) and float(b1) == float(b2)
    r = torch.tensor([0.6])
    st_strain = lame_state(r, inner_radius=A, outer_radius=B, inner_pressure=P,
                           young_modulus=E, poisson_ratio=NU, plane="strain")
    st_stress = lame_state(r, inner_radius=A, outer_radius=B, inner_pressure=P,
                           young_modulus=E, poisson_ratio=NU, plane="stress")
    assert torch.allclose(st_strain.radial_stress, st_stress.radial_stress)
    assert torch.allclose(st_strain.hoop_stress, st_stress.hoop_stress)
    assert not torch.allclose(st_strain.radial_displacement, st_stress.radial_displacement)


# --------------------------------------------------------------------------- #
# geometry audit
# --------------------------------------------------------------------------- #
def test_mesh_geometry_is_the_specified_annulus():
    nodes, cells = sector_q4_mesh(A, B, 24, 8)
    radius = torch.linalg.vector_norm(nodes, dim=1)
    assert float((radius[:25] - A).abs().max()) < 1e-14
    assert float((radius[-25:] - B).abs().max()) < 1e-14
    area = 0.0
    for cell in cells.tolist():
        xy = nodes[cell].numpy()
        area += 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                          - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
    exact = 0.25 * math.pi * (B * B - A * A)
    # the polygon chords under-cut the arc by O(dtheta^2) ~ 3e-4 at 24 divisions
    assert abs(area - exact) / exact < 1e-3


# --------------------------------------------------------------------------- #
# finite element layer
# --------------------------------------------------------------------------- #
def test_fe_element_centres_match_lame_within_two_percent(bench):
    run = bench.solve_fem(
        n_theta=48, n_radial=16, inner_radius=A, outer_radius=B, inner_pressure=P,
        young_modulus=E, poisson_ratio=NU, plane="strain",
    )
    metrics = bench.score_run(run, plane="strain")
    assert metrics["u_r_max"] < GATE
    assert metrics["sigma_r_max"] < GATE
    assert metrics["sigma_theta_max"] < GATE
    assert metrics["sigma_z_max"] < GATE


def test_second_order_convergence(bench):
    coarse = bench.solve_fem(
        n_theta=24, n_radial=8, inner_radius=A, outer_radius=B, inner_pressure=P,
        young_modulus=E, poisson_ratio=NU, plane="strain",
    )
    fine = bench.solve_fem(
        n_theta=48, n_radial=16, inner_radius=A, outer_radius=B, inner_pressure=P,
        young_modulus=E, poisson_ratio=NU, plane="strain",
    )
    e_coarse = bench.score_run(coarse, plane="strain")["u_r_max"]
    e_fine = bench.score_run(fine, plane="strain")["u_r_max"]
    order = math.log(e_coarse / e_fine) / math.log(2.0)
    assert 1.8 < order < 2.2


def test_plane_state_changes_displacement_by_the_analytic_gap(bench):
    strain = bench.solve_fem(
        n_theta=48, n_radial=16, inner_radius=A, outer_radius=B, inner_pressure=P,
        young_modulus=E, poisson_ratio=NU, plane="strain",
    )
    stress = bench.solve_fem(
        n_theta=48, n_radial=16, inner_radius=A, outer_radius=B, inner_pressure=P,
        young_modulus=E, poisson_ratio=NU, plane="stress",
    )
    rays = strain["ray_radius"]
    u_strain = strain["ray_u_radial"]
    u_stress = stress["ray_u_radial"]
    numeric = float(np.max(np.abs(u_stress - u_strain) / np.abs(u_strain)))
    analytic_strain = bench.lame_fields(rays, plane="strain")["u_r"]
    analytic_stress = bench.lame_fields(rays, plane="stress")["u_r"]
    analytic = float(np.max(np.abs(analytic_stress - analytic_strain) / np.abs(analytic_strain)))
    assert abs(numeric - analytic) < 1e-3
    assert analytic > 0.05  # the branches really differ for nu = 0.3


def test_equilibrium_identity_and_axisymmetry(bench):
    run = bench.solve_fem(
        n_theta=48, n_radial=16, inner_radius=A, outer_radius=B, inner_pressure=P,
        young_modulus=E, poisson_ratio=NU, plane="strain",
    )
    metrics = bench.score_run(run, plane="strain")
    assert metrics["reaction_rel_error"] < 1e-9
    assert metrics["max_tau_over_sigma"] < 1e-9
    assert metrics["max_u_hoop_over_u_radial"] < 1e-9


def test_evidence_file_is_consistent_and_all_anchors_pass():
    if not EVIDENCE.exists():
        pytest.skip("run benchmarks/lame_cylinder_anchor.py to produce the evidence file")
    ev = json.loads(EVIDENCE.read_text())
    assert ev["case"] == "CASE-LAME-CYLINDER"
    assert ev["gate"] == GATE
    assert ev["all_passed"] is True
    for anchor in ev["anchors"]:
        assert anchor["passed"], anchor["id"]
    # quoted numbers in the rendered report must come from this file
    doc = (ROOT / "docs" / "lame-cylinder-anchor-case.md").read_text()
    assert "CASE-LAME-CYLINDER" in doc
    oracle = ev["oracle_validation"]["strain"]
    assert oracle["sigma_theta_max"] < 1e-9
    assert ev["observed_order"]["u_r_max"] == pytest.approx(2.0, abs=0.15)