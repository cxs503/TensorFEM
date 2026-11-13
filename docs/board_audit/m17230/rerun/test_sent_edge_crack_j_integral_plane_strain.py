"""Plane-strain verification of the SENT J-integral path (ledger acceptance).

The ledger acceptance for CASE-FEM-JINTEGRAL-EDGE-CRACK grades the normalised
``K_I / (sigma sqrt(pi a))`` against Tada's finite-width correction at
``a/W = 0.3`` in **plane strain**.  The original anchor run (m16291) covered
plane stress only; this file pins the plane-strain invariants under ``pytest``
while the graded ladder lives in
``benchmarks/verified/fem_sent_j_integral/run.py --stage plane_strain``.

Everything except the medium-mesh acceptance test runs on the coarse fan mesh
so the module is exercised in a few seconds.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

from tensorfem.edge_crack_sent import (
    j_integral_subring,
    sent_k_from_j,
    sent_k_reference,
    sent_t6_mesh,
    solve_sent,
    tada_sent_shape_factor,
)
from tensorfem.fatigue.crack_tip_quarter_point import mode_i_k_from_j

WIDTH = 0.10
HEIGHT = 0.20
SIGMA = 100e6
YOUNG = 210e9
NU = 0.3
ALPHA = 0.3  # ledger acceptance point


@pytest.fixture(scope="module")
def coarse_mesh():
    return sent_t6_mesh(
        WIDTH, ALPHA * WIDTH, HEIGHT, n_sectors=24, first_ring=0.08, growth=1.6
    )


@pytest.fixture(scope="module")
def coarse_solve(coarse_mesh):
    return solve_sent(
        coarse_mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU, plane="strain"
    )


def _canonical_station(mesh):
    return min(
        range(1, len(mesh.stations) - 1),
        key=lambda idx: abs(mesh.stations[idx] - 0.15),
    )


# --------------------------------------------------------------------------- #
# handbook reference and the E' convention
# --------------------------------------------------------------------------- #
def test_tada_factor_at_ledger_acceptance_point():
    """F(0.3) = 1.65992 from the Tada/Paris/Irwin polynomial (handbook table 1.66)."""
    assert tada_sent_shape_factor(ALPHA) == pytest.approx(1.65992, abs=5e-5)


def test_effective_modulus_convention():
    """K^2/J must be E (plane stress) and E/(1-nu^2) (plane strain)."""
    j = 1.0e-3
    k_stress = mode_i_k_from_j(j, YOUNG, NU, plane="stress")
    k_strain = mode_i_k_from_j(j, YOUNG, NU, plane="strain")
    assert k_stress**2 / j == pytest.approx(YOUNG, rel=1e-12)
    assert k_strain**2 / j == pytest.approx(YOUNG / (1.0 - NU**2), rel=1e-12)
    assert k_strain / k_stress == pytest.approx(1.0 / math.sqrt(1.0 - NU**2), rel=1e-12)


def test_sent_k_wrapper_delegates_to_the_shipped_helper():
    j = 2.5e-3
    assert sent_k_from_j(j, YOUNG, NU, plane="strain") == pytest.approx(
        mode_i_k_from_j(j, YOUNG, NU, plane="strain"), rel=1e-14
    )


# --------------------------------------------------------------------------- #
# discrete invariants of the plane-strain solve
# --------------------------------------------------------------------------- #
def test_plane_strain_equilibrium_closes(coarse_solve):
    assert coarse_solve.reaction_ratio == pytest.approx(1.0, abs=1e-9)


def test_plane_strain_applied_load_is_sigma_times_width(coarse_solve):
    assert coarse_solve.applied_load_n == pytest.approx(SIGMA * WIDTH, rel=1e-12)


def test_plane_strain_k_matches_tada_on_the_coarse_mesh(coarse_mesh, coarse_solve):
    """Coarse mesh already inside the 5 % ledger tolerance at a/W = 0.3."""
    station = _canonical_station(coarse_mesh)
    j = j_integral_subring(
        coarse_mesh, coarse_solve.displacement, station,
        young_pa=YOUNG, poisson=NU, plane="strain",
    )
    k = sent_k_from_j(j["j_j_m2"], YOUNG, NU, plane="strain")
    reference = sent_k_reference(SIGMA, ALPHA * WIDTH, WIDTH)
    assert abs((k - reference) / reference) < 0.05


def test_plane_strain_path_independence_on_the_inner_fan(coarse_mesh, coarse_solve):
    values = [
        j_integral_subring(
            coarse_mesh, coarse_solve.displacement, station,
            young_pa=YOUNG, poisson=NU, plane="strain",
        )["j_j_m2"]
        for station in range(2, len(coarse_mesh.stations) - 1)
        if coarse_mesh.stations[station] <= 0.35
    ]
    assert min(values) > 0.0
    assert (max(values) - min(values)) / max(values) < 0.03


def test_plane_strain_medium_mesh_meets_the_platform_gate():
    """The graded acceptance point: medium mesh, a/W = 0.3, plane strain, < 3 %."""
    a = ALPHA * WIDTH
    mesh = sent_t6_mesh(WIDTH, a, HEIGHT, n_sectors=48, first_ring=0.05, growth=1.45)
    solved = solve_sent(mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU, plane="strain")
    station = _canonical_station(mesh)
    j = j_integral_subring(
        mesh, solved.displacement, station, young_pa=YOUNG, poisson=NU, plane="strain"
    )
    k = sent_k_from_j(j["j_j_m2"], YOUNG, NU, plane="strain")
    reference = sent_k_reference(SIGMA, a, WIDTH)
    assert abs((k - reference) / reference) < 0.03


def test_plane_strain_k_is_not_the_plane_stress_value(coarse_mesh):
    """The constitutive path must actually change: J differs between the two planes."""
    stress = solve_sent(coarse_mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU, plane="stress")
    strain = solve_sent(coarse_mesh, sigma_pa=SIGMA, young_pa=YOUNG, poisson=NU, plane="strain")
    station = _canonical_station(coarse_mesh)
    j_stress = j_integral_subring(
        coarse_mesh, stress.displacement, station, young_pa=YOUNG, poisson=NU, plane="stress"
    )["j_j_m2"]
    j_strain = j_integral_subring(
        coarse_mesh, strain.displacement, station, young_pa=YOUNG, poisson=NU, plane="strain"
    )["j_j_m2"]
    assert j_stress > 0.0 and j_strain > 0.0
    assert j_strain != pytest.approx(j_stress, rel=1e-6)


# --------------------------------------------------------------------------- #
# product entry: run_sent_j_qualification (ledger acceptance, library level)
# --------------------------------------------------------------------------- #
def test_qualification_entry_passes_the_ledger_point_on_the_qualification_mesh():
    """The product entry itself (not a script) must return a passing bundle."""
    from tensorfem.edge_crack_sent import run_sent_j_qualification

    report = run_sent_j_qualification()
    assert report["anchor"] == "CASE-FEM-JINTEGRAL-EDGE-CRACK"
    assert report["configuration"]["a_over_w"] == pytest.approx(ALPHA)
    assert report["configuration"]["plane"] == "strain"
    k_row = next(row for row in report["evidence"] if row["id"] == "mode_i_k")
    # ledger band 5 %, platform band 3 %: the qualification mesh is inside both
    assert k_row["relative_error"] < 0.03
    assert k_row["passed"] is True
    assert abs(k_row["actual"] / k_row["oracle"] - 1.0) == pytest.approx(
        k_row["relative_error"], rel=1e-12
    )
    assert report["path_independence"]["passed"] is True
    assert report["equilibrium"]["reaction_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert report["passed"] is True


def test_qualification_entry_rejects_an_out_of_range_crack_ratio():
    from tensorfem.edge_crack_sent import run_sent_j_qualification

    with pytest.raises(ValueError):
        run_sent_j_qualification(alpha=0.7)  # outside the Tada fit range
    with pytest.raises(ValueError):
        run_sent_j_qualification(plane="axisymmetric")


def test_qualification_is_wired_into_the_marine_benchmark_suite():
    """L3 integration: the ship-structure suite imports and runs the entry."""
    from tensorfem import marine_benchmark_suite

    assert hasattr(marine_benchmark_suite, "run_sent_j_qualification")
    source = Path(marine_benchmark_suite.__file__).read_text(encoding="utf-8")
    assert "run_sent_j_qualification()" in source
    assert "marine.fracture.sent_j." in source


def test_cli_dispatches_the_sent_j_integral_entry(monkeypatch, capsys):
    """`tensorfem sent-j-integral` is an exit-code-carrying entry point."""
    from tensorfem import cli

    stub = {
        "anchor": "CASE-FEM-JINTEGRAL-EDGE-CRACK",
        "evidence": [{"id": "mode_i_k", "relative_error": 0.0197, "passed": True}],
        "passed": True,
    }
    monkeypatch.setattr(
        "tensorfem.edge_crack_sent.run_sent_j_qualification", lambda: dict(stub)
    )
    monkeypatch.setattr(sys, "argv", ["tensorfem", "sent-j-integral"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["anchor"] == "CASE-FEM-JINTEGRAL-EDGE-CRACK"

    monkeypatch.setattr(
        "tensorfem.edge_crack_sent.run_sent_j_qualification",
        lambda: {**stub, "passed": False},
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
