"""Tests for the linearised stiffened-panel buckling operator and its anchor.

The anchor's own evidence (``benchmarks/verified/fem_panel_buckling/result.json``
and ``gates_verified.json``) is re-derived here: the closed-form oracle is
recomputed inside the test, the finite-element values are recomputed on meshes
small enough for a unit test and compared with the cached evidence, and every
graded gate is checked against the number it claims to grade.  A drifted formula
or a hand-edited evidence file therefore fails the suite instead of passing
silently.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

torch.set_default_dtype(torch.float64)

from tensorfem.stiffened_panel_buckling import (  # noqa: E402
    assembly_buckling,
    bare_plate_buckling,
    classical_plate_buckling_stress,
    panel_buckling,
    panel_cell_mesh,
    plate_cell_mesh,
)
from tensorfem.stiffened_panel_ultimate import StiffenedPanel  # noqa: E402

ANCHOR = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "fem_panel_buckling"

SPAN, SPACING, T = 0.800, 0.400, 0.006
YOUNG, POISSON, YIELD = 206.0e9, 0.3, 355.0e6


def campaign_panel() -> StiffenedPanel:
    return StiffenedPanel(span=SPAN, spacing=SPACING, plate_thickness=T,
                          web_height=0.150, web_thickness=0.010,
                          flange_width=0.050, flange_thickness=0.010,
                          young=YOUNG, poisson=POISSON, yield_stress=YIELD)


# --------------------------------------------------------------------------
# closed form
# --------------------------------------------------------------------------
def test_classical_plate_buckling_reproduces_the_plate_formula():
    result = classical_plate_buckling_stress(SPAN, SPACING, T, YOUNG, POISSON)
    rigidity = YOUNG * T ** 3 / (12.0 * (1.0 - POISSON ** 2))
    expected = 4.0 * math.pi ** 2 * rigidity / (SPACING ** 2 * T)
    assert result["buckling_coefficient"] == pytest.approx(4.0, rel=1e-12)
    assert result["half_waves"] == 2                 # a/b = 2 -> two half waves
    assert result["sigma_cr"] == pytest.approx(expected, rel=1e-12)
    assert result["sigma_cr"] / 1e6 == pytest.approx(167.5664, rel=1e-5)


def test_classical_plate_buckling_rejects_bad_input():
    with pytest.raises(ValueError):
        classical_plate_buckling_stress(-1.0, SPACING, T, YOUNG, POISSON)
    with pytest.raises(ValueError):
        classical_plate_buckling_stress(SPAN, SPACING, T, YOUNG, 0.5)


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------
def test_plate_cell_rejects_unknown_edge_condition():
    with pytest.raises(ValueError):
        plate_cell_mesh(SPAN, SPACING, T, 4, 2, edge_rotation="spring")


def test_bare_plate_converges_from_above_to_the_closed_form():
    reference = classical_plate_buckling_stress(SPAN, SPACING, T, YOUNG, POISSON)["sigma_cr"]
    coarse = bare_plate_buckling(SPAN, SPACING, T, YOUNG, POISSON, 10, 5, modes=1)
    fine = bare_plate_buckling(SPAN, SPACING, T, YOUNG, POISSON, 20, 10, modes=1)
    assert coarse.critical_stress > fine.critical_stress > reference
    assert abs(fine.critical_stress / reference - 1.0) < 0.05
    assert abs(coarse.critical_stress / reference - 1.0) < 0.10
    # a converged linearised buckling load is accurate to far better than the
    # coarse-mesh discretisation error
    assert float(fine.residuals[0]) < 1e-8


def test_bare_plate_first_mode_has_two_half_waves_along_the_load():
    """a/b = 2 means the classical mode is m = 2, i.e. one interior nodal line."""
    result = bare_plate_buckling(SPAN, SPACING, T, YOUNG, POISSON, 20, 10, modes=1)
    nodes = result.nodes
    nx = 20
    row = 5 * (nx + 1)                              # y = b/2
    field = result.modes[0][row:row + nx + 1]
    # count sign changes of the nodal field, skipping the nodal line itself
    signs = [1 if value > 1e-6 * abs(field).max()
             else (-1 if value < -1e-6 * abs(field).max() else 0) for value in field]
    nonzero = [value for value in signs if value != 0]
    assert nonzero[0] != nonzero[-1]
    crossings = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a != b)
    assert crossings == 1


def test_panel_cell_buckles_above_the_idealised_plate():
    panel = campaign_panel()
    result = panel_buckling(panel, 16, 4, modes=2)
    unit = math.pi ** 2 * YOUNG * T ** 3 / (12.0 * (1 - POISSON ** 2)) / (SPACING ** 2 * T)
    assert result.critical_stress > 4.0 * unit       # the stiffener restrains the plate edge
    assert result.critical_stress < YIELD            # but the cell still buckles below yield
    assert float(result.residuals[0]) < 1e-8
    assert result.n_elements == 16 * 4 + 16 + 16


def test_panel_cell_mesh_matches_the_nonlinear_unit_cell():
    """The buckling mesh must be the same cell the nonlinear path integrates."""
    from tensorfem.stiffened_panel_unit_fe import build_panel_unit

    panel = campaign_panel()
    buckling_mesh = panel_cell_mesh(panel, 4, 2)
    case = build_panel_unit(panel, nx=4, ny=2, layers=1, plate_imperfection_ratio=0.0,
                            column_imperfection_ratio=0.0)
    assert torch.allclose(buckling_mesh.nodes, case.nodes)
    assert len(buckling_mesh.fixed) == len(case.fixed_dofs)
    assert torch.equal(buckling_mesh.fixed.sort().values, case.fixed_dofs.sort().values)


# --------------------------------------------------------------------------
# evidence discipline
# --------------------------------------------------------------------------
@pytest.mark.skipif(not (ANCHOR / "result.json").exists(), reason="anchor not run")
def test_anchor_gates_match_the_evidence():
    result = json.loads((ANCHOR / "result.json").read_text())
    gates = json.loads((ANCHOR / "gates_verified.json").read_text())
    by_id = {gate["id"]: gate for gate in gates["gates"]}

    plate = {f"{row['nx']}x{row['ny']}": row for row in result["plate_rows"]}
    oracle = result["plate_oracle"]["sigma_cr_pa"]

    finest = by_id["plate_finest_vs_classical"]
    finest_row = plate[f"{finest['note'].split('mesh ')[1].split(':')[0]}"]
    assert finest["value"] == pytest.approx(abs(finest_row["relative_error"]), rel=1e-12)
    assert finest_row["sigma_cr_pa"] / oracle - 1.0 == pytest.approx(
        finest_row["relative_error"], rel=1e-12)

    gci = by_id["plate_gci_fine"]
    assert gci["value"] == pytest.approx(result["plate_ladder_gci"]["gci_fine"], rel=1e-12)
    assert gci["value"] <= gates["criteria"]["f_mesh_convergence_gci"]

    panel = result["panel_rows"][-1]
    unit = math.pi ** 2 * YOUNG * T ** 3 / (12.0 * (1 - POISSON ** 2)) / (SPACING ** 2 * T)
    assert panel["sigma_cr_pa"] / unit == pytest.approx(panel["k_effective"], rel=1e-9)
    assert by_id["panel_at_least_simply_supported"]["value"] > 0.0
    assert by_id["panel_at_most_clamped"]["value"] > 0.0


@pytest.mark.skipif(not (ANCHOR / "result.json").exists(), reason="anchor not run")
def test_anchor_records_the_geometrically_linear_limitation():
    result = json.loads((ANCHOR / "result.json").read_text())
    path = result["linear_path"]
    # the small-strain layered shell cannot buckle: the trace is exactly linear
    assert path["deflection_proportionality_residual"] < 1e-6
    assert path["max_plastic_strain"] == 0.0
    assert path["applied_over_classical_buckling"] > 0.5
    assert path["has_peak"] is False
    criteria = json.loads((ANCHOR / "gates_verified.json").read_text())["campaign_criteria"]
    assert criteria["(c)_postbuckling_path_vs_experiment"]["status"] == "FAIL"
    assert criteria["(b)_ultimate_vs_Paik2007_experiment"]["status"] == "WITHHELD"


@pytest.mark.skipif(not (ANCHOR / "_cache" / "plate_20x10.json").exists(),
                    reason="anchor cache not present")
def test_cached_plate_level_reproduces_the_operator():
    cached = json.loads((ANCHOR / "_cache" / "plate_20x10.json").read_text())
    recomputed = bare_plate_buckling(SPAN, SPACING, T, YOUNG, POISSON, 20, 10, modes=1)
    assert recomputed.critical_stress == pytest.approx(cached["sigma_cr_pa"], rel=1e-10)


# --------------------------------------------------------------------------
# the limitation itself (documents the missing geometric stiffness)
# --------------------------------------------------------------------------
def test_layered_panel_path_is_geometrically_linear():
    """Characterisation test for the limitation this anchor reports.

    ``stiffened_panel_unit_fe`` is a small-strain layered shell with no
    geometric stiffness, so its tangent cannot lose stability: the transverse
    deflection stays proportional to the load through the classical buckling
    stress.  If the element ever gains a geometric term this test fails and the
    anchor's WITHHELD ultimate-strength criteria must be revisited.
    """
    import numpy as np
    from tensorfem.stiffened_panel_unit_fe import (
        build_panel_unit, trace_panel_unit_path, trace_summary)

    case = build_panel_unit(campaign_panel(), nx=4, ny=2, layers=1,
                            plate_imperfection_ratio=0.1)
    step = 6.0e-4
    trace = trace_panel_unit_path(case, steps=3, step_size=step,
                                  load_scale=step / (0.1 * YIELD), max_iterations=8)
    summary = trace_summary(case, trace)
    loads = np.array([p["average_stress"] for p in summary["points"]])
    deflections = np.array([p["transverse_deflection"] for p in summary["points"]])
    assert len(loads) == 3
    slope = deflections / loads
    assert np.allclose(slope, slope[0], rtol=1e-8)     # exactly proportional
    assert max(p["maximum_plastic_strain"] for p in summary["points"]) == 0.0
    assert summary["has_peak"] is False