"""Tests for the elasto-plastic stiffened-panel unit cell and its Riks driver."""
from __future__ import annotations


import pytest
import torch

from tensorfem.arc_length import ArcLengthProblem, solve_arc_length
from tensorfem.stiffened_panel_ultimate import StiffenedPanel
from tensorfem.stiffened_panel_unit_fe import (
    assemble_panel_unit,
    build_panel_unit,
    trace_panel_unit_path,
    trace_summary,
)

ELASTIC = StiffenedPanel(span=0.8, spacing=0.8, plate_thickness=0.012,
                         web_height=0.18, web_thickness=0.009,
                         flange_width=0.080, flange_thickness=0.012,
                         yield_stress=1.0e12)
ICE_BELT = StiffenedPanel(span=2.4, spacing=0.800, plate_thickness=0.020,
                          web_height=0.20, web_thickness=0.010,
                          flange_width=0.080, flange_thickness=0.012)


def test_unit_cell_geometry_and_reference_load() -> None:
    case = build_panel_unit(ICE_BELT, nx=4, ny=2, layers=2)
    panel = ICE_BELT
    expected_area = (panel.plate_thickness * panel.spacing / 2.0
                     + panel.web_height * panel.web_thickness
                     + 0.5 * panel.flange_width * panel.flange_thickness)
    assert case.reference_area == pytest.approx(expected_area, rel=1e-12)
    assert case.element_counts == (8, 4, 4)
    assert case.n_dofs == 6 * len(case.nodes)
    # Unit average stress: the nodal axial forces sum to the reference area.
    assert float(case.compression_load[0::6].sum()) == pytest.approx(-expected_area,
                                                                    rel=1e-12)
    assert int(len(case.free_dofs())) < case.n_dofs


def test_unit_cell_pressure_load_acts_on_the_plate_only() -> None:
    case = build_panel_unit(ICE_BELT, nx=4, ny=2, layers=2, pressure_ratio=0.0)
    # Only plate nodes carry a transverse nodal force from the pressure field.
    count = 3 * (5 * 3)
    transverse = case.pressure_load[2::6]
    assert float(transverse[:count].abs().sum()) > 0.0
    assert float(transverse[count:].abs().sum()) == pytest.approx(0.0, abs=1e-18)


def test_assembled_tangent_is_symmetric_and_residual_free_at_zero() -> None:
    case = build_panel_unit(ICE_BELT, nx=3, ny=2, layers=2)
    zero = torch.zeros(case.n_dofs, dtype=torch.float64)
    internal, tangent, _stress, _trial = assemble_panel_unit(case, zero, case.virgin_state())
    assert float(internal.abs().max()) == pytest.approx(0.0, abs=1e-12)
    assert float((tangent - tangent.T).abs().max()) < 1e-6 * float(tangent.abs().max())


def test_elastic_axial_stiffness_matches_the_gross_section() -> None:
    case = build_panel_unit(ELASTIC, nx=2, ny=1, layers=1)
    free = case.free_dofs()
    zero = torch.zeros(case.n_dofs, dtype=torch.float64)
    _internal, tangent, _stress, _state = assemble_panel_unit(case, zero, case.virgin_state())
    solve = torch.linalg.solve(tangent[free][:, free], case.reference_load[free])
    displacement = torch.zeros(case.n_dofs, dtype=torch.float64)
    displacement[free] = solve
    stress = 0.01 * ELASTIC.yield_stress
    shortening = -float(displacement[case.loaded_edge_dofs].mean()) * stress
    uniform = stress * ELASTIC.span / ELASTIC.young
    # The uniform traction on an eccentric section also drives bending, so the
    # mean shortening is compared with the bar solution within 40 %.
    assert shortening == pytest.approx(uniform, rel=0.4)


def test_stateful_driver_matches_library_arc_length_in_the_elastic_limit() -> None:
    case = build_panel_unit(ELASTIC, nx=2, ny=1, layers=1)
    free = case.free_dofs()
    ds = 1e-4
    weight = ds / (0.02 * 2.35e8)
    trace = trace_panel_unit_path(case, steps=4, step_size=ds, load_scale=weight,
                                  tolerance=1e-6, max_iterations=10)
    state = case.virgin_state()

    def internal_tangent(vector: torch.Tensor):
        displacement = torch.zeros(case.n_dofs, dtype=torch.float64)
        displacement[free] = vector
        internal, tangent, _stress, _trial = assemble_panel_unit(case, displacement, state)
        return internal[free], tangent[free][:, free]

    problem = ArcLengthProblem(internal_tangent, case.reference_load[free])
    reference = solve_arc_length(problem, torch.zeros(len(free), dtype=torch.float64),
                                 steps=4, step_size=ds, load_scale=weight,
                                 tolerance=1e-6, max_iterations=10, minimum_step=1e-9)
    assert trace.converged and reference.converged
    assert len(trace.points) == len(reference.points) == 4
    for mine, theirs in zip(trace.points, reference.points):
        assert mine.load_factor == pytest.approx(theirs.load_factor, rel=2e-3)
        # solve_arc_length works in the reduced (free) space
        full = torch.zeros(case.n_dofs, dtype=torch.float64)
        full[free] = theirs.displacement
        assert mine.end_shortening == pytest.approx(
            -float(full[case.loaded_edge_dofs].mean()), rel=2e-3)


@pytest.mark.slow
def test_plastic_collapse_path_has_a_limit_point() -> None:
    panel = StiffenedPanel(span=0.8, spacing=0.8, plate_thickness=0.012,
                           web_height=0.18, web_thickness=0.009,
                           flange_width=0.080, flange_thickness=0.012)
    case = build_panel_unit(panel, nx=2, ny=2, layers=2)
    ds = 2.4e-4
    trace = trace_panel_unit_path(case, steps=14, step_size=ds,
                                  load_scale=ds / (0.05 * panel.yield_stress),
                                  tolerance=2e-3, max_iterations=8)
    summary = trace_summary(case, trace)
    assert summary["points_completed"] >= 8
    stresses = [row["stress_over_yield"] for row in summary["points"]]
    assert max(stresses) > 0.2
    assert all(1e-12 <= value < 3.0 for value in stresses)
    assert summary["ultimate_over_yield"] == pytest.approx(max(stresses), rel=1e-12)


def test_trace_summary_reports_the_peak_consistently() -> None:
    case = build_panel_unit(ELASTIC, nx=2, ny=1, layers=1)
    ds = 1e-4
    trace = trace_panel_unit_path(case, steps=3, step_size=ds,
                                  load_scale=ds / (0.02 * 2.35e8),
                                  tolerance=1e-6, max_iterations=8)
    summary = trace_summary(case, trace)
    assert summary["ultimate_stress"] == max(row["average_stress"] for row in summary["points"])
    assert summary["closed_form_panel_ultimate"] == case.panel.panel_ultimate_stress
    assert summary["points_completed"] == len(summary["points"])