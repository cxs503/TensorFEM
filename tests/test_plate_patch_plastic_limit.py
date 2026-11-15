"""Tests for the plate-panel patch-load elastoplastic capacity module.

Anchor ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET``: (1) elastoplastic plate with
isotropic hardening and a low-temperature flow stress, (2) permanent-deflection
post-processing, (3) consistency regression against the elastic solution on the
same mesh.

The closed-form references (Navier series, yield-line upper bound, strip lower
bound) are literature-anchored and tested first; the finite element tests then
pin the elastoplastic path against those references *and* against the same-mesh
elastic response, because the layered Shell4 element carries a known elastic
bias that is quantified in the module docstring and in the case document.
"""
import math
import pytest
import torch

from tensorfem.plate_patch_plastic_limit import (
    STEEL_GRADES,
    analytic_first_yield_pressure,
    elastic_patch_plate,
    flow_stress_table,
    low_temperature_flow_stress,
    patch_plate_mesh,
    plastic_moment,
    run_patch_plastic_path,
    strip_lower_bound,
    surface_extrapolated_mises,
    yield_line_patch_upper_bound,
    _patch_mechanism_volume_ratio,
    _patch_mechanism_volume_ratio_quadrature,
)
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4State,
    solve_layered_shell4_increment,
)

A = B = 1.0
T = 0.02
E, NU, FY = 210e9, 0.3, 355e6
PATCH = (0.5 * A, 0.5 * B)
M_P = plastic_moment(T, FY)


# ---------------------------------------------------------------------------
# closed-form references
# ---------------------------------------------------------------------------

def test_plastic_moment_is_quarter_sigma_t_squared():
    assert plastic_moment(0.02, 355e6) == pytest.approx(355e6 * 0.02 ** 2 / 4)
    with pytest.raises(ValueError):
        plastic_moment(0.0, 355e6)


def test_yield_line_uniform_load_recovers_classical_24mp_over_a2():
    result = yield_line_patch_upper_bound(A, B, T, FY, (A, B))
    assert result["pressure"] == pytest.approx(24 * M_P / A ** 2, rel=1e-12)
    assert result["mechanism_volume"] == pytest.approx(A * B / 3, rel=1e-12)


def test_yield_line_closed_form_matches_quadrature():
    for patch in ((A, B), (0.5 * A, 0.5 * B), (0.25 * A, 0.5 * B), (0.1 * A, 0.1 * B)):
        exact = _patch_mechanism_volume_ratio(A, B, patch)
        numeric = _patch_mechanism_volume_ratio_quadrature(A, B, patch, n=200)
        assert exact == pytest.approx(numeric, rel=2e-2), patch


def test_patch_upper_bound_exceeds_lower_bound_and_grows_with_concentration():
    previous = None
    for fraction in (1.0, 0.5, 0.25):
        patch = (fraction * A, fraction * B)
        upper = yield_line_patch_upper_bound(A, B, T, FY, patch)["pressure"]
        lower = strip_lower_bound(A, B, T, FY, patch)["pressure"]
        assert upper > lower > 0.0
        if previous is not None:
            assert upper > previous
        previous = upper


def test_navier_uniform_load_matches_timoshenko_coefficients():
    q = 1e5
    result = elastic_patch_plate(A, B, T, E, NU, (A, B), q)
    D = E * T ** 3 / (12 * (1 - NU ** 2))
    assert result["deflection"][0] == pytest.approx(0.00406 * q * A ** 4 / D, rel=1e-2)
    assert result["Mx"][0] == pytest.approx(0.0479 * q * A ** 2, rel=1e-2)


def test_navier_patch_load_tends_to_the_point_load_limit():
    """Small patch -> Timoshenko's concentrated-load coefficient 0.0116."""
    D = E * T ** 3 / (12 * (1 - NU ** 2))
    fraction = 0.05
    pressure = 1.0 / (fraction * A * fraction * B)     # unit total load
    w = elastic_patch_plate(A, B, T, E, NU, (fraction * A, fraction * B), pressure)["deflection"][0]
    assert w * D / (1.0 * A ** 2) == pytest.approx(0.0116, rel=2e-2)


def test_first_yield_pressure_scales_with_strength_and_thickness_squared():
    base = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=9, terms=21)["pressure"]
    strong = analytic_first_yield_pressure(A, B, T, E, NU, 2 * FY, PATCH, grid=9, terms=21)["pressure"]
    thick = analytic_first_yield_pressure(A, B, 1.5 * T, E, NU, FY, PATCH, grid=9, terms=21)["pressure"]
    assert strong == pytest.approx(2 * base, rel=1e-9)
    assert thick == pytest.approx(base * 1.5 ** 2, rel=1e-9)   # sigma ~ p/t^2


# ---------------------------------------------------------------------------
# low-temperature flow stress
# ---------------------------------------------------------------------------

def test_flow_stress_reference_and_amplification():
    room = low_temperature_flow_stress("DH36", 20.0)
    assert room.yield_stress == pytest.approx(355e6)
    assert room.ductile and room.amplification == pytest.approx(1.0)
    arctic = low_temperature_flow_stress("DH36", -40.0)
    assert arctic.yield_stress > room.yield_stress
    assert arctic.amplification == pytest.approx(1.08, abs=0.02)
    monotone = [low_temperature_flow_stress("DH36", T_).yield_stress
                for T_ in (20.0, 0.0, -20.0, -40.0, -60.0)]
    assert all(x < y for x, y in zip(monotone, monotone[1:]))


def test_flow_stress_withholds_plasticity_below_dbtt():
    brittle = low_temperature_flow_stress("DH36", -80.0)
    assert brittle.ductile is False
    assert brittle.yield_stress == pytest.approx(355e6)   # no extrapolation
    assert "DBTT" in brittle.note
    with pytest.raises(KeyError):
        low_temperature_flow_stress("NOT-A-GRADE", 0.0)
    assert set(STEEL_GRADES) >= {"AH36", "DH36", "EH36", "FH36"}


def test_flow_stress_table_is_monotone():
    table = flow_stress_table("EH36", [20.0, -20.0, -40.0])
    assert [row["temperature_c"] for row in table] == [20.0, -20.0, -40.0]
    assert table[0]["yield_stress_MPa"] == pytest.approx(355.0)


# ---------------------------------------------------------------------------
# mesh contract
# ---------------------------------------------------------------------------

def test_mesh_rejects_patch_that_does_not_follow_element_edges():
    with pytest.raises(ValueError):
        patch_plate_mesh(A, B, T, 6, 6, young=E, poisson=NU, yield_stress=FY,
                         hardening=2e9, patch=(0.37 * A, 0.5 * B))
    with pytest.raises(ValueError):
        run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                               hardening=2e9, patch=PATCH, load_factors=[1e5],
                               nx=5, ny=5)
    with pytest.raises(ValueError):
        patch_plate_mesh(A, B, T, 6, 6, young=E, poisson=NU, yield_stress=FY,
                         hardening=2e9, patch=PATCH)


def test_patch_pressure_forces_carry_the_right_total_load():
    model, info = patch_plate_mesh(A, B, T, 8, 8, young=E, poisson=NU,
                                   yield_stress=FY, hardening=2e9, patch=PATCH)
    assert float(info["reference_load"].sum()) == pytest.approx(PATCH[0] * PATCH[1])
    assert int((info["reference_load"] != 0).sum()) == 5 * 5


# ---------------------------------------------------------------------------
# finite element path (single small mesh, shared fixture)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def capacity():
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)
    steps = [factor * py["pressure"] for factor in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8)]
    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                  hardening=2e10, patch=PATCH, load_factors=steps,
                                  nx=8, ny=8, layers=3, unload_steps=3)
    probe = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                   hardening=2e10, patch=PATCH,
                                   load_factors=[0.5 * py["pressure"]],
                                   nx=8, ny=8, layers=3)
    return {"analytic": py, "path": path, "probe": probe}


def test_elastic_branch_scales_linearly_with_the_load(capacity):
    steps = capacity["path"].loading[:3]
    ratios = [s.centre_deflection / elastic_patch_plate(A, B, T, E, NU, PATCH, s.load_factor)["deflection"][0] for s in steps]
    assert max(ratios) - min(ratios) < 5e-3
    assert all(s.max_alpha == 0.0 for s in steps)


def test_same_mesh_elastic_bias_is_quantified_and_bounded(capacity):
    probe = capacity["probe"].loading[0]
    analytic = capacity["analytic"]["sigma_eq_per_pa"]
    fe = probe.max_surface_mises / probe.load_factor
    bias = fe / analytic - 1.0
    assert abs(bias) < 0.03


def test_yield_is_detected_on_the_surface_before_any_layer_yields(capacity):
    path = capacity["path"]
    first = path.first_yield_factor
    assert first is not None
    surface_yielded = [s for s in path.loading if s.max_surface_mises >= FY]
    assert surface_yielded
    assert first <= surface_yielded[0].load_factor
    layer_yielded = [s for s in path.loading if s.max_alpha > 0.0]
    assert layer_yielded
    ratio = layer_yielded[0].load_factor / first
    assert 1.0 < ratio < 1.0 / (1.0 - 1.0 / 3) + 0.35


def test_transition_point_matches_same_mesh_elastic_field_within_5pct(capacity):
    path = capacity["path"]
    probe = capacity["probe"].loading[0]
    fe_sigma_per_pa = probe.max_surface_mises / probe.load_factor
    same_mesh_yield = FY / fe_sigma_per_pa
    error = (path.first_yield_factor - same_mesh_yield) / same_mesh_yield
    assert abs(error) < 0.05


def test_transition_point_raw_error_versus_analytic_yield_load(capacity):
    path = capacity["path"]
    analytic = capacity["analytic"]["pressure"]
    error = (path.first_yield_factor - analytic) / analytic
    assert -0.20 < error < 0.05


def test_permanent_deflection_is_positive_and_unloads_monotonically(capacity):
    path = capacity["path"]
    assert path.permanent_deflection is not None
    assert path.permanent_deflection > 0.0
    assert path.peak_factor > capacity["analytic"]["pressure"]
    residuals = [s.centre_deflection for s in path.unloading]
    assert residuals == sorted(residuals, reverse=True)
    assert residuals[-1] == pytest.approx(path.permanent_deflection)


def test_permanent_set_is_a_fraction_of_the_peak_deflection(capacity):
    path = capacity["path"]
    peak = path.loading[-1].centre_deflection
    assert 0.0 < path.permanent_deflection < peak


def test_surface_extrapolation_is_exact_for_linear_stress():
    thickness, layers = 0.02, 5
    dz = thickness / layers
    z = [(-thickness / 2 + (k + 0.5) * dz) for k in range(layers)]
    stress = torch.zeros((1, 4, layers, 3), dtype=torch.float64)
    for k, zk in enumerate(z):
        stress[0, :, k, 0] = 100e6 * zk / (thickness / 2)
    assert surface_extrapolated_mises(stress, thickness) == pytest.approx(100e6, rel=1e-9)
    with pytest.raises(ValueError):
        surface_extrapolated_mises(torch.zeros((1, 4, 1, 3), dtype=torch.float64), thickness)


def test_collapse_load_matches_the_yield_line_bound_after_layering_correction():
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)
    upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"]
    lower = strip_lower_bound(A, B, T, FY, PATCH)["pressure"]
    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                  hardening=1e6, patch=PATCH,
                                  load_factors=[0.6 * py["pressure"], 1.0 * py["pressure"],
                                                1.4 * py["pressure"], 1.8 * py["pressure"],
                                                2.4 * py["pressure"], 3.0 * py["pressure"]],
                                  nx=8, ny=8, layers=3)
    collapse = path.collapse_factor or path.loading[-1].load_factor
    assert lower < collapse < upper
    assert abs(collapse / upper - 1.0) < 0.20


# ---------------------------------------------------------------------------
# 3-level mesh convergence
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh_sweep():
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)
    upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"]
    lower = strip_lower_bound(A, B, T, FY, PATCH)["pressure"]
    rows = []
    for nx in (4, 8, 12):
        steps = [f * py["pressure"] for f in (0.3, 0.9, 1.5, 1.8)]
        path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                      hardening=2e10, patch=PATCH, load_factors=steps,
                                      nx=nx, ny=nx, layers=3, unload_steps=3)
        probe = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                       hardening=2e10, patch=PATCH,
                                       load_factors=[0.5 * py["pressure"]],
                                       nx=nx, ny=nx, layers=3)
        fe_sigma_per_pa = probe.loading[0].max_surface_mises / probe.loading[0].load_factor
        same_mesh_yield = FY / fe_sigma_per_pa
        rows.append({
            "nx": nx, "first": path.first_yield_factor,
            "first_err_same_mesh": (path.first_yield_factor - same_mesh_yield) / same_mesh_yield,
            "collapse": path.collapse_factor or path.loading[-1].load_factor,
            "permanent": path.permanent_deflection, "peak": path.loading[-1].centre_deflection,
        })
    return {"py": py, "upper": upper, "lower": lower, "rows": rows}


def test_collapse_load_converges_monotonically_to_the_upper_bound(mesh_sweep):
    upper = mesh_sweep["upper"]
    lower = mesh_sweep["lower"]
    collapses = [r["collapse"] for r in mesh_sweep["rows"]]
    assert all(c < upper for c in collapses)
    assert all(c > lower for c in collapses)
    assert all(collapses[i] <= collapses[i + 1] for i in range(len(collapses) - 1))


def test_transition_point_stays_within_5pct_at_every_mesh_level(mesh_sweep):
    for r in mesh_sweep["rows"]:
        assert abs(r["first_err_same_mesh"]) < 0.05


def test_permanent_set_is_bounded_and_converges(mesh_sweep):
    for r in mesh_sweep["rows"]:
        assert 0.0 < r["permanent"] < r["peak"]
    perms = [r["permanent"] for r in mesh_sweep["rows"]]
    assert max(perms) < 1.5 * min(perms)


# ---------------------------------------------------------------------------
# Anchor acceptance tests (GAP-PP-3-ICE-STRUCT-PLASTIC-SET)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def anchor_collapse_fine():
    """Near-ideally plastic collapse at 8x8 mesh for the <10% criterion."""
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)
    upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"]
    lower = strip_lower_bound(A, B, T, FY, PATCH)["pressure"]
    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                  hardening=1e6, patch=PATCH,
                                  load_factors=[0.5 * py["pressure"], 1.0 * py["pressure"],
                                                1.5 * py["pressure"], 2.0 * py["pressure"],
                                                2.5 * py["pressure"], 3.0 * py["pressure"]],
                                  nx=8, ny=8, layers=3)
    return {"py": py, "upper": upper, "lower": lower, "path": path}


def test_anchor_collapse_within_10pct_of_upper_bound(anchor_collapse_fine):
    """Anchor: plastic limit load vs yield-line upper bound error < 10%."""
    upper = anchor_collapse_fine["upper"]
    lower = anchor_collapse_fine["lower"]
    path = anchor_collapse_fine["path"]
    collapse = path.collapse_factor or path.loading[-1].load_factor
    assert lower < collapse < upper, f"collapse {collapse:.0f} not in [{lower:.0f}, {upper:.0f}]"
    error = abs(collapse / upper - 1.0)
    assert error < 0.10, f"collapse error vs upper = {error:.3f} (>10%)"


def test_anchor_transition_within_5pct_at_fine_mesh(anchor_collapse_fine):
    """Anchor: elastic->plastic transition vs same-mesh elastic field < 5%."""
    path = anchor_collapse_fine["path"]
    py = anchor_collapse_fine["py"]
    probe = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                   hardening=1e6, patch=PATCH,
                                   load_factors=[0.25 * py["pressure"]],
                                   nx=8, ny=8, layers=3)
    fe_sigma_per_pa = probe.loading[0].max_surface_mises / probe.loading[0].load_factor
    same_mesh_yield = FY / fe_sigma_per_pa
    error = abs(path.first_yield_factor - same_mesh_yield) / same_mesh_yield
    assert error < 0.05, f"transition error at 8x8 = {error:.3f} (>5%)"


def test_anchor_permanent_deflection_positive_at_fine_mesh(anchor_collapse_fine):
    """Anchor: permanent deflection is positive and bounded."""
    py = anchor_collapse_fine["py"]
    path_unload = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                         hardening=1e6, patch=PATCH,
                                         load_factors=[0.5 * py["pressure"], 1.0 * py["pressure"],
                                                       1.5 * py["pressure"], 2.0 * py["pressure"],
                                                       2.5 * py["pressure"], 3.0 * py["pressure"]],
                                         nx=8, ny=8, layers=3, unload_steps=3)
    assert path_unload.permanent_deflection is not None
    assert path_unload.permanent_deflection > 0.0
    peak = path_unload.loading[-1].centre_deflection
    assert path_unload.permanent_deflection < peak


# ---------------------------------------------------------------------------
# Mesh-sweep tooling (scripts/patch_plate_mesh_sweep.py)
#
# The convergence verdicts and every published number are derived from the
# computed rows.  These tests pin that derivation on synthetic rows so that a
# regression back to hard-coded ``True`` flags or a hand-typed table is caught
# without paying for a three-mesh finite-element sweep.
# ---------------------------------------------------------------------------

import importlib.util
import json
import sys
from pathlib import Path


def _mesh_sweep_module():
    root = Path(__file__).resolve().parents[1]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "patch_plate_mesh_sweep", root / "scripts" / "patch_plate_mesh_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_rows(collapse_mpa=(1.55, 1.62, 1.66), permanent_mm=(9.0, 12.0, 13.5)):
    """Rows shaped like the sweep output; values chosen to be hand-checkable."""
    upper = 1.70e6
    lower = 0.76e6
    py = 0.804e6
    rows = []
    for i, (c, d) in enumerate(zip(collapse_mpa, permanent_mm)):
        rows.append({
            "nx": (4, 6, 8)[i],
            "first_yield_factor": 1.02 * py,
            "first_yield_error_vs_analytic": 0.02,
            "first_yield_error_vs_same_mesh": 0.004,
            "collapse_factor": c * 1e6,
            "collapse_measurements": {"intersection_of_tangents": c * 1e6,
                                      "double_deflection": 1.05 * c * 1e6},
            "collapse_spread": 0.05,
            "collapse_is_limit_point": False,
            "ramp_end_factor": 2.4123e6,
            "collapse_factor_layering_corrected": c * 1e6 * (2.0 / 3.0),
            "collapse_error_vs_upper": c * 1e6 / upper - 1.0,
            "collapse_error_vs_upper_corrected": c * 1e6 / upper - 1.0,
            "permanent_deflection": d * 1e-3,
            "permanent_set_valid": True,
            "peak_centre_deflection": 40.0e-3,
            "elastic_bias": 0.104,
        })
    bounds = {"analytic_first_yield_Pa": py,
              "yield_line_upper_Pa": upper, "strip_lower_Pa": lower}
    return rows, bounds


def test_convergence_summary_is_derived_from_the_rows():
    sweep = _mesh_sweep_module()
    rows, bounds = _synthetic_rows()
    sweep.convergence_summary(rows, bounds)     # sanity: fresh rows are consistent
    rows, bounds = _synthetic_rows()
    rows, bounds = _synthetic_rows()
    ok = sweep.convergence_summary(rows, bounds)
    assert ok["collapse_monotone"] is True
    assert ok["collapse_below_upper"] is True
    assert ok["collapse_above_lower"] is True
    assert ok["permanent_set_bounded"] is True
    assert abs(ok["transition_vs_same_mesh_max_abs_pct"] - 0.4) < 1e-9
    assert ok["collapse_is_measured_limit_load"] is True
    # worst row is the coarsest: 1.55 MPa against the 1.70 MPa upper bound
    assert abs(ok["collapse_vs_upper_max_abs_pct"] - abs(100 * (1.55 / 1.70 - 1.0))) < 1e-6
    assert ok["acceptance"]["collapse_vs_upper_lt_10pct"] is True
    # a sweep whose "collapse" is the ramp end must be flagged, not accepted
    for row in rows:
        row["collapse_factor"] = row["ramp_end_factor"]
    flagged = sweep.convergence_summary(rows, bounds)
    assert flagged["collapse_is_measured_limit_load"] is False
    # ... and a non-monotone / out-of-bound sweep must flip those flags, i.e.
    # they are computed, not constants.
    bad_rows, bad_bounds = _synthetic_rows(collapse_mpa=(1.66, 1.60, 1.72))
    bad = sweep.convergence_summary(bad_rows, bad_bounds)
    assert bad["collapse_monotone"] is False
    assert bad["collapse_below_upper"] is False
    # a permanent set above the peak deflection is not "bounded"
    over_rows, _ = _synthetic_rows(permanent_mm=(45.0, 46.0, 47.0))
    assert sweep.convergence_summary(over_rows, bounds)["permanent_set_bounded"] is False


def test_markdown_table_lists_every_mesh_level_with_its_own_numbers():
    sweep = _mesh_sweep_module()
    rows, bounds = _synthetic_rows()
    table = sweep.markdown_table(rows, bounds)
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    assert len(lines) == 2 + len(rows)          # header + separator + one row per level
    for row in rows:
        line = next(ln for ln in lines if ln.startswith(f"| {row['nx']} |"))
        assert f"{row['first_yield_factor'] / 1e6:.4f}" in line
        assert f"{row['collapse_factor'] / 1e6:.4f}" in line
        assert f"{row['ramp_end_factor'] / 1e6:.4f}" in line
        assert f"{1000 * row['permanent_deflection']:.4f}" in line
    for placeholder in ("_（", "| _ |", "TBD", "TODO"):
        assert placeholder not in table
    # the bounds are injected verbatim, not typed
    assert f"{bounds['yield_line_upper_Pa'] / 1e6:.4f}" in table
    assert f"{bounds['strip_lower_Pa'] / 1e6:.4f}" in table


def test_case_doc_reports_unmet_criteria_as_unmet():
    sweep = _mesh_sweep_module()
    rows, bounds = _synthetic_rows()
    summary = sweep.convergence_summary(rows, bounds)
    doc = sweep.case_doc(rows, bounds, summary)
    assert "met" in doc and "NOT met" not in doc
    # inflate the transition error past the 5 % band -> the doc must say NOT met
    for row in rows:
        row["first_yield_error_vs_same_mesh"] = 0.08
    strict = sweep.convergence_summary(rows, bounds)
    doc = sweep.case_doc(rows, bounds, strict)
    assert "NOT met" in doc
    assert f"{strict['transition_vs_same_mesh_max_abs_pct']:.2f} %" in doc


def test_report_payload_embeds_the_figure_and_the_computed_numbers(tmp_path):
    sweep = _mesh_sweep_module()
    rows, bounds = _synthetic_rows()
    summary = sweep.convergence_summary(rows, bounds)
    fig = tmp_path / "fig.png"
    fig.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
    payload = sweep.report_payload(rows, bounds, summary, fig, mission_id=22133)

    assert payload["mission_id"] == 22133
    assert payload["scenario_key"] == "GAP-PP-3-ICE-STRUCT-PLASTIC-SET"
    assert payload["figures"][0]["name"] == sweep.FIG_NAME
    assert payload["figures"][0]["png_base64"]
    md = payload["markdown"]
    for row in rows:
        assert f"{row['collapse_factor'] / 1e6:.4f}" in md
    assert f"{summary['permanent_set_spread_pct']:.2f}%" in md
    assert "_（既有测试已验）" not in md          # the blank-cell defect stays gone
    # honest error judgement section must be present with an explicit yes/no
    assert "是否 <3%" in md
    json.dumps(payload)                           # serialisable as posted


def test_mesh_levels_must_place_the_patch_edge_on_an_element_edge():
    """Regression: nx=6 leaves the patch edge at 1.5 element widths.

    The original 4/6/8 sweep could never produce an nx=6 row - the element
    routine rejects the mesh - so the levels are now validated up front and the
    admissible set is 4/8/12 (nx a multiple of 4).
    """
    sweep = _mesh_sweep_module()
    assert sweep.MESH_LEVELS == (4, 8, 12)
    sweep.assert_patch_compatible(sweep.MESH_LEVELS)          # admissible levels pass
    with pytest.raises(ValueError, match="off the element grid"):
        sweep.assert_patch_compatible((4, 6, 8))
    with pytest.raises(ValueError):
        patch_plate_mesh(A, B, T, 6, 6, young=E, poisson=NU,  # the underlying guard
                         yield_stress=FY, hardening=2e10, patch=PATCH)
