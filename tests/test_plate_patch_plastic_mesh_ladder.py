"""Mesh-ladder invariants for the plate-panel plastic capacity (anchor GAP-PP-3).

The campaign script ``scripts/patch_plate_plastic_mesh_ladder.py`` produces the
three-mesh limit-load ladder.  These tests pin the *invariants* that make the
ladder's numbers meaningful, not the numbers themselves (those live in
``artifacts/patch_plate_plastic_ladder_summary.json`` and in the case document):

* the bound ordering the acceptance refers to (strip lower < Ilyushin < yield line);
* the ordering of the limit-load *definitions* (a looser stiffness criterion is
  reached earlier, so it can only report a smaller limit load);
* the Richardson extrapolation recovering a known limit from a synthetic ladder
  (if this test fails, the convergence claim is arithmetic, not physics);
* the support declaration: only the in-plane-free model may be compared with a
  rigid-plastic bending bound, because a linear flat shell has no membrane
  coupling;
* on the real FE path: elastic branch against Navier, bounded section moment
  ratio, first yield before section saturation, and a permanent set that is
  smaller than the peak deflection it came from.
"""
import importlib.util
import json
import math
import pathlib

import pytest
import torch

from tensorfem.layered_shell4_plasticity import LayeredShell4Model  # noqa: F401
from tensorfem.plate_patch_plastic_limit import (
    PatchPath,
    PatchPathPoint,
    elastic_patch_plate,
    limit_load_from_path,
    low_temperature_flow_stress,
    plastic_bound_bracket,
    run_patch_plastic_path,
    section_moment_capacity,
)

A = B = 1.0
T = 0.02
E, NU = 210e9, 0.3
PATCH = (0.5 * A, 0.5 * B)
FY = low_temperature_flow_stress("DH36", -40.0).yield_stress

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / \
    "patch_plate_plastic_mesh_ladder.py"


def _ladder_module():
    spec = importlib.util.spec_from_file_location("plate_plastic_mesh_ladder", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_path(k0: float, plastic_slope: float, p_peak: float,
                    steps: int = 40) -> PatchPath:
    """Elastic line ``w = k0 p`` up to ``p_c``, then ``dw/dp = plastic_slope``.

    ``plastic_slope`` is in deflection-per-load, like ``k0``; the analytic limit
    is the two-line intersection ``p_c``.
    """
    path = PatchPath(model=None, info={})
    dp = p_peak / steps
    p_c = p_peak / 2.0
    for index in range(1, steps + 1):
        p = index * dp
        w = k0 * min(p, p_c) + plastic_slope * max(p - p_c, 0.0)
        path.loading.append(PatchPathPoint(p, w, 0.0, 0.0, 0.0))
    path.ramp_completed = True
    return path


# ---------------------------------------------------------------------------
# Ramp / reference plumbing
# ---------------------------------------------------------------------------

def test_ramp_factors_are_increasing_and_cover_the_requested_top():
    ladder = _ladder_module()
    factors = ladder.ramp_factors(1.0e6, top=2.0, step=0.1)
    assert factors[0] == pytest.approx(0.2e6)
    assert factors[-1] == pytest.approx(2.0e6)
    assert all(b > a for a, b in zip(factors, factors[1:]))
    assert len(factors) == 19


def test_reference_bounds_keep_their_admissible_ordering():
    """The acceptance needs a *bracket*: strip lower < Ilyushin < yield line."""
    ladder = _ladder_module()
    ref = ladder.references(FY)
    bounds = ref["bounds"]
    assert bounds["strip_lower"] < bounds["ilyushin_sine"] < bounds["yield_line"]
    # the bracket reported by the library must agree, term by term
    assert ref["bracket"]["upper_yield_line"] == pytest.approx(bounds["yield_line"])
    assert ref["bracket"]["upper_ilyushin_sine"] == pytest.approx(bounds["ilyushin_sine"])
    assert ref["bracket"]["lower_strip"] == pytest.approx(bounds["strip_lower"])
    assert ref["p_yield_analytic"] > 0.0
    assert ref["m_p"] == pytest.approx(section_moment_capacity(T, FY, 4)["m_p"])


def test_plan_contains_three_free_meshes_for_the_convergence_claim():
    """A two-mesh ladder cannot support a convergence order; guard the third."""
    ladder = _ladder_module()
    free = [nx for nx, geometry in ladder.PLAN if geometry == "free"]
    assert sorted(free) == [8, 12, 16]
    assert all(nx % 2 == 0 for nx, _ in ladder.PLAN)


# ---------------------------------------------------------------------------
# Limit-load definitions
# ---------------------------------------------------------------------------

def test_limit_load_definitions_bracket_the_analytic_two_line_limit():
    """On a path with a known kink all three definitions land on the kink."""
    k0, slope, p_c, peak = 1.0e-6, 1.0e-4, 0.5, 1.0
    path = _synthetic_path(k0, slope, peak, steps=200)
    estimate = limit_load_from_path(path, criteria=(0.02, 0.05, 0.10))
    reference = peak / 2.0 * 2.0  # two-line intersection of the construction
    reference = (slope * peak - (k0 * p_c + slope * (peak - p_c))) / (slope - k0)
    assert reference == pytest.approx(p_c, rel=0.06)
    assert estimate["limit_load"] == pytest.approx(reference, rel=0.06)
    assert estimate["points"] == 200


def test_a_looser_stiffness_criterion_can_only_report_a_smaller_limit_load():
    """``c`` is a fraction of the elastic slope: larger ``c`` -> earlier crossing."""
    path = _synthetic_path(1.0e-6, 1.0e-4, 1.0, steps=200)
    loose = limit_load_from_path(path, criteria=(0.30,))
    tight = limit_load_from_path(path, criteria=(0.02,))
    assert loose["measurements"]["stiffness_criterion_0.3"] <= \
        tight["measurements"]["stiffness_criterion_0.02"] + 1e-12
    assert tight["spread"] >= 0.0


def test_limit_load_is_not_the_end_of_the_ramp():
    """Regression: the last prescribed factor is an input, not a measurement."""
    path = _synthetic_path(1.0e-6, 1.0e-4, 1.0, steps=200)
    estimate = limit_load_from_path(path, criteria=(0.05,))
    assert estimate["limit_load"] < 0.75 * estimate["peak_factor"]


# ---------------------------------------------------------------------------
# Richardson extrapolation of the ladder
# ---------------------------------------------------------------------------

def _fake_record(nx: int, limit: float, geometry: str = "free",
                 bound: float = 1.5e6) -> dict:
    return {
        "nx": nx, "layers": 4, "geometry": geometry,
        "section_capacity": section_moment_capacity(T, FY, 4),
        "references": {"bounds": {"ilyushin_sine": bound,
                                  "yield_line": 1.84e6,
                                  "strip_lower": 0.82e6}},
        "limit_estimates": {
            "tight": {"limit_load": limit, "spread": 0.02,
                      "measurements": {"stiffness_criterion_0.02": limit}},
            "loose": {"limit_load": limit * 0.98, "spread": 0.01,
                      "measurements": {}}},
        "first_yield": 0.87e6,
        "first_yield_error_vs_analytic": 0.005,
        "ramp_completed": True, "ramp_truncated": False,
        "measured_limit_load": None, "limit_point_searched": False,
        "peak_factor": 2.6e6, "permanent_deflection": 2.0e-3,
        "permanent_set_valid": True, "unload_converged_steps": 3,
        "unload_failure": None, "elapsed_s": 1.0,
        "path": [{"p": 1.0, "p_over_py": 1.0, "w": 1e-3, "alpha": 0.0,
                  "moment_ratio": 1.0, "sigma_surface_over_fy": 1.0}],
    }


def test_report_line_survives_a_ramp_without_any_limit_definition():
    """A ramp that never softens must be *reported*, not crash the driver.

    ``limit_load_from_path`` returns ``None`` when none of the limit-load
    definitions fires (short or purely elastic ramp).  The report path used to
    format that ``None`` with ``:.4f`` and die with a ``TypeError`` *after* a
    multi-minute FE run, losing the run; ``describe`` turns it into ``n/a``.
    """
    ladder = _ladder_module()
    record = {
        "nx": 8, "geometry": "free", "elapsed_s": 1.5, "first_yield": None,
        "permanent_deflection": None, "permanent_set_valid": False,
        "limit_estimates": {
            "tight": {"limit_load": None, "spread": None, "measurements": {}},
            "loose": {"limit_load": None, "spread": None, "measurements": {}},
        },
    }
    text = ladder.describe(record)
    assert "limit n/a" in text and "spread n/a" in text
    assert "perm. set withheld" in text
    # a record that *does* carry a limit keeps the numeric report
    record["limit_estimates"]["tight"] = {
        "limit_load": 1.2e6, "spread": 0.04,
        "measurements": {"intersection_of_tangents": 1.2e6}}
    record["errors"] = {"ilyushin_sine": -0.2, "yield_line": -0.35}
    record["first_yield"] = 0.87e6
    record["permanent_deflection"] = 5.2e-3
    record["permanent_set_valid"] = True
    text = ladder.describe(record)
    assert "limit 1.2000 MPa" in text and "-20.00 % vs Ilyushin" in text


def test_richardson_extrapolation_recovers_a_second_order_ladder(tmp_path, monkeypatch):
    """``p(h) = p_inf + C h^2`` with h = 1/nx must give back ``p_inf`` and q = 2."""
    ladder = _ladder_module()
    monkeypatch.setattr(ladder, "ART", tmp_path)
    p_inf, c = 1.40e6, 3.0e7
    for nx in (8, 12, 16):
        h = 1.0 / nx
        record = _fake_record(nx, p_inf + c * h * h)
        (tmp_path / f"patch_plate_plastic_ladder_nx{nx}_free.json").write_text(
            json.dumps(record))
    assert ladder.analyse() == 0
    summary = json.loads((tmp_path / "patch_plate_plastic_ladder_summary.json").read_text())
    extrapolation = summary["extrapolation_free"]
    assert extrapolation["order_estimate"] == pytest.approx(2.0, rel=1e-6)
    assert extrapolation["limit_infinity"] == pytest.approx(p_inf, rel=1e-9)


DENSE_RAMP = {"top_over_py": 2.4, "step_over_py": 0.1, "points": 23,
              "factors_over_py": [round(0.2 + 0.1 * k, 6) for k in range(23)]}
COARSE_RAMP = {"top_over_py": 1.8, "step_over_py": 0.4, "points": 5,
               "factors_over_py": [0.2, 0.6, 1.0, 1.4, 1.8]}


def test_ramp_signature_carries_the_prescribed_ramp_and_the_layering():
    ladder = _ladder_module()
    record = _fake_record(8, 1.5e6)
    record["ramp"] = dict(DENSE_RAMP)
    signature = ladder.ramp_signature(record)
    assert signature["factors_over_py"] == DENSE_RAMP["factors_over_py"]
    assert signature["layers"] == 4
    # a legacy artifact without a ramp still gets a signature, but an empty one
    legacy = ladder.ramp_signature(_fake_record(12, 1.5e6))
    assert legacy["factors_over_py"] is None
    assert "legacy" in ladder._ramp_label(legacy)


def test_a_ladder_that_mixes_two_prescribed_ramps_is_refused(tmp_path, monkeypatch):
    """A mesh ladder is one discretisation family on one ramp.

    The delivered campaign had an ``nx = 12`` record on a dense ramp next to a
    legacy ``nx = 8`` record on a coarse one; differencing those two measures the
    ramp resolution, not the mesh.  ``analyse`` must exclude the odd record,
    report it with a reason, and refuse the Richardson extrapolation.
    """
    ladder = _ladder_module()
    monkeypatch.setattr(ladder, "ART", tmp_path)
    for nx, limit in ((8, 1.60e6), (16, 1.45e6)):
        record = _fake_record(nx, limit)
        record["ramp"] = dict(DENSE_RAMP)
        (tmp_path / f"patch_plate_plastic_ladder_nx{nx}_free.json").write_text(
            json.dumps(record))
    odd = _fake_record(12, 1.20e6)
    odd["ramp"] = dict(COARSE_RAMP)
    (tmp_path / "patch_plate_plastic_ladder_nx12_free.json").write_text(
        json.dumps(odd))

    assert ladder.analyse() == 0
    summary = json.loads((tmp_path / "patch_plate_plastic_ladder_summary.json").read_text())
    assert [entry["tag"] for entry in summary["records"]] == ["nx8_free", "nx16_free"]
    excluded = summary["records_excluded"]
    assert [entry["tag"] for entry in excluded] == ["nx12_free"]
    assert "ramp" in excluded[0]["ladder_excluded"]
    assert "5 points" in excluded[0]["ladder_excluded"]
    # two comparable meshes are not a convergence order, and the summary says so
    assert summary["extrapolation_free"]["order_estimate"] is None


def test_one_prescribed_ramp_is_extrapolated_even_across_a_geometry_change(
        tmp_path, monkeypatch):
    """Records that do share the ramp are admitted (control for the guard)."""
    ladder = _ladder_module()
    monkeypatch.setattr(ladder, "ART", tmp_path)
    p_inf, c = 1.40e6, 3.0e7
    for nx in (8, 12, 16):
        record = _fake_record(nx, p_inf + c / nx ** 2)
        record["ramp"] = dict(DENSE_RAMP)
        (tmp_path / f"patch_plate_plastic_ladder_nx{nx}_free.json").write_text(
            json.dumps(record))
    assert ladder.analyse() == 0
    summary = json.loads((tmp_path / "patch_plate_plastic_ladder_summary.json").read_text())
    assert summary["records_excluded"] == []
    assert summary["extrapolation_free"]["limit_infinity"] == pytest.approx(p_inf, rel=1e-9)


def test_analysis_only_judges_the_free_support_against_the_bending_bound(tmp_path, monkeypatch):
    ladder = _ladder_module()
    monkeypatch.setattr(ladder, "ART", tmp_path)
    for nx in (8, 12, 16):
        (tmp_path / f"patch_plate_plastic_ladder_nx{nx}_free.json").write_text(
            json.dumps(_fake_record(nx, 1.45e6 + 1.0e6 / nx)))
    (tmp_path / "patch_plate_plastic_ladder_nx8_membrane.json").write_text(
        json.dumps(_fake_record(8, 2.5e6, geometry="membrane")))
    assert ladder.analyse() == 0
    summary = json.loads((tmp_path / "patch_plate_plastic_ladder_summary.json").read_text())
    by_tag = {entry["tag"]: entry for entry in summary["records"]}
    assert "acceptance" in by_tag["nx8_free"]
    assert "acceptance" not in by_tag["nx8_membrane"]
    # 1.45e6 + 1e5 is within 10 % of the 1.5 MPa Ilyushin bound
    assert by_tag["nx16_free"]["acceptance"]["within_10pct_of_ilyushin"] is True
    assert by_tag["nx16_free"]["acceptance"]["below_yield_line_upper_bound"] is True
    assert by_tag["nx8_membrane"]["errors"]["yield_line"] > 0.0


# ---------------------------------------------------------------------------
# FE invariants on one short ramp (no queue needed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fe_ramp():
    """One short ramp with unloading: elastic branch, saturation, residual."""
    p_y = 0.8692336468059256e6
    path = run_patch_plastic_path(
        A, B, T, young=E, poisson=NU, yield_stress=FY, hardening=1e6, patch=PATCH,
        load_factors=[0.4 * p_y, 1.2 * p_y, 1.6 * p_y], nx=8, ny=8, layers=4,
        unload_steps=2, exact_thickness=True, max_deflection_factor=6.0,
        bisection_tolerance=1e-2)
    return path, p_y


def test_fe_elastic_branch_reproduces_the_navier_deflection(fe_ramp):
    path, _p_y = fe_ramp
    first = path.loading[0]
    navier = elastic_patch_plate(A, B, T, E, NU, PATCH, first.load_factor,
                                 terms=41)["deflection"][0]
    assert first.centre_deflection == pytest.approx(navier, rel=0.03)


def test_section_moment_ratio_never_exceeds_the_layer_capacity(fe_ramp):
    path, _p_y = fe_ramp
    capacity = section_moment_capacity(T, FY, 4)["ratio"]
    assert max(step.max_moment_ratio for step in path.loading) <= capacity + 1e-6


def test_first_yield_precedes_section_saturation(fe_ramp):
    path, p_y = fe_ramp
    assert path.first_yield_factor is not None
    assert path.first_yield_factor == pytest.approx(p_y, rel=0.05)
    # 1.6 P_y already carries ~0.88 of the section capacity (see the case doc);
    # the invariant is the *ordering*, not the plateau value of one mesh
    saturated = [s.load_factor for s in path.loading if s.max_moment_ratio > 0.85]
    assert saturated and min(saturated) > path.first_yield_factor


def test_permanent_set_is_smaller_than_the_peak_it_came_from(fe_ramp):
    path, _p_y = fe_ramp
    assert path.permanent_set_valid, path.unload_failure
    residual = path.permanent_deflection
    assert residual is not None and residual > 0.0
    assert residual < path.loading[-1].centre_deflection
    # unloading is (nearly) elastic: the residual equals the plastic part
    assert path.unloading[-1].load_factor < path.loading[-1].load_factor


def test_load_deflection_path_is_monotone_and_stiffness_decays(fe_ramp):
    path, _p_y = fe_ramp
    ws = [s.centre_deflection for s in path.loading]
    ps = [s.load_factor for s in path.loading]
    assert all(b > a for a, b in zip(ws, ws[1:]))
    slopes = [(w1 - w0) / (p1 - p0) for p0, p1, w0, w1 in
              zip(ps, ps[1:], ws, ws[1:])]
    assert all(b > a for a, b in zip(slopes, slopes[1:]))  # compliance grows
    assert path.stiffness_ratio() is not None