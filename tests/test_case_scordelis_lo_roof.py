"""Regression gate for CASE-FEM-SCORDELIS-LO-ROOF.

Fast slice of ``benchmarks/case_scordelis_lo_roof.py``: the two coarse rungs of
the ladder, the rank census, the equilibrium audit, the verdict gates and the
consistency of the committed artifact with a fresh recomputation.  The full
4/8/16 ladder lives in the case script.

The point of the case is that the historical explanation of this benchmark's
50 % flat-facet failure ("faceting cannot capture membrane-bending coupling")
was falsified: the same facets pass once the transverse-shear block is rank
sufficient.  ``test_default_shear_null_fails_closed`` protects that
*inverted* gate, so the failure cannot silently disappear.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "case_scordelis_lo_roof", ROOT / "benchmarks" / "case_scordelis_lo_roof.py"
)
_case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_case)

REFERENCE = _case.REFERENCE
ARTIFACT = ROOT / "benchmarks" / "artifacts" / _case.CASE / "case_result.json"
DOC = ROOT / "docs" / (_case.CASE + ".md")


@pytest.fixture(scope="module")
def coarse():
    return {config: _case.solve(config, 8) for config in ("cyl", "mitc", "sri")}


@pytest.fixture(scope="module")
def census():
    return _case.rank_census()


def test_parameters_are_the_classical_benchmark():
    par = _case.PARAMETERS
    assert (par["length_m"], par["radius_m"], par["half_angle_deg"]) == (50.0, 25.0, 40.0)
    assert (par["thickness_m"], par["young_pa"], par["poisson"], par["vertical_load_pa"]) == (
        0.25, 4.32e8, 0.0, 90.0)
    assert REFERENCE == -0.3024


def test_exact_surface_area_is_analytic():
    # L * R * 2*half_angle, with the angle in radians: 50*25*(4*pi/9)
    assert _case.exact_surface_area() == pytest.approx(50.0 * 25.0 * 4.0 * 3.141592653589793 / 9.0)


def test_facets_under_measure_the_curved_surface():
    from tensorfem.cylindrical_shell_benchmarks import scordelis_lo_cylindrical

    mesh = scordelis_lo_cylindrical(4, 4)
    deficit = _case.facet_area(mesh.nodes, mesh.elements) / _case.exact_surface_area() - 1.0
    assert deficit < 0.0, "planar facets cannot over-measure a convex arc"
    assert abs(deficit) < 0.01
    assert _case.sagitta(16) < 0.1 * _case.PARAMETERS["radius_m"]


def test_metric_consistent_element_reaches_reference_at_8x8(coarse):
    assert abs(coarse["cyl"]["relative_error"]) < 0.03


def test_flat_facet_with_mitc_clears_3pct_while_default_does_not(coarse):
    # The falsified claim: identical facets, identical nodes, identical load.
    assert abs(coarse["mitc"]["relative_error"]) < 0.10
    assert abs(coarse["sri"]["relative_error"]) > 0.20


def test_rank_census_names_the_defect(census):
    by_rule = {c["plate_shear"]: c for c in census}
    assert by_rule["sri"]["transverse_mechanisms"] >= 1, "sri must keep transverse mechanisms"
    assert by_rule["mitc"]["transverse_mechanisms"] == 0, (
        "assumed natural shear must leave only the in-plane/drilling gauge direction"
    )


def test_flat_facet_reaction_balances_the_faceted_load():
    level = _case.solve("mitc", 8)
    assert abs(level["equilibrium_ratio"] - 1.0) < 1e-9


def test_committed_artifact_agrees_with_a_fresh_recomputation(coarse):
    payload = json.loads(ARTIFACT.read_text())
    recorded = next(l for l in payload["ladders"]["mitc"] if l["mesh"] == "8x8")
    assert recorded["probe_displacement"] == pytest.approx(coarse["mitc"]["probe_displacement"], rel=1e-12)
    assert recorded["relative_error"] == pytest.approx(coarse["mitc"]["relative_error"], rel=1e-12)
    assert payload["all_passed"] is True
    assert len(payload["gates"]) == 11


def test_documentation_is_rendered_from_the_artifact():
    payload = json.loads(ARTIFACT.read_text())
    assert DOC.read_text() == _case.render_doc(payload) == (
        ARTIFACT.parent / "case_result.md").read_text()


def test_default_shear_null_fails_closed(census):
    """If the documented sri failure ever disappears, the gates must go red."""
    ladders = {config: [{"mesh": "%dx%d" % (n, n), "probe_displacement": REFERENCE * f,
                         "relative_error": f - 1.0, "formulation": config,
                         "sagitta_over_thickness": 0.0, "facet_area_deficit": 0.0,
                         "equilibrium_ratio": 1.0, "elements": 1, "nodes": 4}
                        for n, f in ((4, 1.02), (8, 1.02), (16, 1.02))]
               for config in ("cyl", "mitc", "sri")}
    gates = _case.build_gates(ladders, census)
    failed = {g["gate"] for g in gates if not g["passed"]}
    assert "null.default_sri_16x16" in failed
    assert "accuracy.flat_facet_mitc_16x16" not in failed


def test_richardson_order_is_positive_on_a_monotone_ladder():
    levels = [{"mesh": "%dx%d" % (n, n), "probe_displacement": -1.0 + 0.1 / n ** 2}
              for n in (4, 8, 16)]
    result = _case.richardson(levels)
    assert result["observed_order"] == pytest.approx(2.0, abs=0.05)
    assert result["richardson_limit"] == pytest.approx(-1.0, abs=1e-12)