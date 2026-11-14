"""Regression gate for CASE-FEM-CANTILEVER-TIP (anchor ``P L^3/(3 E I)``, < 2 %).

The calculation-pool anchor ``FEM-CANTILEVER-TIP`` asks for the tip deflection of a
slender cantilever against Euler-Bernoulli beam theory.  These tests are the cheap
guard rails around the heavy study in ``benchmarks/case_fem_cantilever_tip.py``:

* the anchor is re-derived from the bending-moment field (unit-load / Castigliano
  quadrature), so a typo in ``P L^3/(3 E I)`` cannot pass silently;
* the 1-D Euler-Bernoulli beam solver must reproduce the cubic solution exactly;
* the 1-D Timoshenko solver must match the closed-form shear-flexible deflection;
* the 3-D HEX20 solid must stay inside the 2 % acceptance band on the declared
  reference mesh, and refine towards the anchor when the mesh is coarsened;
* the JSON document that the TensorBench L4 workload consumes must expose the
  declared ``ref``/``got`` pointers and the predeclared tolerance.

Run ``pytest -q tests/test_case_fem_cantilever_tip.py`` (a few seconds).
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "case_fem_cantilever_tip", ROOT / "benchmarks" / "case_fem_cantilever_tip.py"
)
_case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_case)

TOLERANCE = _case.TOLERANCE
REFERENCE_MESH = _case.REFERENCE_MESH


@pytest.fixture(scope="module")
def quick_study() -> dict:
    """One quick study shared by the document-level assertions."""
    return _case.run_study(quick=True)


def test_anchor_rederived_from_moment_field() -> None:
    """``P L^3/(3 E I)`` and the shear term must follow from the unit-load integral."""
    ref = _case.analytic_reference()
    quad = _case.castigliano_integral()
    assert quad["bending_m"] == pytest.approx(ref["euler_bernoulli_m"], rel=1e-12)
    assert quad["shear_m"] == pytest.approx(ref["shear_term_m"], rel=1e-12)
    assert ref["timoshenko_m"] == pytest.approx(
        ref["euler_bernoulli_m"] + ref["shear_term_m"], rel=1e-15)


def test_reference_value_and_tolerance_are_declared() -> None:
    """The case parameters must give the documented 1 cm deflection and a 2 % band."""
    ref = _case.analytic_reference()
    assert ref["euler_bernoulli_m"] == pytest.approx(1.0e-2, rel=1e-12)
    assert TOLERANCE == pytest.approx(0.02)
    assert REFERENCE_MESH == (32, 4)


def test_eb_beam_element_is_exact() -> None:
    """Cubic Hermite EB elements interpolate the cubic bending field exactly."""
    ref = _case.analytic_reference()
    for n in (1, 2, 4, 8):
        row = _case.beam_eb_level(n)
        error = abs(row["tip_deflection_m"] - ref["euler_bernoulli_m"]) / ref["euler_bernoulli_m"]
        assert error < 1e-9, (n, error)


def test_timoshenko_beam_matches_closed_form() -> None:
    """The shear-flexible beam solver reproduces ``PL/(kappa A G)`` on top of bending."""
    ref = _case.analytic_reference()
    row = _case.beam_timoshenko_level(8)
    assert row["tip_deflection_m"] == pytest.approx(ref["timoshenko_m"], rel=1e-9)
    # and it therefore sits above the pure-bending anchor by the shear term
    assert row["tip_deflection_m"] > ref["euler_bernoulli_m"]


def test_solid_coarse_mesh_is_within_band() -> None:
    """A cheap solid mesh already meets the 2 % acceptance criterion."""
    ref = _case.analytic_reference()
    row = _case.solid_level(16, 2)
    error = abs(row["tip_deflection_m"] - ref["euler_bernoulli_m"]) / ref["euler_bernoulli_m"]
    assert error < TOLERANCE, (row, error)


def test_solid_refines_towards_the_anchor() -> None:
    """Coarsening the solid mesh must make the bending error larger, not smaller."""
    ref = _case.analytic_reference()
    coarse = _case.solid_level(8, 1)
    medium = _case.solid_level(16, 2)
    err_coarse = abs(coarse["tip_deflection_m"] - ref["euler_bernoulli_m"])
    err_medium = abs(medium["tip_deflection_m"] - ref["euler_bernoulli_m"])
    assert err_medium < err_coarse
    # single element across the section is deliberately reported as outside the band
    assert err_coarse / ref["euler_bernoulli_m"] > TOLERANCE


def test_declared_reference_mesh_meets_acceptance() -> None:
    """The predeclared reference mesh is the metric the L4 judgement uses."""
    ref = _case.analytic_reference()
    row = _case.solid_level(*REFERENCE_MESH)
    error = abs(row["tip_deflection_m"] - ref["euler_bernoulli_m"]) / ref["euler_bernoulli_m"]
    assert error < TOLERANCE, (row["dofs"], error)


def test_solid_positional_arguments_are_guarded() -> None:
    """Calling the solid model with a stray positional argument must fail loudly."""
    with pytest.raises(TypeError):
        _case.solid_level(32, 2, 2.0)  # third positional used to bind ``length``


def test_workload_document_contract(quick_study: dict) -> None:
    """The JSON the L4 workload prints must carry the declared ref/got/tolerance."""
    doc = dict(quick_study)
    doc.pop("_solid_fields", None)  # solver tensors stay out of the published document
    assert doc["benchmark"]["benchmark_id"] == "CASE-FEM-CANTILEVER-TIP"
    assert doc["benchmark"]["version"] == "1.0.0"
    assert doc["execution"]["status"] == "success"
    assert doc["workload"]["workload_id"] == "benchmarks/case_fem_cantilever_tip.py"

    ref = doc["reference"]["euler_bernoulli_m"]
    got = doc["verdict"]["tip_deflection_m"]
    assert got == pytest.approx(doc["levels"]["solid_reference"]["tip_deflection_m"], rel=1e-15)
    assert doc["verdict"]["reference_value_m"] == pytest.approx(ref, rel=1e-15)
    assert doc["verdict"]["relative_error"] == pytest.approx((got - ref) / ref, rel=1e-9)
    assert doc["verdict"]["tolerance_rel"] == pytest.approx(0.02)
    assert doc["verdict"]["passed"] is True
    assert all(doc["verdict"]["checks"].values()), doc["verdict"]["checks"]

    # the document must survive a JSON round-trip without NaN/inf
    assert json.loads(json.dumps(doc))["verdict"]["passed"] is True