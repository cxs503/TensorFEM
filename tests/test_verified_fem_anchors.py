"""Regression gates for the IN36-N01 verified anchors.

Three independent verification surfaces live under ``benchmarks/verified``:

* ``fem_cantilever``    -- Euler--Bernoulli tip deflection, bending stress and a
  GCI ladder (Q8 plane stress plus a HEX20 solid cross-check),
* ``fem_hertz_contact`` -- sphere--sphere contact radius and peak pressure
  (axisymmetric penalty contact),
* ``fem_plate_modal``   -- first five simply supported plate frequencies
  (Mindlin/Q4 with a shift-invert eigen-solve).

The drivers write ``result.json`` + ``gates_verified.json``; this module

1. re-derives the closed-form oracles in the test (so a drifting formula in a
   driver cannot pass silently),
2. checks the committed evidence is self-consistent, i.e. every value in
   ``gates_verified.json`` equals the number reported in ``result.json`` and
   every gate threshold matches the campaign criterion, and
3. re-runs the cheap anchor end to end (the cantilever ladder costs seconds) so
   the pipeline itself, not just the stored JSON, is exercised.

Any real artifact edit that breaks a gate therefore fails here instead of
passing on a stale file.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "benchmarks" / "verified"
#: Small mesh used for the library-equivalence check: the identity being tested
#: (fast eigensolver vs `solve_plate_vibration`) is mesh independent, and the
#: library's torch eigensolve dominates the runtime.
COARSE_PLATE = (8, 6)


def _load(name: str, path: Path):
    """Import a driver script with its own ``benchmarks/verified`` sys.path entry."""
    if str(VERIFIED) not in sys.path:
        sys.path.insert(0, str(VERIFIED))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cantilever():
    return _load("verified_fem_cantilever", VERIFIED / "fem_cantilever" / "run.py")


@pytest.fixture(scope="module")
def hertz():
    return _load("verified_fem_hertz", VERIFIED / "fem_hertz_contact" / "run.py")


@pytest.fixture(scope="module")
def plate():
    return _load("verified_fem_plate", VERIFIED / "fem_plate_modal" / "run.py")


# ---------------------------------------------------------------------------
# 1. the oracles themselves
# ---------------------------------------------------------------------------


def test_cantilever_oracle_matches_closed_form(cantilever):
    oracle = cantilever.analytic()
    inertia = cantilever.WIDTH * cantilever.DEPTH**3 / 12.0
    expected = cantilever.LOAD * cantilever.LENGTH**3 / (3.0 * cantilever.YOUNG * inertia)
    assert oracle["inertia"] == pytest.approx(inertia, rel=1e-15)
    assert oracle["tip_deflection"] == pytest.approx(expected, rel=1e-15)
    # 10 kN on a 10 m steel cantilever of section 1 x 0.1 m: 190.5 mm.
    assert oracle["tip_deflection"] == pytest.approx(0.190476, rel=1e-5)
    assert oracle["slenderness_L_over_h"] == 100.0
    # Euler--Bernoulli is the right yardstick at this slenderness.
    assert oracle["shear_deflection_over_bending"] < 1e-4


def test_hertz_oracle_matches_textbook_two_sphere_solution(hertz):
    oracle = hertz.hertz_oracle()
    e_star = 210.0e9 / (2.0 * (1.0 - 0.3**2))
    radius = 25.0e-3
    a = (3.0 * 10.0e3 * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    assert oracle["effective_modulus"] == pytest.approx(e_star, rel=1e-14)
    assert oracle["contact_radius"] == pytest.approx(a, rel=1e-14)
    assert oracle["peak_pressure"] == pytest.approx(
        3.0 * 10.0e3 / (2.0 * math.pi * a * a), rel=1e-14
    )
    # Independent magnitudes: a ~ 1.176 mm, p0 ~ 3.454 GPa, approach ~ 55.3 um.
    assert oracle["contact_radius"] == pytest.approx(1.1757e-3, rel=2e-4)
    assert oracle["peak_pressure"] == pytest.approx(3.4544e9, rel=2e-4)
    assert oracle["approach"] == pytest.approx(5.5288e-5, rel=2e-4)


def test_hertz_equivalent_problem_reproduces_the_two_sphere_law(hertz):
    """The axisymmetric model's *inputs* must reproduce the sphere--sphere law.

    ``tensorfem.hertz_axisymmetric.hertz_reference`` knows only a rigid sphere on
    a half-space; feeding it the reduced radius and the reduced modulus must give
    back the two-sphere contact radius, peak pressure and load.  This is the
    equivalence that lets a one-body model verify a two-body contact.
    """
    from tensorfem.hertz_axisymmetric import hertz_reference

    oracle = hertz.hertz_oracle()
    load, radius, peak = hertz_reference(
        oracle["approach"],
        oracle["effective_radius"],
        oracle["model_young_modulus"],
        0.3,
    )
    assert float(radius) == pytest.approx(oracle["contact_radius"], rel=1e-12)
    assert float(peak) == pytest.approx(oracle["peak_pressure"], rel=1e-12)
    assert float(load) == pytest.approx(10.0e3, rel=1e-9)


def test_plate_oracle_matches_navier_solution(plate):
    oracle = plate.navier_frequencies()
    d = 210.0e9 * 0.012**3 / (12.0 * (1.0 - 0.3**2))
    scale = (math.pi / 2.0) * math.sqrt(d / (7850.0 * 0.012))
    expected = [scale * ((m / 1.2) ** 2 + (n / 0.8) ** 2) for m, n in plate.MODE_LABELS]
    assert oracle == pytest.approx(expected, rel=1e-14)
    # Textbook magnitudes for a 1.2 x 0.8 x 0.012 m steel panel.
    assert oracle[0] == pytest.approx(66.586, rel=1e-4)
    assert oracle[4] == pytest.approx(266.346, rel=1e-4)
    # The five modes must be distinct, otherwise a mis-paired mode could hide.
    assert len(set(round(f, 6) for f in oracle)) == 5


# ---------------------------------------------------------------------------
# 2. committed evidence is self-consistent
# ---------------------------------------------------------------------------

ANCHORS = ("fem_cantilever", "fem_hertz_contact", "fem_plate_modal")


@pytest.mark.parametrize("anchor", ANCHORS)
def test_evidence_files_exist_and_declare_the_schema(anchor):
    directory = VERIFIED / anchor
    result = json.loads((directory / "result.json").read_text())
    gates = json.loads((directory / "gates_verified.json").read_text())
    assert gates["schema"] == "tensorfem.verified-gates.v1"
    assert gates["anchor"] == anchor
    assert gates["mission"] == "IN36-N01"
    assert result["mission"] == "IN36-N01"
    assert gates["passed"] is True, gates["failed_gates"]
    assert gates["failed_gates"] == []
    for name in gates["artifacts"]["figures"]:
        path = directory / name
        assert path.exists() and path.stat().st_size > 10_000, f"missing figure {name}"
    assert (directory / "result.json").stat().st_size > 1_000


@pytest.mark.parametrize("anchor", ANCHORS)
def test_every_gate_value_is_reproduced_by_the_result_json(anchor):
    """No gate may carry a number that the result document does not contain."""
    directory = VERIFIED / anchor
    result = json.loads((directory / "result.json").read_text())
    gates = json.loads((directory / "gates_verified.json").read_text())
    numbers = set()

    def collect(node):
        if isinstance(node, dict):
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            numbers.add(float(node))

    collect(result)
    collect(gates["criteria"])
    for gate in gates["gates"]:
        assert gate["passed"] is True, gate
        assert math.isfinite(gate["value"])
        assert any(math.isclose(gate["value"], n, rel_tol=1e-12, abs_tol=0.0) for n in numbers), (
            f"{anchor}: gate {gate['id']} value {gate['value']!r} is not in result.json"
        )
        if gate["operator"] == "<=":
            assert gate["value"] <= gate["threshold"]
        else:
            assert gate["value"] >= gate["threshold"]


def test_campaign_thresholds_are_the_ones_actually_applied():
    cantilever = json.loads((VERIFIED / "fem_cantilever" / "gates_verified.json").read_text())
    hertz = json.loads((VERIFIED / "fem_hertz_contact" / "gates_verified.json").read_text())
    plate = json.loads((VERIFIED / "fem_plate_modal" / "gates_verified.json").read_text())

    assert cantilever["criteria"]["tip_deflection_rel_error_max"] == 0.01
    assert cantilever["criteria"]["gci_fine_max"] == 0.03
    assert cantilever["criteria"]["observed_order_min"] == 1.8

    assert hertz["criteria"]["contact_radius_rel_error_max"] == 0.02
    assert hertz["criteria"]["peak_pressure_rel_error_max"] == 0.03

    assert plate["criteria"]["frequency_rel_error_max"] == 0.01
    for anchor in (cantilever, hertz, plate):
        if "refinement_levels_min" in anchor["criteria"]:
            assert anchor["criteria"]["refinement_levels_min"] >= 3


def test_hertz_evidence_keeps_the_honest_non_monotone_note():
    """The contact radius is not monotone here; the evidence must say so."""
    result = json.loads((VERIFIED / "fem_hertz_contact" / "result.json").read_text())
    convergence = result["convergence"]
    assert convergence["contact_radius_monotone"] is False
    assert convergence["pressure_l2_observed_order_finest_pair"] > 1.5
    assert "not monotone" in convergence["note"]


def test_plate_evidence_records_the_library_equivalence():
    result = json.loads((VERIFIED / "fem_plate_modal" / "result.json").read_text())
    assert result["library_equivalence_max_rel"] < 1e-9
    assert result["verdict"]["min_mode_mac"] > 0.999
    assert result["verdict"]["max_frequency_rel_error"] < 0.01


# ---------------------------------------------------------------------------
# 3. re-run the cheap anchor end to end
# ---------------------------------------------------------------------------


def test_cantilever_anchor_runs_end_to_end(cantilever, tmp_path, monkeypatch):
    monkeypatch.setattr(cantilever, "HERE", tmp_path)
    # Plotting is a rendering concern exercised by the driver run itself; the
    # test wants the solve -> gate path, so the figure step is stubbed to keep
    # the suite inside a normal CI budget.
    monkeypatch.setattr(cantilever, "figures", lambda *args, **kwargs: [])
    # Same ladder family, three levels instead of four: the gates, the
    # Richardson arithmetic and the file contract are all exercised.
    monkeypatch.setattr(cantilever, "Q8_LEVELS", ((10, 1), (20, 2), (40, 4)))
    monkeypatch.setattr(cantilever, "HEX20_LEVELS", ((16, 2, 2), (32, 4, 4)))
    result = cantilever.run()
    assert result["verdict"]["gate_passed"] is True
    assert result["verdict"]["finest_relative_error"] < 0.01
    assert result["q8_richardson"]["observed_order"] >= 1.8
    assert result["q8_richardson"]["gci_fine"] < 0.03
    assert (tmp_path / "gates_verified.json").exists()
    assert len(result["q8_levels"]) >= 3


@pytest.fixture(scope="module")
def coarse_plate_record(plate, tmp_path_factory):
    """Solve the coarsest plate level once, into a throw-away cache."""
    cache = tmp_path_factory.mktemp("plate_cache")
    original = plate.CACHE
    plate.CACHE = cache
    try:
        yield plate.solve_modes(*COARSE_PLATE)
    finally:
        plate.CACHE = original


def test_plate_fast_eigensolver_agrees_with_the_library(
    plate, coarse_plate_record, tmp_path, monkeypatch
):
    """The shift-invert path must solve the same discrete problem as the library."""
    monkeypatch.setattr(plate, "CACHE", tmp_path)
    plate.solve_modes(*COARSE_PLATE)  # refresh the cache in the redirected directory
    difference = plate.library_equivalence(*COARSE_PLATE)
    assert difference < 1e-9


def test_plate_level_is_free_of_mechanisms(coarse_plate_record):
    """Retained frequencies must all be real; a mechanism would appear as zero."""
    assert all(f > 0.0 for f in coarse_plate_record["frequencies_hz"])
    assert coarse_plate_record["spurious_modes_filtered"] >= 0
    assert min(coarse_plate_record["mode_mac"]) > 0.99
