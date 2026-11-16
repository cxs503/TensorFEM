"""Regression tests for the v4 implicit-IBM / Aitken cross-validation harness.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (benchmark
``benchmarks/xval_implicit_ibm_aitken_v4.py``).

The first v4 delivery shipped two rows that were not physics results:

* ``vacuum free-vibration f vs analytic``: ``43.048 Hz`` against a
  ``0.144486 Hz`` reference.  Two harness defects: the free-vibration driver was
  called **without** the section the reference was computed from (the library
  silently fell back to its unit section ``A = 1``, ``I = 1/12``), and the
  analytic Euler-Bernoulli reference **omitted the area A**, so it was low by
  ``1/sqrt(A) = 7.07`` for the ``A = 1/50`` section.
* ``plate f_fsi/f_vac``: ``1.5811`` against ``0.7906``, attributed to "unit
  handling".  The record was ``300`` steps long while the analytic period of the
  case is ``600`` steps, so every period estimator returned the window length
  (``T_analytic / (n dt) = 2``) and the harness fell back to that legacy value
  instead of refusing to report.

These tests pin the formula, the section propagation, the window gate and the
three-way verdict, so neither failure mode can come back silently.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

from tensorfem.ibm_aitken_implicit import BeamProperties  # noqa: E402
from tensorfem.ibm_aitken_added_mass import period_window_ok  # noqa: E402
from xval_implicit_ibm_aitken_v4 import (  # noqa: E402
    TOL,
    euler_bernoulli_f1,
    plate_added_mass_study,
    plate_verdict,
    structure_checks,
    vacuum_verdict,
)

ARTIFACTS = os.path.join(HERE, "..", "benchmarks",
                         "xval_implicit_ibm_aitken_v4_artifacts", "evidence.json")


def _section(t=1.0 / 50.0):
    return BeamProperties(E=1.0e5, A=t, I=t ** 3 / 12.0, rho=1.0,
                          length=1.0, thickness=t)


# ---------------------------------------------------------------------------
def test_euler_bernoulli_reference_carries_the_area():
    """f1 must scale as 1/sqrt(A); the pre-fix formula dropped A entirely."""
    props = _section()
    ref = euler_bernoulli_f1(props.E, props.I, props.rho, props.A, props.length)
    hand = (1.87510407 ** 2) / (2.0 * math.pi) * math.sqrt(
        props.E * props.I / (props.rho * props.A * props.length ** 4))
    assert ref == pytest.approx(hand, rel=1e-14)

    # Closed form for the t = 1/50 section of the benchmark.
    assert ref == pytest.approx(1.0216699, rel=1e-6)

    # What the old harness computed (A omitted) and by how much it was wrong.
    legacy = euler_bernoulli_f1(props.E, props.I, props.rho, 1.0, props.length)
    assert legacy == pytest.approx(0.1444858, rel=1e-6)
    assert legacy / ref == pytest.approx(math.sqrt(props.A), rel=1e-12)

    # The one case where the old formula was accidentally right.
    unit = _section(t=1.0)
    assert euler_bernoulli_f1(unit.E, unit.I, unit.rho, unit.A, unit.length) == \
        pytest.approx(euler_bernoulli_f1(unit.E, unit.I, unit.rho, 1.0, unit.length),
                      rel=1e-14)


def test_vacuum_verdict_is_three_way():
    """An unusable record must be INVALID_WINDOW, never FAIL with a number."""
    ok = vacuum_verdict({"vacuum_window_ok": True, "vacuum_freq_rel_error": 0.001})
    assert ok["verdict"] == "PASS" and ok["pass"] is True

    bad = vacuum_verdict({"vacuum_window_ok": True, "vacuum_freq_rel_error": 0.5})
    assert bad["verdict"] == "FAIL" and bad["pass"] is False

    short = vacuum_verdict({"vacuum_window_ok": False,
                            "vacuum_freq_rel_error": 297.0})
    assert short["verdict"] == "INVALID_WINDOW" and short["pass"] is False

    nan = vacuum_verdict({"vacuum_window_ok": True,
                          "vacuum_freq_rel_error": float("nan")})
    assert nan["verdict"] == "FAIL"


def test_plate_verdict_is_three_way():
    good = plate_verdict({"window_ok_2cyc": True, "period_fit_valid": True,
                          "f_ratio_error": 0.5 * TOL})
    assert good["verdict"] == "PASS" and good["pass"] is True

    bad = plate_verdict({"window_ok_2cyc": True, "period_fit_valid": True,
                         "f_ratio_error": 2.0 * TOL})
    assert bad["verdict"] == "FAIL"

    # The shipped v4 row: error 1.0 with a 0.5-period record.
    short = plate_verdict({"window_ok_2cyc": False, "period_fit_valid": False,
                           "f_ratio_error": 1.0})
    assert short["verdict"] == "INVALID_WINDOW" and short["pass"] is False

    # A rejected damped-sinusoid fit invalidates the row even if the window is long.
    rejected = plate_verdict({"window_ok_2cyc": True, "period_fit_valid": False,
                              "f_ratio_error": 0.0})
    assert rejected["verdict"] == "INVALID_WINDOW"

    errored = plate_verdict({"error": "ValueError: boom"})
    assert errored["verdict"] == "ERROR" and errored["pass"] is False


def test_structure_checks_propagates_section_and_gates_the_window():
    """Short record: the section is the requested one and the row is invalid."""
    props = _section()
    st = structure_checks(n_elements=4, n_steps=40, dt=5.0e-3, props=props)

    assert st["section_consistent"] is True, (
        "free-vibration driver must receive the section the analytic "
        "reference is computed from (pre-fix v4 used the unit section)")
    assert st["section_simulated"]["A"] == pytest.approx(props.A)
    assert st["section_simulated"]["I"] == pytest.approx(props.I)

    # 40 steps x 5e-3 s = 0.2 s against T ~ 0.98 s -> far below the 2-cycle gate.
    assert st["vacuum_window_ok"] is False
    assert math.isnan(st["vacuum_frequency"])
    assert st["f1_analytic"] == pytest.approx(1.0216699, rel=1e-6)
    assert vacuum_verdict(st)["verdict"] == "INVALID_WINDOW"


def test_plate_study_does_not_fall_back_to_the_window_length():
    """A too-short record yields NaN, not the legacy window artifact."""
    pm = plate_added_mass_study(c=4, lattice=(48, 48), n_steps=60)
    assert pm["period_fit_valid"] is False
    assert pm["window_ok_2cyc"] is False
    assert period_window_ok(pm["n_steps"], 1.0, pm["period_reference"], 2.0) is False
    assert math.isnan(pm["f_ratio_measured"])
    # the legacy estimate is still recorded for diagnosis, but never used
    assert pm["period_legacy"] == pytest.approx(60.0, abs=1e-9) or \
        math.isnan(pm["period_legacy"])
    assert plate_verdict(pm)["verdict"] == "INVALID_WINDOW"


@pytest.mark.skipif(not os.path.exists(ARTIFACTS), reason="evidence.json absent")
def test_shipped_evidence_is_self_consistent():
    """The archived full-run evidence must carry the corrected, gated rows."""
    with open(ARTIFACTS) as fh:
        ev = json.load(fh)

    st = ev["structure"]
    assert st["section_consistent"] is True
    assert st["vacuum_window_ok"] is True
    assert st["vacuum_cycles_in_record"] >= 2.0
    assert st["vacuum_freq_rel_error"] <= TOL
    assert st["vacuum_frequency"] != pytest.approx(43.04812834224599, rel=1e-6)

    pm = ev["plate_added_mass"]
    assert pm["period_fit_valid"] is True
    assert pm["n_steps"] * 1.0 >= 2.0 * pm["period_reference"]
    assert plate_verdict(pm)["verdict"] in ("PASS", "FAIL")
    assert pm["f_ratio_measured"] != pytest.approx(1.5811388300841898, rel=1e-6)

    status = ev["status"]
    assert status["invalid_windows"] == []
    # The pre-fix v4 assertion demanded `n_pass == n_checks`, which its own
    # shipped artefact already violated (the plate row was INVALID_WINDOW).  A
    # configuration-difference row must be published as NOT passing instead of
    # being upgraded to PASS to satisfy a counter.
    assert status["n_checks"] == len(ev["table"])
    assert status["n_pass"] == sum(1 for r in ev["table"] if r["pass"])
    for r in ev["table"]:
        if r.get("verdict") == "OUT_OF_REF_BAND":
            assert r["pass"] is False, r
        elif r["kind"] == "relative" and "optimal fixed relaxation" not in r["item"]:
            assert r["pass"] is True, r

# ---------------------------------------------------------------------------
# m22878 adversarial-review additions: the shipped v4 table conflated the
# published LIFE band with the applied sanity window, and rendered iteration
# caps as reference measurements.
def test_force_band_verdict_keeps_reference_and_sanity_band_apart():
    from xval_implicit_ibm_aitken_v4 import force_band_verdict

    # The shipped v4 row: 61.1 % printed next to "ref 11-37 %" and marked PASS.
    shipped = force_band_verdict(0.6111655919181167)
    assert shipped["verdict"] == "OUT_OF_REF_BAND"
    assert shipped["pass"] is False
    assert shipped["ref_band_deviation"] == pytest.approx((0.6111655919181167 - 0.37) / 0.37)

    # Inside the published band -> the only outcome that counts as a pass.
    assert force_band_verdict(0.25)["verdict"] == "IN_BAND"
    assert force_band_verdict(0.25)["pass"] is True

    # Outside the sanity window -> hard fail, not "configuration difference".
    assert force_band_verdict(0.95)["verdict"] == "OUTSIDE_SANITY"
    assert force_band_verdict(0.95)["pass"] is False


@pytest.mark.skipif(not os.path.exists(ARTIFACTS), reason="evidence.json absent")
def test_shipped_table_has_no_iteration_cap_disguised_as_a_reference():
    """No `relative` row may carry the solver's iteration cap as its reference."""
    with open(ARTIFACTS) as fh:
        ev = json.load(fh)
    for row in ev["table"]:
        if row["kind"] != "relative":
            continue
        assert isinstance(row["ref"], (int, float)), row
        assert row["ref"] < 200, f"iteration cap leaked into a relative row: {row}"

    # The honest counterpart of the "Aitken beats fixed relaxation" claim: the
    # optimal fixed factor omega* = 1/(1-g) converges in fewer iterations than
    # Aitken for every gain in the shipped scalar study, so the advantage claim
    # is only true against a deliberately unstable omega = 1.
    by = {(r["gain"], r["mode"]): r for r in ev["aitken_scalar"]}
    for g in sorted({r["gain"] for r in ev["aitken_scalar"]}):
        assert by[(g, "fixed_opt")]["iterations"] < by[(g, "aitken")]["iterations"]
    opt_rows = [r for r in ev["table"] if "optimal fixed relaxation" in r["item"]]
    assert opt_rows and not any(r["pass"] for r in opt_rows)
