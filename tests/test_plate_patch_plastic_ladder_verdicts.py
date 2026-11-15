"""Verdict layer of the plate-panel plastic limit-load ladder (anchor GAP-PP-3).

The case document `docs/CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER.md` states the
acceptance verdicts of the anchor.  Those verdicts must be *derived from the
artifacts*, otherwise the document can claim a pass that the data do not
support.  This module pins that derivation:

* ``ladder_monotone`` must not call a one-mesh or gappy ladder monotone -- a
  missing limit load is not evidence of convergence;
* ``bracket_verdict`` must reject a limit load that leaves the rigid-plastic
  bracket in *either* direction (above an upper bound falsifies the bound or the
  FE model; below a lower bound falsifies the FE model);
* ``converged_limit`` must label a single-mesh number as *not extrapolated*
  instead of presenting it as mesh-independent;
* ``acceptance_table`` must emit ``not_run`` where the artifacts cannot answer,
  and must judge the limit load against the *converged* value only.

All tests are pure-python on synthetic summaries: no FE run, no queue.
"""
import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / \
    "patch_plate_plastic_ladder_figures.py"


def _module():
    spec = importlib.util.spec_from_file_location("patch_plate_plastic_ladder_figures",
                                                  _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOUNDS = {"ilyushin_sine": 1.5151816676552382e6,
          "yield_line": 1.842240e6,
          "strip_lower": 0.8187733333333334e6,
          "johansen_twist": 2.379982e6}


def _entry(nx: int, limit, *, first_yield_error=0.008, spread=0.05,
           geometry="free") -> dict:
    return {"tag": f"nx{nx}_{geometry}", "nx": nx, "geometry": geometry,
            "limit_tight": limit, "limit_loose": limit,
            "criteria_spread_tight": spread,
            "first_yield": 0.876e6, "first_yield_error": first_yield_error,
            "permanent_deflection": 5.21e-3, "permanent_set_valid": True,
            "peak_factor": 1.56e6, "elapsed_s": 10.0, "moment_plateau": 1.0,
            "section_capacity_ratio": 0.98,
            "references": {"bounds": dict(BOUNDS)}}


def _summary(entries, **extra) -> dict:
    return {"records": list(entries), **extra}


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------

def test_monotone_ladder_is_recognised_in_the_decreasing_direction():
    module = _module()
    assert module.ladder_monotone([1.20e6, 1.15e6, 1.13e6]) is True
    assert "decreasing" in module.convergence_trend([1.20e6, 1.15e6, 1.13e6])


def test_increasing_ladder_is_also_monotone_but_labelled_differently():
    module = _module()
    assert module.ladder_monotone([1.10e6, 1.13e6, 1.14e6]) is True
    assert "increasing" in module.convergence_trend([1.10e6, 1.13e6, 1.14e6])


def test_a_non_monotone_ladder_is_not_called_convergent():
    module = _module()
    assert module.ladder_monotone([1.20e6, 1.16e6, 1.18e6]) is False
    assert module.convergence_trend([1.20e6, 1.16e6, 1.18e6]) == "not monotone"


@pytest.mark.parametrize("values", [[1.2e6], [], [1.2e6, None, 1.1e6], [None, None]])
def test_missing_values_are_never_read_as_convergence(values):
    """A single mesh or a ramp on which no criterion fired cannot be monotone."""
    module = _module()
    assert module.ladder_monotone(values) is None


# ---------------------------------------------------------------------------
# bracket admissibility
# ---------------------------------------------------------------------------

def test_a_limit_inside_the_bracket_is_admissible():
    module = _module()
    verdict, detail = module.bracket_verdict(1.20e6, BOUNDS)
    assert verdict == "admissible"
    assert "inside bracket" in detail


def test_a_limit_above_an_upper_bound_is_rejected():
    """Above a rigorous upper bound the bound *or* the FE model is wrong."""
    module = _module()
    verdict, _ = module.bracket_verdict(1.95e6, BOUNDS)
    assert verdict.startswith("violated")
    assert "upper" in verdict


def test_a_limit_below_a_lower_bound_is_rejected():
    """An approximate collapse load cannot sit below a rigorous lower bound."""
    module = _module()
    verdict, _ = module.bracket_verdict(0.75e6, BOUNDS)
    assert verdict.startswith("violated")
    assert "lower" in verdict


def test_bracket_verdict_withholds_without_an_admissible_upper_bound():
    module = _module()
    verdict, _ = module.bracket_verdict(1.2e6, {"strip_lower": 0.8e6})
    assert verdict == "withheld"


# ---------------------------------------------------------------------------
# converged limit
# ---------------------------------------------------------------------------

def test_single_mesh_number_is_reported_as_not_extrapolated():
    module = _module()
    limit, label = module.converged_limit(_summary([_entry(8, 1.1997e6)]))
    assert limit == pytest.approx(1.1997e6)
    assert "not extrapolated" in label


def test_richardson_value_wins_over_the_finest_mesh():
    module = _module()
    summary = _summary([_entry(8, 1.20e6), _entry(12, 1.15e6), _entry(16, 1.13e6)],
                       extrapolation_free={"limit_infinity": 1.11e6,
                                           "order_estimate": 2.0})
    limit, label = module.converged_limit(summary)
    assert limit == pytest.approx(1.11e6)
    assert "Richardson" in label


def test_no_limit_anywhere_gives_none_and_withholds():
    module = _module()
    limit, label = module.converged_limit(_summary([_entry(8, None)]))
    assert limit is None
    assert "no limit load" in label


# ---------------------------------------------------------------------------
# acceptance table
# ---------------------------------------------------------------------------

def _rows(module, summary) -> dict[str, str]:
    rows = module.acceptance_table(summary, summary["records"])
    assert all(row.startswith("| ") and row.endswith(" |") for row in rows)
    return {"\n".join(row.split("|")[1:3]).strip(): row for row in rows}


def test_acceptance_judges_the_limit_against_the_converged_value():
    module = _module()
    summary = _summary([_entry(8, 1.40e6), _entry(12, 1.45e6), _entry(16, 1.47e6)],
                       extrapolation_free={"limit_infinity": 1.50e6,
                                           "order_estimate": 1.4})
    rows = _rows(module, summary)
    row = next(v for k, v in rows.items() if "Ilyushin" in k)
    # the converged 1.50 MPa is within 10 % of the 1.515 MPa sine bound, while
    # the coarse nx = 8 value (1.40 MPa) is not: the converged value must decide
    assert "**met**" in row and "not met" not in row


def test_acceptance_flags_a_limit_load_outside_the_bracket():
    module = _module()
    summary = _summary([_entry(8, 1.30e6), _entry(12, 1.35e6), _entry(16, 1.38e6)],
                       extrapolation_free={"limit_infinity": 2.10e6,
                                           "order_estimate": 1.4})
    rows = _rows(module, summary)
    bracket = next(v for k, v in rows.items() if "bracket" in k and "rigid" in k)
    assert "not met" in bracket and "violated (above upper bound)" in bracket


def test_a_single_mesh_is_reported_as_not_run_not_as_converged():
    module = _module()
    summary = _summary([_entry(8, 1.20e6)])
    rows = _rows(module, summary)
    convergence = next(v for k, v in rows.items() if "mesh convergence" in k)
    assert "not_run" in convergence


def test_a_non_monotone_ladder_fails_the_convergence_item():
    module = _module()
    summary = _summary([_entry(8, 1.20e6), _entry(12, 1.16e6), _entry(16, 1.18e6)])
    rows = _rows(module, summary)
    convergence = next(v for k, v in rows.items() if "mesh convergence" in k)
    assert "**not met**" in convergence and "not monotone" in convergence


def test_missing_first_yield_evidence_is_not_run():
    module = _module()
    summary = _summary([_entry(8, 1.20e6, first_yield_error=None)])
    rows = _rows(module, summary)
    first = next(v for k, v in rows.items() if "first yield" in k)
    assert "not_run" in first


def test_membrane_runs_do_not_appear_in_the_acceptance_table():
    """Only the in-plane-free ladder is judged against the bending bounds."""
    module = _module()
    summary = _summary([_entry(8, 1.20e6),
                        _entry(8, 2.5e6, geometry="membrane")])
    table = "\n".join(module.acceptance_table(summary, summary["records"]))
    assert "membrane" not in table