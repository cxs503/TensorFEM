"""Guards for the CASE FEM-HERTZ-CONTACT truncation-vs-model-bias study.

The anchor certificate of ``fem_hertz_contact_loadcontrol`` raised, and left
open (finding F2), the question of why the indentation error saturates at
``-1.01 %`` on every surface mesh.  ``truncation_extrapolation.py`` answers it by
extrapolating a *depth* ladder, and these tests guard the answer on three levels
(mirroring ``tests/test_fem_structural_benchmarks.py``):

1. **Arithmetic** -- the depth-extrapolation routine is exercised on synthetic
   sequences with a known limit, order and failure mode (no solver involved).
2. **Decision rule** -- the pre-declared rule that separates "unconverged
   truncation" from "model bias" must fire in both directions, so the committed
   verdict cannot be an artefact of the rule.
3. **Evidence self-consistency** -- the committed artifact is re-read and every
   published gate value is recomputed from the committed ladder rows.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "benchmarks" / "verified" / "fem_hertz_contact_loadcontrol"


def _load_study():
    verified = ANCHOR.parent
    for path in (str(verified), str(ANCHOR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(
        "hertz_truncation_extrapolation", ANCHOR / "truncation_extrapolation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STUDY = _load_study()


# ---------------------------------------------------------------------------
# 1. arithmetic
# ---------------------------------------------------------------------------


def test_recovers_known_limit_and_order():
    """e(d) = -1 + 300/d must extrapolate to -1 with observed order 1."""
    depths = [24.0, 48.0, 96.0, 192.0, 384.0]
    errors = [-1.0 + 300.0 / d for d in depths]
    out = STUDY.extrapolate_errors(depths, errors, ratio=2.0)
    assert out["blocked"] is None
    assert out["observed_order"] == pytest.approx(1.0, abs=1e-9)
    assert out["error_infinite_domain_pct"] == pytest.approx(-1.0, abs=1e-9)


def test_decays_to_zero():
    """A pure truncation error e = -200 d^-1.5 extrapolates to zero."""
    depths = [48.0, 96.0, 192.0]
    errors = [-200.0 / d**1.5 for d in depths]
    out = STUDY.extrapolate_errors(depths, errors, ratio=2.0)
    assert out["magnitude_decay"] is True
    assert out["observed_order"] == pytest.approx(1.5, rel=1e-9)
    assert abs(out["error_infinite_domain_pct"]) < 1e-9


def test_non_monotone_is_blocked_not_extrapolated():
    out = STUDY.extrapolate_errors([1.0, 2.0, 4.0], [0.5, 0.8, 0.4], ratio=2.0)
    assert out["blocked"] is not None and "non_monotone" in out["blocked"]
    assert out["error_infinite_domain_pct"] is None


def test_saturated_sequence_is_blocked():
    out = STUDY.extrapolate_errors([1.0, 2.0, 4.0], [1.01, 1.01, 1.01], ratio=2.0)
    assert out["observed_order"] is None
    assert out["error_infinite_domain_pct"] is None
    assert out["blocked"] is not None


def test_truncation_insensitive_quantity_is_blocked():
    """A near-field measure that ignores the boundary must not be extrapolated."""
    out = STUDY.extrapolate_errors(
        [96.0, 192.0, 384.0], [-0.5316, -0.5317, -0.5319], ratio=2.0
    )
    assert out["blocked"] is not None and "truncation_insensitive" in out["blocked"]
    assert out["error_infinite_domain_pct"] is None
    assert out["ladder_spread_pp"] < STUDY.TRUNCATION_INSENSITIVE_PP


def test_rejects_non_uniform_depth_steps():
    with pytest.raises(ValueError):
        STUDY.extrapolate_errors([96.0, 200.0, 384.0], [-1.0, -0.7, -0.55])


def test_rejects_short_ladder():
    with pytest.raises(ValueError):
        STUDY.extrapolate_errors([96.0, 192.0], [-1.0, -0.7])


# ---------------------------------------------------------------------------
# 2. decision rule fires both ways
# ---------------------------------------------------------------------------


def _study_with(plateau, limit, order=1.2):
    """Minimal study payload whose indentation entry carries the wanted limit."""
    depths = [96.0, 192.0, 384.0]
    if limit == 0.0:
        errors = [plateau / 3**i for i in range(3)]
    else:
        # e - limit must decay like d^-order
        errors = [limit + (plateau - limit) / 2 ** (order * i) for i in range(3)]
    entry = {
        "a_over_R": 0.05,
        "h_over_a": 0.015,
        "levels": [
            {"domain_depth_over_a": d, "error_pct": {"indentation": e}}
            for d, e in zip(depths, errors)
        ],
        "indentation": STUDY.extrapolate_errors(depths, errors, 2.0),
        "contact_radius": STUDY.extrapolate_errors(depths, [-0.5, -0.5, -0.5], 2.0),
        "peak_pressure": STUDY.extrapolate_errors(depths, [0.6, 0.6, 0.6], 2.0),
    }
    return {"a_over_R_0.05": entry}


def test_rule_accepts_truncation_hypothesis():
    decision = STUDY.decide(_study_with(-1.01, 0.0))
    assert decision["verdict"] == "H_trunc"
    assert abs(decision["evidence"]["indentation_extrapolated_pct"]) < 1e-9


def test_rule_flags_model_bias():
    """A residual above the declared 0.5 % bound must read as a model bias."""
    decision = STUDY.decide(_study_with(-1.01, -0.9))
    assert decision["verdict"] == "H_bias"
    assert abs(decision["evidence"]["indentation_extrapolated_pct"]) > 0.5


# ---------------------------------------------------------------------------
# 3. committed evidence self-consistency
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact():
    path = ANCHOR / "truncation_extrapolation.json"
    if not path.exists():
        pytest.skip("truncation_extrapolation.json not committed")
    return json.loads(path.read_text())


def test_committed_verdict_is_regraded_by_this_test(artifact):
    """Recompute the decision from the committed rows; it must not drift."""
    regraded = STUDY.decide(artifact["study"])
    assert regraded["verdict"] == artifact["decision"]["verdict"]
    assert regraded["evidence"]["indentation_extrapolated_pct"] == pytest.approx(
        artifact["decision"]["evidence"]["indentation_extrapolated_pct"], rel=1e-9
    )


def test_committed_gate_values_are_recomputable(artifact):
    main = artifact["study"]["a_over_R_0.05"]
    delt = main["indentation"]
    gates = {g["id"]: g for g in artifact["gate_table"]}
    assert set(gates) >= {
        "truncation_extrapolated_indentation",
        "truncation_residual_model_bias",
        "truncation_ladder_monotone_aR005",
        "truncation_order_observed",
        "truncation_insensitive_contact_radius_aR005",
    }
    assert gates["truncation_extrapolated_indentation"]["value"] == pytest.approx(
        abs(delt["error_infinite_domain_pct"]), rel=1e-9
    )
    assert gates["truncation_insensitive_contact_radius_aR005"]["value"] == pytest.approx(
        main["contact_radius"]["ladder_spread_pp"], rel=1e-9
    )
    # a FAIL is a result, not an error: never assert `passed` blindly, assert that
    # the recorded verdict follows from the recorded value.
    for gate in gates.values():
        assert math.isfinite(gate["value"])
        expected = (
            gate["value"] <= gate["threshold"]
            if gate["operator"] == "<="
            else gate["value"] >= gate["threshold"]
        )
        assert gate["passed"] is expected


def test_committed_ladder_is_the_declared_one(artifact):
    criteria = artifact["criteria"]
    delt = artifact["study"]["a_over_R_0.05"]["indentation"]
    assert delt["depths"] == criteria["depths_over_a"]
    assert artifact["study"]["a_over_R_0.05"]["h_over_a"] == criteria["h_over_a"]
    assert delt["magnitude_decay"] is True


def test_anchor_certificate_carries_the_truncation_gates():
    """The anchor certificate must expose these gates, not keep them in a side file."""
    path = ANCHOR / "gates_verified.json"
    if not path.exists():
        pytest.skip("gates_verified.json missing")
    doc = json.loads(path.read_text())
    ids = {g["id"] for g in doc["gates"]}
    assert "truncation_extrapolated_indentation" in ids
    assert doc["passed"] is True