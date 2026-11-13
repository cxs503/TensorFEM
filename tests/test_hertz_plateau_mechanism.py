"""Guards for the CASE-FEM-HERTZ-CONTACT indentation-plateau mechanism study.

The anchor certificate left the ``-1.01 %`` indentation plateau unexplained and
the truncation study extrapolated it to ``-0.428 %`` without naming a mechanism.
``farfield_discretisation.py`` / ``nearfield_resolution.py`` /
``converged_model.py`` test three competing mechanisms (stretched far field,
never-refined vertical near-field spacing, irreducible contact-model bias) with
pre-declared thresholds.

The tests guard the study on three levels:

1. **Mesh invariance** -- "only ``n_zfine`` changed" is the central claim of the
   vertical arm, so it is asserted bit-for-bit on the mesh constructor (radial
   node array identical, vertical band identical, ``dz`` exactly halved).
2. **Decision rule** -- the pre-declared two-sided rule must fire in *both*
   directions on synthetic rows, otherwise the committed verdict would be an
   artefact of the rule rather than of the data.
3. **Evidence self-consistency** -- the committed artifacts are re-read and
   every published error percentage and the published Richardson limit are
   recomputed from the committed rows.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "benchmarks" / "verified" / "fem_hertz_contact_loadcontrol"


def _load(name, filename):
    verified = ANCHOR.parent
    for path in (str(verified), str(ANCHOR), str(ROOT / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(name, ANCHOR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FAR = _load("hertz_farfield_discretisation", "farfield_discretisation.py")
ZRES = _load("hertz_nearfield_resolution", "nearfield_resolution.py")


def _artifact(name):
    with open(ANCHOR / name) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. mesh invariance: the vertical arm changes ONLY the vertical spacing
# ---------------------------------------------------------------------------


def test_vertical_refinement_keeps_the_radial_near_field_identical():
    ref, r24, z24 = FAR.mesh_for(768.0, FAR.H_REL, FAR.BASE_OUTER, FAR.BASE_GROW, n_zfine=24)
    ref2, r96, z96 = FAR.mesh_for(768.0, FAR.H_REL, FAR.BASE_OUTER, FAR.BASE_GROW, n_zfine=96)
    assert np.array_equal(r24, r96), "radial nodes must be bit-identical"
    assert ref.contact_radius == ref2.contact_radius
    # the fine band (top 4a) is the only region touched by n_zfine
    band = FAR.BAND * ref.contact_radius
    top24 = z24 >= -band - 1e-12
    top96 = z96 >= -band - 1e-12
    assert top96.sum() == 4 * (top24.sum() - 1) + 1, "fine band must be refined 4x, same vertices"
    # bottom of the domain and the stretched far field are untouched
    assert z24[-1] == z96[-1]
    dz24 = z24[0] - z24[1]
    dz96 = z96[0] - z96[1]
    assert dz96 == pytest.approx(dz24 / 4.0, rel=1e-12)


def test_comparison_groups_scale_with_the_domain_depth_not_the_near_field():
    """The h/a ladder never refined dz -- that is the study's premise."""
    ref, r, z = FAR.mesh_for(96.0, 0.0075, FAR.BASE_OUTER, FAR.BASE_GROW, n_zfine=24)
    dz = (z[0] - z[1]) / ref.contact_radius
    a = ref.contact_radius
    # the fine band (4a) is cut into a FIXED 24 layers -> dz = a/6, independent of h/a
    assert dz == pytest.approx(FAR.BAND / 24.0, rel=1e-9)
    # radial spacing of the same case is an order of magnitude finer
    assert (r[1] - r[0]) / a == pytest.approx(0.0075, rel=1e-9)
    assert dz / ((r[1] - r[0]) / a) > 10.0


# ---------------------------------------------------------------------------
# 2. decision rules fire in both directions
# ---------------------------------------------------------------------------


def _arm(e_delta, n_zfine=24):
    return {
        "label": f"dz_{n_zfine}",
        "n_zfine": n_zfine,
        "near_field_dz_over_a": 1.0 / n_zfine,
        "near_field_dr_over_a": 0.015,
        "n_elements": 1000,
        "wall_s": 1.0,
        "error_pct": {
            "contact_radius_linear": -0.5,
            "indentation": e_delta,
            "peak_pressure_consistent": 0.5,
        },
    }


def test_vertical_rule_accepts_h_zres_when_the_offset_collapses():
    rows = [_arm(-0.48, 24), _arm(-0.25, 48), _arm(-0.10, 96), _arm(-0.09, 96)]
    assert ZRES.combined_verdict(rows)["accepted"] == "H_zres"


def test_vertical_rule_accepts_h_bias_when_refinement_does_nothing():
    rows = [_arm(-0.48, 24), _arm(-0.479, 48), _arm(-0.478, 96), _arm(-0.478, 96)]
    assert ZRES.combined_verdict(rows)["accepted"] == "H_bias"


def test_far_field_rule_accepts_h_far_when_refinement_collapses_the_offset():
    def far_row(e, n):
        return {"error_pct": {"indentation": e}, "n_elements": n}

    rows = [far_row(-0.48, 9040), far_row(-0.20, 14112), far_row(-0.08, 26752)]
    out = FAR.verdict(rows, {"e_inf_pct": -0.4278}, {"e_inf_pct": -0.05})
    assert out["accepted"] == "H_far"
    rows = [far_row(-0.4804, 9040), far_row(-0.4581, 14112), far_row(-0.4609, 26752)]
    out = FAR.verdict(rows, {"e_inf_pct": -0.4278}, {"e_inf_pct": -0.39})
    assert out["accepted"] == "H_bias"


# ---------------------------------------------------------------------------
# 3. committed evidence is self-consistent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["farfield_discretisation.json", "nearfield_resolution.json",
             "converged_model.json"]
)
def test_artifact_reports_reproducible_percentages(name):
    doc = _artifact(name)
    groups = []
    if "farfield_ladder" in doc:
        groups.append(doc["farfield_ladder"])
    if "depth_ladder_refined" in doc:
        groups.append(doc["depth_ladder_refined"])
    if "arms" in doc:
        groups.append(doc["arms"])
    if "depth_ladder_dz_a24" in doc:
        groups.append(doc["depth_ladder_dz_a24"])
    if doc.get("vertical_ladder_a48"):
        groups.append([doc["vertical_ladder_a48"]])
    assert groups
    for rows in groups:
        for row in rows:
            assert row["load_residual"] <= 1e-9
            assert row["penetration_over_delta"] <= 1e-5
            for key, ref_key in (("indentation", "indentation"),
                                 ("contact_radius_linear", "contact_radius"),
                                 ("peak_pressure_consistent", "peak_pressure")):
                got = row["numeric"][key]
                ref = row["analytic"][ref_key]
                assert row["error_pct"][key] == pytest.approx(
                    100.0 * (got / ref - 1.0), rel=1e-9, abs=1e-9
                )


def test_published_richardson_limit_is_recomputed_from_the_rows():
    doc = _artifact("converged_model.json")
    rows = doc["depth_ladder_dz_a24"]
    assert [r["depth_over_a"] for r in rows] == [96.0, 192.0, 384.0, 768.0]
    from _harness import richardson  # noqa: E402  (benchmarks/verified on sys.path)

    errs = [r["error_pct"]["indentation"] for r in rows]
    res = richardson(errs[-3], errs[-2], errs[-1], 2.0)
    assert doc["extrapolation"]["e_inf_pct"] == pytest.approx(
        res["richardson_limit"], rel=1e-9
    )
    assert doc["extrapolation"]["richardson"]["observed_order"] == pytest.approx(
        res["observed_order"], rel=1e-9
    )
    # the study's whole point: the vertical refinement moves the extrapolated
    # residual, i.e. the prior verdict was not mechanism-free
    assert doc["share_of_prior_bias_explained_pct"] > 10.0


def test_vertical_arm_monotone_in_the_refinement_variable():
    doc = _artifact("nearfield_resolution.json")
    arms = doc["arms"]
    dz = [a["near_field_dz_over_a"] for a in arms[:3]]
    ed = [abs(a["error_pct"]["indentation"]) for a in arms[:3]]
    assert dz == sorted(dz, reverse=True)
    assert ed == sorted(ed, reverse=True), "|e_delta| must decay with dz -> 0"
    assert all(a["near_field_dr_over_a"] == arms[0]["near_field_dr_over_a"] for a in arms)


def test_far_field_arm_is_flat_and_rejects_its_hypothesis():
    doc = _artifact("farfield_discretisation.json")
    ladder = doc["farfield_ladder"]
    assert [r["n_grow"] for r in ladder] == sorted(r["n_grow"] for r in ladder)
    assert doc["arm_verdict"]["shift_pp"] <= doc["criteria"]["h_bias_equivalence_pp"]
    assert doc["arm_verdict"]["accepted"] == "H_bias"


def test_case_pool_gates_of_the_anchor_still_hold():
    """This study must not invalidate the certified case-pool numbers."""
    with open(ANCHOR / "gates_verified.json") as fh:
        gates = json.load(fh)
    assert gates["passed"] is True
    assert not gates["failed_gates"]
    by_id = {g["id"]: g for g in gates["gates"]}
    for gid in ("contact_radius_aR005", "indentation_aR005",
                "contact_radius_aR010", "indentation_aR010"):
        assert abs(by_id[gid]["value"]) <= 3.0
    assert len(gates["gates"]) >= 20