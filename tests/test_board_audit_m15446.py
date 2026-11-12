"""Guards the m15446 board-audit verdict against silent drift.

The verdict (L2) rests on three machine-checkable facts; if any of them changes
upstream (benchmark row registered, module wired into the solver, artifacts
published to main) this test fails loudly instead of letting a stale verdict
stand.
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY = os.path.join(HERE, "docs", "board_audit", "m15645", "verify_output.json")


def _v():
    with open(VERIFY) as fh:
        return json.load(fh)


def test_recomputation_agrees_with_artifact_flags():
    """Our independent pass/fail must match the artifact's own flags (no mislabelling found)."""
    v = _v()
    assert v["criteria"]["flag_disagreements"] == []
    assert v["criteria"]["n_passed"] + v["criteria"]["n_failed"] == v["criteria"]["n_criteria"] == 10


def test_verdict_level_is_what_the_ladder_gives():
    v = _v()
    assert v["level"]["ladder"]["L1_tests"] is True
    assert v["level"]["ladder"]["L2_impl+entry+run_artifacts"] is True
    assert v["level"]["assigned_level"] == 2


def test_l3_and_l4_blockers_are_real():
    v = _v()
    # no main-solver caller -> L3 blocked
    assert v["integration"]["main_solver_callers"] == []
    assert v["integration"]["exported_in_package_init"] is False
    # no registered benchmark referee row for this anchor -> L4 blocked
    assert v["benchmark"]["registered_rows"] == 0


def test_run_artifacts_are_hashed_and_archived():
    v = _v()
    files = {os.path.basename(f["path"]): f for f in v["artifacts"]["files"]}
    assert set(files) == {"repro.json", "fig_size_effect.png", "fig_load_cmod.png"}
    for f in files.values():
        assert f["exists"] and len(f["sha256"]) == 64 and os.path.exists(os.path.join(HERE, f["copied_to"]))


def test_key_failed_criteria_are_not_silently_passing():
    v = _v()
    got = {c["criterion"]: c for c in v["criteria"]["checks"]}
    # the three physically damning ones
    assert got["K10_gf"]["got"] > 0.5 and not got["K10_gf"]["passed_recomputed"]
    assert got["K4_lefm"]["got"] < 0.6 and not got["K4_lefm"]["passed_recomputed"]
    assert got["K3_balance"]["got"] > 1e10 and not got["K3_balance"]["passed_recomputed"]