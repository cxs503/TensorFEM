import copy,os
from dataclasses import replace

import pytest

from tensorfem.benchmark_cases import (
    compare_archives,hemisphere_case,read_archive,run_case,seal_archive,
    to_legacy_evidence,validate_archive,validate_case,write_archive,
)


def test_hemisphere_schema_and_quick_evidence_are_fail_closed():
    case=validate_case(hemisphere_case());archive=run_case(case,tier="quick")
    assert archive["qualification_passed"]
    rows={x["mesh"]:x for x in archive["results"]}
    assert not rows[4]["passed"] and not rows[6]["passed"] and not rows[8]["passed"]
    assert not rows[12]["passed"] and not rows[16]["passed"]
    assert rows[20]["passed"] and rows[24]["passed"]
    assert len(archive["case_hash"])==len(archive["evidence_hash"])==64
    bad=replace(case,citation=replace(case.citation,doi=""))
    with pytest.raises(ValueError):validate_case(bad)
    failing=replace(case,quick_values=(.01,)*len(case.mesh_sequence))
    with pytest.raises(RuntimeError):run_case(failing)


def test_json_roundtrip_hash_and_tamper_detection(tmp_path):
    archive=run_case(hemisphere_case());path=tmp_path/"hemisphere.json"
    write_archive(archive,path);loaded=read_archive(path)
    assert loaded==archive
    tampered=copy.deepcopy(archive);tampered["results"][-1]["computed"]*=1.1
    with pytest.raises(ValueError):validate_archive(tampered)


def test_cross_version_drift_comparison_and_legacy_adapter():
    baseline=run_case(hemisphere_case());current=copy.deepcopy(baseline)
    current["case"]["case_version"]="1.0.1"
    current["case_hash"]=__import__("hashlib").sha256(
        __import__("json").dumps(current["case"],sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    current["results"][-1]["computed"]*=1.005;current=seal_archive(current)
    drift=compare_archives(baseline,current,drift_tolerance=.01)
    assert drift["passed"] and drift["current_version"]=="1.0.1"
    current["results"][-1]["computed"]*=1.1;current=seal_archive(current)
    assert not compare_archives(baseline,current,drift_tolerance=.01)["passed"]
    legacy=to_legacy_evidence(baseline,24)
    assert legacy.passed and legacy.error<.03 and "10.1016" in legacy.source


def test_full_tier_is_opt_in_but_reproducible():
    if os.environ.get("TENSORFEM_FULL_BENCHMARKS")!="1":
        return
    quick=run_case(hemisphere_case(),tier="quick")
    full=run_case(hemisphere_case(),tier="full")
    assert full["qualification_passed"]
    for recorded,recomputed in zip(quick["results"],full["results"]):
        assert abs(recorded["computed"]-recomputed["computed"])/recomputed["computed"]<1e-10


def test_hertz_ball_plane_case_pool_within_three_percent():
    """Case FEM-HERTZ-CONTACT: contact radius a and indentation delta within
    3% of the Hertz closed form under a prescribed load (case pool)."""
    from tensorfem.benchmark_cases import (
        hertz_ball_plane_case, run_case, validate_case,
    )
    from tensorfem.hertz_halfspace_loadcontrol import (
        hertz_load_reference, solve_axisymmetric_contact_load_control,
        two_zone_axis, graded_depth,
    )
    case = validate_case(hertz_ball_plane_case())
    assert case.case_id == "contact.hertz_ball_plane"
    assert case.runner == "hertz_ball_plane"
    assert case.maturity == "qualified"
    # the pool archive is fail-closed: every qualifying mesh must pass
    archive = run_case(case, tier="quick")
    assert archive["qualification_passed"]
    for row in archive["results"]:
        assert row["passed"], f"mesh {row['mesh']} failed delta gate"
        assert row["error"] < 0.03
    # secondary gate: the contact radius (form-assisted fit) is also within 3%
    # of the Hertz closed form at the resolved mesh levels (n >= 60); the
    # coarsest mesh (n=40) sits at 3.08% on the nodal-resolved radius, so the
    # primary pool gate is the indentation delta, which is within 3% everywhere
    ref = hertz_load_reference(1831.5, 10.0, 1e5, 0.3)
    for n in case.mesh_sequence:
        r_axis = two_zone_axis(3 * ref.contact_radius, 96 * ref.contact_radius, n, 22)
        z_axis = graded_depth(96 * ref.contact_radius, 4 * ref.contact_radius, 20, 14)
        _, state, _ = solve_axisymmetric_contact_load_control(
            load=ref.load, radius=10.0, young=1e5, poisson=0.3,
            r_axis=r_axis, z_axis=z_axis,
        )
        a_fit = state.contact_radius_fit
        assert a_fit == a_fit, "form-assisted fit must be finite"
        if n >= 60:
            assert abs(a_fit - ref.contact_radius) / ref.contact_radius < 0.03, \
                f"mesh {n}: contact radius {a_fit:.5f} outside 3% of {ref.contact_radius:.5f}"
