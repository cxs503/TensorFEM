import copy,os

import pytest

from tensorfem.benchmark_cases import seal_archive
from tensorfem.shell_benchmark_suite import (
    classical_shell_cases,compare_suites,read_suite,run_classical_shell_suite,
    seal_suite,validate_suite,write_suite,
)


def test_quick_classical_suite_has_traceable_coarse_failures_and_fine_passes():
    suite=run_classical_shell_suite(tier="quick")
    assert suite["passed"] and len(suite["suite_hash"])==64 and len(suite["cases"])==4
    for archive in suite["cases"]:
        rows=archive["results"]
        assert any(not x["passed"] for x in rows) # retained coarse negative evidence
        assert all(x["passed"] for x in rows if x["role"]=="qualification")
        assert archive["case"]["citation"]["doi"].startswith("10.")
        assert archive["case"]["capability_boundaries"]


def test_suite_json_hash_and_tamper_fail_closed(tmp_path):
    suite=run_classical_shell_suite();path=tmp_path/"shell-suite.json"
    write_suite(suite,path);assert read_suite(path)==suite
    bad=copy.deepcopy(suite);bad["cases"][0]["results"][-1]["computed"]*=2
    with pytest.raises(ValueError):validate_suite(bad)


def test_per_case_per_mesh_drift_and_suite_version():
    old=run_classical_shell_suite();new=copy.deepcopy(old);new["suite_version"]="1.0.1"
    target=new["cases"][-1];target["results"][-1]["computed"]*=1.005
    new["cases"][-1]=seal_archive(target);new=seal_suite(new)
    report=compare_suites(old,new,drift_tolerance=.01)
    assert report["passed"] and report["current_version"]=="1.0.1"
    target=new["cases"][-1];target["results"][-1]["computed"]*=1.1
    new["cases"][-1]=seal_archive(target);new=seal_suite(new)
    report=compare_suites(old,new,drift_tolerance=.01)
    assert not report["passed"]
    assert any(not x["passed"] for x in report["cases"][target["case"]["case_id"]]["results"])


def test_full_classical_suite_is_release_opt_in():
    if os.environ.get("TENSORFEM_FULL_SHELL_BENCHMARKS")!="1":return
    quick=run_classical_shell_suite(tier="quick");full=run_classical_shell_suite(tier="full")
    assert full["passed"]
    q={x["case"]["case_id"]:x for x in quick["cases"]}
    for actual in full["cases"]:
        recorded=q[actual["case"]["case_id"]]
        for a,b in zip(actual["results"],recorded["results"]):
            assert abs(a["computed"]-b["computed"])/abs(a["computed"])<1e-10
