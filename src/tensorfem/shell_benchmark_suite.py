"""Versioned classical-shell benchmark cases and suite evidence archive."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,math

from .benchmark_cases import (
    BenchmarkCase,Citation,EVIDENCE_VERSION,ReferenceQuantity,SCHEMA_VERSION,
    compare_archives,hemisphere_case,run_case,validate_archive,
)

SUITE_SCHEMA="tensorfem.benchmark-suite/1.0"


def scordelis_case():
    return BenchmarkCase(SCHEMA_VERSION,"shell.scordelis_lo","1.0.0","Scordelis--Lo cylindrical roof",
        Citation("MacNeal and Harder","A proposed standard set of problems to test finite element accuracy",
                 "Finite Elements in Analysis and Design 1, 3--20",1985,"10.1016/0168-874X(85)90003-4"),
        {"surface":"full cylindrical roof","length":50.,"radius":25.,"half_angle_deg":40.,"thickness":.25},
        {"young_modulus":4.32e8,"poisson_ratio":0.},
        ("rigid diaphragm ends restrain global transverse and vertical translation","one axial rigid-mode gauge"),
        ("uniform global downward surface load 90",),(2,4,6,8,12,16),
        ReferenceQuantity("midspan free-edge vertical displacement",-.3024,"length","signed"),
        "absolute_relative",.03,"qualified",
        ("linear cylindrical shell","metric-consistent Q4","not a general arbitrary-curvature nonlinear shell"),
        "scordelis_lo",(-9.661737106432787,-.21673053837090422,-.2855005478054662,
                        -.2967296525273086,-.30137424009241787,-.3024690512968621),(8,12,16))


def pinched_cylinder_case():
    return BenchmarkCase(SCHEMA_VERSION,"shell.pinched_cylinder","1.0.0","MacNeal--Harder pinched cylinder",
        Citation("MacNeal and Harder","A proposed standard set of problems to test finite element accuracy",
                 "Finite Elements in Analysis and Design 1, 3--20",1985,"10.1016/0168-874X(85)90003-4"),
        {"model":"symmetry octant","full_length":600.,"radius":300.,"thickness":3.},
        {"young_modulus":3e6,"poisson_ratio":.3},
        ("midspan and circumferential reflection symmetry","rigid end diaphragm"),
        ("quarter of unit radial pinch load at octant corner",),(3,4,6,8,12),
        ReferenceQuantity("loaded-point radial displacement",-1.8248e-5,"length","signed"),
        "absolute_relative",.03,"qualified",
        ("linear cylindrical Q4 response","octant symmetry","nonlinear 3x3 integration check is separate"),
        "pinched_cylinder",(-1.798886022029942e-5,-1.7726598846606177e-5,-1.7267332546022934e-5,
                            -1.7636652834589215e-5,-1.8068243904498985e-5),(12,))


def pure_bending_case():
    exact=20/math.pi
    return BenchmarkCase(SCHEMA_VERSION,"shell.large_rotation_pure_bending","1.0.0","90-degree shell-strip pure bending",
        Citation("Bathe and Bolourchi","Large displacement analysis of three-dimensional beam structures",
                 "International Journal for Numerical Methods in Engineering 14, 961--986",1979,"10.1002/nme.1620140703"),
        {"strip_length":10.,"width":1.,"thickness":.1,"prescribed_end_rotation_deg":90.},
        {"young_modulus":1.2e6,"poisson_ratio":0.},
        ("root cross-section clamped","rotation prescribed linearly along strip"),
        ("displacement-controlled pure end rotation",),(1,2,4),
        ReferenceQuantity("quarter-circle tip vertical coordinate",exact,"length","signed"),
        "absolute_relative",.03,"qualified",
        ("geometrically nonlinear rotation-controlled shell strip","linear elastic material","not load-controlled post-buckling"),
        "pure_bending_90deg",(7.0710678118654755,6.532814824384769,6.407288619431961),(2,4))


def classical_shell_cases():return (scordelis_case(),pinched_cylinder_case(),hemisphere_case(),pure_bending_case())


def _canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def _hash(x):return hashlib.sha256(_canonical(x).encode()).hexdigest()


def seal_suite(payload):
    clean={k:v for k,v in payload.items() if k!="suite_hash"}
    return {**clean,"suite_hash":_hash(clean)}


def validate_suite(archive):
    if archive.get("suite_schema")!=SUITE_SCHEMA or not archive.get("cases"):raise ValueError("invalid suite archive")
    if archive.get("suite_hash")!=_hash({k:v for k,v in archive.items() if k!="suite_hash"}):raise ValueError("suite hash mismatch")
    ids=[]
    for case in archive["cases"]:validate_archive(case);ids.append(case["case"]["case_id"])
    if ids!=sorted(ids) or len(ids)!=len(set(ids)):raise ValueError("suite cases must be unique and sorted")
    if archive.get("passed")!=all(x["qualification_passed"] for x in archive["cases"]):raise ValueError("suite status mismatch")
    return archive


def run_classical_shell_suite(*,tier="quick",suite_version="1.0.0"):
    evidence=sorted((run_case(c,tier=tier) for c in classical_shell_cases()),key=lambda x:x["case"]["case_id"])
    payload={"suite_schema":SUITE_SCHEMA,"suite_id":"shell.classical_suite","suite_version":suite_version,
             "tier":tier,"passed":all(x["qualification_passed"] for x in evidence),"cases":evidence}
    archive=seal_suite(payload);validate_suite(archive)
    if not archive["passed"]:raise RuntimeError("classical shell suite failed closed")
    return archive


def write_suite(archive,path):validate_suite(archive);Path(path).write_text(json.dumps(archive,indent=2,sort_keys=True),encoding="utf-8")
def read_suite(path):return validate_suite(json.loads(Path(path).read_text(encoding="utf-8")))


def compare_suites(baseline,current,*,drift_tolerance=.01):
    validate_suite(baseline);validate_suite(current)
    if baseline["suite_id"]!=current["suite_id"]:raise ValueError("different suite IDs")
    old={x["case"]["case_id"]:x for x in baseline["cases"]};new={x["case"]["case_id"]:x for x in current["cases"]}
    if set(old)!=set(new):raise ValueError("suite case membership changed")
    cases={key:compare_archives(old[key],new[key],drift_tolerance=drift_tolerance) for key in sorted(old)}
    return {"suite_id":baseline["suite_id"],"baseline_version":baseline["suite_version"],
            "current_version":current["suite_version"],"tolerance":drift_tolerance,
            "passed":all(x["passed"] for x in cases.values()),"cases":cases}
