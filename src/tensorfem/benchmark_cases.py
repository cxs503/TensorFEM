"""Versioned benchmark case schema, evidence archives and drift comparison."""
from __future__ import annotations
from dataclasses import asdict,dataclass,replace
from pathlib import Path
import hashlib,json


SCHEMA_VERSION="tensorfem.benchmark-case/1.0"
EVIDENCE_VERSION="tensorfem.benchmark-evidence/2.0"


@dataclass(frozen=True)
class Citation:
    authors: str
    title: str
    publication: str
    year: int
    doi: str


@dataclass(frozen=True)
class ReferenceQuantity:
    name: str
    value: float
    unit: str
    sign_policy: str


@dataclass(frozen=True)
class BenchmarkCase:
    schema_version: str
    case_id: str
    case_version: str
    title: str
    citation: Citation
    geometry: dict
    material: dict
    boundary_conditions: tuple[str,...]
    loads: tuple[str,...]
    mesh_sequence: tuple[int,...]
    quantity: ReferenceQuantity
    error_strategy: str
    tolerance: float
    maturity: str
    capability_boundaries: tuple[str,...]
    runner: str
    quick_values: tuple[float,...]
    qualifying_meshes: tuple[int,...]

    def to_dict(self):return asdict(self)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


def _hash(value):return hashlib.sha256(_canonical(value).encode()).hexdigest()


def validate_case(case: BenchmarkCase):
    if case.schema_version!=SCHEMA_VERSION:raise ValueError("unsupported benchmark case schema")
    if not case.case_id or not case.case_version or not case.citation.doi.startswith("10."):
        raise ValueError("case identity and DOI are required")
    if not case.citation.authors or not case.citation.title or case.citation.year<1900:raise ValueError("incomplete citation")
    if not case.geometry or not case.material or not case.boundary_conditions or not case.loads:raise ValueError("physical definition is incomplete")
    if tuple(sorted(set(case.mesh_sequence)))!=case.mesh_sequence or len(case.mesh_sequence)<2:raise ValueError("mesh sequence must be unique and increasing")
    if len(case.quick_values)!=len(case.mesh_sequence):raise ValueError("one quick value is required per mesh")
    if not set(case.qualifying_meshes)<=set(case.mesh_sequence):raise ValueError("qualifying mesh is absent")
    if not case.quantity.unit or case.quantity.value==0 or case.quantity.sign_policy not in ("signed","magnitude"):
        raise ValueError("invalid reference quantity")
    if case.error_strategy!="absolute_relative" or not 0<case.tolerance<=.03:raise ValueError("unsupported error policy")
    if case.maturity not in ("experimental","qualified","stable"):raise ValueError("invalid maturity")
    if not case.capability_boundaries or case.runner not in ("hemisphere_18deg",):raise ValueError("missing capability boundary or runner")
    return case


def case_hash(case):validate_case(case);return _hash(case.to_dict())


def hemisphere_case():
    return BenchmarkCase(SCHEMA_VERSION,"shell.hemisphere_18deg","1.0.0",
        "MacNeal--Harder hemispherical shell with 18-degree hole",
        Citation("MacNeal and Harder","A proposed standard set of problems to test finite element accuracy",
                 "Finite Elements in Analysis and Design 1, 3--20",1985,"10.1016/0168-874X(85)90003-4"),
        {"midsurface":"quarter hemisphere","radius":10.,"hole_polar_angle_deg":18.,"thickness":.04},
        {"young_modulus":6.825e7,"poisson_ratio":.3},
        ("meridional symmetry constraints","one vertical rigid-translation gauge"),
        ("alternating unit radial equator point loads",),(4,6,8,12,16),
        ReferenceQuantity("loaded-equator radial displacement",.0924,"length","magnitude"),
        "absolute_relative",.03,"qualified",
        ("linear small-strain quarter model","projected MITC-like Q4 shell","not nonlinear doubly-curved shell"),
        "hemisphere_18deg",(.061187831142486326,.07505940982508462,.08597612461589936,
                            .09176267439836297,.09320280296640622),(12,16))


def _computed(case,tier):
    if tier=="quick":return case.quick_values
    if tier!="full":raise ValueError("tier must be quick or full")
    if case.runner=="hemisphere_18deg":
        from .spherical_shell import hemisphere_with_hole
        return tuple(hemisphere_with_hole(n).displacement for n in case.mesh_sequence)
    raise ValueError("runner is not available")


def seal_archive(payload):
    clean={k:v for k,v in payload.items() if k!="evidence_hash"}
    return {**clean,"evidence_hash":_hash(clean)}


def validate_archive(archive):
    if archive.get("evidence_schema")!=EVIDENCE_VERSION:raise ValueError("unsupported evidence schema")
    if archive.get("case_hash")!=_hash(archive["case"]):raise ValueError("case hash mismatch")
    if archive.get("evidence_hash")!=_hash({k:v for k,v in archive.items() if k!="evidence_hash"}):raise ValueError("evidence hash mismatch")
    if not archive.get("results"):raise ValueError("empty benchmark evidence")
    return archive


def run_case(case: BenchmarkCase,*,tier="quick"):
    validate_case(case);values=_computed(case,tier);results=[];reference=case.quantity.value
    for mesh,value in zip(case.mesh_sequence,values):
        computed=abs(float(value)) if case.quantity.sign_policy=="magnitude" else float(value)
        error=abs(computed-reference)/abs(reference)
        results.append({"mesh":mesh,"computed":computed,"reference":reference,"unit":case.quantity.unit,
                        "error":error,"tolerance":case.tolerance,"passed":error<case.tolerance,
                        "role":"qualification" if mesh in case.qualifying_meshes else "convergence"})
    by_mesh={x["mesh"]:x for x in results}
    qualified=all(by_mesh[n]["passed"] for n in case.qualifying_meshes)
    case_document=json.loads(_canonical(case.to_dict()))
    payload={"evidence_schema":EVIDENCE_VERSION,"case":case_document,"case_hash":case_hash(case),
             "tier":tier,"qualification_passed":qualified,"results":results}
    archive=seal_archive(payload);validate_archive(archive)
    if case.maturity in ("qualified","stable") and not qualified:raise RuntimeError("qualified benchmark failed closed")
    return archive


def write_archive(archive,path):
    validate_archive(archive);Path(path).write_text(json.dumps(archive,indent=2,sort_keys=True),encoding="utf-8")


def read_archive(path):return validate_archive(json.loads(Path(path).read_text(encoding="utf-8")))


def compare_archives(baseline,current,*,drift_tolerance=.01):
    validate_archive(baseline);validate_archive(current)
    if baseline["case"]["case_id"]!=current["case"]["case_id"]:raise ValueError("cannot compare different cases")
    old={x["mesh"]:x for x in baseline["results"]};new={x["mesh"]:x for x in current["results"]}
    common=sorted(set(old)&set(new))
    if not common or not 0<drift_tolerance<=.03:raise ValueError("invalid drift comparison")
    rows=[]
    for mesh in common:
        drift=abs(new[mesh]["computed"]-old[mesh]["computed"])/max(abs(old[mesh]["computed"]),1e-30)
        rows.append({"mesh":mesh,"baseline":old[mesh]["computed"],"current":new[mesh]["computed"],"drift":drift,"passed":drift<drift_tolerance})
    return {"case_id":baseline["case"]["case_id"],"baseline_version":baseline["case"]["case_version"],
            "current_version":current["case"]["case_version"],"tolerance":drift_tolerance,
            "passed":all(x["passed"] for x in rows),"results":rows}


def to_legacy_evidence(archive,mesh=16):
    """Adapt one qualified result to the existing BenchmarkEvidence contract."""
    validate_archive(archive);row=next((x for x in archive["results"] if x["mesh"]==mesh),None)
    if row is None:raise ValueError("mesh is absent from archive")
    from .benchmark_registry import BenchmarkEvidence
    c=archive["case"];citation=c["citation"]
    return BenchmarkEvidence(c["case_id"],"curved shell",c["quantity"]["name"],row["unit"],
        f"{citation['authors']} ({citation['year']}), DOI:{citation['doi']}",row["computed"],row["reference"],
        row["error"],row["tolerance"],row["passed"])
