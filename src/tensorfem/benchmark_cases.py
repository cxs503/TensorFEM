"""Versioned benchmark case schema, evidence archives and drift comparison.

Adds JSON round-trip (from_dict / from_json / to_json) and a units converter
supporting SI, mm-tonne-s and mm-kg-s unit systems for FEM benchmark cases.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass,replace
from pathlib import Path
import hashlib,json,math
from typing import Any,Dict,Optional


SCHEMA_VERSION="tensorfem.benchmark-case/1.0"
EVIDENCE_VERSION="tensorfem.benchmark-evidence/2.0"


# ---------------------------------------------------------------------------
# Unit-system registry & converter
# ---------------------------------------------------------------------------

# Canonical (SI) base units: length=m, mass=kg, time=s, force=N, pressure=Pa
_UNIT_SYSTEMS: Dict[str, Dict[str, float]] = {
    # length(m), mass(kg), time(s), force(N), pressure(Pa)
    "SI":          {"length": 1.0, "mass": 1.0,      "time": 1.0, "force": 1.0,   "pressure": 1.0},
    "mm-tonne-s":  {"length": 1e-3, "mass": 1e3,     "time": 1.0, "force": 1.0,   "pressure": 1e6},
    "mm-kg-s":     {"length": 1e-3, "mass": 1.0,      "time": 1.0, "force": 1.0,   "pressure": 1e6},
    "m-kg-s":      {"length": 1.0,   "mass": 1.0,     "time": 1.0, "force": 1.0,   "pressure": 1.0},
    "inch-lbf-s":  {"length": 0.0254, "mass": 0.45359237, "time": 1.0, "force": 4.4482216152605, "pressure": 6894.757293168},
}

# Dimensional exponents for common FEM quantities (L, M, T)
_DIM_EXP: Dict[str, tuple] = {
    "length":       (1, 0, 0),
    "area":         (2, 0, 0),
    "volume":       (3, 0, 0),
    "mass":         (0, 1, 0),
    "density":      (-3, 1, 0),
    "time":         (0, 0, 1),
    "velocity":     (1, 0, -1),
    "acceleration": (1, 0, -2),
    "force":        (1, 1, -2),
    "pressure":     (-1, 1, -2),
    "stress":       (-1, 1, -2),
    "young_modulus":(-1, 1, -2),
    "energy":       (2, 1, -2),
    "moment":       (2, 1, -2),
    "frequency":    (0, 0, -1),
}

# Map human-readable unit tags to dimensional category
_UNIT_TAG_MAP: Dict[str, str] = {
    "m": "length", "mm": "length", "inch": "length", "in": "length",
    "kg": "mass", "tonne": "mass", "t": "mass", "lb": "mass",
    "s": "time", "ms": "time",
    "N": "force", "kN": "force", "MN": "force", "lbf": "force",
    "Pa": "pressure", "MPa": "pressure", "GPa": "pressure", "psi": "pressure",
    "kg/m^3": "density", "kg/mm^3": "density", "tonne/mm^3": "density",
    "length": "length", "mass": "mass", "time": "time",
    "force": "force", "pressure": "pressure", "stress": "stress",
    "young_modulus": "young_modulus", "density": "density",
}

# Explicit conversion factors to SI for common unit tags
_TAG_TO_SI: Dict[str, float] = {
    "m": 1.0, "mm": 1e-3, "inch": 0.0254, "in": 0.0254,
    "kg": 1.0, "tonne": 1e3, "t": 1e3, "lb": 0.45359237,
    "s": 1.0, "ms": 1e-3,
    "N": 1.0, "kN": 1e3, "MN": 1e6, "lbf": 4.4482216152605,
    "Pa": 1.0, "MPa": 1e6, "GPa": 1e9, "psi": 6894.757293168,
    "kg/m^3": 1.0, "kg/mm^3": 1e9, "tonne/mm^3": 1e12,
}


def _dim_factor(unit_system: str, dim_category: str) -> float:
    """Return factor to convert a quantity in *unit_system* to SI."""
    sys_factors = _UNIT_SYSTEMS[unit_system]
    exp = _DIM_EXP.get(dim_category)
    if exp is None:
        return 1.0
    L, M, T = exp
    return (sys_factors["length"] ** L *
            sys_factors["mass"] ** M *
            sys_factors["time"] ** T)


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a scalar *value* between two unit tags via SI.

    Supports: m/mm/inch, kg/tonne/lb, s/ms, N/kN/MN/lbf, Pa/MPa/GPa/psi,
    and dimensional categories (length, mass, force, pressure, density,
    young_modulus, stress).
    """
    if from_unit == to_unit:
        return value
    # Direct tag conversion
    f_si = _TAG_TO_SI.get(from_unit)
    t_si = _TAG_TO_SI.get(to_unit)
    if f_si is not None and t_si is not None:
        return value * f_si / t_si
    # Dimensional category conversion (assumes SI base)
    cat = _UNIT_TAG_MAP.get(from_unit, from_unit)
    exp = _DIM_EXP.get(cat)
    if exp is not None:
        # Both are dimensional categories; treat as SI-to-SI identity
        # unless they differ, in which case we try tag lookup
        if from_unit == to_unit:
            return value
    # Fallback: return unchanged (unknown units)
    return value


def convert_case_units(case_dict: dict, from_system: str, to_system: str) -> dict:
    """Return a *copy* of a case dict with numeric fields converted between
    unit systems.  Only geometry, material and quick_values are touched;
    mesh_sequence, case_id, etc. are preserved verbatim.
    """
    out = dict(case_dict)
    # Geometry: length-type values
    geo = dict(out.get("geometry", {}))
    for k, v in geo.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            # Heuristic: keys containing radius, thickness, length, depth,
            # width, height, diameter are lengths
            if any(tok in k.lower() for tok in ("radius", "thickness", "length",
                                                  "depth", "width", "height",
                                                  "diameter", "span", "gap")):
                f = _dim_factor(from_system, "length") / _dim_factor(to_system, "length")
                geo[k] = v * f
    out["geometry"] = geo

    # Material: young_modulus (pressure dim), density
    mat = dict(out.get("material", {}))
    for k, v in mat.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if "young" in k.lower() or "modulus" in k.lower():
                f = _dim_factor(from_system, "pressure") / _dim_factor(to_system, "pressure")
                mat[k] = v * f
            elif "density" in k.lower() or "rho" in k.lower():
                f = _dim_factor(from_system, "density") / _dim_factor(to_system, "density")
                mat[k] = v * f
    out["material"] = mat

    # quick_values: same dimension as the reference quantity
    qv = out.get("quick_values", [])
    if qv:
        qty = out.get("quantity", {})
        unit_tag = qty.get("unit", "length")
        cat = _UNIT_TAG_MAP.get(unit_tag, unit_tag)
        f = _dim_factor(from_system, cat) / _dim_factor(to_system, cat)
        out["quick_values"] = tuple(v * f for v in qv)

    # Reference quantity value
    qty = dict(out.get("quantity", {}))
    if "value" in qty:
        unit_tag = qty.get("unit", "length")
        cat = _UNIT_TAG_MAP.get(unit_tag, unit_tag)
        f = _dim_factor(from_system, cat) / _dim_factor(to_system, cat)
        qty["value"] = qty["value"] * f
    out["quantity"] = qty

    return out


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Citation:
    authors: str
    title: str
    publication: str
    year: int
    doi: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Citation":
        return cls(
            authors=str(d["authors"]),
            title=str(d["title"]),
            publication=str(d["publication"]),
            year=int(d["year"]),
            doi=str(d["doi"]),
        )


@dataclass(frozen=True)
class ReferenceQuantity:
    name: str
    value: float
    unit: str
    sign_policy: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReferenceQuantity":
        return cls(
            name=str(d["name"]),
            value=float(d["value"]),
            unit=str(d["unit"]),
            sign_policy=str(d["sign_policy"]),
        )


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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkCase":
        """Reconstruct a BenchmarkCase from a plain dict (JSON-decoded)."""
        return cls(
            schema_version=str(d["schema_version"]),
            case_id=str(d["case_id"]),
            case_version=str(d["case_version"]),
            title=str(d["title"]),
            citation=Citation.from_dict(d["citation"]) if isinstance(d["citation"], dict) else d["citation"],
            geometry=dict(d["geometry"]),
            material=dict(d["material"]),
            boundary_conditions=tuple(d["boundary_conditions"]),
            loads=tuple(d["loads"]),
            mesh_sequence=tuple(int(x) for x in d["mesh_sequence"]),
            quantity=ReferenceQuantity.from_dict(d["quantity"]) if isinstance(d["quantity"], dict) else d["quantity"],
            error_strategy=str(d["error_strategy"]),
            tolerance=float(d["tolerance"]),
            maturity=str(d["maturity"]),
            capability_boundaries=tuple(d["capability_boundaries"]),
            runner=str(d["runner"]),
            quick_values=tuple(float(x) for x in d["quick_values"]),
            qualifying_meshes=tuple(int(x) for x in d["qualifying_meshes"]),
        )

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True,
                          ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkCase":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(text))

    def with_units_converted(self, from_system: str, to_system: str) -> "BenchmarkCase":
        """Return a new BenchmarkCase with geometry/material/quick_values
        converted between unit systems.  The case_id and schema_version are
        preserved; a suffix is appended to case_version to record the conversion.
        """
        d = convert_case_units(self.to_dict(), from_system, to_system)
        d["case_version"] = f"{d['case_version']}+{from_system}2{to_system}"
        return self.from_dict(d)


# ---------------------------------------------------------------------------
# Canonicalisation & hashing
# ---------------------------------------------------------------------------

def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)


def _hash(value):return hashlib.sha256(_canonical(value).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

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
    runners=("hemisphere_18deg","scordelis_lo","pinched_cylinder","pure_bending_90deg","hertz_ball_plane")
    if not case.capability_boundaries or case.runner not in runners:raise ValueError("missing capability boundary or runner")
    return case


def case_hash(case):validate_case(case);return _hash(case.to_dict())


# ---------------------------------------------------------------------------
# Built-in cases
# ---------------------------------------------------------------------------

def hemisphere_case():
    return BenchmarkCase(SCHEMA_VERSION,"shell.hemisphere_18deg","2.0.0",
        "MacNeal--Harder hemispherical shell with 18-degree hole",
        Citation("MacNeal and Harder","A proposed standard set of problems to test finite element accuracy",
                 "Finite Elements in Analysis and Design 1, 3--20",1985,"10.1016/0168-874X(85)90003-4"),
        {"midsurface":"quarter hemisphere","radius":10.,"hole_polar_angle_deg":18.,"thickness":.04},
        {"young_modulus":6.825e7,"poisson_ratio":.3},
        ("meridional symmetry constraints","one vertical rigid-translation gauge"),
        ("alternating unit radial equator point loads",),(4,6,8,12,16,20,24),
        ReferenceQuantity("loaded-equator radial displacement",.0924,"length","magnitude"),
        "absolute_relative",.03,"qualified",
        ("linear small-strain quarter model","projected MITC-like Q4 shell","not nonlinear doubly-curved shell"),
        "hemisphere_18deg",(.1015827104289639,.09873806252562636,.09720500004786899,
                            .0958018428478389,.09520900652004827,.0948885396800606,
                            .09468459406197954),(20,24))


def _computed(case,tier):
    if tier=="quick":return case.quick_values
    if tier!="full":raise ValueError("tier must be quick or full")
    if case.runner=="hemisphere_18deg":
        from .spherical_shell import hemisphere_with_hole
        return tuple(hemisphere_with_hole(n).displacement for n in case.mesh_sequence)
    if case.runner=="scordelis_lo":
        from .cylindrical_shell_benchmarks import scordelis_lo_cylindrical
        return tuple(scordelis_lo_cylindrical(n,n).probe_displacement for n in case.mesh_sequence)
    if case.runner=="pinched_cylinder":
        from .pinched_cylinder import solve_pinched_cylinder_linear
        return tuple(solve_pinched_cylinder_linear(n).displacement for n in case.mesh_sequence)
    if case.runner=="pure_bending_90deg":
        from .large_rotation_shell_benchmark import pure_bending_shell
        return tuple(float(pure_bending_shell(n).tip[2]) for n in case.mesh_sequence)
    if case.runner=="hertz_ball_plane":
        from .hertz_halfspace_loadcontrol import (
            hertz_load_reference, solve_axisymmetric_contact_load_control,
            two_zone_axis, graded_depth,
        )
        ref = hertz_load_reference(1831.5, 10.0, 1e5, 0.3)
        out = []
        for n in case.mesh_sequence:
            r_axis = two_zone_axis(3*ref.contact_radius, 96*ref.contact_radius, n, 22)
            z_axis = graded_depth(96*ref.contact_radius, 4*ref.contact_radius, 20, 14)
            _, state, _ = solve_axisymmetric_contact_load_control(
                load=ref.load, radius=10.0, young=1e5, poisson=0.3,
                r_axis=r_axis, z_axis=z_axis)
            out.append(float(state.prescribed_indentation))
        return tuple(out)
    raise ValueError("runner is not available")


# ---------------------------------------------------------------------------
# Archive seal / validate / run / IO
# ---------------------------------------------------------------------------

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
    """Adapt one recorded result to the existing BenchmarkEvidence contract."""
    validate_archive(archive);row=next((x for x in archive["results"] if x["mesh"]==mesh),None)
    if row is None:raise ValueError("mesh is absent from archive")
    from .benchmark_registry import BenchmarkEvidence
    c=archive["case"];citation=c["citation"]
    return BenchmarkEvidence(c["case_id"],"curved shell",c["quantity"]["name"],row["unit"],
        f"{citation['authors']} ({citation['year']}), DOI:{citation['doi']}",row["computed"],row["reference"],
        row["error"],row["tolerance"],row["passed"])


def hertz_ball_plane_case():
    """Hertz rigid sphere on an elastic half-space, load-controlled.

    Case FEM-HERTZ-CONTACT: contact radius ``a`` and indentation ``delta``
    under a prescribed normal load ``P`` against the Hertz closed form
    (Johnson 1985 Sec. 4.2, rigid-indenter limit).  The primary pool quantity
    is the indentation ``delta`` (robustly within 3% at every mesh level); the
    contact radius ``a`` is recovered from the same load-controlled state and
    cross-checked in the acceptance test.
    """
    return BenchmarkCase(
        SCHEMA_VERSION, "contact.hertz_ball_plane", "1.0.0",
        "Hertz rigid sphere on elastic half-space: contact radius and indentation under prescribed load",
        Citation("K. L. Johnson", "Contact Mechanics",
                 "Cambridge University Press, Sec. 4.2 (Hertz contact)", 1985, "10.1017/cbo9781139171731"),
        {"body":"axisymmetric half-space (r>=0, z<=0)","tip":"rigid sphere","radius":10.0,
         "contact_zone":"r in [0, 3a]","patch_depth":"96 a"},
        {"young_modulus":1.0e5,"poisson_ratio":0.3},
        ("bottom fully fixed","outer radius radially fixed","axis node radially fixed by symmetry"),
        ("prescribed normal load P = 1831.5 (2 pi integrated)",),
        (40, 60, 80, 100, 120, 160, 200),
        ReferenceQuantity("indentation depth", 0.024999983333330558, "length", "magnitude"),
        "absolute_relative", 0.03, "qualified",
        ("linear small-strain axisymmetric half-space","load-controlled ALM contact",
         "not finite-slip or rolling contact"),
        "hertz_ball_plane",
        (0.024722, 0.024736, 0.024726, 0.024730, 0.024727, 0.024727, 0.024727),
        (40, 60, 80, 100, 120, 160, 200),
    )
