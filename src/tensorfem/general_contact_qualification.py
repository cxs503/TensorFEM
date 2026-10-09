"""Fail-closed composite qualification for low-order 3-D surface contact."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .dynamic_contact_extended import UnifiedEvent, coulomb_impact
from .mortar_contact3d import self_contact_candidates

SCHEMA="tensorfem.general-contact3d-composite/1.0"
SOURCES={
    "curved_frictionless":"tensorfem.curved-surface-contact3d-qualification/1.0",
    "curved_frictional":"tensorfem.frictional-surface-mesh-qualification/1.0",
    "curved_finite_strain_friction":"tensorfem.curved-finite-strain-friction-qualification/1.0",
    "hertz_3d":"tensorfem.hertz-3d-qualification/1.0",
}

def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)

def _read(path,expected):
    try: report=json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError) as error:
        raise ValueError(f"missing or unreadable contact evidence: {path}") from error
    if report.get("schema")!=expected: raise ValueError(f"invalid contact evidence schema: {path}")
    digest=report.get("evidence_sha256")
    if not isinstance(digest,str) or len(digest)!=64: raise ValueError(f"missing evidence SHA-256: {path}")
    body={k:v for k,v in report.items() if k!="evidence_sha256"}
    candidates=[body]
    # The finite-strain runner measures wall time outside the physics report;
    # its canonical physics hash intentionally excludes that timing metadata.
    if "wall_time_seconds" in body:
        candidates.append({k:v for k,v in body.items() if k!="wall_time_seconds"})
    if digest not in {hashlib.sha256(_canonical(v).encode()).hexdigest() for v in candidates}:
        raise ValueError(f"contact evidence SHA-256 mismatch: {path}")
    return report

def _boundary_evidence():
    d=torch.float64
    vertices=torch.tensor([[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,0.],
                           [.05,0.,.02],[.05,1.,.02]],dtype=d)
    faces=torch.tensor([[0,2,1],[1,2,3],[2,4,3],[3,4,5]])
    pairs=self_contact_candidates(vertices,faces,search_distance=.08)
    self_ok=(0,3) in pairs and all(not(set(faces[i].tolist())&set(faces[j].tolist())) for i,j in pairs)
    positions=torch.tensor([[-1.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,1.,0.]],dtype=d)
    event=UnifiedEvent("edge-edge",.5,torch.tensor([0,1]),torch.tensor([.5,.5],dtype=d),
        torch.tensor([2,3]),torch.tensor([.5,.5],dtype=d),torch.tensor([0.,0.,1.],dtype=d))
    velocity=torch.tensor([[1.,0.,-1.],[1.,0.,-1.],[0.,0.,0.],[0.,0.,0.]],dtype=d)
    impact=coulomb_impact(event,velocity,torch.ones(4,dtype=d),friction=.3)
    force=float(torch.linalg.vector_norm(impact.impulses.sum(0)))
    moment=float(torch.linalg.vector_norm(torch.linalg.cross(positions,impact.impulses).sum(0)))
    return {"self_contact_candidate_filter":self_ok,"impact_force_imbalance":force,
        "impact_moment_imbalance":moment,
        "impact_energy_nonincrease":bool(impact.kinetic_after<=impact.kinetic_before)}

def qualify_general_contact3d(*,curved_frictionless,curved_frictional,
    curved_finite_strain_friction,hertz_3d):
    paths={"curved_frictionless":curved_frictionless,"curved_frictional":curved_frictional,
        "curved_finite_strain_friction":curved_finite_strain_friction,"hertz_3d":hertz_3d}
    reports={name:_read(paths[name],schema) for name,schema in SOURCES.items()}
    errors=[]
    errors.append(float(reports["curved_frictionless"]["mesh_sequence"][-1]["relative_error"]))
    friction=reports["curved_frictional"]
    errors.extend((float(friction["mesh_sequence"][-1]["normal_relative_error"]),
                   float(friction["mesh_sequence"][-1]["coulomb_relative_error"])))
    finite=reports["curved_finite_strain_friction"]
    errors.extend(float(row["coulomb_relative_error"]) for row in finite["mesh_sequence"])
    hertz=reports["hertz_3d"]
    hertz_errors=hertz.get("errors")
    if not isinstance(hertz_errors,dict) or not hertz_errors: raise ValueError("Hertz evidence has no error metrics")
    errors.extend(float(v) for v in hertz_errors.values())
    maximum=max(errors)
    rollback=all(bool(reports[name].get("rollback_exact")) for name in
                 ("curved_frictionless","curved_frictional","curved_finite_strain_friction"))
    boundaries=_boundary_evidence()
    passed=(all(bool(report.get("passed",report.get("status")=="qualified")) for report in reports.values())
        and all(math.isfinite(v) for v in errors) and maximum<=.03 and rollback
        and boundaries["self_contact_candidate_filter"]
        and boundaries["impact_force_imbalance"]<1e-10
        and boundaries["impact_moment_imbalance"]<1e-10
        and boundaries["impact_energy_nonincrease"])
    clean={"schema":SCHEMA,"passed":passed,
        "scope":"general_surface_to_surface" if passed else "blocked",
        "qualified_scope":"quasi-static low-order faceted, nonmatching, double-deformable 3-D surface-to-surface contact with friction and small/finite-strain subsets",
        "maximum_relative_error":maximum,"rollback_exact":rollback,
        "source_evidence":{n:{"schema":r["schema"],"evidence_sha256":r["evidence_sha256"]} for n,r in reports.items()},
        "boundary_evidence":boundaries,
        "excluded_capabilities":["self-contact solve (candidate generation only)","dynamic impact solve (independent capability)","production-scale segmentation/search"]}
    return {**clean,"evidence_sha256":hashlib.sha256(_canonical(clean).encode()).hexdigest()}
