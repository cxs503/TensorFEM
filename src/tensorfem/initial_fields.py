"""Validated transfer of initial imperfection and residual-stress fields.

The transfer is deliberately solver independent: model IDs are preferred and
coordinate matching is an explicit fallback.  Stress components follow the
TensorFEM global Cartesian order ``(xx, yy, zz, xy, yz, xz)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping
import torch

from .result_db import ResultDB
from .result_db_v2 import FieldSpec, ResultDBv2, ResultFrame, ResultStep

STRESS_COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "xz")
VECTOR_COMPONENTS = ("x", "y", "z")
_LENGTH_UNITS = {"m", "mm", "cm", "dimensionless"}
_STRESS_UNITS = {"Pa", "kPa", "MPa", "GPa", "dimensionless"}


@dataclass(frozen=True)
class StructuralInitialFields:
    """Initial fields mapped into the target model ordering."""
    node_ids: torch.Tensor
    element_ids: torch.Tensor
    connectivity: torch.Tensor
    imperfection: torch.Tensor | None = None
    residual_stress: torch.Tensor | None = None
    residual_location: Literal["node", "element", "integration_point"] | None = None
    units: Mapping[str, str] | None = None
    coordinate_system: str = "global Cartesian"

    def validate(self) -> "StructuralInitialFields":
        n, e = len(self.node_ids), len(self.element_ids)
        if self.imperfection is not None:
            if self.imperfection.shape != (n, 3) or not torch.isfinite(self.imperfection).all():
                raise ValueError("imperfection must be finite [nnode,3]")
        if self.residual_stress is not None:
            expected = n if self.residual_location == "node" else e
            if self.residual_location not in ("node", "element", "integration_point"):
                raise ValueError("residual stress location is required")
            if self.residual_stress.ndim not in (2, 3) or len(self.residual_stress) != expected:
                raise ValueError("residual stress has incompatible entity count")
            if self.residual_stress.shape[-1] != 6 or not torch.isfinite(self.residual_stress).all():
                raise ValueError("residual stress must have six finite Cartesian components")
            if self.residual_location != "integration_point" and self.residual_stress.ndim != 2:
                raise ValueError("only integration-point stress may have an IP axis")
        return self

    def residual_resultant(self, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Return component-wise weighted stress resultant (not a force without areas)."""
        if self.residual_stress is None:
            raise ValueError("no residual stress field")
        value = self.residual_stress
        if value.ndim == 3:
            value = value.mean(dim=1)
        w = torch.ones(len(value), dtype=value.dtype, device=value.device) if weights is None else weights.to(value)
        if w.shape != (len(value),) or not torch.isfinite(w).all() or torch.any(w <= 0):
            raise ValueError("weights must be positive finite entity weights")
        return torch.sum(value * w[:, None], dim=0)

    def assert_self_equilibrated(self, weights: torch.Tensor | None = None, *, rtol: float = 1e-8,
                                 atol: float = 1e-10) -> torch.Tensor:
        resultant = self.residual_resultant(weights)
        value = self.residual_stress
        if value is None:  # pragma: no cover - guarded above
            raise ValueError("no residual stress field")
        if value.ndim == 3:
            value = value.mean(dim=1)
        w = torch.ones(len(value), dtype=value.dtype, device=value.device) if weights is None else weights.to(value)
        scale = torch.sum(torch.abs(value) * w[:, None], dim=0)
        if torch.any(torch.abs(resultant) > atol + rtol * scale):
            raise ValueError(f"residual stress is not self-equilibrated: {resultant.tolist()}")
        return resultant


@dataclass(frozen=True)
class ImportedStripResponse:
    shortening: float
    scale: float
    maximum_amplitude: float
    membrane_force: float
    equilibrium_residual: float


def solve_imported_elastic_strip(fields: StructuralInitialFields, *, shortening: float,
                                 young: float, bending_stiffness: float) -> ImportedStripResponse:
    """Reduced structural solve driven by the complete imported nodal fields.

    The piecewise-periodic profile gradient supplies the von Karman membrane
    strain and every axial residual-stress value participates in equilibrium.
    """
    profile,residual=imperfect_strip_inputs(fields)
    if len(profile)<8: raise ValueError("field-driven strip requires at least eight nodes")
    if shortening<=0 or young<=0 or bending_stiffness<=0: raise ValueError("solve parameters must be positive")
    if float(torch.max(torch.abs(profile)))==0: raise ValueError("imperfection profile must be nonzero")
    fields.assert_self_equilibrated()
    dx=1./len(profile);gradient=(torch.roll(profile,-1)-torch.roll(profile,1))/(2*dx)
    shape=.5*gradient**2;norm=torch.mean(profile**2)
    def evaluate(scale):
        stress=residual+young*(shortening+shape*(scale*scale-1.))
        force=torch.mean(stress)
        equilibrium=bending_stiffness*norm*(scale-1.)-torch.mean(stress*gradient**2)*scale
        return equilibrium,force
    grid=torch.linspace(.02,8.,6000,dtype=profile.dtype,device=profile.device)
    lo=grid[0];flo=evaluate(lo)[0];bracket=None
    for hi in grid[1:]:
        fhi=evaluate(hi)[0]
        if float(flo*fhi)<=0: bracket=(lo,hi);break
        lo,flo=hi,fhi
    if bracket is None: raise RuntimeError("could not bracket imported-field strip equilibrium")
    lo,hi=bracket
    for _ in range(60):
        mid=.5*(lo+hi);fm=evaluate(mid)[0]
        if float(evaluate(lo)[0]*fm)<=0:hi=mid
        else:lo=mid
    scale=.5*(lo+hi);equilibrium,force=evaluate(scale)
    return ImportedStripResponse(float(shortening),float(scale),float(torch.max(torch.abs(profile))*scale),
                                 float(force),float(torch.abs(equilibrium)))


def _positions(source_ids, target_ids, source_coordinates, target_coordinates, tolerance, missing):
    lookup = {int(v): i for i, v in enumerate(source_ids.tolist())}
    positions = []; used = set()
    for i, value in enumerate(target_ids.tolist()):
        if int(value) in lookup:
            position=lookup[int(value)]
            if position in used: raise ValueError(f"source entity {value} maps more than once")
            positions.append(position);used.add(position);continue
        if source_coordinates is not None and target_coordinates is not None:
            distance = torch.linalg.vector_norm(source_coordinates - target_coordinates[i], dim=1)
            minimum=torch.min(distance);candidates=torch.nonzero(torch.isclose(distance,minimum,rtol=1e-12,atol=1e-14)).flatten()
            if len(candidates)!=1: raise ValueError(f"ambiguous coordinate match for entity ID {value}")
            j = int(candidates[0])
            if float(distance[j]) <= tolerance or missing == "nearest":
                if j in used: raise ValueError(f"coordinate mapping reuses source entity {int(source_ids[j])}")
                positions.append(j);used.add(j);continue
        if missing == "zero": positions.append(-1)
        else: raise KeyError(f"cannot map entity ID {value}")
    return positions


def _mapped(values, positions, target_like):
    shape = (len(positions), *values.shape[1:]); out = torch.zeros(shape, dtype=target_like.dtype, device=target_like.device)
    valid = [i for i, p in enumerate(positions) if p >= 0]
    if valid:
        source = torch.tensor([positions[i] for i in valid], dtype=torch.long, device=values.device)
        out[torch.tensor(valid, device=out.device)] = values[source].to(out)
    return out


def import_initial_fields(db: ResultDB | ResultDBv2, *, target_node_ids: torch.Tensor,
                          target_element_ids: torch.Tensor, target_connectivity: torch.Tensor,
                          target_coordinates: torch.Tensor | None = None,
                          imperfection_field: str = "initial_imperfection",
                          residual_stress_field: str = "residual_stress", step: str | None = None,
                          frame: int | None = None, missing: Literal["error", "zero", "nearest"] = "error",
                          coordinate_tolerance: float = 1e-8) -> StructuralInitialFields:
    """Import and map initial fields from an in-memory ResultDB v1 or v2.

    IDs take precedence. Coordinate matching applies to nodes only and requires
    a ``coordinates`` node field in the source database.
    """
    if missing not in ("error", "zero", "nearest") or coordinate_tolerance < 0:
        raise ValueError("invalid missing policy or coordinate tolerance")
    db.validate()
    source_coordinates = None; imperfection = stress = None; location = None; units = {}
    if isinstance(db, ResultDB):
        source_coordinates = db.node_fields.get("coordinates")
        if imperfection_field in db.node_fields:
            imperfection = db.node_fields[imperfection_field]; units[imperfection_field] = db.units.get(imperfection_field, "dimensionless")
        if residual_stress_field in db.node_fields:
            stress=db.node_fields[residual_stress_field]; location="node"
        elif residual_stress_field in db.element_fields:
            stress=db.element_fields[residual_stress_field]; location="element"
        if stress is not None: units[residual_stress_field]=db.units.get(residual_stress_field,"dimensionless")
    else:
        selected_step = next((s for s in db.steps if step is None or s.name == step), None)
        if selected_step is None: raise KeyError(step)
        selected_frame = next((f for f in reversed(selected_step.frames) if frame is None or f.index == frame), None)
        if selected_frame is None: raise KeyError(frame)
        source_coordinates = selected_frame.fields.get("node", {}).get("coordinates")
        if imperfection_field in selected_frame.fields.get("node", {}):
            imperfection=selected_frame.fields["node"][imperfection_field]; units[imperfection_field]=db.field_specs[imperfection_field].unit
        spec=db.field_specs.get(residual_stress_field)
        if spec and residual_stress_field in selected_frame.fields.get(spec.location, {}):
            stress=selected_frame.fields[spec.location][residual_stress_field]; location=spec.location; units[residual_stress_field]=spec.unit
            if tuple(spec.components) != STRESS_COMPONENTS: raise ValueError("residual stress components must be xx,yy,zz,xy,yz,xz")
    if imperfection is None and stress is None: raise KeyError("database contains no requested initial fields")
    if imperfection is not None:
        if units[imperfection_field] not in _LENGTH_UNITS: raise ValueError("unsupported imperfection unit")
        pos=_positions(db.node_ids,target_node_ids,source_coordinates,target_coordinates,coordinate_tolerance,missing)
        imperfection=_mapped(imperfection,pos,target_coordinates if target_coordinates is not None else target_node_ids.to(torch.float64))
    if stress is not None:
        if units[residual_stress_field] not in _STRESS_UNITS: raise ValueError("unsupported residual-stress unit")
        ids=db.node_ids if location=="node" else db.element_ids; targets=target_node_ids if location=="node" else target_element_ids
        pos=_positions(ids,targets,source_coordinates if location=="node" else None,target_coordinates if location=="node" else None,coordinate_tolerance,missing)
        stress=_mapped(stress,pos,target_coordinates if target_coordinates is not None else targets.to(torch.float64))
    return StructuralInitialFields(target_node_ids.clone(),target_element_ids.clone(),target_connectivity.clone(),
                                   imperfection,stress,location,units,db.coordinate_system).validate()


def initial_fields_result_db(fields: StructuralInitialFields, *, coordinates: torch.Tensor,
                             step: str = "Initial") -> ResultDBv2:
    """Create a ResultDB v2 frame for restart/audit round trips."""
    fields.validate()
    if coordinates.shape != (len(fields.node_ids), 3): raise ValueError("coordinates must be [nnode,3]")
    specs={"coordinates":FieldSpec("node",(fields.units or {}).get("coordinates","m"),VECTOR_COMPONENTS)}
    values={"node":{"coordinates":coordinates.clone()}}
    if fields.imperfection is not None:
        specs["initial_imperfection"]=FieldSpec("node",(fields.units or {}).get("initial_imperfection","dimensionless"),VECTOR_COMPONENTS)
        values["node"]["initial_imperfection"]=fields.imperfection.clone()
    if fields.residual_stress is not None:
        loc=fields.residual_location; specs["residual_stress"]=FieldSpec(loc,(fields.units or {}).get("residual_stress","dimensionless"),STRESS_COMPONENTS)
        values.setdefault(loc,{})["residual_stress"]=fields.residual_stress.clone()
    return ResultDBv2(fields.node_ids.clone(),fields.element_ids.clone(),fields.connectivity.clone(),specs,
                      (ResultStep(step,(ResultFrame(0,0.,0.,values),)),),coordinate_system=fields.coordinate_system).validate()


def imperfect_strip_inputs(fields: StructuralInitialFields, *, transverse_component: int = 2,
                            stress_component: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract nodal strip profiles for a reduced structural analysis.

    Residual stress must be nodal because interpolation from elements/IPs is a
    solver-specific projection.  The returned arrays retain every nodal value.
    """
    fields.validate()
    if not 0 <= transverse_component < 3 or not 0 <= stress_component < 6:
        raise ValueError("component index out of range")
    if fields.imperfection is None: raise ValueError("no imperfection field")
    if fields.residual_stress is None or fields.residual_location != "node":
        raise ValueError("strip input requires nodal residual stress")
    return fields.imperfection[:,transverse_component].clone(),fields.residual_stress[:,stress_component].clone()


def run_initial_field_transfer_qualification() -> dict[str, object]:
    """Exercise ID reordering, self-balance and field-sensitive equilibrium."""
    dtype=torch.float64;n=16
    ids=torch.arange(100,100+n);element_ids=torch.tensor([7])
    connectivity=ids.reshape(1,-1);x=torch.arange(n,dtype=dtype)/n
    coordinates=torch.stack((x,torch.zeros_like(x),torch.zeros_like(x)),1)
    imperfection=torch.zeros((n,3),dtype=dtype)
    imperfection[:,2]=0.02*torch.sin(2*torch.pi*x)
    residual=torch.zeros((n,6),dtype=dtype)
    # Match the squared-gradient harmonic so translating this self-balanced
    # fabrication stress changes its correlation with the imperfection field.
    residual[:,0]=0.2*torch.cos(4*torch.pi*x)
    order=torch.arange(n-1,-1,-1)
    db=ResultDB(ids[order],element_ids,connectivity,
        {"coordinates":coordinates[order],"initial_imperfection":imperfection[order],
         "residual_stress":residual[order]},
        units={"coordinates":"m","initial_imperfection":"m","residual_stress":"MPa"})
    fields=import_initial_fields(db,target_node_ids=ids,target_element_ids=element_ids,
        target_connectivity=connectivity,target_coordinates=coordinates)
    mapping_error=max(float(torch.max(torch.abs(fields.imperfection-imperfection))),
                      float(torch.max(torch.abs(fields.residual_stress-residual))))
    balance=float(torch.max(torch.abs(fields.assert_self_equilibrated())))
    response=solve_imported_elastic_strip(fields,shortening=0.002,young=100.,bending_stiffness=2.)
    shifted=StructuralInitialFields(ids,element_ids,connectivity,imperfection,
        torch.roll(residual,3,0),"node",fields.units)
    shifted.assert_self_equilibrated()
    shifted_response=solve_imported_elastic_strip(
        shifted,shortening=0.002,young=100.,bending_stiffness=2.)
    sensitivity=abs(response.membrane_force-shifted_response.membrane_force)
    passed=(mapping_error<1e-14 and balance<1e-12 and
            response.equilibrium_residual<1e-10 and sensitivity>1e-5)
    if not passed: raise AssertionError("initial-field transfer qualification failed")
    return {"scope":"ResultDB-to-reduced-strip field transfer; not Shell4/IP projection",
            "mapping_max_error":mapping_error,"residual_resultant":balance,
            "equilibrium_residual":response.equilibrium_residual,
            "field_distribution_response_difference":sensitivity,"passed":True}
