"""Conservative current-surface exchange for six-DOF shell nodes (SI).

This is a load/velocity interface, not a fluid solver or a finite-rotation
constitutive model. Rebuild the map from the current midsurface when it moves.
Point forces and their offset couples use the transpose of velocity mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .plate import _shape

D = torch.float64
SCHEMA = "tensor-solver.surface-exchange.v1"
UNITS = {"position": "m", "velocity": "m/s", "angular_velocity": "rad/s",
         "time": "s", "force": "N", "moment": "N*m", "traction": "Pa"}


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("time_s must be finite nonnegative SI time")
    return float(value)


def _finite(value, shape, name):
    t = torch.as_tensor(value, dtype=D, device="cpu")
    if t.shape != shape or not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return t


@dataclass(frozen=True)
class SurfaceLoad:
    time_s: float
    geometry_sha256: str
    source: str
    generalized_force: torch.Tensor
    sample_force: torch.Tensor


class QuadSurfaceExchange:
    """Four Gauss samples per Q4, including shell face offset couples.

    This CPU adapter copies inputs to CPU. Nodes are current midsurface positions
    in global XYZ. Areas are midsurface areas; face offsets define force lever
    arms, not a reconstructed curved outer-skin area. Each sample represents
    an area, not a point pressure. Positive face_offset_m follows connectivity
    normals; positive pressure_pa acts opposite those normals. No extrapolation
    or nearest-node assignment is silently introduced.
    """

    def __init__(self, nodes, elements, *, face_offset_m=0.):
        self.nodes = torch.as_tensor(nodes, dtype=D, device="cpu").clone()
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3 or not bool(torch.isfinite(self.nodes).all()):
            raise ValueError("nodes must be finite [n,3] current SI positions")
        raw = torch.as_tensor(elements,device="cpu")
        if raw.ndim != 2 or raw.shape[1] != 4 or raw.numel() == 0 or raw.dtype not in (torch.int32, torch.int64):
            raise ValueError("elements require integer [e,4] connectivity")
        self.elements = raw.to(torch.long).clone()
        if int(raw.min()) < 0 or int(raw.max()) >= len(self.nodes):
            raise ValueError("surface connectivity outside node range")
        if any(len(set(e)) != 4 for e in raw.tolist()):
            raise ValueError("repeated Q4 corner")
        offset = torch.as_tensor(face_offset_m, dtype=D, device="cpu")
        if offset.ndim == 0:
            offset = offset.expand(len(raw))
        if offset.shape != (len(raw),) or not bool(torch.isfinite(offset).all()):
            raise ValueError("face_offset_m must be finite scalar or per-element vector")
        self.face_offset_m=offset.clone()
        ids, weights, positions, normals, areas, offsets = [], [], [], [], [], []
        gp = 3**-.5
        for e, distance in zip(self.elements, offset):
            xyz = self.nodes[e]
            _,center_dn=_shape(xyz.new_tensor(0.),xyz.new_tensor(0.))
            center_t=xyz.T@center_dn
            center_n=torch.linalg.cross(center_t[:,0],center_t[:,1])
            if float(torch.linalg.vector_norm(center_n))<=torch.finfo(D).eps:
                raise ValueError("folded or degenerate Q4 surface")
            for xi,eta in ((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)):
                _,corner_dn=_shape(xyz.new_tensor(xi),xyz.new_tensor(eta));t=xyz.T@corner_dn
                if float(torch.dot(torch.linalg.cross(t[:,0],t[:,1]),center_n))<=0:
                    raise ValueError("folded or degenerate Q4 surface")
            for xi, eta in ((-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp)):
                N, dn = _shape(xyz.new_tensor(xi), xyz.new_tensor(eta))
                tangents = xyz.T @ dn
                cross = torch.linalg.cross(tangents[:,0], tangents[:,1])
                area = torch.linalg.vector_norm(cross)
                if float(area) <= torch.finfo(D).eps:
                    raise ValueError("degenerate Q4 surface Jacobian")
                normal = cross / area
                r = distance * normal
                ids.append(e);weights.append(N);positions.append(N @ xyz + r)
                normals.append(normal);areas.append(area);offsets.append(r)
        self.ids = torch.stack(ids);self.weights = torch.stack(weights)
        self.positions = torch.stack(positions);self.normals = torch.stack(normals)
        self.areas = torch.stack(areas);self.offsets = torch.stack(offsets)
        data = {"nodes_m":self.nodes.tolist(), "elements":self.elements.tolist(),
                "face_offset_m":offset.tolist(), "frame":"global_xyz", "schema":SCHEMA}
        self.geometry_sha256 = hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self._integrity_sha256=self._digest()

    def _digest(self):
        h=hashlib.sha256()
        for t in (self.nodes,self.elements,self.face_offset_m,self.ids,self.weights,self.positions,self.normals,self.areas,self.offsets):
            h.update(t.detach().contiguous().numpy().tobytes())
        h.update(self.geometry_sha256.encode())
        return h.hexdigest()

    def _check_integrity(self):
        if self._digest()!=self._integrity_sha256:
            raise ValueError("surface map mutated; rebuild from current geometry")

    def velocities(self, nodal_velocity):
        self._check_integrity()
        v = _finite(nodal_velocity, (len(self.nodes),6), "nodal_velocity")
        samples = v[self.ids]
        translation = torch.einsum('pa,pai->pi',self.weights,samples[:,:,:3])
        rotation = torch.einsum('pa,pai->pi',self.weights,samples[:,:,3:])
        return translation + torch.linalg.cross(rotation,self.offsets)

    def point_loads(self, sample_force, *, time_s, source, frame="global_xyz", units="N"):
        self._check_integrity()
        if frame != "global_xyz" or units != "N" or source not in ("TensorLBM","TensorFVM","TensorDEM","verification"):
            raise ValueError("loads require declared solver source, global_xyz and N")
        f = _finite(sample_force, self.positions.shape, "sample_force").clone()
        total = torch.zeros((len(self.nodes),6),dtype=D)
        moment = torch.linalg.cross(self.offsets,f)
        for a in range(4):
            total[:,:3].index_add_(0,self.ids[:,a],self.weights[:,a,None]*f)
            total[:,3:].index_add_(0,self.ids[:,a],self.weights[:,a,None]*moment)
        return SurfaceLoad(_time(time_s),self.geometry_sha256,source,total,f)

    def tractions(self, traction_pa, *, time_s, source, units="Pa"):
        if units != "Pa":
            raise ValueError("surface traction must use Pa")
        t = _finite(traction_pa,self.positions.shape,"traction_pa")
        return self.point_loads(t*self.areas[:,None],time_s=time_s,source=source)

    def pressure(self, pressure_pa, *, time_s, source):
        p = _finite(pressure_pa,(len(self.areas),),"pressure_pa")
        return self.tractions(-p[:,None]*self.normals,time_s=time_s,source=source)

    def validate_load(self, load, *, expected_time_s):
        if not isinstance(load,SurfaceLoad) or load.geometry_sha256 != self.geometry_sha256:
            raise ValueError("load belongs to a different surface geometry")
        expected = _time(expected_time_s)
        if abs(_time(load.time_s)-expected) > 1e-12*max(1.,abs(expected)):
            raise ValueError("stale or mismatched load time")
        _finite(load.generalized_force,(len(self.nodes),6),"generalized_force")
        expected_load = self.point_loads(load.sample_force,time_s=load.time_s,source=load.source)
        if not torch.equal(load.generalized_force,expected_load.generalized_force):
            raise ValueError("generalized loads inconsistent with point-force mapping")

    def audit(self, load, nodal_velocity, *, expected_time_s):
        self.validate_load(load,expected_time_s=expected_time_s)
        v = _finite(nodal_velocity,(len(self.nodes),6),"nodal_velocity")
        nodal = load.generalized_force
        force = nodal[:,:3].sum(0)
        moment = (torch.linalg.cross(self.nodes,nodal[:,:3])+nodal[:,3:]).sum(0)
        reference_force = load.sample_force.sum(0)
        reference_moment = torch.linalg.cross(self.positions,load.sample_force).sum(0)
        nodal_power = float((nodal*v).sum())
        sample_power = float((load.sample_force*self.velocities(v)).sum())
        return {"force_N":force.tolist(),"moment_N_m":moment.tolist(),
                "force_transfer_error_N":float(torch.linalg.vector_norm(force-reference_force)),
                "moment_transfer_error_N_m":float(torch.linalg.vector_norm(moment-reference_moment)),
                "nodal_power_W":nodal_power,"sample_power_W":sample_power,
                "power_transfer_error_W":abs(nodal_power-sample_power)}

    def motion_record(self,nodal_velocity,*,time_s):
        return {"schema":SCHEMA,"time_s":_time(time_s),"frame":"global_xyz","units":UNITS,
                "source":"TensorFEM","geometry_sha256":self.geometry_sha256,
                "sample_ids":list(range(len(self.areas))),"positions_m":self.positions.tolist(),
                "velocities_m_s":self.velocities(nodal_velocity).tolist(),
                "normals":self.normals.tolist(),"area_weights_m2":self.areas.tolist()}


def validate_physics_ownership(*, ice_owners, resolved_fluid, simplified_water_terms):
    """One ice model per region; reduced water loads must not duplicate CFD."""
    if set(ice_owners) not in ({"TensorDEM"},{"TensorFEM-reference"}) or len(ice_owners)!=1:
        raise ValueError("one declared ice owner is required; hybrid regions need a separate contract")
    if not isinstance(resolved_fluid,bool):
        raise ValueError("resolved_fluid must be a boolean")
    if resolved_fluid and simplified_water_terms:
        raise ValueError("resolved fluid cannot silently duplicate buoyancy/added-mass/drag terms")
    return {"ice_owner":ice_owners[0],"resolved_fluid":resolved_fluid,
            "simplified_water_terms":list(simplified_water_terms)}


class PlanarEmbedding:
    """Explicit SI XY-to-XYZ embedding; never silently rename vertical axes."""
    def __init__(self,origin_m,axes):
        self.origin=_finite(origin_m,(3,),"origin_m").clone()
        self.axes=_finite(axes,(3,2),"axes").clone()
        if not torch.allclose(self.axes.T@self.axes,torch.eye(2,dtype=D),rtol=0.,atol=1e-12):
            raise ValueError("embedding axes must be orthonormal")

    def positions(self,xy_m):
        p=torch.as_tensor(xy_m,dtype=D,device="cpu")
        if p.ndim!=2 or p.shape[1]!=2 or not bool(torch.isfinite(p).all()):
            raise ValueError("planar positions must be finite [n,2]")
        return p@self.axes.T+self.origin

    def vectors(self,xy):
        v=torch.as_tensor(xy,dtype=D,device="cpu")
        if v.ndim!=2 or v.shape[1]!=2 or not bool(torch.isfinite(v).all()):
            raise ValueError("planar vectors must be finite [n,2]")
        return v@self.axes.T

    def project_load(self,xyz_force):
        f=torch.as_tensor(xyz_force,dtype=D,device="cpu")
        if f.ndim!=2 or f.shape[1]!=3 or not bool(torch.isfinite(f).all()):
            raise ValueError("forces must be finite [n,3]")
        planar=f@self.axes
        if not torch.allclose(planar@self.axes.T,f,rtol=1e-12,atol=1e-12):
            raise ValueError("2-D DEM cannot receive an out-of-plane force")
        return planar
