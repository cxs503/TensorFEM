"""Point forces to a reference Q4 shell, retaining application-point moments.

Projection chooses a carrier element, not a physical contact law. The explicit
lever arm retains the original force location; it is not a pressure inference.
"""
from dataclasses import dataclass
import hashlib
import math
import torch
from .surface_coupling import QuadSurfaceExchange,_finite,_time,D
from .plate import _shape

@dataclass(frozen=True)
class PointLoad:
    time_s: float
    geometry_sha256: str
    source: str
    sample_force: torch.Tensor
    generalized_force: torch.Tensor

class PointSurfaceExchange:
    def __init__(self,nodes,elements,points,*,maximum_offset_m):
        if isinstance(maximum_offset_m,bool) or not isinstance(maximum_offset_m,(int,float)) or not math.isfinite(maximum_offset_m) or maximum_offset_m<=0:
            raise ValueError('explicit positive maximum offset required')
        surface=QuadSurfaceExchange(nodes,elements)
        self.nodes=surface.nodes;self.elements=surface.elements
        self.points=torch.as_tensor(points,dtype=D,device='cpu').clone()
        if self.points.ndim!=2 or self.points.shape[1]!=3 or len(self.points)==0 or not bool(torch.isfinite(self.points).all()):raise ValueError('finite nonempty point coordinates required')
        self.maximum_offset_m=float(maximum_offset_m)
        xyz=self.nodes[self.elements]
        faces=[];shapes=[];anchors=[];coordinates=[]
        # Small CPU interface: bounded Gauss-Newton projection on every carrier.
        for point in self.points:
            natural=torch.zeros((len(xyz),2),dtype=D)
            for _ in range(20):
                weights=[];derivatives=[]
                for xi,eta in natural:
                    N,dN=_shape(xi,eta);weights.append(N);derivatives.append(dN)
                N=torch.stack(weights);dN=torch.stack(derivatives)
                x=torch.einsum('ea,eai->ei',N,xyz)
                jac=torch.einsum('eai,eab->eib',xyz,dN)
                change=torch.linalg.pinv(jac)@(x-point)[:,:,None]
                next_natural=(natural-change.squeeze(-1)).clamp(-1,1)
                if float((next_natural-natural).abs().max())<1e-13:
                    natural=next_natural;break
                natural=next_natural
            N=torch.stack([_shape(xi,eta)[0] for xi,eta in natural])
            x=torch.einsum('ea,eai->ei',N,xyz)
            face=int(torch.linalg.vector_norm(x-point,dim=1).argmin())
            faces.append(face);shapes.append(N[face]);anchors.append(x[face]);coordinates.append(natural[face])
        self.face_ids=torch.tensor(faces,dtype=torch.long)
        self.ids=self.elements[self.face_ids]
        self.weights=torch.stack(shapes);self.anchors=torch.stack(anchors)
        self.natural_coordinates=torch.stack(coordinates)
        self.offsets=self.points-self.anchors
        self.distances=torch.linalg.vector_norm(self.offsets,dim=1)
        if float(self.distances.max())>self.maximum_offset_m:raise ValueError('force point outside declared mapping distance')
        self.geometry_sha256=self._digest()

    def _digest(self):
        h=hashlib.sha256()
        for t in (self.nodes,self.elements,self.points,self.face_ids,self.ids,self.weights,self.anchors,self.natural_coordinates,self.offsets,self.distances):h.update(t.detach().contiguous().numpy().tobytes())
        h.update(repr(self.maximum_offset_m).encode());return h.hexdigest()

    def _check(self):
        if self._digest()!=self.geometry_sha256:raise ValueError('point map mutated; rebuild geometry')

    def velocities(self,nodal_velocity):
        self._check();v=_finite(nodal_velocity,(len(self.nodes),6),'nodal velocity')[self.ids]
        linear=torch.einsum('pa,pai->pi',self.weights,v[:,:,:3])
        angular=torch.einsum('pa,pai->pi',self.weights,v[:,:,3:])
        return linear+torch.linalg.cross(angular,self.offsets)

    def point_loads(self,force,*,time_s,source):
        self._check()
        if source not in ('TensorLBM','TensorDEM','TensorFVM','verification'):raise ValueError('unknown load owner')
        f=_finite(force,self.points.shape,'point force').clone()
        g=torch.zeros((len(self.nodes),6),dtype=D)
        couple=torch.linalg.cross(self.offsets,f)
        for a in range(4):
            g[:,:3].index_add_(0,self.ids[:,a],self.weights[:,a,None]*f)
            g[:,3:].index_add_(0,self.ids[:,a],self.weights[:,a,None]*couple)
        return PointLoad(_time(time_s),self.geometry_sha256,source,f,g)

    def audit(self,load,velocity,*,expected_time_s):
        self._check()
        if not isinstance(load,PointLoad) or load.geometry_sha256!=self.geometry_sha256 or abs(_time(load.time_s)-_time(expected_time_s))>1e-12:raise ValueError('stale or mismatched point load')
        rebuilt=self.point_loads(load.sample_force,time_s=load.time_s,source=load.source)
        if not torch.equal(load.generalized_force,rebuilt.generalized_force):raise ValueError('inconsistent point loads')
        velocity=_finite(velocity,(len(self.nodes),6),'nodal velocity')
        g=load.generalized_force
        f=g[:,:3].sum(0);m=(torch.linalg.cross(self.nodes,g[:,:3])+g[:,3:]).sum(0)
        return {'force_N':f.tolist(),'moment_N_m':m.tolist(),
                'force_error_N':float(torch.linalg.vector_norm(f-load.sample_force.sum(0))),
                'moment_error_N_m':float(torch.linalg.vector_norm(m-torch.linalg.cross(self.points,load.sample_force).sum(0))),
                'power_error_W':abs(float((g*velocity).sum()-(load.sample_force*self.velocities(velocity)).sum())),
                'maximum_carrier_offset_m':float(self.distances.max()),'physical_accuracy_qualified':False}

    def motion_record(self,velocity,*,time_s):
        return {'schema':'tensorfem.point-motion/1','time_s':_time(time_s),'frame':'global_xyz',
                'units':{'position':'m','velocity':'m/s'},'geometry_sha256':self.geometry_sha256,
                'force_points_m':self.points.tolist(),'velocities_m_s':self.velocities(velocity).tolist(),
                'carrier_element_ids':self.face_ids.tolist(),'carrier_offsets_m':self.offsets.tolist()}
