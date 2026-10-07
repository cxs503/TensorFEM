"""AD tangent through the accepted adaptive finite-plastic substep tree."""
from dataclasses import dataclass
import time,torch
from .adaptive_global_plasticity import AdaptiveGlobalState
from .nonproportional_plasticity import integrate_adaptive
from .solid3d import _gradient
@dataclass(frozen=True)
class ADMetrics: paths:int; seconds:float
def assemble_ad_global(model,u,committed,*,yield_stress,hardening,material_rtol=2e-5,tangent=True):
 t=time.perf_counter();x=model.reference_nodes[model.elements];dn=x.new_tensor([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
 g,d=_gradient(x,dn,"AD plastic TET4");v=d/6; ed=torch.stack(tuple(3*model.elements+i for i in range(3)),2).reshape(-1,12).long()
 internal=torch.zeros_like(u);K=u.new_zeros((model.n_dofs,model.n_dofs)); Ps=[];points=[];added=[];paths=0
 for e,dofs in enumerate(ed):
  def response(q):
   F=torch.eye(3,dtype=q.dtype,device=q.device)+q.reshape(4,3).T@g[e]
   r=integrate_adaptive(F,committed.points[e],model.young,model.poisson,yield_stress,hardening,rtol=material_rtol)
   return (r.first_piola@g[e].T).T.reshape(-1)*v[e]
  q=u[dofs];fe=response(q);paths+=1
  F=torch.eye(3,dtype=q.dtype,device=q.device)+q.reshape(4,3).T@g[e]
  r=integrate_adaptive(F,committed.points[e],model.young,model.poisson,yield_stress,hardening,rtol=material_rtol);paths+=1
  internal.index_add_(0,dofs,fe)
  if tangent:
   ke=torch.autograd.functional.jacobian(response,q);paths+=1
   if not bool(torch.isfinite(ke).all()):raise RuntimeError("AD plastic tangent is non-finite")
   K.index_put_((dofs[:,None],dofs[None,:]),ke,accumulate=True)
  Ps.append(r.first_piola);points.append(r.state);added.append(r.dissipation*v[e])
 a=torch.stack(added);state=AdaptiveGlobalState(tuple(points),committed.cumulative_dissipation+a)
 return internal,K,torch.stack(Ps),state,a,ADMetrics(paths,time.perf_counter()-t)
def benchmark_tangents(model,u,state,**kw):
 from .adaptive_global_plasticity import assemble_adaptive_global
 _,Ka,_,_,_,ma=assemble_ad_global(model,u,state,**kw)
 t=time.perf_counter();_,Kr,_,_,_,_=assemble_adaptive_global(model,u,state,**kw);tr=time.perf_counter()-t
 return {"ad_seconds":ma.seconds,"oracle_seconds":tr,"speedup":tr/ma.seconds,
 "relative_error":float(torch.linalg.vector_norm(Ka-Kr)/torch.linalg.vector_norm(Kr)),"ad_paths":ma.paths,"oracle_paths":49*len(model.elements)}
