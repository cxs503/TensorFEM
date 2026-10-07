"""Fail-closed hybrid implicit/adaptive-AD finite-plastic tangent."""
from dataclasses import dataclass
import torch
from .implicit_plastic_tangent import implicit_update,implicit_material_tangent
from .nonproportional_plasticity import integrate_adaptive
from .adaptive_global_plasticity import AdaptiveGlobalState
from .solid3d import _gradient
@dataclass(frozen=True)
class HybridMetrics:
 strategies:tuple[str,...]; path_counts:tuple[int,...]; fallbacks:tuple[str,...]; estimated_errors:tuple[float,...]
def assemble_hybrid(model,u,committed,*,yield_stress,hardening,material_rtol=2e-5,qualification=.01,tangent=True):
 x=model.reference_nodes[model.elements];dn=x.new_tensor([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
 g,d=_gradient(x,dn,"hybrid plastic TET4");v=d/6;ed=torch.stack(tuple(3*model.elements+i for i in range(3)),2).reshape(-1,12).long()
 internal=torch.zeros_like(u);K=u.new_zeros((model.n_dofs,model.n_dofs));Ps=[];pts=[];added=[];strategies=[];counts=[];reasons=[];errors=[]
 for e,dofs in enumerate(ed):
  q=u[dofs];F=torch.eye(3,dtype=q.dtype,device=q.device)+q.reshape(4,3).T@g[e]
  adaptive=integrate_adaptive(F,committed.points[e],model.young,model.poisson,yield_stress,hardening,rtol=material_rtol)
  Pi,si,di=implicit_update(F,model.young,model.poisson,yield_stress,hardening,committed.points[e].plastic)
  pe=float(torch.linalg.vector_norm(Pi-adaptive.first_piola)/max(float(torch.linalg.vector_norm(adaptive.first_piola)),1.))
  se=float(torch.linalg.vector_norm(si.plastic_gradient-adaptive.state.plastic.plastic_gradient)/max(float(torch.linalg.vector_norm(adaptive.state.plastic.plastic_gradient)),1.))
  estimate=max(pe,se); qualified=adaptive.accepted_substeps==2 and estimate<qualification
  B=q.new_zeros((9,12))
  for a in range(4):
   for i in range(3):
    for J in range(3):B[3*i+J,3*a+i]=g[e,a,J]
  if qualified:
   P,C,state,diss,_=implicit_material_tangent(F,model.young,model.poisson,yield_stress,hardening,committed.points[e].plastic)
   ke=B.T@C.reshape(9,9)@B*v[e] if tangent else q.new_zeros((12,12));strategy="implicit";count=2;reason=""
   point=type(committed.points[e])(state,F.clone())
  else:
   def force(local):
    Ft=torch.eye(3,dtype=local.dtype,device=local.device)+local.reshape(4,3).T@g[e]
    r=integrate_adaptive(Ft,committed.points[e],model.young,model.poisson,yield_stress,hardening,rtol=material_rtol)
    return (r.first_piola@g[e].T).T.reshape(-1)*v[e]
   ke=torch.autograd.functional.jacobian(force,q) if tangent else q.new_zeros((12,12))
   if tangent and not bool(torch.isfinite(ke).all()):raise RuntimeError("hybrid adaptive AD tangent is non-finite")
   P=adaptive.first_piola;point=adaptive.state;diss=adaptive.dissipation;strategy="adaptive_ad";count=2
   reason="substeps" if adaptive.accepted_substeps>2 else "local_error"
  fe=(P@g[e].T).T.reshape(-1)*v[e];internal.index_add_(0,dofs,fe)
  if tangent:K.index_put_((dofs[:,None],dofs[None,:]),ke,accumulate=True)
  Ps.append(P);pts.append(point);added.append(diss*v[e]);strategies.append(strategy);counts.append(count);reasons.append(reason);errors.append(estimate)
 a=torch.stack(added);state=AdaptiveGlobalState(tuple(pts),committed.cumulative_dissipation+a)
 return internal,K,torch.stack(Ps),state,a,HybridMetrics(tuple(strategies),tuple(counts),tuple(reasons),tuple(errors))
