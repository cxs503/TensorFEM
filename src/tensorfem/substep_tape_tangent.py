"""Reproducible accepted-substep tape and chained finite-plastic tangent."""
from dataclasses import dataclass
import torch
from .finite_strain_plasticity import update_multiplicative_j2
from .nonproportional_plasticity import AdaptiveFiniteJ2State
@dataclass(frozen=True)
class SubstepTape:
 fractions:tuple[float,...];rtol:float;max_local_error:float
class RetapeRequired(RuntimeError):pass
def _err(a,b):
 fp=torch.linalg.vector_norm(a.plastic_gradient-b.plastic_gradient)/max(float(torch.linalg.vector_norm(b.plastic_gradient).detach()),1.)
 al=torch.abs(a.alpha-b.alpha)/max(abs(float(b.alpha.detach())),1.)
 return float(torch.maximum(fp,al).detach())
def build_tape(Ftarget,state,young,poisson,sy,H,*,rtol=2e-5,max_depth=12):
 F0=state.deformation_gradient;accepted=[];worst=0.
 def walk(ta,tb,base,depth):
  nonlocal worst
  Fa=F0+ta*(Ftarget-F0);Fb=F0+tb*(Ftarget-F0);tm=.5*(ta+tb);Fm=F0+tm*(Ftarget-F0)
  _,sf,_,_=update_multiplicative_j2(Fb,young,poisson,sy,H,base)
  _,sm,_,_=update_multiplicative_j2(Fm,young,poisson,sy,H,base);_,sh,_,_=update_multiplicative_j2(Fb,young,poisson,sy,H,sm)
  e=_err(sf,sh);worst=max(worst,e)
  if e<=rtol:accepted.extend((tm,tb));return sh
  if depth>=max_depth:raise RetapeRequired(f"cannot build tape: error={e:g}")
  left=walk(ta,tm,base,depth+1);return walk(tm,tb,left,depth+1)
 walk(0.,1.,state.plastic,0)
 return SubstepTape(tuple(accepted),rtol,worst)
def replay_tape(Ftarget,state,tape,young,poisson,sy,H,*,validate=True,safety=1.05):
 F0=state.deformation_gradient;base=state.plastic;previous=0.;diss=F0.new_zeros(());P=None
 # Pairs are the two accepted half steps of every leaf interval.
 for n in range(0,len(tape.fractions),2):
  tm,tb=tape.fractions[n:n+2];Fa=F0+previous*(Ftarget-F0);Fm=F0+tm*(Ftarget-F0);Fb=F0+tb*(Ftarget-F0)
  if validate:
   _,sf,_,_=update_multiplicative_j2(Fb,young,poisson,sy,H,base)
   _,sm,d1,_=update_multiplicative_j2(Fm,young,poisson,sy,H,base);P,sh,d2,y=update_multiplicative_j2(Fb,young,poisson,sy,H,sm)
   if _err(sf,sh)>safety*tape.rtol:raise RetapeRequired("accepted path left its local error domain")
  else:
   _,sm,d1,_=update_multiplicative_j2(Fm,young,poisson,sy,H,base);P,sh,d2,y=update_multiplicative_j2(Fb,young,poisson,sy,H,sm)
  base=sh;diss=diss+d1+d2;previous=tb
 return P,AdaptiveFiniteJ2State(base,Ftarget.clone()),diss,y
def taped_material_tangent(Ftarget,state,tape,young,poisson,sy,H):
 # Validate once without building the derivative graph, then replay exactly.
 replay_tape(Ftarget,state,tape,young,poisson,sy,H,validate=True)
 fn=lambda F:replay_tape(F,state,tape,young,poisson,sy,H,validate=False)[0]
 P,new,diss,y=replay_tape(Ftarget,state,tape,young,poisson,sy,H,validate=False)
 C=torch.autograd.functional.jacobian(fn,Ftarget)
 if not bool(torch.isfinite(C).all()):raise RuntimeError("taped chain tangent is non-finite")
 return P,C,new,diss,y
