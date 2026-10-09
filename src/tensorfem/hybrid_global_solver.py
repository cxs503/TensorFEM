"""Global cyclic Newton driver for hybrid/AD/Richardson plastic tangents."""
from dataclasses import dataclass
from typing import Sequence,Literal
import torch
from .adaptive_global_plasticity import AdaptiveGlobalState,assemble_adaptive_global
from .ad_plastic_tangent import assemble_ad_global
from .hybrid_plastic_tangent import assemble_hybrid
from .nonlinear_step import StepState,IncrementRecord,solve_adaptive
@dataclass(frozen=True)
class HybridGlobalResult:
 load_factor:float;displacement:torch.Tensor;reaction:torch.Tensor;state:AdaptiveGlobalState
 increment_dissipation:torch.Tensor;increments:tuple[IncrementRecord,...]
 assembly_calls:int;material_paths:int;strategy_counts:dict[str,int];fallback_counts:dict[str,int]
def solve_hybrid_global_path(model,force,targets:Sequence[float],*,yield_stress,hardening,
 initial:HybridGlobalResult|None=None,strategy:Literal['hybrid','ad','richardson']='hybrid',material_rtol=3e-5,
 initial_increment=.1,minimum_increment=1e-4,maximum_increment=.2,tolerance=1e-8,max_iterations=18):
 force=force.to(model.reference_nodes);fixed=model.fixed_dofs.to(device=force.device,dtype=torch.long)
 mask=torch.ones(model.n_dofs,dtype=torch.bool,device=force.device);mask[fixed]=False;free=torch.arange(model.n_dofs,device=force.device)[mask]
 factor=0. if initial is None else initial.load_factor;u=torch.zeros_like(force) if initial is None else initial.displacement.clone()
 state=AdaptiveGlobalState.virgin(model) if initial is None else initial.state;outputs=[]
 assemblers={'hybrid':assemble_hybrid,'ad':assemble_ad_global,'richardson':assemble_adaptive_global}
 if strategy not in assemblers:raise ValueError('unknown tangent strategy')
 for value in targets:
  target=float(value);start=float(factor);delta=target-start;before=state.cumulative_dissipation.clone();calls=0;paths=0;sc={};fc={}
  def evaluate(q,progress,base):
   nonlocal calls,paths
   trial=u.clone().index_copy(0,free,q);fn=assemblers[strategy]
   fi,k,_,trial_state,_,metrics=fn(model,trial,base,yield_stress=yield_stress,hardening=hardening,material_rtol=material_rtol)
   calls+=1
   if strategy=='hybrid':
    paths+=sum(metrics.path_counts)
    for s in metrics.strategies:sc[s]=sc.get(s,0)+1
    for r in metrics.fallbacks:
     if r:fc[r]=fc.get(r,0)+1
   elif strategy=='ad':paths+=metrics.paths;sc['adaptive_ad']=sc.get('adaptive_ad',0)+len(model.elements)
   else:paths+=49*len(model.elements);sc['richardson']=sc.get('richardson',0)+len(model.elements)
   return (start+delta*progress)*force[free]-fi[free],k[free][:,free],trial_state
  advanced=solve_adaptive(evaluate,StepState(0.,u[free].clone(),state),target_factor=1.,initial_increment=initial_increment,
   minimum_increment=minimum_increment,maximum_increment=maximum_increment,tolerance=tolerance,max_iterations=max_iterations)
  u=u.clone().index_copy(0,free,advanced.displacement);state=advanced.material_state;factor=target
  fi=assemblers[strategy](model,u,state,yield_stress=yield_stress,hardening=hardening,material_rtol=material_rtol,tangent=False)[0]
  outputs.append(HybridGlobalResult(factor,u.clone(),fi-factor*force,state,state.cumulative_dissipation-before,
   tuple(advanced.history),calls,paths,dict(sc),dict(fc)))
 return tuple(outputs)
