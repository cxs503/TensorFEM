import torch
from tensorfem.ad_plastic_tangent import assemble_ad_global,benchmark_tangents
from tensorfem.adaptive_global_plasticity import AdaptiveGlobalState,assemble_adaptive_global
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model
from tensorfem.nonproportional_plasticity import integrate_adaptive
D=torch.float64
def setup():
 x=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
 m=FiniteStrainTet4Model(x,torch.tensor([[0,1,2,3]]),1000.,.3,torch.tensor([0,1,2,4,5,6,8,9]))
 s=AdaptiveGlobalState.virgin(m);F1=torch.tensor([[1.05,.18,.02],[.01,.97,.04],[0.,.02,1.01]],dtype=D)
 p=integrate_adaptive(F1,s.points[0],1000.,.3,35.,60.,rtol=3e-5).state;s=AdaptiveGlobalState((p,),s.cumulative_dissipation)
 F2=torch.tensor([[1.14,.25,.04],[.03,.91,.08],[.01,.04,1.03]],dtype=D);u=((F2-torch.eye(3,dtype=D))@x.T).T.reshape(-1)
 return m,s,u
def test_ad_matches_oracle_and_preserves_gates():
 m,s,u=setup();_,Ka,_,trial,diss,metrics=assemble_ad_global(m,u,s,yield_stress=35.,hardening=60.,material_rtol=3e-5)
 _,Kr,_,_,_,_=assemble_adaptive_global(m,u,s,yield_stress=35.,hardening=60.,material_rtol=3e-5)
 assert float(torch.linalg.vector_norm(Ka-Kr)/torch.linalg.vector_norm(Kr))<.03
 assert metrics.paths==3 and float(diss.min())>=0 and abs(float(torch.linalg.det(trial.points[0].plastic.plastic_gradient))-1)<1e-10
def test_ad_newton_not_slower_and_checkpoint_immutable():
 m,s,ut=setup();free=torch.tensor([3,7,10,11]);target=assemble_ad_global(m,ut,s,yield_stress=35.,hardening=60.,material_rtol=3e-5,tangent=False)[0][free]
 def solve(ad):
  u=ut*.8
  for it in range(1,9):
   fn=assemble_ad_global if ad else assemble_adaptive_global;f,k,*_=fn(m,u,s,yield_stress=35.,hardening=60.,material_rtol=3e-5);r=target-f[free]
   if float(torch.linalg.vector_norm(r))<1e-8:return it,u
   u[free]+=torch.linalg.solve(k[free][:,free],r)
  raise RuntimeError
 ia,ua=solve(True);ir,ur=solve(False);assert ia<=ir and torch.linalg.vector_norm(ua[free]-ur[free])<1e-7
 before=s.points[0].plastic.plastic_gradient.clone();assemble_ad_global(m,ut,s,yield_stress=35.,hardening=60.,material_rtol=3e-5);assert torch.equal(before,s.points[0].plastic.plastic_gradient)
def test_measured_reduction():
 m,s,u=setup();e=benchmark_tangents(m,u,s,yield_stress=35.,hardening=60.,material_rtol=3e-5)
 assert e['relative_error']<.03 and e['ad_paths']<e['oracle_paths']/10 and e['speedup']>1.1
