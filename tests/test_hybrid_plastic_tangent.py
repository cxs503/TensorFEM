import torch
from tensorfem.hybrid_plastic_tangent import assemble_hybrid
from tensorfem.adaptive_global_plasticity import AdaptiveGlobalState,assemble_adaptive_global
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model
from tensorfem.nonproportional_plasticity import integrate_adaptive
D=torch.float64
def setup(scale):
 x=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D);m=FiniteStrainTet4Model(x,torch.tensor([[0,1,2,3]]),1000.,.3,torch.tensor([0,1,2]))
 s=AdaptiveGlobalState.virgin(m);F0=torch.tensor([[1.04,.12,.01],[.02,.98,.03],[0.,.01,1.]],dtype=D);p=integrate_adaptive(F0,s.points[0],1000.,.3,35.,60.,rtol=3e-5).state;s=AdaptiveGlobalState((p,),s.cumulative_dissipation)
 Ft=torch.tensor([[1.08,.16,.02],[.03,.95,.04],[.01,.02,1.01]],dtype=D);F=F0+scale*(Ft-F0);u=((F-torch.eye(3,dtype=D))@x.T).T.reshape(-1);return m,s,u
def test_small_implicit_large_ad_and_both_match_oracle():
 for scale,expected in ((.05,"implicit"),(1.,"adaptive_ad")):
  m,s,u=setup(scale);_,Kh,_,trial,diss,met=assemble_hybrid(m,u,s,yield_stress=35.,hardening=60.,material_rtol=3e-5)
  _,Kr,_,_,_,_=assemble_adaptive_global(m,u,s,yield_stress=35.,hardening=60.,material_rtol=3e-5)
  assert met.strategies==(expected,)
  assert float(torch.linalg.vector_norm(Kh-Kr)/torch.linalg.vector_norm(Kr))<.03
  assert float(diss.min())>=0 and abs(float(torch.linalg.det(trial.points[0].plastic.plastic_gradient))-1)<1e-10
def test_noncoaxial_newton_no_regression_and_restart_immutable():
 m,s,ut=setup(1.);free=torch.tensor([3,7,10,11]);target=assemble_hybrid(m,ut,s,yield_stress=35.,hardening=60.,material_rtol=3e-5,tangent=False)[0][free]
 def solve(hybrid):
  u=ut*.8
  for it in range(1,9):
   fn=assemble_hybrid if hybrid else assemble_adaptive_global;f,k,*_=fn(m,u,s,yield_stress=35.,hardening=60.,material_rtol=3e-5);r=target-f[free]
   if float(torch.linalg.vector_norm(r))<1e-8:return it,u
   u[free]+=torch.linalg.solve(k[free][:,free],r)
  raise RuntimeError
 ih,uh=solve(True);ir,ur=solve(False);assert ih<=ir and torch.linalg.vector_norm(uh[free]-ur[free])<1e-7
 before=s.points[0].plastic.plastic_gradient.clone();assemble_hybrid(m,ut,s,yield_stress=35.,hardening=60.,material_rtol=3e-5);assert torch.equal(before,s.points[0].plastic.plastic_gradient)
