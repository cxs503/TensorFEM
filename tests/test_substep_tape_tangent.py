import torch,pytest
from tensorfem.nonproportional_plasticity import virgin_adaptive_state,integrate_adaptive,richardson_algorithmic_tangent
from tensorfem.substep_tape_tangent import build_tape,replay_tape,taped_material_tangent,RetapeRequired
D=torch.float64;M=(1000.,.3,35.,60.)
def case(scale=1.):
 s=virgin_adaptive_state();F1=torch.tensor([[1.05,.18,.02],[.01,.97,.04],[0.,.02,1.01]],dtype=D);s=integrate_adaptive(F1,s,*M,rtol=3e-5).state
 Ft=torch.tensor([[1.14,.25,.04],[.03,.91,.08],[.01,.04,1.03]],dtype=D);return s,F1+scale*(Ft-F1)
def test_taped_chain_matches_adaptive_and_richardson_oracles():
 s,F=case();t=build_tape(F,s,*M,rtol=3e-5);P,C,new,diss,y=taped_material_tangent(F,s,t,*M)
 adaptive=integrate_adaptive(F,s,*M,rtol=3e-5);Cr=richardson_algorithmic_tangent(F,s,*M,rtol=3e-5)
 assert torch.linalg.vector_norm(P-adaptive.first_piola)/torch.linalg.vector_norm(adaptive.first_piola)<.03
 assert torch.linalg.vector_norm(C-Cr)/torch.linalg.vector_norm(Cr)<.03
 assert len(t.fractions)>2 and float(diss)>=0 and abs(float(torch.linalg.det(new.plastic.plastic_gradient))-1)<1e-10
def test_tape_neighborhood_gate_and_transaction():
 s,F=case(.15);t=build_tape(F,s,*M,rtol=3e-5);before=s.plastic.plastic_gradient.clone()
 replay_tape(F,s,t,*M);_,large=case(2.5)
 with pytest.raises(RetapeRequired):replay_tape(large,s,t,*M)
 assert torch.equal(before,s.plastic.plastic_gradient)
def test_taped_newton_converges():
 s,F=case(.7);t=build_tape(F,s,*M,rtol=3e-5);target=float(replay_tape(F,s,t,*M)[0][0,1]);x=float(F[0,1])*.9
 for it in range(8):
  X=F.clone();X[0,1]=x
  try:P,C,*_=taped_material_tangent(X,s,t,*M)
  except RetapeRequired:t=build_tape(X,s,*M,rtol=3e-5);P,C,*_=taped_material_tangent(X,s,t,*M)
  r=float(P[0,1])-target
  if abs(r)<1e-8:break
  x-=r/float(C[0,1,0,1])
 assert abs(r)<1e-8 and it<7
