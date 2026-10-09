import torch
from tensorfem.finite_strain_plasticity import MultiplicativeJ2State,update_multiplicative_j2
from tensorfem.implicit_plastic_tangent import implicit_material_tangent,implicit_update,tet4_implicit_response
from tensorfem.nonproportional_plasticity import AdaptiveFiniteJ2State,richardson_algorithmic_tangent
D=torch.float64;MAT=(1000.,.3,35.,60.)
def virgin():return MultiplicativeJ2State(torch.eye(3,dtype=D),torch.zeros((),dtype=D))
def oracle(F,state,h=3e-6):
 C=torch.empty((3,3,3,3),dtype=D)
 for k in range(3):
  for l in range(3):
   p=F.clone();m=F.clone();p[k,l]+=h;m[k,l]-=h
   C[:,:,k,l]=(implicit_update(p,*MAT,state)[0]-implicit_update(m,*MAT,state)[0])/(2*h)
 return C
def test_implicit_matches_two_oracles_noncoaxial():
 F0=torch.tensor([[1.04,.12,.01],[.02,.98,.03],[0.,.01,1.]],dtype=D)
 _,state,_,_=update_multiplicative_j2(F0,*MAT,virgin());F=torch.tensor([[1.08,.16,.02],[.03,.95,.04],[.01,.02,1.01]],dtype=D)
 P,C,trial,diss,_=implicit_material_tangent(F,*MAT,state);Cr=oracle(F,state)
 assert torch.linalg.vector_norm(C-Cr)/torch.linalg.vector_norm(Cr)<3e-3
 # Existing spectral return is a second stress oracle for this qualified increment.
 Ps,_,_,_=update_multiplicative_j2(F,*MAT,state)
 assert torch.linalg.vector_norm(P-Ps)/torch.linalg.vector_norm(Ps)<.03
 assert float(diss)>=0 and abs(float(torch.linalg.det(trial.plastic_gradient))-1)<1e-10
 # Local-domain comparison to the adaptive Richardson/AD-path oracle.
 Flocal=F0+.1*(F-F0)
 _,Ci,_,_,_=implicit_material_tangent(Flocal,*MAT,state)
 Ca=richardson_algorithmic_tangent(Flocal,AdaptiveFiniteJ2State(state,F0),*MAT,rtol=3e-5)
 assert torch.linalg.vector_norm(Ci-Ca)/torch.linalg.vector_norm(Ca)<.03
def test_near_repeated_spectrum_is_finite_and_transition_newton():
 state=virgin();F=torch.diag(torch.tensor([1.08,1.08,.94],dtype=D));P,C,*_=implicit_material_tangent(F,*MAT,state)
 assert torch.isfinite(C).all()
 target=float(P[0,0]);x=1.05
 for it in range(8):
  X=torch.diag(torch.tensor([x,1.08,.94],dtype=D));p,c,*_=implicit_material_tangent(X,*MAT,state);r=float(p[0,0])-target
  if abs(r)<1e-9:break
  x-=r/float(c[0,0,0,0])
 assert abs(r)<1e-9 and it<7
def test_tet_chain_rule_matches_force_difference():
 g=torch.tensor([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D);u=torch.zeros(12,dtype=D);u[3]=.09;u[7]=-.02
 f,K,*_=tet4_implicit_response(u,g,1/6,*MAT,virgin());h=4e-6;up=u.clone();um=u.clone();up[3]+=h;um[3]-=h
 fp=tet4_implicit_response(up,g,1/6,*MAT,virgin())[0];fm=tet4_implicit_response(um,g,1/6,*MAT,virgin())[0]
 assert torch.linalg.vector_norm((fp-fm)/(2*h)-K[:,3])/torch.linalg.vector_norm(K[:,3])<3e-3
