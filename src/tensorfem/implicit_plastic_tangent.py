"""Branch-robust local implicit tangent for multiplicative logarithmic J2."""
from dataclasses import dataclass
import time,torch
from .finite_strain_plasticity import MultiplicativeJ2State
Tensor=torch.Tensor
_X=(.019855071751,.101666761293,.237233795042,.408282678752,.591717321248,.762766204958,.898333238707,.980144928249)
_W=(.050614268145,.111190517227,.156853322939,.181341891689,.181341891689,.156853322939,.111190517227,.050614268145)
def smooth_spd_log(A):
 """Eight-point integral log, smooth for repeated positive eigenvalues."""
 I=torch.eye(3,dtype=A.dtype,device=A.device);B=A-I; out=torch.zeros_like(A)
 for x,w in zip(_X,_W):out=out+w*torch.linalg.solve(I+x*B,B)
 return .5*(out+out.T)
def implicit_update(F,young,poisson,yield_stress,hardening,state):
 if float(torch.linalg.det(F).detach())<=0:raise ValueError("inverted F")
 G=young/(2*(1+poisson));K=young/(3*(1-2*poisson));I=torch.eye(3,dtype=F.dtype,device=F.device)
 Fe=F@torch.linalg.inv(state.plastic_gradient);E=.5*smooth_spd_log(Fe.T@Fe);dev=E-torch.trace(E)/3*I
 Mdev=2*G*dev;q=torch.sqrt(1.5*torch.sum(Mdev*Mdev));f=q-(yield_stress+hardening*state.alpha)
 if bool(f.detach()>0):
  gamma=f/(3*G+hardening);N=1.5*Mdev/q;Fp=torch.matrix_exp(gamma*N)@state.plastic_gradient;alpha=state.alpha+gamma
 else:gamma=torch.zeros_like(q);Fp=state.plastic_gradient;alpha=state.alpha
 trial=MultiplicativeJ2State(Fp,alpha);Fe=F@torch.linalg.inv(Fp);E=.5*smooth_spd_log(Fe.T@Fe)
 M=K*torch.trace(E)*I+2*G*(E-torch.trace(E)/3*I)
 P=torch.linalg.inv(Fe).T@M@torch.linalg.inv(Fp).T
 diss=gamma*(yield_stress+hardening*(state.alpha+.5*gamma))
 return P,trial,diss
def implicit_material_tangent(F,young,poisson,yield_stress,hardening,state):
 """dP/dF after scalar consistency elimination; one primal AD trace."""
 fn=lambda X:implicit_update(X,young,poisson,yield_stress,hardening,state)[0]
 start=time.perf_counter();P,trial,diss=implicit_update(F,young,poisson,yield_stress,hardening,state)
 C=torch.autograd.functional.jacobian(fn,F);elapsed=time.perf_counter()-start
 if not bool(torch.isfinite(C).all()):raise RuntimeError("implicit tangent is non-finite")
 return P,C,trial,diss,elapsed
def tet4_implicit_response(local_u,gradient,volume,young,poisson,yield_stress,hardening,state):
 F=torch.eye(3,dtype=local_u.dtype,device=local_u.device)+local_u.reshape(4,3).T@gradient
 P,C,trial,diss,elapsed=implicit_material_tangent(F,young,poisson,yield_stress,hardening,state)
 force=(P@gradient.T).T.reshape(-1)*volume
 # Exact chain-rule assembly from material dP/dF.
 B=local_u.new_zeros((9,12))
 for a in range(4):
  for i in range(3):
   for J in range(3):B[3*i+J,3*a+i]=gradient[a,J]
 A=C.reshape(9,9);K=B.T@A@B*volume
 return force,K,trial,diss,elapsed
