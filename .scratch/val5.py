import sys, math; sys.path.insert(0,'src')
import torch as T
from tensorfem.hetenyi_reference import beam_state_propagator
D=T.float64
EI=604.8; k_b=4.0e5; lam=(k_b/(4*EI))**0.25; Lc=1/lam; P=1000.0
L=2.0*Lc; xl=L/2
P0=beam_state_propagator(lam*xl)
jump=T.zeros(4,dtype=D); jump[3]=P/(EI*lam**3)
# z(L) = P0 (P0 z0 + jump)
def ztot(z0):
    z1=P0@z0+jump
    return P0@z1
# solve BCs: free free
# rows: z0[2]=0,z0[3]=0 ; zL[2]=0,zL[3]=0
import torch
M=torch.zeros((4,4),dtype=D); r=torch.zeros(4,dtype=D)
M[0,2]=1; M[1,3]=1
# symbolic: ztot is affine -> compute matrix
cols=[]
for j in range(4):
    e=torch.zeros(4,dtype=D); e[j]=1
    cols.append(ztot(e))
A=torch.stack(cols,dim=1); b=ztot(torch.zeros(4,dtype=D))
M[2,:]=A[2,:]; M[3,:]=A[3,:]; r[2]=-b[2]; r[3]=-b[3]
z0=torch.linalg.solve(M,r)
zL=A@z0+b
print("z0",z0.tolist())
print("zL",zL.tolist())
print("w0",z0[0].item(),"wL",zL[0].item())
print("cond", torch.linalg.cond(M).item())
