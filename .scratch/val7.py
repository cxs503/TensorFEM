import sys, math; sys.path.insert(0,'src')
import torch as T
from tensorfem.hetenyi_reference import beam_state_propagator
D=T.float64
EI=604.8; k_b=4.0e5; lam=(k_b/(4*EI))**0.25; Lc=1/lam; P=1000.0
L=2.0*Lc; xl=L/2
stations=[0.0, xl, L]
p_acc=T.eye(4,dtype=D); b_acc=T.zeros(4,dtype=D)
per=[(p_acc.clone(), b_acc.clone())]
for i in range(2):
    if i>0:
        b_acc=b_acc.clone(); b_acc[3]=b_acc[3]+P/(EI*lam**3)
        print("after jump b=",b_acc.tolist())
    prop=beam_state_propagator(lam*(stations[i+1]-stations[i]))
    b_acc=prop@b_acc
    p_acc=prop@p_acc
    per.append((p_acc.clone(),b_acc.clone()))
A_final=p_acc; b_final=b_acc
print("b_final",b_final.tolist())
rows=[T.tensor([0.,0.,1.,0.],dtype=D),T.tensor([0.,0.,0.,1.],dtype=D),A_final[2,:].clone(),A_final[3,:].clone()]
M=T.stack(rows); r=T.tensor([0.,0.,-b_final[2],-b_final[3]],dtype=D)
z0=T.linalg.solve(M,r)
print("z0",z0.tolist())
print("jump expected", P/(EI*lam**3))
