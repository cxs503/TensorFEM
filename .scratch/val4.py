import sys, math; sys.path.insert(0,'src')
import torch as T
from tensorfem.hetenyi_reference import finite_beam_bvp, infinite_beam_point_load, foundation_reaction_integral
D=T.float64
EI=604.8; k_b=4.0e5; lam=(k_b/(4*EI))**0.25; Lc=1/lam; P=1000.0
L=2.0*Lc
x=T.linspace(0,L,11,dtype=D)
s=finite_beam_bvp(x,beam_length=L,lam=lam,k_b=k_b,EI=EI,point_loads=[(L/2,P)])
print("x/L   w            infinite")
ref=infinite_beam_point_load(x,L/2,P,lam,k_b)
for a,b,c in zip(x.tolist(),s.deflection.tolist(),ref.tolist()):
    print(f"{a/L:.2f} {b: .6e} {c: .6e}")
print("state0",s.state_at_origin.tolist())
print("V",s.shear.tolist()[:3], s.shear.tolist()[-3:])
print("M",s.moment.tolist()[:3], s.moment.tolist()[-3:])
