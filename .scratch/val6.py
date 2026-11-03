import sys, math; sys.path.insert(0,'src')
import torch as T
import tensorfem.hetenyi_reference as H
D=T.float64
EI=604.8; k_b=4.0e5; lam=(k_b/(4*EI))**0.25; Lc=1/lam; P=1000.0
L=2.0*Lc; xl=L/2
src = open('src/tensorfem/hetenyi_reference.py').read()
print("_END_CONDITION_ROWS", H._END_CONDITION_ROWS)
print("free rows", H._END_CONDITION_ROWS["free"])
x=T.linspace(0,L,3,dtype=D)
s=H.finite_beam_bvp(x,beam_length=L,lam=lam,k_b=k_b,EI=EI,point_loads=[(xl,P)])
print("z0",s.state_at_origin.tolist())
print("ends",s.end_conditions)
print("w",s.deflection.tolist())
print("w'' (lam^2 z3) check: M0/EI",float(s.moment[0]/EI), "V0/EI", float(s.shear[0]/EI))
print("lam*L", lam*L)
