import sys, math; sys.path.insert(0,'src')
import torch as T
from tensorfem.hetenyi_reference import (finite_beam_bvp, finite_beam_series,
    infinite_beam_point_load, infinite_beam_uniform_load, foundation_reaction_integral)
D = T.float64
E, b, t = 2.1e11, 0.02, 0.012
I = b*t**3/12; EI = E*I
k_found = 2.0e7   # N/m^3
k_b = k_found*b
lam = (k_b/(4*EI))**0.25
Lc = 1/lam
print("EI",EI,"k_b",k_b,"lam",lam,"1/lam",Lc)

# 1) infinite-beam limit: long beam, point load at centre
L = 20*Lc
x = T.linspace(0.12*L, 0.88*L, 41, dtype=D)
for ea, eb in [("free","free"),("clamped","clamped"),("simply_supported","simply_supported")]:
    s = finite_beam_bvp(x, beam_length=L, lam=lam, k_b=k_b, EI=EI, point_loads=[(L/2, 1.0e3)], end_a=ea, end_b=eb)
    ref = infinite_beam_point_load(x, L/2, 1.0e3, lam, k_b)
    print(f"limit {ea:16s}/{eb:16s} max rel {float(((s.deflection-ref)/ref).abs().max()):.3e}")

# 2) simply supported vs Fourier series (uniform load)
L = 3.0*Lc
q0 = 500.0
x = T.linspace(0.05*L, 0.95*L, 41, dtype=D)
s = finite_beam_bvp(x, beam_length=L, lam=lam, k_b=k_b, EI=EI, distributed_loads=[(0.0,L,q0)], end_a="simply_supported", end_b="simply_supported")
ser = finite_beam_series(x, L, lam, k_b, EI, [], boundary="simply_supported", n_terms=200)
# series needs uniform load: check signature
print("series sig ok, len", ser.shape, float(ser.abs().max()))
