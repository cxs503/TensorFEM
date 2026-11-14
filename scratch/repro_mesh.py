import numpy as np
from tensorfem.beam_large_deflection import solve_corotational_beam, _assemble
L=1.0; b=1.0; t=0.1; E=1.2e6; EI=E*b*t**3/12
for n in (2,4,8,16):
    for alpha in (1.0,):
        s=solve_corotational_beam(n_elements=n, length=L, ea=1e6*EI, ei=EI, tip_force=alpha*EI/L**2)
        print(f"n={n:3d} alpha={alpha} conv={s.converged} iters={s.iterations} resid={s.residual_norm:.3e} tip_u={s.tip_x:.6f} tip_v={s.tip_y:.6f}")
