"""Convention-free SERR oracle: G = -(1/b) dU/da at fixed total opening."""
import numpy as np
from tensorfem.cohesive3d_dcb import (build_dcb_model, DCBGeometry, solve_dcb,
                                      beam_serr, beam_compliance)

L, b, h, E, nu = 30e-3, 3e-3, 3e-3, 70e9, 0.30
delta = 1e-8
rows = []
for a0 in (15e-3, 14e-3, 13e-3):          # crack length from loaded end
    g = DCBGeometry(length=L, width=b, arm_thickness=h, young_modulus=E,
                    poisson_ratio=nu, initial_crack=a0, nx=60, ny=1, nz=2)
    m = build_dcb_model(g)
    r = solve_dcb(m, delta_max=delta, n_steps=1, max_iter=30)
    U = r.stats["final_strain_energy"]
    load_rep = r.curve[-1]["load"]
    P_arm = 2.0 * U / delta
    rows.append((a0, U, load_rep, P_arm, g.bonded_length))
    print(f"a0={a0*1e3:5.1f}mm  U={U:.6e}  load_reported={load_rep:.6f} "
          f"P_arm(2U/delta)={P_arm:.6f}  ratio={load_rep/P_arm:.6f} "
          f"n_iface={m.n_iface}")

a = np.array([r[0] for r in rows]); U = np.array([r[1] for r in rows])
P = np.array([r[3] for r in rows])
# G = -(1/b) dU/da at fixed delta  (central difference on non-uniform? uniform here)
G_num = -(U[2] - U[0]) / (a[2] - a[0]) / b
print(f"\nG_numeric (energy route, delta fixed) = {G_num:.4f} J/m^2")
am = a[1]; Pm = P[1]
classic = 12.0 * Pm**2 * am**2 / (E * b**2 * h**3)
their_parm = beam_serr(am, 300.0, DCBGeometry(length=L, width=b, arm_thickness=h,
                                             young_modulus=E, poisson_ratio=nu,
                                             initial_crack=am), modulus=E,
                       root_rotation=False, shear=False) if False else None
print(f"classic 12*P_arm^2*a^2/(E b^2 h^3) = {classic:.4f} J/m^2 "
      f"(ratio G_num/classic = {G_num/classic:.4f})")
# their beam_serr with the *reported* load (-> G = 6*P^2 a^2/(E b^2 h^3))
their = 6.0 * (2*Pm)**2 * am**2 / (E * b**2 * h**3)
print(f"their 6*P_rep^2*a^2/(E b^2 h^3)   = {their:.4f} J/m^2 "
      f"(ratio G_num/their = {G_num/their:.4f})")
print(f"6*P_arm^2*a^2/(E b^2 h^3)         = {6*Pm**2*am**2/(E*b**2*h**3):.4f}")
