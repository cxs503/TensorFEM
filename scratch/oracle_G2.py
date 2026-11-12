"""Slender-arm convention-free G oracle (bending dominance)."""
import numpy as np
from tensorfem.cohesive3d_dcb import build_dcb_model, DCBGeometry, solve_dcb, beam_compliance

E, nu = 70e9, 0.30
L, b, h = 60e-3, 3e-3, 1.0e-3
delta = 1e-8
def run(a0, nx=120):
    g = DCBGeometry(length=L, width=b, arm_thickness=h, young_modulus=E,
                    poisson_ratio=nu, initial_crack=a0, nx=nx, ny=1, nz=2)
    m = build_dcb_model(g)
    r = solve_dcb(m, delta_max=delta, n_steps=1, max_iter=30)
    return r.stats["final_strain_energy"], r.curve[-1]["load"], g

a_list = [40e-3, 38e-3, 36e-3]
out = [run(a) for a in a_list]
for a, (U, load, g) in zip(a_list, out):
    P_arm = 2*U/delta
    print(f"a0={a*1e3:5.1f}mm U={U:.4e} load={load:.6f} P_arm={P_arm:.6f} ratio={load/P_arm:.4f}")
U = np.array([o[0] for o in out]); a = np.array(a_list); P = np.array([2*o[0]/delta for o in out])
G_num = -(U[2]-U[0])/(a[2]-a[0])/b
am, Pm = a[1], P[1]
cl = 12*Pm**2*am**2/(E*b**2*h**3)
g = out[1][2]

# full beam prediction with root rotation + shear, per-arm convention (C_arm = 2*beam_compliance)
def C_arm(cr):
    return 2.0*beam_compliance(cr, g, modulus=E)
dC = (C_arm(am+0.25e-3)-C_arm(am-0.25e-3))/0.5e-3
G_beam_arm = Pm**2/(2*b)*dC
G_their_convention = 6*(2*Pm)**2*am**2/(E*b**2*h**3)   # module's beam_serr(load=load)
print(f"G_num={G_num:.4f}  classic_a^3={cl:.4f} (r={G_num/cl:.3f})  "
      f"beam_full_per-arm={G_beam_arm:.4f} (r={G_num/G_beam_arm:.3f})  "
      f"module beam_serr(2P)={G_their_convention:.4f} (r={G_num/G_their_convention:.3f})")
