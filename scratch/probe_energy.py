import numpy as np
from tensorfem.cohesive3d_dcb import build_dcb_model, DCBGeometry, solve_dcb

g = DCBGeometry(nx=6, ny=1, nz=1)
m = build_dcb_model(g)
print("n_dofs", m.n_dofs, "n_iface", m.n_iface)
# tiny opening: strictly linear elastic, no damage -> W == 0.5 u^T K u, diss == 0
for dmax in (1e-7, 1e-6):
    r = solve_dcb(m, delta_max=dmax, n_steps=4, max_iter=30)
    s = r.stats
    print(f"delta_max={dmax:.1e} steps={s['steps']} conv={s['converged_all']} "
          f"W={s['final_external_work']:.6e} Es={s['final_strain_energy']:.6e} "
          f"Wi={s['final_interface_work']:.6e} rel={s['energy_balance_rel']:.4f}")
