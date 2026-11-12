import numpy as np
from tensorfem.cohesive3d_dcb import build_dcb_model, DCBGeometry, solve_dcb

g = DCBGeometry(nx=6, ny=1, nz=1)
m = build_dcb_model(g)
r = solve_dcb(m, delta_max=1e-6, n_steps=4, max_iter=30)
u, hist = r.displacement, r.history
f, K, hnew, diag = m.interface_response(u, hist)
sep, tra, area = diag["separation"], diag["traction"], diag["area"]
t_dot_d = np.einsum("egk,egk->eg", tra, sep)     # traction . separation per gp
E_coh = float(np.sum(0.5 * t_dot_d * area))
print("interface area total", area.sum(), "E_coh(0.5 t.d A) =", E_coh)
print("solid Es =", r.stats["final_strain_energy"], "W =", r.stats["final_external_work"])
print("Es+E_coh =", r.stats["final_strain_energy"] + E_coh, " ratio W/(Es+Ecoh) =",
      r.stats["final_external_work"] / (r.stats["final_strain_energy"] + E_coh))
print("path Wi =", r.stats["final_interface_work"], " vs E_coh =", E_coh)
print("sep max", np.abs(sep).max(), "trac max", np.abs(tra).max())
print("area per gp", area[0])
print("curve deltas", [c["disp"] for c in r.curve])
print("curve loads", [c["load"] for c in r.curve])
