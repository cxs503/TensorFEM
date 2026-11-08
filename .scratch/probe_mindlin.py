import sys, time, math
import numpy as np
sys.path.insert(0, "src")
from tensorfem.hertz_mindlin import (
    hertz_cylinder_reference, block_mesh, plane_strain_stiffness,
    _ElasticSolver, _dirichlet_dofs, solve_indentation, fit_hertz_profile,
    cattaneo_stick_half_width,
)

E, NU, R = 210e9, 0.3, 0.05
DELTA = 2e-5
ref = hertz_cylinder_reference(young=E, poisson=NU, radius=R, indentation=DELTA)
print("ref b=%.6e p0=%.6e P=%.6e" % (ref.contact_half_width, ref.peak_pressure, ref.load_per_length))
b = ref.contact_half_width

t0 = time.time()
nx, ny = 120, 60
nodes, elems = block_mesh(nx, ny, half_width=30 * b, depth=20 * b, grading=2.0)
K = plane_strain_stiffness(nodes, elems, E, NU)
fixed = _dirichlet_dofs(nodes, ny)
solver = _ElasticSolver(K, fixed)
print("mesh", nodes.shape, "dofs", K.shape[0], "assemble %.1fs" % (time.time() - t0))

surf = np.arange(0, len(nodes), ny + 1)          # i*(ny+1)+0
x_surf = nodes[surf, 0]
cand_mask = np.abs(x_surf) <= 6 * b
surf_c = surf[cand_mask]
x_c = x_surf[cand_mask]
order = np.argsort(np.abs(x_c))
surf_c, x_c = surf_c[order], x_c[order]
w = np.zeros(len(x_c))
w[1:-1] = 0.5 * (x_c[2:] - x_c[:-2])
w[0] = 0.5 * (x_c[1] - x_c[0]); w[-1] = 0.5 * (x_c[-1] - x_c[-2])

t0 = time.time()
res = solve_indentation(nodes, elems, K, solver, surf_c, x_c, radius=R, indentation=DELTA)
print("normal solve %.1fs it=%d conv=%s" % (time.time() - t0, res.iterations, res.converged))
p = res.normal_force / w
fit = fit_hertz_profile(x_c[res.active], p[res.active], b)
print("P_fe=%.6e (ref %.6e) err %+.3f%%" % (res.load, ref.load_per_length, 100 * (res.load / ref.load_per_length - 1)))
print("b_fit=%.6e err %+.3f%%   p0_fit=%.6e err %+.3f%%" % (
    fit["half_width"], 100 * (fit["half_width"] / b - 1),
    fit["peak_pressure"], 100 * (fit["peak_pressure"] / ref.peak_pressure - 1)))
print("p_max_raw=%.4e err %+.3f%%" % (p.max(), 100 * (p.max() / ref.peak_pressure - 1)))
print("complementarity residual %.3e" % res.complementarity_residual)
print("nactive", res.active.sum(), "gap min %.3e" % res.gap.min())
print("pressure nodes:", np.array2string(p[res.active], precision=2))