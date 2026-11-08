import sys, time, math
import numpy as np
sys.path.insert(0, "src")
import tensorfem.hertz_mindlin as hm

E, NU, R = 210e9, 0.3, 0.05
P_TARGET = 3.625e6   # N/m
E_STAR = E / (1 - NU ** 2)
b_ref = math.sqrt(4 * P_TARGET * R / (math.pi * E_STAR))
p0_ref = 2 * P_TARGET / (math.pi * b_ref)
print("load-controlled 2D Hertz: P=%.4e b=%.6e p0=%.6e" % (P_TARGET, b_ref, p0_ref))


def run(nx, ny, delta, half_width_factor=30.0, depth_factor=20.0, verbose=False):
    nodes, elems = hm.block_mesh(nx, ny, half_width_factor * b_ref, depth_factor * b_ref, grading=2.0)
    K = hm.plane_strain_stiffness(nodes, elems, E, NU)
    solver = hm._ElasticSolver(K, hm._dirichlet_dofs(nodes, ny))
    surf = np.arange(0, len(nodes), ny + 1)
    xs = nodes[surf, 0]
    m = np.abs(xs) <= 6 * b_ref
    sc, xc = surf[m], xs[m]
    o = np.argsort(np.abs(xc))
    sc, xc = sc[o], xc[o]
    w = np.zeros(len(xc)); w[1:-1] = 0.5 * (xc[2:] - xc[:-2])
    w[0] = 0.5 * (xc[1] - xc[0]); w[-1] = 0.5 * (xc[-1] - xc[-2])
    res = hm.solve_indentation(nodes, elems, K, solver, sc, xc, radius=R, indentation=delta,
                               max_candidates=len(xc))
    p = res.normal_force / w
    fit = hm.fit_hertz_profile(xc[res.active], p[res.active], b_ref * 0.5)
    return res, fit, p, xc, sc, solver, nodes, elems


def load_controlled(nx, ny, tol=1e-4, iters=40):
    lo, hi = 1e-7, 1e-3
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res, *_ = run(nx, ny, mid)
        if res.load < P_TARGET:
            lo = mid
        else:
            hi = mid
        if abs(res.load / P_TARGET - 1) < tol:
            break
    return mid


for nx, ny in [(60, 40), (100, 60), (160, 100)]:
    t0 = time.time()
    delta = load_controlled(nx, ny)
    res, fit, p, xc, sc, solver, nodes, elems = run(nx, ny, delta)
    ext = np.abs(xc[res.active]).max()
    print("mesh %3dx%3d delta=%.4e P=%.4e (%.2f%%) ext=%.4e (%.2f%%) bfit=%.4e (%.2f%%) "
          "p0fit=%.4e (%.2f%%) pmax=%.4e (%.2f%%) nact=%d resid=%.1e  %.1fs" % (
              nx, ny, delta, res.load, 100 * (res.load / P_TARGET - 1), ext,
              100 * (ext / b_ref - 1), fit["half_width"], 100 * (fit["half_width"] / b_ref - 1),
              fit["peak_pressure"], 100 * (fit["peak_pressure"] / p0_ref - 1),
              p.max(), 100 * (p.max() / p0_ref - 1), res.active.sum(),
              res.complementarity_residual, time.time() - t0))