"""Exploratory 4: strong-coupling multi-DOF beam -- residual sequence, omega trace, gain spectrum."""
import numpy as np
from tensorfem.ibm_aitken_implicit import ImplicitIBMFSI, BeamProperties
from explore2 import build


def trace(mode="aitken", omega=1.0, maxit=8, rho=1.0, dt=1.0, amp=0.05,
          n_elements=10, lat=(64, 64), thin=1.0):
    r = []
    for m in range(1, maxit + 1):
        beam, fluid, solver = build(n_elements=n_elements, amp=amp, fluid_rho=rho,
                                    lattice=lat, dt=dt)
        # allow a thin section: A and thickness scale the structural mass
        beam.props = BeamProperties(E=beam.props.E, A=beam.props.A * thin,
                                    I=beam.props.I * thin ** 3, rho=beam.props.rho,
                                    length=beam.props.length,
                                    thickness=beam.props.thickness * thin)
        beam._assemble_constant_matrices()
        solver.relax_mode = mode
        solver.fixed_omega = omega
        solver.sub_tol = 0.0
        solver.max_sub_iter = m
        try:
            rec = solver.step(1)
            r.append((rec["residual"], rec["omega"]))
        except Exception as e:
            r.append((float("nan"), float("nan")))
            break
    return r


def gain_spectrum(rho=1.0, dt=1.0, amp=0.05, n_elements=10, lat=(64, 64),
                  thin=1.0, h_rel=1e-6):
    """Static d u_new / d u about the released state, velocity held at the base value."""
    beam, fluid, solver = build(n_elements=n_elements, amp=amp, fluid_rho=rho,
                                lattice=lat, dt=dt)
    beam.props = BeamProperties(E=beam.props.E, A=beam.props.A * thin,
                                I=beam.props.I * thin ** 3, rho=beam.props.rho,
                                length=beam.props.length,
                                thickness=beam.props.thickness * thin)
    beam._assemble_constant_matrices()
    rho_f, uf = fluid.macroscopic(include_force=False)
    free = beam.free
    u0, v0, a0 = beam.u.copy(), beam.v.copy(), beam.a.copy()
    vb = v0.reshape(-1, 3)[:, :2].copy()

    def G(y):
        beam.u = y.copy()
        solver.ibm.move(beam.markers())
        rho_i, mom_i = solver.ibm.interpolate(rho_f, uf)
        f_lag = solver.ibm.force(rho_i, mom_i, vb, dt=dt)
        load = beam.apply_marker_loads(-f_lag)
        out = beam.newmark_solve(load, dt)[0]
        return out.copy()

    base = G(u0)
    h = h_rel * max(1.0, float(np.abs(u0).max()))
    A = np.zeros((len(free), len(free)))
    for c, j in enumerate(free):
        y = u0.copy()
        y[j] += h
        beam.v, beam.a = v0.copy(), a0.copy()
        A[:, c] = (G(y) - base)[free] / h
    M = A - np.eye(len(free))
    ev = np.linalg.eigvals(M)
    return A, M, ev, base[free] - u0[free]


if __name__ == "__main__":
    for tag, kw in [
            ("rho1 dt1 thin1", dict(rho=1.0, dt=1.0, thin=1.0)),
            ("rho1 dt1 thin0.3", dict(rho=1.0, dt=1.0, thin=0.3)),
            ("rho1 dt0.3 thin1", dict(rho=1.0, dt=0.3, thin=1.0)),
    ]:
        tr = trace(**kw)
        print(tag, "res:", ["%.2e" % r for r, _ in tr])
        print("      om:", ["%.3f" % o for _, o in tr])
        A, M, ev, r0 = gain_spectrum(**kw)
        g = ev + 1.0
        ag = np.abs(g)
        order = np.argsort(-ag)
        print("      |g| top5:", np.round(ag[order][:5], 4),
              " bottom:", np.round(ag[order][-3:], 4),
              " spread:", round(float(ag.max() / max(ag.min(), 1e-30)), 2))
        print("      g top5 real:", np.round(g[order][:5].real, 4),
              " R=-g:", np.round(-g[order][:5].real, 4))
        print("      rank(M):", np.linalg.matrix_rank(M, tol=1e-8 * np.abs(M).max()),
              " sv top5:", np.round(np.linalg.svd(M, compute_uv=False)[:5], 4))