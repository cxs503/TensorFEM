"""Exploratory 6: conditioned gain probe -- beam spectrum + 1-DOF plate rank."""
import numpy as np
from tensorfem.ibm_aitken_implicit import ImplicitIBMFSI, BeamProperties
from explore2 import build


def gain_beam(rho=1.0, dt=0.1, thin=1.0, amp=0.05, n_elements=10, lat=(64, 64),
              h_rel=1e-3, tol=1e-14, max_iter=12):
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
        out, ok, nit, res = beam.newmark_solve(load, dt, tol=tol, max_iter=max_iter)
        return out.copy(), nit, ok

    base, nit0, _ = G(u0)
    out = {}
    for hr in (h_rel, 0.5 * h_rel):
        h = hr * max(1.0, float(np.abs(u0).max()))
        A = np.zeros((len(free), len(free)))
        for c, j in enumerate(free):
            y = u0.copy()
            y[j] += h
            beam.v, beam.a = v0.copy(), a0.copy()
            col, _, _ = G(y)
            beam.v, beam.a = v0.copy(), a0.copy()
            A[:, c] = (col - base)[free] / h
        out[hr] = A
    A1, A2 = out[h_rel], out[0.5 * h_rel]
    ridges = float(np.abs(A1 - A2).max() / max(np.abs(A1).max(), 1e-30))
    M = A1 - np.eye(len(free))
    ev = np.linalg.eigvals(M)
    return dict(A=A1, M=M, ev=ev, richardson=ridges, newton_iters=nit0,
                umax=float(np.abs(beam.u).max()))


def report(tag, d):
    g = d["ev"] + 1.0
    ag = np.abs(g)
    order = np.argsort(-ag)
    print(f"--- {tag}: newton_iters={d['newton_iters']} richardson={d['richardson']:.2e}")
    print("   |g| top8:", np.round(ag[order][:8], 5))
    print("   |g| bottom4:", np.round(ag[order][-4:], 5),
          " spread:", round(float(ag.max() / ag.min()), 3))
    print("   Re(g) top8:", np.round(g[order][:8].real, 4))
    sv = np.linalg.svd(d["M"], compute_uv=False)
    print("   rank(tol 1e-8):", np.linalg.matrix_rank(d["M"], tol=1e-8 * sv[0]),
          " sv:", np.round(sv[:6], 4), " sv_ratio:", round(float(sv[0] / sv[-1]), 2))


if __name__ == "__main__":
    for tag, kw in [("beam rho1 dt0.1", dict(rho=1.0, dt=0.1)),
                    ("beam rho0.5 dt5e-3", dict(rho=0.5, dt=5e-3)),
                    ("beam rho1 dt0.1 thin0.3", dict(rho=1.0, dt=0.1, thin=0.3))]:
        report(tag, gain_beam(**kw))