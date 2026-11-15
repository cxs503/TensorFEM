"""Exploratory 7: (u,v) state map of the beam sub-iteration + validation of its cost prediction."""
import numpy as np
from tensorfem.ibm_aitken_implicit import (
    ImplicitIBMFSI, BeamProperties, AitkenRelaxation,
)
from explore2 import build

REF = 28.8  # beam length in the lattice (scale = 0.45*64); recomputed below


def state_map(rho=1.0, dt=0.1, thin=1.0, amp=0.05, n_elements=10, lat=(64, 64),
              h_rel=1e-3, tol=1e-14, max_iter=12):
    beam, fluid, solver = build(n_elements=n_elements, amp=amp, fluid_rho=rho,
                                lattice=lat, dt=dt)
    beam.props = BeamProperties(E=beam.props.E, A=beam.props.A * thin,
                                I=beam.props.I * thin ** 3, rho=beam.props.rho,
                                length=beam.props.length,
                                thickness=beam.props.thickness * thin)
    beam._assemble_constant_matrices()
    ref = beam.props.length
    rho_f, uf = fluid.macroscopic(include_force=False)
    free = beam.free
    nf = len(free)
    u0, v0 = beam.u.copy(), beam.v.copy()
    a0 = beam.a.copy()

    def step(y, v):
        beam.u = y.copy()
        beam.v = v.copy()
        solver.ibm.move(beam.markers())
        rho_i, mom_i = solver.ibm.interpolate(rho_f, uf)
        f_lag = solver.ibm.force(rho_i, mom_i, v.reshape(-1, 3)[:, :2], dt=dt)
        load = beam.apply_marker_loads(-f_lag)
        out = beam.newmark_solve(load, dt, tol=tol, max_iter=max_iter)[0]
        return out.copy(), beam.v.copy()

    # base: the state the real loop starts from (predict -> u0, v0 = 0)
    b_u, b_v = step(u0, v0)
    beam.v, beam.a = v0.copy(), a0.copy()
    h = h_rel * max(1.0, float(np.abs(u0).max()))
    hh = 0.5 * h

    def blocks(h):
        Auu = np.zeros((nf, nf)); Auv = np.zeros((nf, nf))
        Avu = np.zeros((nf, nf)); Avv = np.zeros((nf, nf))
        for c, j in enumerate(free):
            y = u0.copy(); y[j] += h
            o, vv = step(y, v0)
            eff = (o[free] - b_u[free]) / h
            Auu[:, c] = eff
            Avu[:, c] = (vv[free] - b_v[free]) / h
            v = v0.copy(); v[j] += h
            o, vv = step(u0, v)
            Auv[:, c] = (o[free] - b_u[free]) / h
            Avv[:, c] = (vv[free] - b_v[free]) / h
        return Auu, Auv, Avu, Avv

    b1 = blocks(h)
    b2 = blocks(hh)
    rich = max(float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-30))
               for x, y in zip(b1, b2))
    return dict(blocks=b1, richardson=rich, free=free, ref=ref,
                r1=b_u[free] - u0[free], u0=u0, v0=v0)


def simulate(d, omega, mode="fixed", tol=1e-8, maxit=200, omega0=None):
    """Run the sub-iteration loop on the *extracted linear map* (no solver)."""
    Auu, Auv, Avu, Avv = d["blocks"]
    ref, nf = d["ref"], len(d["free"])
    y, v = np.zeros(nf), np.zeros(nf)          # perturbations about the base state
    om = omega if omega0 is None else omega0
    ait = AitkenRelaxation(relax_max=1.0, relax0=om)
    r_prev, om_applied, hist = None, float("nan"), []
    for it in range(1, maxit + 1):
        u_new = Auu @ y + Auv @ v
        v_new = Avu @ y + Avv @ v
        r = u_new - y
        res = float(np.linalg.norm(r)) / (ref * np.sqrt(nf))
        if res <= tol:
            return it, res, hist
        if mode == "aitken" and it >= 2:
            om = ait.omega_for(r_prev, r, omega_prev=om_applied)
        om_applied = om
        hist.append(om)
        r_prev = r.copy()
        y = y + om * (u_new - y)
        v = v_new
    return maxit, res, hist


def spectral(d, omega):
    Auu, Auv, Avu, Avv = d["blocks"]
    nf = Auu.shape[0]
    L = np.zeros((2 * nf, 2 * nf))
    L[:nf, :nf] = (1.0 - omega) * np.eye(nf) + omega * Auu
    L[:nf, nf:] = omega * Auv
    L[nf:, :nf] = Avu
    L[nf:, nf:] = Avv
    ev = np.linalg.eigvals(L)
    return float(np.abs(ev).max()), ev


if __name__ == "__main__":
    for tag, kw in [("rho1 dt0.1", dict(rho=1.0, dt=0.1)),
                    ("rho0.5 dt0.1", dict(rho=0.5, dt=0.1)),
                    ("rho1 dt0.1 thin0.3", dict(rho=1.0, dt=0.1, thin=0.3))]:
        d = state_map(**kw)
        print(f"--- {tag}: richardson={d['richardson']:.2e} |r1|norm="
              f"{np.linalg.norm(d['r1']):.3e}")
        for w in (1.0, 0.5, 0.2, 0.05):
            n, res, _ = simulate(d, w, mode="fixed")
            rho, ev = spectral(d, w)
            print(f"    fixed w={w:<5}: n={n:<4} res={res:.1e}  rho(L)={rho:.5f}"
                  f"  n_pred={max(1, int(np.log(1e-8)/np.log(rho))+1) if rho>0 else 'na'}")
        n, res, hist = simulate(d, 1.0, mode="aitken")
        print(f"    aitken: n={n} res={res:.1e} omega_hist={np.round(hist[:4], 4)}")