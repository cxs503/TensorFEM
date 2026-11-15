"""Exploratory: residual sequence + interface-gain spectrum of the 48-DOF beam solver."""
import math
import numpy as np

from tensorfem.ibm_aitken_implicit import (
    BeamProperties, CorotationalBeam2D, ImplicitIBMFSI, IBMInterface, LBMFluid2D,
)

BOX = 64
DT = 1.0
AMP = 0.05


def build(n_elements=16, amp=AMP, fluid_rho=1.0, box=BOX):
    beam = CorotationalBeam2D(n_elements=n_elements, props=BeamProperties(
        E=1.0e5, A=1.0, I=1.0 / 12.0, rho=1.0, length=1.0, thickness=1.0))
    beam.set_initial_deflection(amp)
    scale = 0.45 * float(box)
    beam.x0 = beam.x0 * scale + np.array([0.15 * box, 0.5 * box])
    beam.props = BeamProperties(E=beam.props.E / scale, A=beam.props.A * scale,
                                I=beam.props.I * scale ** 3, rho=beam.props.rho,
                                length=beam.props.length * scale,
                                thickness=beam.props.thickness * scale)
    beam._assemble_constant_matrices()
    beam.set_initial_deflection(amp * scale)
    fluid = LBMFluid2D((box, box), omega=1.8, rho0=fluid_rho)
    return beam, fluid


def residual_sequence(beam, fluid, nmax=8, n_warm=0, mode="aitken", omega=1.0):
    seq = []
    for m in range(1, nmax + 1):
        b, f = build(beam.n_elements)
        solver = ImplicitIBMFSI(b, f, dt=DT, ref_length=b.props.length,
                                sub_tol=1e-8, max_sub_iter=40,
                                relax_mode="aitken")
        if n_warm:
            solver.run(n_warm)
        solver.relax_mode = mode
        solver.fixed_omega = omega
        solver.sub_tol = 0.0
        solver.max_sub_iter = m
        solver.step(1)
        seq.append(float(solver.residuals[-1]) if hasattr(solver, "residuals")
                   else float("nan"))
    return seq


def gain_matrix(n_elements=16, h_rel=1e-6):
    """Static dG/du of the interface map about the released state (velocity frozen)."""
    beam, fluid = build(n_elements)
    solver = ImplicitIBMFSI(beam, fluid, dt=DT, ref_length=beam.props.length)
    rho, u = fluid.macroscopic(include_force=False)
    free = beam.free
    u0 = beam.u.copy()
    v0 = beam.v.copy()
    a0 = beam.a.copy()

    def G(y):
        beam.u = y.copy()
        solver.ibm.move(beam.markers())
        rho_i, mom_i = solver.ibm.interpolate(rho, u)
        f_lag = solver.ibm.force(rho_i, mom_i, v0.reshape(-1, 3)[:, :2].copy(), dt=DT)
        load = beam.apply_marker_loads(-f_lag)
        out, _ok, _it, _res = beam.newmark_solve(load, DT)
        return out.copy()

    base = G(u0)
    beam.v = v0.copy()
    beam.a = a0.copy()
    h = h_rel * max(1.0, float(np.abs(u0).max()))
    A = np.zeros((beam.n_dof, beam.n_dof))
    for j in free:
        y = u0.copy()
        y[j] += h
        col = G(y)
        beam.v = v0.copy()
        beam.a = a0.copy()
        A[:, j] = (col - base) / h
    bfs = A[np.ix_(free, free)]
    M = bfs - np.eye(len(free))
    ev = np.linalg.eigvals(M)
    g = ev + 1.0
    return bfs, M, ev, g, base, u0


if __name__ == "__main__":
    beam, fluid = build(16)
    print("free dofs:", len(beam.free), "n_dof:", beam.n_dof)
    print("u0 max:", float(np.abs(beam.u).max()))
    seq = residual_sequence(beam, fluid, nmax=8)
    print("aitken residual norms:", ["%.3e" % s for s in seq])
    if all(s > 0 for s in seq):
        print("ratios:", ["%.4f" % (seq[i + 1] / seq[i]) for i in range(len(seq) - 1)])
    bfs, M, ev, g, base, u0 = gain_matrix(16)
    print("g spectrum (real,imag), sorted by |g|:")
    order = np.argsort(-np.abs(g))
    for i in order[:8]:
        print("   |g|=%.5f  g=%.5f%+.5fj   R=-g=%.5f" % (
            abs(g[i]), g[i].real, g[i].imag, -g[i].real))
    absg = np.abs(g)
    print("|g| min/max over all 48:", absg.min(), absg.max(),
          " spread ratio:", absg.max() / absg.min())
    print("top-5 |g| / dominant:", (absg[order] / absg[order][0])[:5])
    print("rank(A-I) numerical:", np.linalg.matrix_rank(M, tol=1e-8 * np.abs(M).max()))
    sv = np.linalg.svd(M, compute_uv=False)
    print("singular values of M (top 8):", np.round(sv[:8], 5))