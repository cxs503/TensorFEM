"""Exploratory 2: stability + sub-iteration cost + gain spectrum of the flexural beam solver."""
import math
import numpy as np

from tensorfem.ibm_aitken_implicit import (
    BeamProperties, CorotationalBeam2D, ImplicitIBMFSI, LBMFluid2D,
)


def build(n_elements=10, amp=0.01, fluid_rho=0.2, lattice=(160, 120), dt=5.0e-3,
          relax_max=1.0):
    beam = CorotationalBeam2D(n_elements=n_elements, props=BeamProperties(
        E=1.0e5, A=1.0, I=1.0 / 12.0, rho=1.0, length=1.0, thickness=1.0))
    beam.set_initial_deflection(amp)
    scale = 0.45 * float(lattice[0])
    beam.x0 = beam.x0 * scale + np.array([0.15 * lattice[0], 0.5 * lattice[1]])
    beam.props = BeamProperties(E=beam.props.E / scale, A=beam.props.A * scale,
                                I=beam.props.I * scale ** 3, rho=beam.props.rho,
                                length=beam.props.length * scale,
                                thickness=beam.props.thickness * scale)
    beam._assemble_constant_matrices()
    beam.set_initial_deflection(amp * scale)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=fluid_rho)
    solver = ImplicitIBMFSI(beam, fluid, dt=dt, dx=1.0, ref_length=beam.props.length,
                            sub_tol=1e-8, max_sub_iter=40, relax_max=relax_max)
    return beam, fluid, solver


def residuals_for_m(m=1, n_warm=0, **kw):
    """Residual of the m-th sub-iteration, via the public max_sub_iter switch."""
    beam, fluid, solver = build(**kw)
    solver.sub_tol = 1e-8
    solver.max_sub_iter = 40
    if n_warm:
        solver.run(n_warm)
    solver.sub_tol = 0.0
    solver.max_sub_iter = m
    rec = solver.step(n_warm + 1)
    return rec["residual"], rec


if __name__ == "__main__":
    for n_elem, amp, frho, lat, dt in [
            (10, 0.01, 0.2, (160, 120), 5.0e-3),
            (10, 0.005, 0.1, (160, 120), 5.0e-3),
            (6, 0.01, 0.2, (160, 120), 5.0e-3),
    ]:
        beam, fluid, solver = build(n_elements=n_elem, amp=amp, fluid_rho=frho,
                                    lattice=lat, dt=dt)
        try:
            rec = solver.step(1)
            print(f"n={n_elem} amp={amp} frho={frho}: it={rec['sub_iterations']} "
                  f"res={rec['residual']:.2e} om={rec['omega']:.4f} "
                  f"tip={rec['tip'][1]:.4f}")
            hist = solver.run(8, t_start=2)
            print("   steps:", [h["sub_iterations"] for h in hist],
                  "res:", ["%.1e" % h["residual"] for h in hist])
        except Exception as e:
            print(f"n={n_elem} amp={amp}: ERR {e}")
    # residual sequence at m = 1..8 for the first config
    seq = []
    for m in range(1, 9):
        r, _ = residuals_for_m(m=m, n_elements=10, amp=0.01, fluid_rho=0.2)
        seq.append(r)
    print("seq:", ["%.3e" % s for s in seq])
    print("ratios:", ["%.4f" % (seq[i + 1] / seq[i]) for i in range(len(seq) - 1)])