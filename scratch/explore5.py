"""Exploratory 5: locate a stable Aitken-stable beam case with a non-negligible first residual."""
import numpy as np
from tensorfem.ibm_aitken_implicit import ImplicitIBMFSI, BeamProperties
from explore2 import build


def case(rho, dt, thin, amp=0.05, n_elements=10, lat=(64, 64), mode="aitken",
         omega=1.0, maxit=200, tol=1e-8):
    beam, fluid, solver = build(n_elements=n_elements, amp=amp, fluid_rho=rho,
                                lattice=lat, dt=dt)
    beam.props = BeamProperties(E=beam.props.E, A=beam.props.A * thin,
                                I=beam.props.I * thin ** 3, rho=beam.props.rho,
                                length=beam.props.length,
                                thickness=beam.props.thickness * thin)
    beam._assemble_constant_matrices()
    solver.relax_mode = mode
    solver.fixed_omega = omega
    solver.sub_tol = tol
    solver.max_sub_iter = maxit
    try:
        rec = solver.step(1)
        return ("ok", rec["sub_iterations"], rec["residual"], rec["omega"],
                float(np.abs(beam.u).max()))
    except Exception as e:
        return ("ERR", solver.n_maxit_hits, float("nan"), float("nan"),
                float(np.abs(beam.u).max()))


if __name__ == "__main__":
    print("%-22s %-5s %-4s %-9s %-7s %-9s" % ("config", "stat", "its", "res", "omega", "umax"))
    for thin in (1.0, 0.3):
        for rho in (0.5, 1.0):
            for dt in (0.03, 0.1, 0.3):
                s, its, res, om, um = case(rho, dt, thin)
                print("%-22s %-5s %-4s %-9.2e %-7.3f %-9.3f" % (
                    f"rho{rho} dt{dt} t{thin}", s, its, res, om, um))
    print("\nfixed omega at the strongest stable candidates:")
    for thin in (1.0, 0.3):
        for w in (0.5, 0.2, 0.05):
            s, its, res, om, um = case(1.0, 0.1, thin, mode="fixed", omega=w)
            print("%-22s %-5s %-4s %-9.2e %-7.3f %-9.3f" % (
                f"rho1 dt0.1 t{thin} w{w}", s, its, res, om, um))