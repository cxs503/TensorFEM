"""Exploratory 3: find a strongly coupled, multi-mode, Aitken-stable beam case."""
import numpy as np
from tensorfem.ibm_aitken_implicit import ImplicitIBMFSI
from explore2 import build


def probe(tag, n_elements, amp, frho, lat, dt, mode="aitken", omega=1.0, steps=6):
    beam, fluid, solver = build(n_elements=n_elements, amp=amp, fluid_rho=frho,
                                lattice=lat, dt=dt)
    solver.relax_mode = mode
    solver.fixed_omega = omega
    try:
        rec = solver.step(1)
        its = [rec["sub_iterations"]]
        hist = solver.run(steps - 1, t_start=2)
        its += [h["sub_iterations"] for h in hist]
        print(f"{tag}: ok its={its} first_res={rec['residual']:.2e} "
              f"om_last={hist[-1]['omega']:.3f} n_fail={solver.n_maxit_hits}")
    except Exception as e:
        print(f"{tag}: DIVERGED {e} (n_maxit_hits={solver.n_maxit_hits})")


if __name__ == "__main__":
    cases = [
        ("rho0.5 dt1e-2 a0.05", 10, 0.05, 0.5, (160, 120), 1.0e-2),
        ("rho1.0 dt1e-2 a0.05", 10, 0.05, 1.0, (160, 120), 1.0e-2),
        ("rho1.0 dt5e-3 a0.05", 10, 0.05, 1.0, (160, 120), 5.0e-3),
        ("rho1.0 dt5e-3 short", 10, 0.05, 1.0, (80, 60), 5.0e-3),
        ("rho1.0 dt5e-3 shorter", 8, 0.05, 1.0, (48, 48), 5.0e-3),
        ("rho2.0 dt5e-3 shorter", 8, 0.05, 2.0, (48, 48), 5.0e-3),
    ]
    for tag, n, a, r, lat, dt in cases:
        probe(tag, n, a, r, lat, dt)
    for tag, n, a, r, lat, dt in cases[-3:]:
        probe(tag + " [fixed 0.3]", n, a, r, lat, dt, mode="fixed", omega=0.3)