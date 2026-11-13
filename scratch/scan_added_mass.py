"""Quick scan: coupled frequency + stability vs added-mass ratio (scratch)."""
import json
import math
import sys

sys.path.insert(0, 'src')
from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    analytic_added_mass, coupled_free_vibration)

HALF = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
BOX = int(sys.argv[2]) if len(sys.argv) > 2 else 160
MODE = sys.argv[3] if len(sys.argv) > 3 else "aitken"
NSTEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 600
TARGET_PERIOD = 120.0

m_a = analytic_added_mass(HALF, 1.0)
out = []
for ratio in (0.15, 0.35, 0.60, 1.00, 1.57):
    m_s = m_a / ratio
    k = (m_s + m_a) * (2.0 * math.pi / TARGET_PERIOD) ** 2
    rec = coupled_free_vibration(
        n_steps=NSTEPS, half_width=HALF, center=(BOX / 2.0, BOX / 2.0),
        lattice=(BOX, BOX), mass=m_s, stiffness=k, y0=0.2, rho_f=1.0,
        relax_mode=MODE, fixed_omega=1.0, sub_tol=1e-6, max_sub_iter=25)
    out.append({k2: rec[k2] for k2 in (
        'relax_mode', 'added_mass_ratio', 'mass', 'stiffness', 'period',
        'period_fft', 'frequency', 'frequency_ref', 'frequency_ratio',
        'frequency_ratio_ref', 'frequency_error_pct',
        'added_mass_measured', 'steps_completed', 'diverged',
        'diverged_reason', 'mean_sub_iterations', 'max_sub_iterations',
        'maxit_hits', 'converged_all')})
    print(json.dumps(out[-1]), flush=True)
print(json.dumps({"half_width": HALF, "box": BOX, "mode": MODE,
                  "cases": out}))