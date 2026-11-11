"""Independent reference check: high-order scikit-fem convergence toward the published 23.96."""
import sys, pathlib
sys.path.insert(0, '/root/tp-agent/work/m15387/benchmarks')
from cook_membrane_xval import _skfem_run
REF = 23.96
for n in (16, 24, 32, 48, 64):
    r = _skfem_run(n, "quad")
    print(f"quad2 n={n:3d} ndof={r['ndof']:6d} uy_mid={r['uy_mid']:.6f} err={abs(r['uy_mid']-REF)/REF*100:.4f}%  energy={r['energy']:.6f}")
