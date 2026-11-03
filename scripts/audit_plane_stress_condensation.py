"""Evidence figures for the plane-stress sigma_zz condensation round-off fix.

Left : |sigma_zz| history of the local Newton loop at the stalled arc-length
       trial strain, replayed twice: the historical absolute acceptance
       thresholds (dashed, never reached) and the round-off-scaled ones.
Right: plane-stress stress path of the material point against an independent
       bisection root of sigma_zz(eps_zz) = 0.
"""
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from tensorfem.layered_shell4_plasticity import (LayeredShell4Model,
                                                 LayeredShell4State,
                                                 _tensor_from_inplane,
                                                 plane_stress_j2_update)
from tensorfem.plasticity import update_j2, j2_consistent_tangent

D = torch.float64
STRAIN = [3.368958579694938, 10.018737002848292, 1.115051018219515]
OUT = pathlib.Path('docs/assets/plane_stress_condensation')
OUT.mkdir(parents=True, exist_ok=True)

nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], dtype=D)
model = LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]), 200000., .3, .08,
                           250., 1400., layers=3)
committed = LayeredShell4State.virgin(model).points[0][0][0]
value = torch.tensor(STRAIN, dtype=D)
scale = max(model.young, model.yield_stress, 1.)
eps64 = torch.finfo(D).eps


def history(accept_scaled: bool, iterations=40):
    zz = -model.poisson/(1-model.poisson)*(value[0]+value[1])
    residual, zz_values = [], []
    for _ in range(iterations):
        full = _tensor_from_inplane(value, zz)
        sigma, trial, dgamma = update_j2(full, model.young, model.poisson,
                                         model.yield_stress, model.hardening, committed)
        material = j2_consistent_tangent(full, model.young, model.poisson,
                                         model.hardening, committed, dgamma)
        r = float(sigma[2, 2])
        residual.append(abs(r)); zz_values.append(float(zz))
        if accept_scaled:
            if abs(r) <= 256.*eps64*max(scale, float(sigma.abs().max())):
                break
        elif abs(r) <= 1e-15*scale:
            break
        zz = zz - r/float(material[2, 2])
    return residual, zz_values


old_residual, _ = history(False)
new_residual, _ = history(True)

# Independent bisection root of sigma_zz = 0 and the plane-stress path.
def sigma_zz(zz):
    full = _tensor_from_inplane(value, torch.as_tensor(float(zz), dtype=D))
    sigma, _, _ = update_j2(full, model.young, model.poisson,
                            model.yield_stress, model.hardening, committed)
    return float(sigma[2, 2])


centre = float(-model.poisson/(1-model.poisson)*(value[0]+value[1]))
lo = hi = centre
f_lo = f_hi = sigma_zz(centre)
step = max(1., .5*abs(centre))
for _ in range(200):
    if not f_lo < 0.:
        lo -= step; f_lo = sigma_zz(lo)
    if not f_hi > 0.:
        hi += step; f_hi = sigma_zz(hi)
    if f_lo < 0. < f_hi:
        break
    step *= 2.
for _ in range(200):
    mid = .5*(lo+hi)
    if sigma_zz(mid) < 0.:
        lo = mid
    else:
        hi = mid
root = .5*(lo+hi)

def bisect_root(strain):
    centre = float(-model.poisson/(1-model.poisson)*(strain[0]+strain[1]))
    lo = hi = centre
    f_lo = f_hi = sigma_zz_at(strain, centre)
    step = max(1., .5*abs(centre))
    for _ in range(120):
        if not f_lo < 0.:
            lo -= step; f_lo = sigma_zz_at(strain, lo)
        if not f_hi > 0.:
            hi += step; f_hi = sigma_zz_at(strain, hi)
        if f_lo < 0. < f_hi:
            break
        step *= 2.
    for _ in range(120):
        mid = .5*(lo+hi)
        if sigma_zz_at(strain, mid) < 0.:
            lo = mid
        else:
            hi = mid
    return .5*(lo+hi)


def sigma_zz_at(strain, zz):
    full = _tensor_from_inplane(strain, torch.as_tensor(float(zz), dtype=D))
    sigma, _, _ = update_j2(full, model.young, model.poisson,
                            model.yield_stress, model.hardening, committed)
    return float(sigma[2, 2])


path, reference_path = [], []
for i in range(41):
    strain = torch.tensor([0.002*i/40., -0.004*i/40., 0.], dtype=D)
    stress, _, _ = plane_stress_j2_update(strain, model, committed)
    path.append((float(strain[1]), float(stress[1])))
    root_i = bisect_root(strain)
    full = _tensor_from_inplane(strain, torch.as_tensor(root_i, dtype=D))
    sigma, _, _ = update_j2(full, model.young, model.poisson,
                            model.yield_stress, model.hardening, committed)
    reference_path.append((float(strain[1]), float(sigma[1, 1])))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].semilogy(range(1, len(old_residual)+1), old_residual, 'o-', ms=3,
                 label='historical bound $10^{-15}E$ (stalls)')
axes[0].semilogy(range(1, len(new_residual)+1), new_residual, 's-', ms=3,
                 label='round-off scaled bound')
axes[0].axhline(1e-15*scale, ls='--', lw=1, color='k')
axes[0].axhline(256.*eps64*max(scale, 2e4), ls=':', lw=1, color='tab:red')
axes[0].set_xlabel('local Newton iteration')
axes[0].set_ylabel(r'$|\sigma_{zz}|$ [MPa]')
axes[0].set_title(r'$\varepsilon=(3.369,\,10.019,\,1.115)$, $\Delta\gamma=13.8$')
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

axes[1].plot([p[0] for p in path], [p[1] for p in path], 'o', ms=2.5,
             label='condensed plane-stress point (fixed code)')
axes[1].plot([p[0] for p in reference_path], [p[1] for p in reference_path], '-',
             lw=1, label='independent bisection root')
axes[1].set_xlabel(r'$\varepsilon_y$'); axes[1].set_ylabel(r'$\sigma_y$ [MPa]')
axes[1].set_title('plane-stress path, $\\sigma_y=250$, $H=1400$')
axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)

fig.tight_layout()
fig.savefig(OUT/'condensation_roundoff.png', dpi=140)

evidence = {
    'benchmark': 'plane_stress_sigma_zz_condensation_roundoff',
    'stalled_strain': STRAIN,
    'historical_bound_sigma_zz': 1e-15*scale,
    'new_bound_sigma_zz': 256.*eps64*max(scale, 2e4),
    'old_residual_history': old_residual,
    'new_residual_history': new_residual,
    'old_final_residual': old_residual[-1],
    'new_final_residual': new_residual[-1],
    'old_iterations': len(old_residual),
    'new_iterations': len(new_residual),
    'bisection_root_eps_zz': root,
    'bisection_residual': sigma_zz(root),
    'path_max_abs_difference': max(abs(a[1]-b[1]) for a, b in zip(path, reference_path)),
}
(OUT/'evidence.json').write_text(json.dumps(evidence, indent=2))
print(json.dumps({k: v for k, v in evidence.items()
                  if not isinstance(v, list)}, indent=2))