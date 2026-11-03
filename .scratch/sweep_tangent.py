"""Sweep: current nested-FD condensed tangent vs Richardson-extrapolated response."""
import sys
sys.path.insert(0, "src")
import torch
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4Model, LayeredShell4State, plane_stress_j2_update)

D = torch.float64
nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], dtype=D)
model = LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]), 200000., .3, .1, 250., 1200.)
state = LayeredShell4State.virgin(model).points[0][0][0]


def col(strain, j, h):
    e = torch.zeros(3, dtype=D); e[j] = 1.
    return (plane_stress_j2_update(strain + h*e, model, state, tangent=False)[0]
            - plane_stress_j2_update(strain - h*e, model, state, tangent=False)[0]) / (2*h)


def richardson(strain, j):
    b = 1e-4
    c1, c2, c3 = col(strain, j, b), col(strain, j, b/2), col(strain, j, b/4)
    return (4*c2 - c1)/3


def ref_tangent(strain):
    return torch.stack([richardson(strain, j) for j in range(3)], 1)


print(f"{'sigma_xx[MPa]':>12} {'yield?':>7} {'tangent rel err':>16} {'sigma_zz resid':>15}")
for target in (100., 200., 245., 249., 250., 251., 255., 260., 280., 310., 400.):
    alpha = max((target-250.)/1200., 0.)
    strain = torch.tensor([target/model.young+alpha,
                           -model.poisson*target/model.young-alpha/2, 0.], dtype=D)
    tan = plane_stress_j2_update(strain, model, state)[1]
    ref = ref_tangent(strain)
    err = float(torch.linalg.matrix_norm(tan-ref)/torch.linalg.matrix_norm(ref))
    # recompute condensed sigma_zz residual via the module's own local Newton
    from tensorfem.plasticity import update_j2
    zz = torch.tensor(-model.poisson/(1-model.poisson)*float(strain[0]+strain[1]), dtype=D)
    best = 1e9
    for _ in range(80):
        T = torch.stack((torch.stack((strain[0], .5*strain[2], torch.zeros((), dtype=D))),
                         torch.stack((.5*strain[2], strain[1], torch.zeros((), dtype=D))),
                         torch.stack((torch.zeros((), dtype=D), torch.zeros((), dtype=D), zz))))
        s = update_j2(T, model.young, model.poisson, model.yield_stress, model.hardening, state)[0]
        r = float(s[2, 2]); best = min(best, abs(r))
        h = 1e-7
        def sz(v):
            U = T.clone(); U[2, 2] = v
            return update_j2(U, model.young, model.poisson, model.yield_stress, model.hardening, state)[0][2, 2]
        d = (sz(zz+h)-sz(zz-h))/(2*h)
        zz = zz - r/d
    print(f"{target:12.1f} {'yes' if alpha>0 else 'no':>7} {err:16.3e} {best:15.3e}")