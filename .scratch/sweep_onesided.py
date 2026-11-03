"""Branch-preserving (one-sided) reference for the condensed plane-stress tangent."""
import sys
sys.path.insert(0, "src")
import torch
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4Model, LayeredShell4State, plane_stress_j2_update)

D = torch.float64
nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], dtype=D)
model = LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]), 200000., .3, .1, 250., 1200.)
state = LayeredShell4State.virgin(model).points[0][0][0]


def one_sided(strain, direction, sign):
    """Forward difference in `direction` that stays inside the current branch."""
    def f(h):
        return plane_stress_j2_update(strain + sign*h*direction, model, state, tangent=False)[0]
    # pick step so the stress perturbation stays below 0.25 MPa (yield margin aware)
    h1, h2, h3 = 1e-6, 5e-7, 2.5e-7
    d1 = (f(h1)-f(0.))/h1
    d2 = (f(h2)-f(0.))/h2
    return 2*d2 - d1  # Richardson


print(f"{'sxx[MPa]':>9} {'branch':>8} {'current-FD dir err':>20} {'one-sided ref':>28}")
for target in (100., 200., 240., 245., 249., 249.9, 250., 251., 260., 310.):
    alpha = max((target-250.)/1200., 0.)
    strain = torch.tensor([target/model.young+alpha,
                           -model.poisson*target/model.young-alpha/2, 0.], dtype=D)
    tan = plane_stress_j2_update(strain, model, state)[1]
    # reference in the +xx and -xx directions (one-sided, branch preserving)
    ex = torch.tensor([1., 0., 0.], dtype=D)
    up = one_sided(strain, ex, +1.)
    down = one_sided(strain, ex, -1.)
    e_plus = float((tan @ ex)[0] - up[0]) / float(up[0])
    e_minus = float((tan @ ex)[0] - down[0]) / float(down[0])
    print(f"{target:9.1f} {'plastic' if alpha>0 else 'elastic':>8} "
          f"{e_plus:20.3e} {e_minus:28.3e}")
    print(f"{'':9} {'ref+':>8} {up.tolist()}  {'ref-':>8} {down.tolist()}  code={ (tan@ex).tolist()}")