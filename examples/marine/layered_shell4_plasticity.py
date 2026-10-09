"""Two-element Shell4 membrane yielding and unload demonstration."""
import torch

from tensorfem.layered_shell4_plasticity import (
    LayeredShell4Model, solve_layered_shell4_path,
)


nodes = torch.tensor([[0.,0.,0.], [1.,0.,0.], [2.,0.,0.],
                      [0.,1.,0.], [1.,1.,0.], [2.,1.,0.]], dtype=torch.float64)
model = LayeredShell4Model(nodes, torch.tensor([[0,1,4,3], [1,2,5,4]]),
                           young=200000., poisson=.3, thickness=.1,
                           yield_stress=250., hardening=1500., layers=5)
fixed = []
for node, xyz in enumerate(nodes):
    fixed.extend((6*node+2, 6*node+3, 6*node+4, 6*node+5))
    if float(xyz[0]) == 0.:
        fixed.extend((6*node, 6*node+1))
fixed = torch.tensor(sorted(set(fixed)))
load = torch.zeros(model.n_dofs, dtype=torch.float64)
load[12] = load[30] = 18.
path = solve_layered_shell4_path(model, load, fixed, [.25, .5, .75, 1., .5, 0.])
for step in path:
    print(f"factor={step.load_factor:4.2f}  tip={step.displacement[12]:.8f}  "
          f"max_alpha={float(step.alpha.max()):.8f}  residual={step.residual_norm:.3e}")
