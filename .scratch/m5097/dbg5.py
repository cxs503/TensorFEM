import torch, math
from tensorfem.marine_panel_ultimate_fe import build_panel_case
from tensorfem.layered_shell4_plasticity import assemble_layered_shell4, LayeredShell4State
t=1.0/58.0
case = build_panel_case(4, length=1.0, width=1.0, thickness=t)
m=case.model
mask = torch.ones(m.n_dofs, dtype=torch.bool); mask[case.fixed_dofs]=False
free = torch.arange(m.n_dofs)[mask]
u = torch.zeros(m.n_dofs, dtype=torch.float64)
internal, K, s, tr = assemble_layered_shell4(m, u, LayeredShell4State.virgin(m))
Kff=K[free][:,free]; f=case.reference_load
pred = torch.linalg.solve(Kff, f[free])
sig = 0.05*m.yield_stress
scale = sig
print("loaded-edge mean u_x:", float(scale*pred[torch.searchsorted(free, torch.tensor([6*n for n in case.loaded_nodes]))].mean()) if hasattr(case,'loaded_nodes') else "n/a")
print("expected delta:", -1.0*sig/m.young)
print("max |pred|*sig:", float((pred.abs().max())*sig))
# per-node u_x on right edge
import numpy as np
right = [n for n in range(len(m.nodes)) if abs(float(m.nodes[n,0])-1.0)<1e-9]
vals=[float(scale*pred[int(torch.nonzero(free==6*n).flatten())]) for n in right]
print("right edge u_x (per node):", vals)
