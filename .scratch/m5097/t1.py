import time, torch, math
from tensorfem.marine_panel_ultimate_fe import build_panel_case
from tensorfem.layered_shell4_plasticity import assemble_layered_shell4, LayeredShell4State
case = build_panel_case(4)
m = case.model
print("dofs", m.n_dofs, "elems", len(m.elements))
u = torch.zeros(m.n_dofs, dtype=torch.float64)
st = LayeredShell4State.virgin(m)
t0 = time.time()
for _ in range(5):
    internal, K, stress, trial = assemble_layered_shell4(m, u, st)
print("assemble time per call", (time.time()-t0)/5)
