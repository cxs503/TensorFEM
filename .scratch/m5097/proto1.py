import math, time, torch
from tensorfem.marine_panel_ultimate_fe import build_panel_case
from tensorfem.layered_shell4_plasticity import assemble_layered_shell4, LayeredShell4State
from tensorfem.arc_length import ArcLengthProblem, solve_arc_length

t = 1.0/58.0
case = build_panel_case(4, length=1.0, width=1.0, thickness=t)
m = case.model
fixed = case.fixed_dofs
mask = torch.ones(m.n_dofs, dtype=torch.bool); mask[fixed] = False
free = torch.arange(m.n_dofs)[mask]
f = case.reference_load
print("dofs", m.n_dofs, "free", len(free), "elems", len(m.elements))

state = LayeredShell4State.virgin(m)
holder = {"trial": state}
def internal_tangent(u_red):
    u = torch.zeros(m.n_dofs, dtype=torch.float64)
    u[free] = u_red
    internal, K, stress, trial = assemble_layered_shell4(m, u, state)
    holder["trial"] = trial
    return internal[free], K[free][:, free]

load_scale = 1e-9
prob = ArcLengthProblem(internal_tangent, f[free])
t0 = time.time()
res = solve_arc_length(prob, torch.zeros(len(free), dtype=torch.float64), steps=8, step_size=0.1,
                       load_scale=load_scale, tolerance=1e-3, max_iterations=8,
                       minimum_step=1e-4, blame=None) if False else solve_arc_length(
    prob, torch.zeros(len(free), dtype=torch.float64), steps=8, step_size=0.1,
    load_scale=load_scale, tolerance=1e-3, max_iterations=8, minimum_step=1e-4)
print("elapsed", time.time()-t0, "converged", res.converged, "points", len(res.points))
squash = m.yield_stress*t*1.0
for p in res.points:
    print(round(p.load_factor/squash,4), float(p.displacement[torch.nonzero(free==case.probe if hasattr(case,'probe') else free[:1]).flatten()]))
