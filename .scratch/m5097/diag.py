import math, sys, time, json, torch
from tensorfem.marine_panel_ultimate_fe import build_panel_case
from tensorfem.layered_shell4_plasticity import assemble_layered_shell4, LayeredShell4State
from tensorfem.arc_length import ArcLengthProblem, solve_arc_length

beta = float(sys.argv[1]); steps = int(sys.argv[2])
b = 1.0
E, sy = 210e9, 250e6
t = b*math.sqrt(sy/E)/beta
case = build_panel_case(4, length=1.0, width=b, thickness=t, layers=5)
m = case.model
mask = torch.ones(m.n_dofs, dtype=torch.bool); mask[case.fixed_dofs] = False
free = torch.arange(m.n_dofs)[mask]
f = case.reference_load
state = LayeredShell4State.virgin(m)
def it(u_red):
    u = torch.zeros(m.n_dofs, dtype=torch.float64); u[free] = u_red
    internal, K, s, tr = assemble_layered_shell4(m, u, state)
    return internal[free], K[free][:, free]
prob = ArcLengthProblem(it, f[free])
squash = sy*t*b
t0=time.time()
res = solve_arc_length(prob, torch.zeros(len(free), dtype=torch.float64), steps=steps,
                       step_size=0.05, load_scale=1e-9, tolerance=1e-3, max_iterations=10,
                       minimum_step=1e-5)
out={"beta":beta,"t":t,"converged":res.converged,"elapsed":time.time()-t0,
     "path":[[p.load_factor/squash, float(p.displacement[2*((4//2)*(5)+4//2)])] for p in res.points]}
print(json.dumps({k:v for k,v in out.items() if k!="path"}))
print(json.dumps(out["path"]))
