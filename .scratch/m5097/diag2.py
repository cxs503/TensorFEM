import math, sys, time, json, torch
from tensorfem.marine_panel_ultimate_fe import build_panel_case
from tensorfem.layered_shell4_plasticity import assemble_layered_shell4, LayeredShell4State
from tensorfem.arc_length import ArcLengthProblem, solve_arc_length

beta = float(sys.argv[1]); steps = int(sys.argv[2]); layers=int(sys.argv[3])
b = 1.0
E, sy = 210e9, 250e6
t = b*math.sqrt(sy/E)/beta
case = build_panel_case(4, length=1.0, width=b, thickness=t, layers=layers)
m = case.model
mask = torch.ones(m.n_dofs, dtype=torch.bool); mask[case.fixed_dofs] = False
free = torch.arange(m.n_dofs)[mask]
f = case.reference_load
squash = sy*t*b
state = LayeredShell4State.virgin(m)
def it(u_red):
    u = torch.zeros(m.n_dofs, dtype=torch.float64); u[free] = u_red
    internal, K, s, tr = assemble_layered_shell4(m, u, state)
    return internal[free], K[free][:, free]
prob = ArcLengthProblem(it, f[free])
u_char = t/100.0
load_scale = u_char/squash
t0=time.time()
ds = float(sys.argv[4])
res = solve_arc_length(prob, torch.zeros(len(free), dtype=torch.float64), steps=steps,
                       step_size=ds, load_scale=load_scale, tolerance=1e-3, max_iterations=10,
                       minimum_step=ds/100, maximum_step=4*ds)
out={"beta":beta,"t":t,"converged":res.converged,"elapsed":time.time()-t0,"load_scale":load_scale,
     "faulkner": min(1.0, 2/beta-1/beta**2) if beta>1 else 1.0,
     "path":[[round(p.load_factor/squash,4), round(float(p.displacement[2*((4//2)*(5)+4//2)]),6), p.iterations] for p in res.points]}
print(json.dumps({k:v for k,v in out.items() if k!="path"}))
print(json.dumps(out["path"]))
