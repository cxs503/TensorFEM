import torch
from tensorfem.stiffened_panel_ultimate import StiffenedPanel
import tensorfem.stiffened_panel_unit_fe as fe
EL = StiffenedPanel(span=0.8, spacing=0.8, plate_thickness=0.012, web_height=0.18,
                    web_thickness=0.009, flange_width=0.08, flange_thickness=0.012, yield_stress=1e12)
case = fe.build_panel_unit(EL, nx=2, ny=1, layers=1)
free = case.free_dofs(); ref = case.reference_load; scale=float(torch.linalg.vector_norm(ref[free]))
state = case.virgin_state()
u = torch.zeros(case.n_dofs, dtype=torch.float64); load=0.0
ds=1e-4; weight=ds/(0.02*2.35e8)
internal, K, _, _ = fe.assemble_panel_unit(case, u, state)
pred = torch.linalg.solve(K[free][:,free], ref[free])
direction = torch.cat((pred, pred.new_tensor([1.0]))); direction/=torch.linalg.vector_norm(direction)
print("direction last", float(direction[-1]), "scale", scale, "dl", ds*float(direction[-1])/weight)
trial_u=u.clone(); trial_u[free]+=ds*direction[:-1]; trial_load=ds*float(direction[-1])/weight
for it in range(1,11):
    internal, tangent, _, trial = fe.assemble_panel_unit(case, trial_u, state)
    res = internal - trial_load*ref
    rn = float(torch.linalg.vector_norm(res[free]))/scale
    du=trial_u-u; dl=trial_load
    cons = float(torch.dot(du[free],du[free])) + (weight*dl)**2 - ds**2
    print(it, "rn", rn, "cons", cons, "tol*ds", 1e-6*ds)
    M = torch.zeros((len(free)+1,)*2, dtype=torch.float64)
    M[:-1,:-1]=tangent[free][:,free]; M[:-1,-1]=-ref[free]; M[-1,:-1]=2*du[free]; M[-1,-1]=2*weight**2*dl
    rhs=-torch.cat((res[free], res.new_tensor([cons])))
    try: corr=torch.linalg.solve(M,rhs)
    except Exception as e: print("solve fail", e); break
    trial_u[free]+=corr[:-1]; trial_load+=float(corr[-1])
