import torch
from tensorfem.stiffened_panel_ultimate import StiffenedPanel
import tensorfem.stiffened_panel_unit_fe as fe
panel = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.020, web_height=0.20,
                       web_thickness=0.010, flange_width=0.08, flange_thickness=0.012)
case = fe.build_panel_unit(panel, nx=4, ny=2, layers=3)
free = case.free_dofs(); ref = case.reference_load; scale = float(torch.linalg.vector_norm(ref[free]))
state = case.virgin_state()
u = torch.zeros(case.n_dofs, dtype=torch.float64); load=0.0
ds = 4e-4; weight = ds/(0.05*235e6)
internal, K, _, _ = fe.assemble_panel_unit(case, u, state)
Kff = K[free][:,free]
pred = torch.linalg.solve(Kff, ref[free])
direction = torch.cat((pred, pred.new_tensor([1.0]))); direction = direction/torch.linalg.vector_norm(direction)
print("direction last", float(direction[-1]), "translation part", float(direction[:-1].abs().max()))
trial_u = u.clone(); trial_u[free] += ds*direction[:-1]
trial_load = ds*float(direction[-1])/weight
print("trial_load/235e6 =", trial_load/235e6)
for it in range(1, 13):
    internal, tangent, _, trial = fe.assemble_panel_unit(case, trial_u, state)
    residual = internal - trial_load*ref
    rn = float(torch.linalg.vector_norm(residual[free]))/scale
    du = trial_u - u; dl = trial_load - 0.0
    constraint = float(torch.dot(du[free], du[free])) + (weight*dl)**2 - ds**2
    print(it, "res", rn, "cons", constraint)
    matrix = torch.zeros((len(free)+1,)*2, dtype=torch.float64)
    matrix[:-1,:-1] = tangent[free][:,free]; matrix[:-1,-1] = -ref[free]
    matrix[-1,:-1] = 2*du[free]; matrix[-1,-1] = 2*weight**2*dl
    rhs = -torch.cat((residual[free], residual.new_tensor([constraint])))
    corr = torch.linalg.solve(matrix, rhs)
    trial_u[free] += corr[:-1]; trial_load += float(corr[-1])
    print("   corr du max", float(corr[:-1].abs().max()), "dlam", float(corr[-1])/235e6)
