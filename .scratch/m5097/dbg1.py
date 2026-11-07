import torch
from tensorfem.stiffened_panel_ultimate import StiffenedPanel
import tensorfem.stiffened_panel_unit_fe as fe
panel = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.020, web_height=0.20,
                       web_thickness=0.010, flange_width=0.08, flange_thickness=0.012)
case = fe.build_panel_unit(panel, nx=4, ny=2, layers=3)
free = case.free_dofs(); ref = case.reference_load
state = case.virgin_state()
print("|f| =", float(torch.linalg.vector_norm(ref[free])))
print("sum f_x =", float(ref[0::6].sum()), "refA", case.reference_area)
u = torch.zeros(case.n_dofs, dtype=torch.float64)
internal, K, s, trial = fe.assemble_panel_unit(case, u, state)
print("|int| at u=0:", float(torch.linalg.vector_norm(internal)))
Kff = K[free][:, free]
print("cond est:", float(torch.linalg.cond(Kff)))
pred = torch.linalg.solve(Kff, ref[free])
print("pred max", float(pred.abs().max()))
# elastic check: displacement under 0.1 sigma_y
u2 = torch.zeros(case.n_dofs, dtype=torch.float64); u2[free] = 0.1*235e6*pred
int2, K2, _, _ = fe.assemble_panel_unit(case, u2, state)
res2 = int2 - 0.1*235e6*ref
print("linear consistency residual:", float(torch.linalg.vector_norm(res2[free]))/ (0.1*235e6*float(torch.linalg.vector_norm(ref[free]))))
