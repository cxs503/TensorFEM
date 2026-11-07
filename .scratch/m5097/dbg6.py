import torch
from tensorfem.stiffened_panel_ultimate import StiffenedPanel
import tensorfem.stiffened_panel_unit_fe as fe
panel = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.020, web_height=0.20,
                       web_thickness=0.010, flange_width=0.08, flange_thickness=0.012)
case = fe.build_panel_unit(panel, nx=4, ny=2, layers=3)
free = case.free_dofs(); ref = case.reference_load
u = torch.zeros(case.n_dofs, dtype=torch.float64)
internal, K, _, _ = fe.assemble_panel_unit(case, u, case.virgin_state())
pred = torch.linalg.solve(K[free][:,free], ref[free])
sig = 0.05*panel.yield_stress
u_sol = torch.zeros(case.n_dofs, dtype=torch.float64); u_sol[free] = sig*pred
print("mean u_x loaded edge:", float(u_sol[case.loaded_edge_dofs].mean()), "expected:", -2.4*sig/2.06e11)
print("|ref| (N per Pa):", float(torch.linalg.vector_norm(ref)), "sum:", float(ref[0::6].sum()), "half area:", case.reference_area)
print("max pred:", float(pred.abs().max()), "at dof", int(free[int(torch.argmax(pred.abs()))]))
