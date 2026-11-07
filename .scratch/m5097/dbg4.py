import torch
from tensorfem.stiffened_panel_ultimate import StiffenedPanel
import tensorfem.stiffened_panel_unit_fe as fe
panel = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.020, web_height=0.20,
                       web_thickness=0.010, flange_width=0.08, flange_thickness=0.012)
case = fe.build_panel_unit(panel, nx=4, ny=2, layers=3)
free = case.free_dofs(); ref = case.reference_load
u = torch.zeros(case.n_dofs, dtype=torch.float64)
internal, K, _, _ = fe.assemble_panel_unit(case, u, case.virgin_state())
Kff = K[free][:,free]
pred = torch.linalg.solve(Kff, ref[free])
idx = int(torch.argmax(pred.abs()))
dof = int(free[idx]); node, comp = dof//6, dof%6
print("max pred", float(pred[idx]), "dof", dof, "node", node, "comp", comp, "coord", case.nodes[node].tolist())
print("top5:", [(int(free[int(i)]), float(pred[int(i)])) for i in torch.topk(pred.abs(), 5).indices])
eig = torch.linalg.eigvalsh(Kff)
print("min eig", float(eig[0]), "max eig", float(eig[-1]), "ratio", float(eig[-1]/eig[0]))
# axial response check: uniform compression 11.75 MPa -> strain 5.7e-5, u_x = 2.4*5.7e-5 =1.37e-4
sig = 0.05*235e6
u_sol = torch.zeros(case.n_dofs, dtype=torch.float64); u_sol[free] = sig*pred
loaded_mean = float(u_sol[case.loaded_edge_dofs].mean())
print("mean u_x at loaded edge for 11.75 MPa:", loaded_mean, "expected ~", -2.4*sig/2.06e11)
print("probe w:", float(u_sol[case.probe_dof]))
