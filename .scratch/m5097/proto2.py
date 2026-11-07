import time, torch, math
from tensorfem.stiffened_panel_ultimate import StiffenedPanel
from tensorfem.stiffened_panel_unit_fe import build_panel_unit, assemble_panel_unit
panel = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.020, web_height=0.20,
                       web_thickness=0.010, flange_width=0.08, flange_thickness=0.012)
print(panel.summary())
case = build_panel_unit(panel, nx=4, ny=2, layers=3)
print("dofs", case.n_dofs, "free", len(case.free_dofs()), "elements", case.element_counts,
      "fixed", len(case.fixed_dofs), "refA", case.reference_area)
u = torch.zeros(case.n_dofs, dtype=torch.float64)
t0=time.time(); internal, K, stress, trial = assemble_panel_unit(case, u, case.virgin_state()); print("assemble", time.time()-t0)
print("symmetric", float((K-K.T).abs().max()))
free = case.free_dofs(); f = case.reference_load
lin = torch.linalg.solve(K[free][:,free], f[free])
print("elastic max |du| under unit avg stress", float(lin.abs().max()))
