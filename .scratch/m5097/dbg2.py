import torch
from tensorfem.stiffened_panel_ultimate import StiffenedPanel
import tensorfem.stiffened_panel_unit_fe as fe
from tensorfem.layered_shell4_plasticity import element_response, LayeredShell4State
panel = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.020, web_height=0.20,
                       web_thickness=0.010, flange_width=0.08, flange_thickness=0.012)
case = fe.build_panel_unit(panel, nx=4, ny=2, layers=3)
state = case.virgin_state()
model = case.plate_model
conn = model.elements[0]; ids = fe._element_dofs(conn)
rng = torch.Generator().manual_seed(0)
d = torch.randn(24, dtype=torch.float64, generator=rng)*1e-4
force, K, stress, trial = element_response(model, 0, d, state.points[0][0])
print("element consistency:", float(torch.linalg.vector_norm(K@d - force))/float(torch.linalg.vector_norm(force)))
# scale sensitivity
for scale in (1e-6,1e-5,1e-4,1e-3):
    dd = d*scale/1e-4
    f1, K1, _, _ = element_response(model, 0, dd, state.points[0][0])
    print(scale, float(torch.linalg.vector_norm(K1@dd - f1))/float(torch.linalg.vector_norm(f1)))
