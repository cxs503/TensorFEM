"""Sequential steady heat conduction and thermal expansion of a bar."""
import torch
from tensorfem.thermal import ThermalModel, solve_steady_thermal, solve_thermoelastic_bar

D=torch.float64
nodes=torch.linspace(0.,1.,11,dtype=D)[:,None]
elements=torch.stack((torch.arange(10),torch.arange(1,11)),1)
thermal=ThermalModel(nodes,elements,torch.tensor(45.,dtype=D),torch.tensor(7800.,dtype=D),
    torch.tensor(470.,dtype=D),torch.tensor([0,10]),torch.tensor([100.,20.],dtype=D))
temperature=solve_steady_thermal(thermal)
mechanical=solve_thermoelastic_bar(nodes,elements,210e9,.0012,12e-6,temperature,20.,torch.tensor([0]))
print(f"temperature range: {temperature.min():.3f} .. {temperature.max():.3f} K")
print(f"free-end expansion: {mechanical.displacement[-1]:.6e} m")
