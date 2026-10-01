"""Q4 cantilever verification example; run with PYTHONPATH=src."""
import torch
from tensorfem.continuum import ContinuumModel, rectangular_q4_mesh, solve_continuum

L, H, T, E, P = 10.0, 1.0, 0.2, 200e9, -1000.0
nodes, elements = rectangular_q4_mesh(L, H, 40, 8)
forces = torch.zeros(2*len(nodes), dtype=torch.float64)
right = torch.where(nodes[:,0] == L)[0]
forces[2*right+1] = P/8
forces[2*right[[0,-1]]+1] *= 0.5
left = torch.where(nodes[:,0] == 0)[0]
fixed = torch.stack((2*left,2*left+1),1).reshape(-1)
result = solve_continuum(ContinuumModel(nodes,elements,torch.tensor(E),torch.tensor(0.3),torch.tensor(T),forces,fixed))
tip = result.displacement[2*right[len(right)//2]+1].item()
reference = P*L**3/(3*E*(T*H**3/12))
print({"tip": tip, "reference": reference, "relative_error": abs(tip/reference-1)})
