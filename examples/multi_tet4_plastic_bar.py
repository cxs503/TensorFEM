"""Twelve-TET4 small-strain J2 load/unload demonstration."""
import torch
from tensorfem.global_plasticity import solve_tet4_j2_path
from tensorfem.solid_plasticity import Tet4J2Model

# Import-free compact construction matching the documented unit-bar benchmark.
nodes=torch.tensor([(x,y,z) for x in (0.,.5,1.) for z in (0.,1.) for y in (0.,1.)],dtype=torch.float64)
pat=((0,6,4,7),(0,2,6,7),(0,3,2,7),(0,1,3,7),(0,5,1,7),(0,4,5,7)); elements=[]
for s in (0,1):
    mp=tuple(4*s+i for i in range(8)); elements += [tuple(mp[i] for i in t) for t in pat]
fixed=[]
for i,(x,y,z) in enumerate(nodes.tolist()):
    if x==0: fixed.append(3*i)
    if y==0: fixed.append(3*i+1)
    if z==0: fixed.append(3*i+2)
model=Tet4J2Model(nodes,torch.tensor(elements),200000.,.3,250.,10000.,torch.tensor(sorted(set(fixed))))
force=torch.zeros(model.n_dofs,dtype=torch.float64); tip=torch.nonzero(nodes[:,0]==1.)[:,0]
force[3*tip]=400.*torch.tensor([1/3,1/6,1/6,1/3],dtype=torch.float64)
for result in solve_tet4_j2_path(model,force,[.625,1.,0.]):
    print(result.load_factor,float(result.displacement[3*tip].mean()),float(result.stress[:,0].mean()))
