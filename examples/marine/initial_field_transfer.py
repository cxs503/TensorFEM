"""Map fabrication fields by model IDs and check residual-stress balance."""
import torch
from tensorfem.initial_fields import import_initial_fields, imperfect_strip_inputs
from tensorfem.result_db import ResultDB

ids=torch.tensor([101,102,103,104]);elements=torch.tensor([11]);connectivity=torch.tensor([[101,102,103,104]])
coordinates=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]],dtype=torch.float64)
imperfection=torch.zeros((4,3),dtype=torch.float64);imperfection[:,2]=torch.tensor([0.,.002,0.,-.002])
stress=torch.zeros((4,6),dtype=torch.float64);stress[:,0]=torch.tensor([80.,-80.,80.,-80.])
db=ResultDB(ids,elements,connectivity,{"coordinates":coordinates,
    "initial_imperfection":imperfection,"residual_stress":stress},
    units={"coordinates":"m","initial_imperfection":"m","residual_stress":"MPa"})
initial=import_initial_fields(db,target_node_ids=ids,target_element_ids=elements,
    target_connectivity=connectivity,target_coordinates=coordinates)
initial.assert_self_equilibrated()
strip_imperfection,strip_residual_stress=imperfect_strip_inputs(initial)
print({"imperfection":strip_imperfection.tolist(),"residual_stress":strip_residual_stress.tolist()})
