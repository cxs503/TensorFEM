"""Print the MacNeal--Harder pinched-cylinder convergence sequence."""
import json
from tensorfem.pinched_cylinder import solve_pinched_cylinder_linear

rows=[]
for n in (4,6,8,12):
    r=solve_pinched_cylinder_linear(n)
    rows.append({"mesh":f"{n}x{n}","displacement":r.displacement,
                 "reference":r.reference,"relative_error":r.relative_error})
print(json.dumps(rows,indent=2))
