"""Run the MacNeal--Harder hemisphere-with-hole convergence audit."""
import json
from tensorfem.spherical_shell import hemisphere_with_hole

rows=[]
for n in (4,6,8,12,16):
    r=hemisphere_with_hole(n)
    rows.append({"mesh":f"{n}x{n}","displacement":r.displacement,
                 "reference":r.reference,"relative_error":r.relative_error})
print(json.dumps(rows,indent=2))
