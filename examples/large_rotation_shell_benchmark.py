"""Run the 90-degree pure-bending shell convergence sequence."""
import json
from tensorfem.large_rotation_shell_benchmark import pure_bending_shell
rows=[]
for n in (1,2,4):
    r=pure_bending_shell(n);rows.append({"mesh":f"{n}x1","tip":r.tip.tolist(),
        "exact_tip":r.exact_tip.tolist(),"tip_error":r.tip_error,
        "reaction_moment":r.applied_moment,"exact_moment":r.exact_moment,
        "moment_error":r.moment_error,"path":r.result.path})
print(json.dumps(rows,indent=2))
