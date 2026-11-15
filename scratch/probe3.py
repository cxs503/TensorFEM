import sys, time
sys.path.insert(0, "/root/tp-agent/work/m21707/src")
sys.path.insert(0, "/root/tp-agent/work/m21707")
import numpy as np
import tensorfem.patch_from_contact as P
import tensorfem.ice_panel_patch_contact as IC
from tools.verify_patch_from_contact_navier import navier_crosscheck

t0 = time.time()
out = navier_crosscheck(meshes=(16,), contact_grid=33)
print("elapsed %.1fs" % (time.time() - t0))
for r in out["rows"]:
    print(r["patch"], r["plate_nx"], "mine %.3f ic %.3f nav %.3f -> mine_vs_nav %.3f%% ic_vs_nav %.3f%%" % (
        r["mine_sigma_x_MPa"], r["ic_fem_sigma_Mx_MPa"], r["navier_sigma_Mx_MPa"],
        r["mine_vs_navier_pct"], r["ic_fem_vs_navier_pct"]))

# where is the peak?
A = 1.0; T = 0.01; E = 2.06e11; NU = 0.3
half = 0.2; force = 4.0e4
p0 = force / (4 * half * half)
xs = np.linspace(0.5 - half, 0.5 + half, 9); ys = xs.copy()
XX, YY = np.meshgrid(xs, ys)
uni_data = P.ContactPressureData(node_positions=np.column_stack([XX.ravel(), YY.ravel()]),
                                 normal_pressures=np.full(XX.size, p0), contact_area=0.16)
mine, res = P.contact_pressure_to_plate_load(uni_data, 16, A, E, NU, T, 0.0)
st = P.peak_stresses(mine)
print("P peak sigma_x at", st["sigma_x_at_m"], st["sigma_x_MPa"], "vm at", st["sigma_vm_at_m"], st["sigma_vm_MPa"])
uni_patch = IC.uniform_patch_pressure(total_force=force, centre=(0.5, 0.5), half_x=half, half_y=half, panel=(A, A), grid=33)
nodes, el, f, info = IC.patch_from_contact(uni_patch, 16, 16, A, A)
sol = IC.solve_rect_panel(16, 16, A, A, young=E, poisson=NU, thickness=T, load=f)
s2 = sol.peak_stress()
print("IC peak", s2["peak_sigma_Mx_at_m"], s2["sigma_Mx_MPa"], "vm", s2["peak_sigma_vm_at_m"], s2["sigma_vm_MPa"])
print("IC peak with margin=2", sol.peak_stress(2)["sigma_Mx_MPa"], sol.peak_stress(2)["peak_sigma_Mx_at_m"])
print("IC centre stress_at", sol.stress_at(0.5, 0.5))
print("P centre?", )