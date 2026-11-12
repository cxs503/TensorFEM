import json, base64, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorfem.cohesive3d_dcb import (DCBGeometry, build_dcb_model, solve_dcb, beam_serr)

E, NU, L, B, H = 70e9, 0.30, 60e-3, 3e-3, 1e-3
DELTA = 1e-8
def elastic(crack):
    g = DCBGeometry(length=L, width=B, arm_thickness=H, young_modulus=E, poisson_ratio=NU,
                    initial_crack=crack, nx=120, ny=1, nz=2)
    r = solve_dcb(build_dcb_model(g), delta_max=DELTA, n_steps=4, max_iter=30)
    return g, r.stats["final_strain_energy"], r.curve[-1]["load"], r.stats
cracks = (40e-3, 38e-3, 36e-3)
runs = [elastic(a) for a in cracks]
a = np.array(cracks); U = np.array([r[1] for r in runs]); P = np.array([r[2] for r in runs])
g_fe = -(U[2]-U[0])/(a[2]-a[0])/B
am, pm = a[1], P[1]
g_new = beam_serr(pm, am, runs[1][0], modulus=E, root_rotation=True)
g_old = 6.0*pm**2*am**2/(E*B**2*H**3)

probe = json.load(open('benchmarks/cohesive3d_dcb_displacement_control_probe.json'))
ev = {
 "defect": "GAP-FEM-3D-COHESIVE load-convention / energy-accounting repair",
 "file": "src/tensorfem/cohesive3d_dcb.py",
 "energy_balance_rel": {
   "before_small_model_delta_1e-6": 0.4667,
   "after_small_model_delta_1e-6": 0.0001,
   "before_specimen_probe_documented": 0.499,
   "after_specimen_probe": probe["peak"]["energy_balance_rel"],
 },
 "external_work_vs_strain_energy_elastic": {
   "before_W_over_Es": 1.875, "after_W_over_Es": 1.0000000000001,
 },
 "serr_oracle_slim_geometry": {
   "G_fixed_grip_FE_energy_derivative": g_fe,
   "G_repaired_beam_serr_3P2a2": g_new,
   "G_prerepair_beam_serr_6P2a2": g_old,
   "ratio_FE_over_repaired": g_fe/g_new,
   "ratio_FE_over_prerepair": g_fe/g_old,
 },
 "lefm_peak_load": {
   "prerepair_N": probe["peak"]["lefm_peak_load_N"]/np.sqrt(2.0),
   "repaired_N": probe["peak"]["lefm_peak_load_N"],
   "fem_peak_N": probe["peak"]["nx60_peak_load_N"],
   "prerepair_rel_error": probe["peak"]["nx60_peak_load_N"]/(probe["peak"]["lefm_peak_load_N"]/np.sqrt(2.0))-1,
   "repaired_rel_error": probe["peak"]["nx60_peak_load_N"]/probe["peak"]["lefm_peak_load_N"]-1,
 },
 "compliance_calibration_still_passes": probe["compliance_calibration"]["all_passed"],
}
json.dump(ev, open('docs/GAP-FEM-3D-COHESIVE-energy-convention-evidence.json','w'), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
ax[0].bar(["before\n(small)","after\n(small)","before\n(specimen)","after\n(specimen)"],
          [0.4667, 1e-4, 0.499, probe["peak"]["energy_balance_rel"]],
          color=["#c0392b","#27ae60","#c0392b","#27ae60"])
ax[0].set_yscale("log"); ax[0].set_ylabel(r"$|W_{ext}-U_s-W_{int}|/W_{ext}$")
ax[0].set_title("(a) Energy balance residual"); ax[0].axhline(0.03, ls="--", c="k", lw=0.8)
ax[1].bar(["FE fixed-grip\n$-dU/da/b$","repaired\n$3P^2a^2/Eb^2h^3$","pre-repair\n$6P^2a^2/Eb^2h^3$"],
          [g_fe, g_new, g_old], color=["#2980b9","#27ae60","#c0392b"])
ax[1].set_ylabel(r"$G_I$ [J/m$^2$]  ($\times10^{-10}$)"); ax[1].set_yticks([g_fe, g_new, g_old],
   ["%.2f"%g_fe*1 if False else f"{g_fe:.2e}", f"{g_new:.2e}", f"{g_old:.2e}"])
ax[1].tick_params(axis='y', labelsize=7)
ax[1].set_title("(b) SERR oracle (slim DCB, $a$=38 mm)")
ax[2].bar(["LEFM\npre-repair","FE peak","LEFM\nrepaired"],
          [probe["peak"]["lefm_peak_load_N"]/np.sqrt(2.0), probe["peak"]["nx60_peak_load_N"],
           probe["peak"]["lefm_peak_load_N"]], color=["#c0392b","#2980b9","#27ae60"])
ax[2].set_ylabel("$P$ [N]"); ax[2].set_title("(c) Peak load: sign of the LEFM gap flips")
fig.tight_layout(); fig.savefig('docs/assets/gap-fem-3d-cohesive-energy-convention.png', dpi=135)
print(json.dumps(ev, indent=1))
b64 = base64.b64encode(open('docs/assets/gap-fem-3d-cohesive-energy-convention.png','rb').read()).decode()
open('/tmp/fig_b64.txt','w').write(b64)
