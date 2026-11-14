"""Figure for docs/CASE-FEM-ELASTIC-RESTRAINT-TRIBUTARY.md (reads the evidence JSON)."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ev = json.loads(pathlib.Path("docs/data/elastic_restraint_tributary_evidence.json").read_text())
out = pathlib.Path("docs/figures"); out.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

# (a) mesh dependence
b, a = ev["mesh_study_before_fix"], ev["mesh_study_after_fix"]
for beta, col in (("beta_0.2", "C0"), ("beta_0.5", "C1")):
    ax1.plot(b[beta]["nx"], b[beta]["w_norm"], "o--", color=col, mfc="white",
             label=f"beta={beta.split('_')[1]}  before fix")
    ax1.plot(a[beta]["nx"], a[beta]["w_norm"], "o-", color=col,
             label=f"beta={beta.split('_')[1]}  after fix")
ax1.axhline(0.00406, color="0.4", lw=1, ls=":")
ax1.axhline(0.00126, color="0.4", lw=1, ls=":")
ax1.text(8.2, 0.00403, "SSSS 0.00406", fontsize=8, color="0.3")
ax1.text(8.2, 0.00123, "CCCC 0.00126", fontsize=8, color="0.3")
ax1.set_xlabel("elements per side, nx")
ax1.set_ylabel(r"$w_c D/(q a^4)$")
ax1.set_title("(a) beta must be mesh independent")
ax1.legend(fontsize=7, loc="center right")
ax1.grid(alpha=0.3)

# (b) strip edge-moment accuracy
r = ev["edge_moment_recovery_convergence"]["rows"]
labels = [f"{x['mesh'][0]}x{x['mesh'][1]}" for x in r]
idx = range(len(r))
w = 0.36
ax2.bar([i - w / 2 for i in idx], [x["M_extrapolated_err_pct"] for x in r], w,
        label=r"extrapolated $B^T D B$", color="C3")
ax2.bar([i + w / 2 for i in idx],
        [x.get("M_spring_err_pct", 0.0) for x in r], w,
        label=r"spring $M_n=c_r\theta_n$", color="C2")
for i, x in enumerate(r):
    ax2.text(i - w / 2, x["M_extrapolated_err_pct"] - 3,
             f"{x['M_extrapolated_err_pct']:.0f}%", ha="center", fontsize=8)
    if "M_spring_err_pct" in x:
        ax2.text(i + w / 2, -3, f"{x['M_spring_err_pct']:.2f}%", ha="center", fontsize=8)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xticks(list(idx)); ax2.set_xticklabels(labels)
ax2.set_ylabel("edge moment error vs closed form [%]")
ax2.set_title("(b) restrained-edge moment, strip beta=0.5 (exact 277.78 N m/m)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(out / "elastic_restraint_tributary.png", dpi=160)
print("wrote", out / "elastic_restraint_tributary.png")
