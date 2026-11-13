"""Audit figure for m17663: reproduced rows -> 3 panels (numbers come from
reproduced_numbers.json / scratch rows.json, no hand copying)."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
num = json.loads((HERE / "reproduced_numbers.json").read_text())
ROWS = pathlib.Path("/root/tp-agent/work/m17663/.scratch/m17604_repro/logs/rows.json")
rows = json.loads(ROWS.read_text())
rs = list(num["rows"].values())

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
styles = {("TensorFEM", 1): ("#1f77b4", "o", "HEX8 (in-tree)"),
          ("TensorFEM", 2): ("#1f77b4", "s", "HEX20 (in-tree, selected)"),
          ("scikit-fem", 1): ("#d62728", "^", "TET4 (skfem)"),
          ("scikit-fem", 2): ("#d62728", "v", "TET10 (skfem, projected)")}
for a, key, title in ((ax[0], "worst_m1", "M1: Lame at volume-averaged radius"),
                      (ax[1], "worst_m2", "M2: exact layer integral")):
    for (arm, order), (c, m, name) in styles.items():
        sel = [r for r in num["rows"].values() if r["arm"] == arm and r["order"] == order]
        if not sel:
            continue
        sel.sort(key=lambda r: r["dofs"])
        a.plot([r["dofs"] for r in sel], [100 * r[key] for r in sel],
               marker=m, color=c, ls="--" if arm == "scikit-fem" else "-", ms=6, label=name)
    a.axhline(3, color="r", ls=":", label="selection claim 3%")
    a.axhline(5, color="gray", ls="-.", label="case line 5%")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("degrees of freedom"); a.set_ylabel("worst probe error [%]")
    a.set_title(title); a.grid(alpha=.3, which="both"); a.legend(fontsize=7)

# panel (c): stress field along radius, FE element values vs Lame closed form
r = np.linspace(8.0, 10.0, 400)
E, NU, P, A_, B_ = 210e9, 0.3, 1e6, 8.0, 10.0
lam = E * NU / ((1 + NU) * (1 - 2 * NU)); mu = E / (2 * (1 + NU))
C1 = P * A_**3 / (B_**3 - A_**3); C2 = P * A_**3 * B_**3 / (2 * (B_**3 - A_**3))
sr = C1 - 2 * C2 / r**3
st = C1 + C2 / r**3
vm = np.sqrt(0.5 * ((sr - st) ** 2 + (st - 0) ** 2 + (0 - sr) ** 2))
ax[2].plot(r, sr / 1e6, "k-", lw=1.6, label="Lame sigma_r")
ax[2].plot(r, vm / 1e6, "k--", lw=1.2, label="Lame von Mises")
row = rows["tf_o2_n4"]
cen = np.asarray(row["fields_centres"]); rad = np.linalg.norm(cen, axis=1)
o = np.argsort(rad)
ax[2].plot(rad[o], np.asarray(row["fields_von_mises"])[o] / 1e6, "s", ms=2.5,
           alpha=.45, color="#1f77b4", label="FE HEX20 n=4 von Mises")
ax[2].set_xlabel("radius r [m]"); ax[2].set_ylabel("stress [MPa]")
ax[2].set_title("field check: HEX20 vs Lamé (p=1 MPa)"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=7)
fig.tight_layout()
out = HERE / "fig_xval_audit.png"
fig.savefig(out, dpi=140)
print("wrote", out, out.stat().st_size)
