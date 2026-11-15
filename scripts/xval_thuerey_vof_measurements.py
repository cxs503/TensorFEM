"""Measurements for XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE (programmatic, no hand copy)."""
from __future__ import annotations
import base64, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from tensorfem import lbm_free_surface_thuerey as L

OUT = pathlib.Path("artifacts"); FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
res: dict = {}

# ---- 1. exchange-operator invariants (closed cluster, no flow) -------------
def cluster(ny=9, nx=9):
    s = L.FreeSurfaceLBM(L.FreeSurfaceConfig(nx=nx, ny=ny))
    s.flags[1:-1, 1:-1] = L.INTERFACE; s.mass[1:-1, 1:-1] = 0.5
    rng = np.random.default_rng(3); s.f = rng.random((9, ny, nx)) * 0.05 + 0.02
    s.f[:, s.flags == L.GAS] = 0.0; return s

s = cluster(); dm = s._mass_flux(s.f.copy()); iface = s.flags == L.INTERFACE
res["antisymmetry_abs_sum"] = float(dm[iface].sum()); res["antisymmetry_scale"] = float(np.abs(dm[iface]).sum())
ok = cluster(); ok.mass[4, 4] = 1.37; m0 = ok.mass.sum(); ok._exchange(np.full_like(ok.mass, 1.0))
res["exchange_mass_delta"] = float(ok.mass.sum() - m0); res["exchange_unplaced"] = float(ok.unplaced_mass)
nv = cluster(); nv.cfg.mass_exchange = False; nv.mass[4, 4] = 1.37; n0 = nv.mass.sum()
nv._exchange(np.full_like(nv.mass, 1.0))
res["clipping_mass_delta"] = float(nv.mass.sum() - n0); res["clipping_unplaced"] = float(nv.unplaced_mass)

# ---- 2. quiescent pool -----------------------------------------------------
cfg = L.FreeSurfaceConfig(nx=32, ny=24, gravity=0.0, nu=2e-3)
pool = L.FreeSurfaceLBM(cfg); pool.solid_bottom(1); pool.fill_block(2, 18, 1, 18)
p0 = pool.total_mass()
for _ in range(200): pool.step()
res["quiescent_drift"] = (pool.total_mass() - p0) / p0
res["quiescent_unplaced"] = pool.unplaced_mass

# ---- 3. hydrostatic column -------------------------------------------------
g = 1.0e-5
cfg = L.FreeSurfaceConfig(nx=32, ny=56, gravity=g, nu=2e-3)
h = L.FreeSurfaceLBM(cfg); h.solid_bottom(1); h.fill_block(1, 31, 1, 41)
h0 = h.total_mass()
for _ in range(1500): h.step()
rho, ux, uy = h.macroscopic(); ys = np.arange(4, 39); prof = rho[4:39, 16]
slope = float(np.polyfit(ys, prof, 1)[0]); ana = 3.0 * g
res["hydro_slope"] = slope; res["hydro_slope_analytic"] = ana
res["hydro_slope_err_rel"] = abs(slope - ana) / ana
res["hydro_umax"] = float(np.abs(uy).max())
res["hydro_mass_drift"] = (h.total_mass() - h0) / h0
res["hydro_profile"] = {"y": ys.tolist(), "rho": prof.tolist(),
                        "analytic": (1.0 + ana * (41.0 - ys)).tolist()}

# ---- 4. dam break vs Ritter -----------------------------------------------
g, hh, T = 1.0e-5, 40.0, 1200.0
cfg = L.FreeSurfaceConfig(nx=160, ny=60, gravity=g, nu=2e-3)
d = L.FreeSurfaceLBM(cfg); d.solid_bottom(1); d.fill_block(2, 40, 1, 41)
m0 = d.total_mass(); fronts, times = [], []
for k in range(1, int(T) + 1):
    d.step()
    if k % 100 == 0:
        fronts.append(d.front_position()); times.append(float(k))
col = d.column_height(); xs = np.arange(160)
rit = L.ritter_surface(xs, T, g, hh, dam_x=40.0)
ref = float(L.ritter_front(T, g, hh, dam_x=40.0))
res["dam_front_lbm"] = d.front_position(); res["dam_front_ritter"] = ref
res["dam_front_err_rel"] = abs(d.front_position() - ref) / (ref - 40.0)
res["dam_mass_drift"] = (d.total_mass() - m0) / m0
res["dam_times"] = times; res["dam_fronts"] = fronts
res["dam_ritter_fronts"] = [float(L.ritter_front(t, g, hh, dam_x=40.0)) for t in times]
res["dam_profile"] = {"x": xs[38:70].tolist(), "lbm": col[38:70].tolist(),
                      "ritter": rit[38:70].tolist()}

# ---- 5. figures ------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def b64(path):
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(res["dam_profile"]["x"], res["dam_profile"]["lbm"], "o-", ms=3, label="free-surface LBM + Thürey exchange")
ax[0].plot(res["dam_profile"]["x"], res["dam_profile"]["ritter"], "k--", label="Ritter (1892) exact")
ax[0].set_xlabel("x [lattice]"); ax[0].set_ylabel("liquid height h [lattice]")
ax[0].set_title("dam break, t = 1200 steps"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(res["dam_times"], res["dam_fronts"], "o-", label="LBM front")
ax[1].plot(res["dam_times"], res["dam_ritter_fronts"], "k--", label="Ritter front 2√(g h0) t")
ax[1].set_xlabel("time [lattice steps]"); ax[1].set_ylabel("front x")
ax[1].set_title("front position (documented FAIL)"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "xval_thuerey_dambreak.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(res["hydro_profile"]["rho"], res["hydro_profile"]["y"], "o", ms=3, label="LBM density (pressure)")
ax.plot(res["hydro_profile"]["analytic"], res["hydro_profile"]["y"], "k--", label="1 + 3 g (h − y)")
ax.set_xlabel("ρ (pressure)"); ax.set_ylabel("y [lattice]")
ax.set_title("resting column, 1500 steps"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "xval_thuerey_hydrostatic.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.bar(["Thürey exchange", "clipping"], [res["exchange_mass_delta"], res["clipping_mass_delta"]],
       color=["#2b7", "#c33"])
ax.axhline(0, color="k", lw=.8); ax.set_ylabel("Δ(mass) over one overflow event")
ax.set_title("surplus 0.37 must move, not vanish")
fig.tight_layout(); fig.savefig(FIG / "xval_thuerey_exchange.png", dpi=130); plt.close(fig)

res["figures"] = {p.name: b64(p) for p in FIG.glob("xval_thuerey_*.png")}
(OUT / "xval_thuerey_vof_measurements.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items()
                  if not isinstance(v, (dict, list))}, indent=1))
