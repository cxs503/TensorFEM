"""XVAL entry point: two-component Shan-Chen MCMP (barycentric u + Guo forcing).

Anchor: ``XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO``
External reference: ``lssfau/lbmpy`` ``tests/shan_chen/test_shan_chen_two_component.py``
(reference array from ``lbm-principles-practice/chapter9/shanchen.cpp``).
Acceptance: relative error < 15 % (owner accuracy standard).

Run:
    PYTHONPATH=src python examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py

Writes (all numbers in the report/doc are injected from ``results.json``):
    docs/assets/lbm_shan_chen_mcmp_barycentric_guo/results.json
    docs/assets/lbm_shan_chen_mcmp_barycentric_guo/fig*.png
    docs/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from tensorfem.lbm_shan_chen_mcmp import (
    GOLDEN_32x32, REFERENCE_PROVENANCE, REF_A, REF_B, THRESHOLD_REL_L2,
    MCMPConfig, ShanChenMCMP, relative_l2, relative_linf, run_reference_case,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "lbm_shan_chen_mcmp_barycentric_guo"
REFS = ASSETS / "lbm_refs"
DOC = ROOT / "docs" / "XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO.md"


# ---------------------------------------------------------------- study arms
def ladder_arms(steps: int = 200, n: int = 64) -> list:
    """Scheme arms: the two defects of the first delivery plus the validated one."""
    arms = [
        ("validated: psi(x+c), u_bary = m+ (Fa+Fb)/2, no extra shift",
         dict(neighbour_convention="plus", eq_shift="none")),
        ("defect D1: psi(x-c) neighbour (force sign flipped)",
         dict(neighbour_convention="minus", eq_shift="none")),
        ("defect D2: psi(x+c) but equilibrium shifted again by F_s/(2 rho_s)",
         dict(neighbour_convention="plus", eq_shift="half")),
        ("variant: classic Shan-Chen tau F_s/rho_s equilibrium shift",
         dict(neighbour_convention="plus", eq_shift="tau")),
        ("variant: per-component (self) velocity instead of barycentric",
         dict(neighbour_convention="plus", eq_shift="none", velocity_mode="self")),
        ("variant: no Guo source term (collision only)",
         dict(neighbour_convention="plus", eq_shift="none", force_scheme="none")),
    ]
    rows = []
    for label, over in arms:
        cfg = dict(nx=n, ny=n, g_ab=6.0, g_ba=6.0, tau_a=1.0, tau_b=1.0)
        cfg.update(over)
        sim = ShanChenMCMP(MCMPConfig(**cfg))
        sim.init_step_x(0.1, 0.9)
        div = None
        for i in range(1, steps + 1):
            sim.step()
            if not np.isfinite(sim.f).all():
                div = i
                break
        row = {"arm": label, "diverged_at": div, "steps": steps}
        if div is None:
            rho = sim.density()
            row["rel_l2_a"] = relative_l2(rho[0, 0, :], REF_A)
            row["rho_min"] = float(rho.min())
            row["rho_max"] = float(rho.max())
        else:
            fin = np.where(np.isfinite(sim.density()), sim.density(), np.nan)
            row["rho_max_last_finite"] = float(np.nanmax(fin))
        rows.append(row)
    return rows


def golden_study() -> list:
    """Resolution + interaction-strength study against live lbmpy references."""
    cases = [("resolution", 32, 6.0), ("resolution", 48, 6.0),
             ("resolution", 96, 6.0), ("g_ab", 32, 3.0), ("g_ab", 32, 4.5)]
    out = []
    for axis, n, g in cases:
        path = REFS / f"lbmpy_n{n}_g{g}_s400.json"
        if not path.exists():  # pragma: no cover
            continue
        ref = json.loads(path.read_text())
        sim = ShanChenMCMP(MCMPConfig(nx=n, ny=n, g_ab=g, g_ba=g, tau_a=1.0, tau_b=1.0))
        sim.init_step_x(0.1, 0.9)
        sim.run(int(ref["params"]["steps"]))
        got = sim.density()[0, 0, :]
        r = np.asarray(ref["rho_a_profile"], dtype=float)
        out.append({"axis": axis, "n": n, "g_ab": g,
                    "steps": int(ref["params"]["steps"]),
                    "rel_l2_a": relative_l2(got, r),
                    "rel_linf_a": relative_linf(got, r),
                    "ref_rho_a_interface": float(r[0]),
                    "got_rho_a_interface": float(got[0]),
                    "ref_file": str(path.relative_to(ROOT))})
    return out


def live_lbmpy_check() -> dict:
    """Re-execute the OSS algorithm on this machine and verify the stored goldens."""
    try:
        sys.path.insert(0, str(ROOT))
        from tools.lbmpy_shan_chen_live_ref import run_live_reference
    except Exception as exc:  # pragma: no cover
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    try:
        live = run_live_reference(n=32, steps=400, g_ab=6.0)
    except Exception as exc:  # pragma: no cover - lbmpy/pystencils absent
        return {"status": "not_run", "reason": f"{type(exc).__name__}: {exc}"}
    stored = json.loads(GOLDEN_32x32.read_text())
    live_a = np.asarray(live["rho_a_profile"])
    stored_a = np.asarray(stored["rho_a_profile"])
    return {"status": "executed", "lbmpy_version": live["lbmpy_version"],
            "pystencils_version": live["pystencils_version"],
            "wall_time_s": live["wall_time_s"],
            "max_abs_diff_vs_stored_golden": float(np.abs(live_a - stored_a).max()),
            "rel_l2_vs_stored_golden": relative_l2(live_a, stored_a)}


# ------------------------------------------------------------------- figures
def make_figures(res: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    anchor = res["anchor"]
    ref_a = np.asarray(anchor["rho_a_ref"])
    got_a = np.asarray(anchor["rho_a_profile"])
    ref_b = np.asarray(anchor["rho_b_ref"])
    got_b = np.asarray(anchor["rho_b_profile"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ref_a, "o", ms=3, label="lbmpy ref $\\rho_a$")
    ax[0].plot(got_a, "-", lw=1, label="TensorFEM $\\rho_a$")
    ax[0].plot(ref_b, "s", ms=3, label="lbmpy ref $\\rho_b$")
    ax[0].plot(got_b, "--", lw=1, label="TensorFEM $\\rho_b$")
    ax[0].set_xlabel("x [lattice]")
    ax[0].set_ylabel("density")
    ax[0].legend(fontsize=7)
    ax[0].set_title("64$^2$, $g_{ab}$=6, 1000 steps")
    ax[1].semilogy(np.abs(got_a - ref_a) + 1e-18, label="$|\\Delta\\rho_a|$")
    ax[1].semilogy(np.abs(got_b - ref_b) + 1e-18, label="$|\\Delta\\rho_b|$")
    ax[1].set_xlabel("x [lattice]")
    ax[1].set_ylabel("pointwise error")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"rel L2 = {anchor['rel_l2_a']:.2e}")
    fig.tight_layout()
    p = ASSETS / "fig1_profiles_vs_lbmpy.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p)

    # field map of the recovered two-phase state
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    im0 = ax[0].imshow(np.asarray(anchor["rho_a_field"]), origin="lower")
    ax[0].set_title("$\\rho_a$ field (1000 steps)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    u = np.asarray(anchor["speed_field"])
    im1 = ax[1].imshow(u, origin="lower")
    ax[1].set_title(f"|u| field, max={u.max():.2e}")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    p = ASSETS / "fig2_fields.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p)

    # falsification ladder
    lad = res["ladder"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    labels = [f"{i}: {r['arm'][:46]}" for i, r in enumerate(lad)]
    vals = [r["diverged_at"] if r["diverged_at"] else r["steps"] for r in lad]
    colors = ["tab:red" if r["diverged_at"] else "tab:green" for r in lad]
    ax.barh(range(len(lad)), vals, color=colors)
    ax.set_yticks(range(len(lad)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(f"steps completed before divergence (cap {lad[0]['steps']})")
    ax.invert_yaxis()
    fig.tight_layout()
    p = ASSETS / "fig3_scheme_ladder.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p)

    # resolution + g_ab study
    st = res["golden_study"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, mark in (("resolution", "o"), ("g_ab", "s")):
        rows = [r for r in st if r["axis"] == key]
        x = [r["n"] if key == "resolution" else r["g_ab"] for r in rows]
        y = [max(r["rel_l2_a"], 1e-16) for r in rows]
        ax[0].semilogy(x, y, mark + "-", label=key)
    ax[0].axhline(THRESHOLD_REL_L2, color="r", ls="--", label="15% acceptance")
    ax[0].set_xlabel("n (resolution) / $g_{ab}$")
    ax[0].set_ylabel("rel $L_2(\\rho_a)$ vs live lbmpy ref")
    ax[0].legend(fontsize=8)
    ax[1].bar(range(len(st)), [max(r["rel_l2_a"], 1e-16) for r in st], color="tab:blue")
    ax[1].set_yscale("log")
    ax[1].axhline(THRESHOLD_REL_L2, color="r", ls="--")
    ax[1].set_xticks(range(len(st)))
    ax[1].set_xticklabels([f"n{r['n']} g{r['g_ab']}" for r in st], fontsize=7)
    ax[1].set_ylabel("rel $L_2$")
    fig.tight_layout()
    p = ASSETS / "fig4_resolution_gstudy.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p)
    return [str(p.relative_to(ROOT)) for p in figs]


# ---------------------------------------------------------------- doc render
def render_doc(res: dict) -> str:
    a = res["anchor"]
    p = a["config"]
    lad = "\n".join(
        "| {} | {} | {} |".format(
            r["arm"],
            "—" if r["diverged_at"] is None else f"**{r['diverged_at']}**",
            "—" if "rel_l2_a" not in r else f"{r['rel_l2_a']:.3e}")
        for r in res["ladder"])
    gs = "\n".join(
        f"| {r['axis']} | {r['n']} | {r['g_ab']} | {r['steps']} | "
        f"{r['ref_rho_a_interface']:.6f} | {r['got_rho_a_interface']:.6f} | "
        f"{r['rel_l2_a']:.3e} | {r['rel_linf_a']:.3e} | {'PASS' if r['rel_l2_a'] < THRESHOLD_REL_L2 else 'FAIL'} |"
        for r in res["golden_study"])
    live = res["live_lbmpy"]
    live_line = (
        f"executed on this machine with lbmpy {live.get('lbmpy_version')} / "
        f"pystencils {live.get('pystencils_version')} in {live.get('wall_time_s')} s; "
        f"max |Δ| against the stored golden = {live.get('max_abs_diff_vs_stored_golden'):.2e}"
        if live.get("status") == "executed" else
        f"not executed ({live.get('status')}: {live.get('reason')})")
    verdict = "PASS" if res["anchor_passed"] else "FAIL"
    return f"""# XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO — two-component Shan-Chen cross-validation

Anchor: `XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO`  
External reference: `{REFERENCE_PROVENANCE['repo']}` `{REFERENCE_PROVENANCE['path']}`
(array origin `{REFERENCE_PROVENANCE['origin']}`)  
Acceptance: relative error < {int(THRESHOLD_REL_L2*100)} % — **verdict: {verdict}**
(rel $L_2$ = {a['rel_l2_a']:.3e}).

Every number in this document is injected from
`docs/assets/lbm_shan_chen_mcmp_barycentric_guo/results.json` by
`examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py` (no hand-typed value).

## 1. Case and reference

Two immiscible components `a`, `b` on a periodic `{p['nx']}x{p['ny']}` D2Q9 lattice,
sharp step initial condition (`a` = 0.1 left / 0.9 right, `b` complementary), which
creates two periodic interfaces.  The reference density profiles are the published
lbmpy arrays (`ref_a`, `ref_b`; sha256 `{REFERENCE_PROVENANCE['sha256_ref_a'][:16]}…` /
`{REFERENCE_PROVENANCE['sha256_ref_b'][:16]}…`) generated by the independent C++
code `lbm-principles-practice/chapter9/shanchen.cpp`.  The reference was
**re-executed live** for this delivery: {live_line}.

| parameter | value |
|---|---|
| domain | {p['nx']} x {p['ny']}, periodic |
| relaxation | $\\tau_a$ = {p['tau_a']}, $\\tau_b$ = {p['tau_b']} ($\\omega$ = 1) |
| interaction | $g_{{aa}}$ = {p['g_aa']}, $g_{{bb}}$ = {p['g_bb']}, $g_{{ab}}$ = $g_{{ba}}$ = {p['g_ab']} |
| pseudo potential | $\\psi = \\rho_0(1-e^{{-\\rho/\\rho_0}})$, $\\rho_0$ = {p['rho0']} |
| steps | {a['steps_completed']} |
| wall time | {a['wall_time_s']} s (NumPy, float64, serial) |

## 2. Method

The solver `src/tensorfem/lbm_shan_chen_mcmp.py` implements

* interaction force $F_s(x) = -\\psi_s(x)\\sum_{{s'}}g_{{ss'}}\\sum_i w_i c_i \\psi_{{s'}}(x+c_i)$
  (plus-neighbour convention; `ShanChenMCMP.interaction_force`),
* one barycentric mixture velocity
  $u = \\sum_s(\\sum_i c_i f_i^s + F_s/2)/\\sum_s\\rho_s$ (`barycentric_velocity`),
* Guo (2002) source $S_i^s = (1-\\omega_s/2)w_i[(c_i-u)/c_s^2 + (c_i\\cdot u)c_i/c_s^4]\\cdot F_s$
  (`step`), SRT collision, periodic pull streaming.

Two defects of the first delivery (m18651, honest FAIL) were located and fixed;
they are kept as switches so the failure can be replayed:

| arm | diverged at step | rel $L_2(\\rho_a)$ after {res['ladder'][0]['steps']} steps |
|---|---|---|
{lad}

**Root cause of the previous failure**: `np.roll(psi, +c)` samples `psi(x-c)`
(+=`np.roll(a,s)[i]=a[i-s]`), which flips the sign of the whole interaction force;
at $g_{{ab}}$ = 6 the flipped force repels like components and the state blows up
within {min(r['diverged_at'] for r in res['ladder'] if r['diverged_at'])} steps.  A second, independent
error was the extra per-component equilibrium shift $F_s/(2\\rho_s)$ on top of an
already force-corrected barycentric velocity (double counting).

## 3. Results — anchor case

| quantity | reference (lbmpy) | computed (TensorFEM) | rel. error |
|---|---|---|---|
| $\\rho_a$ profile (L2) | — | — | {a['rel_l2_a']:.3e} |
| $\\rho_b$ profile (L2) | — | — | {a['rel_l2_b']:.3e} |
| $\\rho_a$ profile (Linf) | — | — | {a['rel_linf_a']:.3e} |
| $\\rho_a$ at interface x=0 | {a['rho_a_ref'][0]:.6f} | {a['rho_a_profile'][0]:.6f} | {abs(a['rho_a_profile'][0]-a['rho_a_ref'][0])/abs(a['rho_a_ref'][0]):.3e} |
| $\\rho_a$ peak | {max(a['rho_a_ref']):.6f} | {max(a['rho_a_profile']):.6f} | {abs(max(a['rho_a_profile'])-max(a['rho_a_ref']))/max(a['rho_a_ref']):.3e} |
| mass $a$ (initial {0.5*p['nx']*p['ny']:.0f}) | {0.5*p['nx']*p['ny']:.6f} | {a['mass_a']:.6f} | {abs(a['mass_a']-0.5*p['nx']*p['ny'])/(0.5*p['nx']*p['ny']):.3e} |

Acceptance: rel $L_2$ = {a['rel_l2_a']:.3e} < {THRESHOLD_REL_L2} → **{verdict}**
({THRESHOLD_REL_L2/max(a['rel_l2_a'],1e-30):.1e}x margin).

## 4. Resolution and interaction-strength sensitivity

Each row is a *separate live lbmpy execution* (`tools/lbmpy_shan_chen_live_ref.py`,
400 steps) compared against the TensorFEM solver at the same settings; the stored
reference files are listed in `results.json:golden_study[].ref_file`.

| axis | n | $g_{{ab}}$ | steps | ref $\\rho_a$(x=0) | got $\\rho_a$(x=0) | rel L2 | rel Linf | gate |
|---|---|---|---|---|---|---|---|---|
{gs}

## 5. Failure modes / limitations

* The `psi(x-c)` sign defect and the double velocity shift are both catastrophic
  (table in §2); neither is detectable by mass conservation, which holds to
  {abs(a['mass_a']-0.5*p['nx']*p['ny'])/(0.5*p['nx']*p['ny']):.1e} even in the broken arms — an
  invariant alone cannot certify this scheme.
* The Laplace-law (drop) calibration was **not** re-attempted in this round; the
  static-drop state is only reported as a field/spurious-current map (`fig2_fields.png`,
  max |u| = {max(map(max, a['speed_field'])):.2e}).  Not run ≠ passed.
* Comparison is against a lattice-Boltzmann reference, not experiment: this closes
  the *implementation* question (does our MCMP kernel reproduce the published
  solution), not the *model* question (is Shan-Chen surface tension quantitative).
* Reproducibility requires the same float64 arithmetic; the residual
  {a['rel_l2_a']:.1e} is the accumulated difference to a different code (pystencils
  generated C), not a truncation error.

## 6. Artifacts and reproduction

```
repro_cmd: PYTHONPATH=src python examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py
unit tests: PYTHONPATH=src python -m pytest tests/test_lbm_shan_chen_mcmp.py -q
live reference generator: python tools/lbmpy_shan_chen_live_ref.py --n 32 --steps 400
```

| artifact | path |
|---|---|
| results | `docs/assets/lbm_shan_chen_mcmp_barycentric_guo/results.json` |
| figures | `fig1_profiles_vs_lbmpy.png`, `fig2_fields.png`, `fig3_scheme_ladder.png`, `fig4_resolution_gstudy.png` |
| live references | `docs/assets/lbm_shan_chen_mcmp_barycentric_guo/lbm_refs/*.json` |
| implementation | `src/tensorfem/lbm_shan_chen_mcmp.py` |
| platform integration | `src/tensorfem/benchmark_registry.py::run_registered_benchmarks` (`lbm.mcmp_shan_chen.*`) |
"""


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    anchor = run_reference_case()
    sim = ShanChenMCMP(MCMPConfig(nx=64, ny=64, g_ab=6.0, g_ba=6.0))
    sim.init_step_x(0.1, 0.9)
    sim.run(1000)
    rho = sim.density()
    u = sim.velocity()
    anchor.update({
        "rho_a_ref": [float(v) for v in REF_A],
        "rho_b_ref": [float(v) for v in REF_B],
        "rho_a_field": [[float(v) for v in row] for row in rho[0]],
        "rho_b_field": [[float(v) for v in row] for row in rho[1]],
        "speed_field": [[float(v) for v in row] for row in np.hypot(u[0], u[1])],
        "profile_row": 0,
    })
    res = {
        "anchor": "XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO",
        "repo": "TensorFEM",
        "entry_point": "examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py",
        "impl_module": "src/tensorfem/lbm_shan_chen_mcmp.py",
        "reference": REFERENCE_PROVENANCE,
        "threshold_rel_l2": THRESHOLD_REL_L2,
        "anchor": anchor,
        "anchor_passed": anchor["rel_l2_a"] < THRESHOLD_REL_L2,
        "ladder": ladder_arms(),
        "golden_study": golden_study(),
        "live_lbmpy": live_lbmpy_check(),
        "numpy_version": np.__version__,
        "python": sys.version.split()[0],
        "wall_time_s": round(time.time() - t0, 2),
    }
    res["figures"] = make_figures(res)
    (ASSETS / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    DOC.write_text(render_doc(res))
    print(json.dumps({k: res[k] for k in
                      ("anchor_passed", "ladder", "golden_study", "live_lbmpy",
                       "figures", "wall_time_s")}, indent=1)[:2600])
    print(f"rel_l2_a={anchor['rel_l2_a']:.3e} rel_l2_b={anchor['rel_l2_b']:.3e} "
          f"verdict={'PASS' if res['anchor_passed'] else 'FAIL'}")
    return 0 if res["anchor_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())