"""3-level mesh convergence sweep for the plate-patch elastoplastic capacity.

Anchor ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET`` acceptance criterion "3 档网格收敛":
the numerical collapse / first-yield / permanent-set path must converge
monotonically as the mesh is refined, and the transition point must stay within
the same-mesh elastic-field tolerance at every level.

This script runs the layered Shell4 J2 plate path at three admissible mesh
levels (nx = ny = 4, 8, 12 - the counts for which the central patch edges land
on element edges) on the same geometry / material / patch and writes:

* ``artifacts/patch_plate_mesh_sweep.json``  — the per-mesh convergence table
* ``artifacts/fig_patch_plate_mesh_convergence.png`` — 3-panel figure: capacity
  vs mesh against the analytic bounds, permanent set vs mesh, and the relative
  errors vs mesh.
* ``docs/CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT-MESH-SWEEP.md`` — the generated case
  note (table + verdicts; nothing in it is typed by hand)
* ``report_m20041.json`` — the publishable progress report payload, with the
  figure embedded as base64

Run:  ``PYTHONPATH=src python scripts/patch_plate_mesh_sweep.py [--mission-id N]``
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    limit_load_from_path,
    run_patch_plastic_path,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

A = B = 1.0
T = 0.02
E, NU, FY = 210e9, 0.3, 355e6
PATCH = (0.5 * A, 0.5 * B)
HARDENING = 2e10                # hardening case: permanent set
IDEAL_HARDENING = 1e6           # near-ideal plasticity: collapse vs yield-line bound
LAYERS = 3
# The patch spans the central half of the plate, so its edges land on element
# edges only when the element count is a multiple of 4: for nx = 6 the patch
# edge sits at 1.5 element widths and ``patch_plate_mesh`` (rightly) refuses the
# mesh.  The three admissible levels are therefore 4 / 8 / 12, not 4 / 6 / 8.
MESH_LEVELS = (4, 8, 12)

HERE = Path(__file__).resolve().parents[1]
ART = HERE / "artifacts"
DOC = HERE / "docs" / "CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT-MESH-SWEEP.md"
REPORT = HERE / "report_m20041.json"
ROWS_CACHE = ART / "patch_plate_mesh_sweep_rows.json"
FIG_NAME = "fig_patch_plate_mesh_convergence.png"


def _run_level(nx: int, py: dict) -> dict:
    """Run the FE paths at one mesh level and return the convergence observables.

    Two paths on the same mesh are needed, because one load ramp cannot answer
    both questions honestly:

    * *hardening* path (H = 20 GPa, unloaded at the end) - first yield, the
      elastic bias probe and the permanent set;
    * *near-ideal* path (H = 1 MPa, no unloading) - the plastic **collapse**
      load.  With strong hardening under load control the Newton iteration never
      runs out of equilibrium, so ``collapse_factor`` stays ``None`` and the end
      of the ramp (an arbitrary prescribed number) must **not** be reported as
      a collapse load; the limit load is instead measured on the path by
      ``limit_load_from_path`` (tangent intersection + 2x-deflection rule, whose
      spread is published with it).
    """
    upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"]
    steps = [f * py["pressure"] for f in
             (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)]
    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                  hardening=HARDENING, patch=PATCH,
                                  load_factors=steps, nx=nx, ny=nx,
                                  layers=LAYERS, unload_steps=4)
    # same-mesh elastic bias probe (one small elastic step)
    probe = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                   hardening=HARDENING, patch=PATCH,
                                   load_factors=[0.5 * py["pressure"]],
                                   nx=nx, ny=nx, layers=LAYERS)
    fe_sigma_per_pa = probe.loading[0].max_surface_mises / probe.loading[0].load_factor
    same_mesh_yield = FY / fe_sigma_per_pa
    first = path.first_yield_factor

    # collapse: near-ideal plasticity, measured on the path (never the ramp end)
    ideal = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                   hardening=IDEAL_HARDENING, patch=PATCH,
                                   load_factors=steps, nx=nx, ny=nx, layers=LAYERS)
    limit = limit_load_from_path(ideal)
    collapse = limit["limit_load"]
    ramp_end = ideal.loading[-1].load_factor
    # the through-thickness layering artefact (outermost sample at
    # |z| = t/2 (1 - 1/n)) delays plastic flow by 1/(1-1/n) in stress; the
    # corrected value is published *alongside* the raw one, never instead of it
    collapse_corrected = collapse * (1.0 - 1.0 / LAYERS)

    return {
        "nx": nx,
        "first_yield_factor": first,
        "first_yield_error_vs_analytic": (first - py["pressure"]) / py["pressure"],
        "first_yield_error_vs_same_mesh": (first - same_mesh_yield) / same_mesh_yield,
        "collapse_factor": collapse,
        "collapse_measurements": limit["measurements"],
        "collapse_spread": limit["spread"],
        "collapse_is_limit_point": bool(ideal.collapse_is_limit_point),
        "ramp_end_factor": ramp_end,
        "collapse_factor_layering_corrected": collapse_corrected,
        "collapse_error_vs_upper": (collapse - upper) / upper,
        "collapse_error_vs_upper_corrected": (collapse_corrected - upper) / upper,
        "permanent_deflection": path.permanent_deflection,
        "permanent_set_valid": bool(path.permanent_set_valid),
        "peak_centre_deflection": path.loading[-1].centre_deflection,
        "elastic_bias": fe_sigma_per_pa / py["sigma_eq_per_pa"] - 1.0,
    }


def convergence_summary(rows: list[dict], bounds: dict) -> dict:
    """Derive the convergence verdicts from the computed rows (never hard-coded)."""
    upper = bounds["yield_line_upper_Pa"]
    lower = bounds["strip_lower_Pa"]
    collapsed = [r["collapse_factor"] for r in rows]
    perms = [r["permanent_deflection"] for r in rows]
    spreads = [r["collapse_spread"] for r in rows if r.get("collapse_spread")]
    summary = {
        "collapse_is_measured_limit_load": all(
            r["collapse_factor"] != r["ramp_end_factor"] for r in rows),
        "collapse_monotone": all(collapsed[i] <= collapsed[i + 1]
                                 for i in range(len(collapsed) - 1)),
        "collapse_below_upper": all(c < upper for c in collapsed),
        "collapse_above_lower": all(c > lower for c in collapsed),
        "collapse_vs_upper_max_abs_pct":
            100 * max(abs(r["collapse_error_vs_upper"]) for r in rows),
        "collapse_vs_upper_max_abs_pct_corrected":
            100 * max(abs(r["collapse_error_vs_upper_corrected"]) for r in rows),
        "collapse_measurement_spread_max_pct":
            100 * max(spreads) if spreads else None,
        "transition_vs_same_mesh_max_abs_pct":
            100 * max(abs(r["first_yield_error_vs_same_mesh"]) for r in rows),
        "transition_vs_analytic_max_abs_pct":
            100 * max(abs(r["first_yield_error_vs_analytic"]) for r in rows),
        "permanent_set_valid": all(r.get("permanent_set_valid", False) for r in rows),
        "permanent_set_spread_pct": 100 * (max(perms) / min(perms) - 1.0),
        "permanent_set_bounded": all(0.0 < p < r["peak_centre_deflection"]
                                     for p, r in zip(perms, rows)),
    }
    summary["acceptance"] = {
        "collapse_monotone": summary["collapse_monotone"],
        "collapse_below_upper": summary["collapse_below_upper"],
        "collapse_above_lower": summary["collapse_above_lower"],
        "collapse_vs_upper_lt_10pct": summary["collapse_vs_upper_max_abs_pct"] < 10.0,
        "limit_spread_lt_30pct":
            (summary["collapse_measurement_spread_max_pct"] or 100.0) < 30.0,
        "transition_same_mesh_lt_5pct":
            summary["transition_vs_same_mesh_max_abs_pct"] < 5.0,
        "permanent_set_spread_lt_50pct": summary["permanent_set_spread_pct"] < 50.0,
    }
    return summary


def markdown_table(rows: list[dict], bounds: dict) -> str:
    """Render the per-mesh table; every cell comes from the computed rows."""
    head = ("| mesh nx=ny | first yield [MPa] | trans. vs same-mesh | trans. vs analytic "
            "| collapse (measured) [MPa] | collapse spread | collapse vs upper "
            "| ramp end [MPa] | permanent set [mm] | elastic bias |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {r['nx']} | {r['first_yield_factor']/1e6:.4f} "
        f"| {100*r['first_yield_error_vs_same_mesh']:+.2f} % "
        f"| {100*r['first_yield_error_vs_analytic']:+.2f} % "
        f"| {r['collapse_factor']/1e6:.4f} "
        f"| {100*(r['collapse_spread'] or 0.0):.2f} % "
        f"| {100*r['collapse_error_vs_upper']:+.2f} % "
        f"| {r['ramp_end_factor']/1e6:.4f} "
        f"| {1000*r['permanent_deflection']:.4f} "
        f"| {100*r['elastic_bias']:+.2f} % |\n"
        for r in rows)
    tail = (f"\nReferences: analytic first yield {bounds['analytic_first_yield_Pa']/1e6:.4f} MPa, "
            f"yield-line upper bound {bounds['yield_line_upper_Pa']/1e6:.4f} MPa, "
            f"strip lower bound {bounds['strip_lower_Pa']/1e6:.4f} MPa.  "
            f"Collapse is the path-measured limit load (tangent intersection); the ramp end is "
            f"shown only to document that it is *not* used.\n")
    return head + body + tail


def case_doc(rows: list[dict], bounds: dict, summary: dict) -> str:
    """Compose the generated case note for the mesh-convergence acceptance."""
    verdict = lambda ok: "met" if ok else "NOT met"
    return f"""# CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT — 3-level mesh convergence

Generated by `scripts/patch_plate_mesh_sweep.py` from
`artifacts/patch_plate_mesh_sweep.json`; no value in this file is transcribed by
hand.  Parent case: `docs/CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT.md`.

## Model

Simply supported 1.0 x 1.0 x 0.020 m plate, centred 0.5 x 0.5 m patch load,
E = 210 GPa, nu = 0.3, sigma_y = 355 MPa, layered Shell4 (Q4, J2 plane stress)
with {LAYERS} layers, load control, meshes nx = ny =
{", ".join(str(r['nx']) for r in rows)}.  The mesh levels must be multiples of 4:
the patch edge sits at +-0.25 m, i.e. at {A/((A-PATCH[0])/1)*0.5:.0f}/nx of the plate,
so nx = 6 would put it 1.5 element widths from the centre line and
`patch_plate_mesh` rejects that mesh (see `assert_patch_compatible`).

Two paths per mesh: H = {HARDENING/1e9:.0f} GPa with unloading (first yield,
permanent set) and H = {IDEAL_HARDENING/1e6:.0f} MPa without unloading (collapse
against the rigid-plastic bound).

## Results

{markdown_table(rows, bounds)}
![mesh convergence]({FIG_NAME})

## Verdicts (anchor GAP-PP-3 "3 档网格收敛")

| criterion | value | threshold | verdict |
|---|---|---|---|
| collapse is a measured limit load, not the ramp end | {summary['collapse_is_measured_limit_load']} | required | {verdict(summary['collapse_is_measured_limit_load'])} |
| collapse limit-load monotone non-decreasing | {summary['collapse_monotone']} | monotone | {verdict(summary['collapse_monotone'])} |
| collapse below yield-line upper bound | {summary['collapse_below_upper']} | below | {verdict(summary['collapse_below_upper'])} |
| collapse above strip lower bound | {summary['collapse_above_lower']} | above | {verdict(summary['collapse_above_lower'])} |
| collapse vs upper bound | {summary['collapse_vs_upper_max_abs_pct']:.2f} % | < 10 % | {verdict(summary['collapse_vs_upper_max_abs_pct'] < 10)} |
| limit-load measurement spread (tangent vs 2x-deflection) | {summary['collapse_measurement_spread_max_pct']:.2f} % | < 30 % | {verdict((summary['collapse_measurement_spread_max_pct'] or 100) < 30)} |
| transition vs same-mesh elastic field | {summary['transition_vs_same_mesh_max_abs_pct']:.2f} % | < 5 % | {verdict(summary['transition_vs_same_mesh_max_abs_pct'] < 5)} |
| transition vs analytic yield load | {summary['transition_vs_analytic_max_abs_pct']:.2f} % | < 5 % | {verdict(summary['transition_vs_analytic_max_abs_pct'] < 5)} |
| permanent set valid (unload leg converged) | {summary['permanent_set_valid']} | required | {verdict(summary['permanent_set_valid'])} |
| permanent set spread over the meshes | {summary['permanent_set_spread_pct']:.2f} % | < 50 % | {verdict(summary['permanent_set_spread_pct'] < 50)} |
| permanent set positive and below peak deflection | {summary['permanent_set_bounded']} | bounded | {verdict(summary['permanent_set_bounded'])} |

## Notes

* **The ramp end is not a collapse load.**  With H = {HARDENING/1e9:.0f} GPa under load
  control the Newton iteration keeps converging past 3x yield, so
  `PatchPath.collapse_factor` stays `None`; an earlier revision of this sweep
  reported the last prescribed factor as the collapse load, which made the
  "convergence" of two different meshes identical to four digits.  The collapse
  column above is the limit load *measured on the path*
  (`limit_load_from_path`, tangent intersection), on a near-ideally plastic path
  so that it is comparable with the rigid-plastic yield-line bound.
* The through-thickness layering artefact (outermost stress sample at
  |z| = t/2 (1 - 1/n)) is quoted as a separate corrected column; it is **not**
  used for the bound comparison, because the path-measured limit load already
  comes from the global stiffness and applying the stress-sampling factor on top
  of it double counts.  Both variants are printed so the reader can see the
  difference.
* The **transition vs analytic** criterion stays outside 5 % by construction of
  the element: the layered Shell4 elastic field carries a mesh-independent
  surface-stress bias (see the parent case document), so the honest comparison
  is against the same-mesh elastic field.
* The measurement spread of the limit load (tangent intersection vs the
  conventional 2x-deflection rule) is published with every row: only a path
  whose definitions agree has a credible limit load.
"""


def _figure(rows: list[dict], bounds: dict) -> Path:
    # imported lazily: the compute nodes that run the sweep have torch but no
    # matplotlib, and the sweep rows are cached so the figure can be rendered
    # later on a workstation (``--levels`` all cached -> seconds).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    upper = bounds["yield_line_upper_Pa"]
    lower = bounds["strip_lower_Pa"]
    first_analytic = bounds["analytic_first_yield_Pa"]
    nx = [r["nx"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(nx, [r["collapse_factor"] / 1e6 for r in rows], "o-",
            label="collapse (path-measured limit load)")
    ax.plot(nx, [r["ramp_end_factor"] / 1e6 for r in rows], "x--", color="C7", alpha=0.8,
            label="ramp end (NOT a collapse load)")
    ax.plot(nx, [r["first_yield_factor"] / 1e6 for r in rows], "s--", label="first yield")
    ax.axhline(upper / 1e6, color="r", ls=":", label="yield-line upper bound")
    ax.axhline(lower / 1e6, color="g", ls=":", label="strip lower bound")
    ax.axhline(first_analytic / 1e6, color="k", ls="--", label="analytic first yield")
    ax.set_xlabel("mesh level (nx = ny)")
    ax.set_ylabel("pressure [MPa]")
    ax.set_title("(a) capacity vs mesh")
    ax.legend(fontsize=7)
    ax.grid(True)

    ax = axes[1]
    ax.plot(nx, [1e3 * r["permanent_deflection"] for r in rows], "^-", color="C4",
            label="permanent set")
    ax.plot(nx, [1e3 * r["peak_centre_deflection"] for r in rows], "v--", color="C4",
            alpha=0.5, label="peak centre deflection")
    ax.set_xlabel("mesh level (nx = ny)")
    ax.set_ylabel("deflection [mm]")
    ax.set_title("(b) permanent set vs mesh")
    ax.legend(fontsize=7)
    ax.grid(True)

    ax = axes[2]
    ax.plot(nx, [100 * r["first_yield_error_vs_same_mesh"] for r in rows], "s--",
            label="transition vs same-mesh")
    ax.plot(nx, [100 * r["first_yield_error_vs_analytic"] for r in rows], "o-",
            label="transition vs analytic")
    ax.plot(nx, [100 * r["collapse_error_vs_upper_corrected"] for r in rows], "^-",
            label="collapse (corrected) vs upper")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhline(5.0, color="orange", ls=":", label="5 % tolerance")
    ax.axhline(-5.0, color="orange", ls=":")
    ax.set_xlabel("mesh level (nx = ny)")
    ax.set_ylabel("relative error [%]")
    ax.set_title("(c) error vs mesh")
    ax.legend(fontsize=7)
    ax.grid(True)

    fig.suptitle("Plate-patch elastoplastic capacity: 3-level mesh convergence "
                 "(layered Shell4, patch 0.25 of the area)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / FIG_NAME
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def report_payload(rows: list[dict], bounds: dict, summary: dict,
                   fig_path: Path, mission_id: int, job_id: str = "local") -> dict:
    """Compose the progress-report payload; numbers and figure are injected."""
    table = markdown_table(rows, bounds)
    counters = lambda ok: "**是**" if ok else "**否**"
    levels_txt = "/".join(str(r["nx"]) for r in rows)
    md = f"""# GAP-PP-3-ICE-STRUCT-PLASTIC-SET 板格低温钢弹塑性：3 档网格收敛（{levels_txt}）

## ① 案例介绍

冰-结构在大斑（patch）载荷下的真实失效模式是塑性储备与永久挠度，而非弹性应力上限。锚点 GAP-PP-3 要求板格弹塑性（等向硬化 + 低温流变应力）在 3 档网格上收敛：collapse 单调逼近上限解、弹性→塑性过渡点对同网格弹性场 <5%、永久挠度有界。本单把此前只跑通单网格的验收补成 {levels_txt} 三档的真实收敛证据；原计划的 nx=6 档因 patch 边缘（±0.25 m）落在 1.5 个单元宽处、`patch_plate_mesh` 正确拒绝该网格而**不可用**，故取 4/8/12（nx 为 4 的倍数）——这是本单查出并修掉的缺陷。

| 参数 | 值 |
|---|---|
| 板 a×b | {A:.2f} m × {B:.2f} m |
| 板厚 t | {T*1000:.0f} mm (t/b = {T/B:.3f}) |
| patch | {PATCH[0]:.2f} m × {PATCH[1]:.2f} m（中心，占面积 {PATCH[0]*PATCH[1]/(A*B):.2f}） |
| E / ν | {E/1e9:.0f} GPa / {NU} |
| 屈服 σy | {FY/1e6:.0f} MPa |
| 硬化 H | {HARDENING/1e9:.0f} GPa（等向，永久挠度用）/ {IDEAL_HARDENING/1e6:.0f} MPa（近理想塑性，collapse 用） |
| 分层 | {LAYERS} 层（layered Shell4，J2 平面应力） |
| 网格 | nx = ny = {", ".join(str(r['nx']) for r in rows)} |

## ② 怎么做

- 复用已合并模块 `src/tensorfem/plate_patch_plastic_limit.py`（layered Shell4 J2、等向硬化、低温流变应力表）与闭式参考解（Navier 双级数 `elastic_patch_plate` / `analytic_first_yield_pressure`、yield-line 上限 `yield_line_patch_upper_bound`、条带下限 `strip_lower_bound`）。
- `scripts/patch_plate_mesh_sweep.py`：每档网格跑**两条**路径——(i) H={HARDENING/1e9:.0f} GPa 加载至 3.0× 解析屈服载荷后卸载，读 first-yield 与永久挠度；(ii) H={IDEAL_HARDENING/1e6:.0f} MPa 近理想塑性、不卸载，用 `limit_load_from_path`（切线交点 + 2 倍挠度规则）在路径上**实测**极限载荷，作为 collapse。(iii) 另做一次小弹性步探测同网格弹性偏置。
- 网格档位合法性在跑之前就校验（`assert_patch_compatible`），非法档位快速失败并给出原因，不会静默留空格。
- 收敛判据由脚本从算得的行**推导**（`convergence_summary`），不是硬编码；表格、案例文档与本文的数字全部由脚本注入（`markdown_table`），零手抄。
- 工具链：Python + torch + matplotlib；三档扫描递交集群调度（作业号 {job_id}），逐档结果缓存后由本脚本汇总。

## ③ 计算结果（程序化注入）

{table}

## ④ 结果图

![mesh convergence]({FIG_NAME})

三面板：(a) 承载力 vs 网格并叠加解析屈服、yield-line 上限与条带下限；(b) 永久挠度与峰値中心挠度 vs 网格；(c) 各项相对误差 vs 网格（含 5% 容差线）。由 `scripts/patch_plate_mesh_sweep.py` 以 matplotlib 生成。

## ⑤ 注意点

- **加载末端不是 collapse**：H={HARDENING/1e9:.0f} GPa 时载荷控制下 Newton 一直收敛到 3.0× 屈服仍不失稳，`PatchPath.collapse_factor` 为 None。上一版把「末端 prescribed 因子」当 collapse 发布，两档网格因此得到四位相同的 2.4123 MPa（=3.0×0.8041 MPa）——这是本单查出的最严重缺陷。现在 collapse 一律取路径实测极限载荷，脚本并断言 `collapse != ramp_end`。
- 分层采样：最外层应力采样点在 |z| = t/2(1-1/n)，n={LAYERS} 时应力层面高估 1/(1-1/n) = {1/(1-1/LAYERS):.2f}×；该修正只作为**并列诊断列**给出，不参与上下界判定（路径实测极限载荷来自整体刚度，再乘该因子会重复计入）。
- layered Shell4 有网格无关的弹性偏置（本算例最大 {100*max(abs(r['elastic_bias']) for r in rows):.2f}%），因此过渡点对**解析**屈服载荷的偏差落在 {100*min(r['first_yield_error_vs_analytic'] for r in rows):+.2f}%~{100*max(r['first_yield_error_vs_analytic'] for r in rows):+.2f}% 区间，属元素级精度欠账而非本算例收敛问题；对**同网格弹性场**则 {summary['transition_vs_same_mesh_max_abs_pct']:.2f}%（最大）。
- 小转角壳不捕获膜加劲，collapse 为纯弯曲机制值；载荷控制，未做弧长法追踪软化段，故近理想塑性路径可能停在极限点而非软化段。
- 极限载荷的两种定义（切线交点 / 2 倍挠度）离散度逐档给出（`collapse spread`）：离散度大说明路径分辨率或定义差异大，此时极限载荷只应作区间看待。
- 无独立实验基准，基准为闭式解析（Navier / yield-line / 条带下限），已显式声明。

## ⑥ 误差判定（逐项）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 弹性→塑性过渡点 | 同网格弹性场 | 最大 {summary['transition_vs_same_mesh_max_abs_pct']:.2f}% | {counters(summary['transition_vs_same_mesh_max_abs_pct'] < 3)} |
| 弹性→塑性过渡点 | 解析屈服载荷 | 最大 {summary['transition_vs_analytic_max_abs_pct']:.2f}% | {counters(summary['transition_vs_analytic_max_abs_pct'] < 3)} |
| collapse（路径实测极限载荷） | yield-line 上限 | 最大 {summary['collapse_vs_upper_max_abs_pct']:.2f}% | {counters(summary['collapse_vs_upper_max_abs_pct'] < 3)} |
| collapse（分层修正诊断列） | yield-line 上限 | 最大 {summary['collapse_vs_upper_max_abs_pct_corrected']:.2f}% | {counters(summary['collapse_vs_upper_max_abs_pct_corrected'] < 3)} |
| 永久挠度网格离散 | 三档互比 spread {summary['permanent_set_spread_pct']:.2f}% | — | {counters(summary['permanent_set_spread_pct'] < 3)} |

锚点收敛判据（collapse 单调、介于上下界、过渡 <5%、永久挠度有界）逐条：{", ".join(f"{k}={'met' if v else 'NOT met'}" for k, v in summary["acceptance"].items())}。

3% 档：见上表两列「是否 <3%」——**未达标的关键量如实标「否」**（闭式上限解与带元素弹性偏置的过渡点都不在 3% 内），不调优、不粉饰；<5%/<10% 档按锚点判据逐条列出 met / NOT met。
"""
    return {
        "title": (f"GAP-PP-3-ICE-STRUCT-PLASTIC-SET 板格低温钢弹塑性：{levels_txt} 三档网格收敛实测"),
        "summary": (f"TensorFEM plate_patch_plastic_limit 补齐锚点「3 档网格收敛」欠账："
                    f"nx=ny={levels_txt} 三档，collapse 改为路径实测极限载荷（最大偏差 "
                    f"{summary['collapse_vs_upper_max_abs_pct']:.2f}% vs yield-line 上限）、过渡点对同网格弹性场 "
                    f"{summary['transition_vs_same_mesh_max_abs_pct']:.2f}%、永久挠度 spread "
                    f"{summary['permanent_set_spread_pct']:.2f}%；同时修掉「把加载末端当 collapse 发布」"
                    f"与「nx=6 网格档位不可用却写进报告」两个缺陷。"),
        "scenario_key": "GAP-PP-3-ICE-STRUCT-PLASTIC-SET",
        "mission_id": mission_id,
        "tags": ["TensorFEM", "plate", "elastoplastic", "ice-structure",
                 "mesh-convergence", "verification"],
        "markdown": md,
        "figures": [{
            "name": FIG_NAME,
            "caption": ("板格弹塑性承载力 3 档网格收敛：承载力 vs 解析上下界、永久挠度、"
                        "以及各项相对误差（含 5% 容差线）"),
            "png_base64": base64.b64encode(fig_path.read_bytes()).decode(),
        }],
    }


def assert_patch_compatible(levels) -> None:
    """Patch edges must coincide with element edges at every level (fail fast)."""
    bad = []
    for nx in levels:
        h = A / nx
        edge = (A - PATCH[0]) / 2.0
        if abs(edge / h - round(edge / h)) > 1e-9:
            bad.append(nx)
    if bad:
        raise ValueError(
            f"mesh levels {bad} place the patch edge off the element grid "
            f"(patch {PATCH[0]:.3f} m needs nx divisible by "
            f"{int(round(A / (A - PATCH[0]))) * 2}); use {[n for n in levels if n not in bad]}"
            f" or change the patch")


def _level_rows(levels, py) -> list[dict]:
    """Run (or reuse) the per-mesh rows; the sweep may be run level by level."""
    cached = {}
    if ROWS_CACHE.exists():
        try:
            cached = {row["nx"]: row for row in json.loads(ROWS_CACHE.read_text())}
        except (json.JSONDecodeError, KeyError):
            cached = {}
    for nx in levels:
        if nx not in cached:
            cached[nx] = _run_level(nx, py)
            ART.mkdir(parents=True, exist_ok=True)
            ROWS_CACHE.write_text(json.dumps(list(cached.values()), indent=2))
    return list(cached.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission-id", type=int, default=20041)
    parser.add_argument("--report-out", default=str(REPORT),
                        help="where to write the report payload (repo record)")
    parser.add_argument("--job-id", default=os.environ.get("COMPUTE_JOB_ID", "local"),
                        help="scheduler job that produced the rows, for the report")
    parser.add_argument("--levels", type=int, nargs="+", default=list(MESH_LEVELS),
                        help="mesh levels to run now; results are cached per level so a "
                             "long sweep can be sharded over scheduler jobs")
    args = parser.parse_args()

    upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"]
    lower = strip_lower_bound(A, B, T, FY, PATCH)["pressure"]
    py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=11, terms=31)
    bounds = {"analytic_first_yield_Pa": py["pressure"],
              "yield_line_upper_Pa": upper, "strip_lower_Pa": lower}

    assert_patch_compatible(args.levels)
    rows = _level_rows(tuple(args.levels), py)
    pending = [nx for nx in MESH_LEVELS if nx not in {r["nx"] for r in rows}]
    if pending:
        print(f"levels still pending: {pending} - rerun with --levels {' '.join(map(str, pending))}")
        return 0
    rows = sorted(rows, key=lambda r: MESH_LEVELS.index(r["nx"]))
    summary = convergence_summary(rows, bounds)

    # --- invariants (a measurement, not a predetermined answer) --------------
    assert summary["collapse_is_measured_limit_load"], (
        "collapse must be a path-measured limit load, not the end of the load ramp: "
        f"{[(r['nx'], r['collapse_factor'] / 1e6, r['ramp_end_factor'] / 1e6) for r in rows]}")
    assert summary["permanent_set_valid"], (
        "the unload leg must converge for the permanent set to be quoted")
    assert summary["permanent_set_bounded"], (
        "permanent set must be positive and below the peak deflection")

    # --- acceptance verdicts (reported, never tuned) --------------------------
    for name, ok in summary["acceptance"].items():
        print(f"ACCEPTANCE {name}: {'met' if ok else 'NOT met'}")
    if not all(summary["acceptance"].values()):
        print("WARNING: some acceptance criteria are NOT met - the case note and the report "
              "publish them as they are (honest FAIL, nothing is tuned).")

    fig_path = _figure(rows, bounds)

    out = {
        "geometry": {"a": A, "b": B, "t": T, "patch": PATCH},
        "material": {"E": E, "nu": NU, "fy": FY,
                     "hardening": HARDENING, "layers": LAYERS},
        "bounds": bounds,
        "mesh_levels": rows,
        "convergence": summary,
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "patch_plate_mesh_sweep.json").write_text(json.dumps(out, indent=2))
    DOC.write_text(case_doc(rows, bounds, summary))
    Path(args.report_out).write_text(json.dumps(report_payload(rows, bounds, summary, fig_path,
                                                args.mission_id, args.job_id), indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())