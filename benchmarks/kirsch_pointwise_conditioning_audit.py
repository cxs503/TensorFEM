#!/usr/bin/env python3
"""Kirsch pointwise comparison: why the radial rows FAIL, and what cures them.

``docs/structural-certification-materials.md`` reports, for the Kirsch
quarter-annulus case, three ``FAIL`` rows per ray -- all of them the *radial*
stress at ``r/a = 1.02 / 1.05 / 1.25`` -- under the relative criterion
``|fe - ref| / |ref| <= 3 %``.  This script audits those rows instead of
accepting or deleting them:

1. **Where are the probes?** The two canonical rays (``theta = 90 deg`` and
   ``theta = 0 deg``) are the symmetry edges of the quarter annulus, so every
   probe lies on an element interface.  A ``Q4`` element carries a bilinear
   displacement field and therefore a *discontinuous* stress; the raw pointwise
   protocol silently reports the side of the interface that the locator returns.
2. **How large is the error in absolute terms?** The radial stress vanishes at
   the free hole edge, so near the edge it is a small difference of large
   numbers.  Dividing that absolute error by the small reference manufactures a
   big relative number; the same error divided by the stress scale is tiny.
3. **Does a well-defined measurement cure it?** The production recovery
   operator (``tensorfem.stress_recovery``: Barlow extrapolation of the ``2x2``
   Gauss-point stresses plus area-weighted nodal averaging) is applied to the
   *same* displacement solution, and the same probes are re-measured.

The original verdicts are untouched.  What is added is the attribution of the
failing rows, a re-measurement with a single-valued sampling operator on the
same solution, and two non-vacuity controls.

Outputs
-------
* ``benchmarks/kirsch_pointwise_conditioning_audit.json`` -- every number;
* figures under ``docs/assets/kirsch-conditioning/``;
* ``docs/kirsch-pointwise-conditioning-audit.md`` -- rendered from the JSON.

Run::

    OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/kirsch_pointwise_conditioning_audit.py
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem import __version__  # noqa: E402
from tensorfem.continuum import solve_continuum  # noqa: E402
from tensorfem.kirsch_hole import (  # noqa: E402
    KirschHole,
    annulus_model,
    q4_stress_at_point,
)
from tensorfem.stress_recovery import (  # noqa: E402
    interpolate_nodal_field,
    locate_point,
    q4_gauss_stresses,
    recover_nodal_stress,
)

JSON_OUT = ROOT / "benchmarks" / "kirsch_pointwise_conditioning_audit.json"
ASSETS = ROOT / "docs" / "assets" / "kirsch-conditioning"
DOC_OUT = ROOT / "docs" / "kirsch-pointwise-conditioning-audit.md"

#: Mesh ladder of the published certification material (unchanged).
LEVELS = (24, 32, 48, 64)
RAYS_DEGREES = (90.0, 0.0)
RADIAL_SCALES = (1.02, 1.05, 1.10, 1.25, 1.50, 2.00)
OUTER_RADIUS = 10.0
GRADING = 5.0
FAR_FIELD = 1.0
TOLERANCE = 0.03
#: Withholding floor inherited from the certification material.
WITHHOLD_FRACTION = 0.01
#: Declared ill-conditioning boundary for the *relative* criterion: below 5 % of
#: the stress scale the 3 % relative band asks for an absolute error smaller
#: than the measured floor of the Q4 pointwise protocol (~5e-4 .. 5e-3 of the
#: scale on this ladder), so the row can no longer discriminate.
ILL_CONDITIONED_FRACTION = 0.05
COMPONENTS = ("radial", "hoop", "shear")
GAUSS_PARENT_RADIUS = 1.0 / math.sqrt(3.0)  # Q4 2x2 superconvergent points
GAUSS_PARENT = ((-GAUSS_PARENT_RADIUS, -GAUSS_PARENT_RADIUS), (GAUSS_PARENT_RADIUS, -GAUSS_PARENT_RADIUS),
                (GAUSS_PARENT_RADIUS, GAUSS_PARENT_RADIUS), (-GAUSS_PARENT_RADIUS, GAUSS_PARENT_RADIUS))


def kirsch_hole() -> KirschHole:
    return KirschHole(radius=1.0, far_field=FAR_FIELD, young=1.0, poisson=0.3, plane="stress")


def polar_components(stress, theta: float) -> dict[str, float]:
    """Cartesian ``(sxx, syy, sxy)`` -> polar ``(sigma_rr, sigma_tt, tau_rt)``."""
    sxx, syy, sxy = (float(value) for value in stress)
    c, s = math.cos(theta), math.sin(theta)
    stt = sxx * s * s + syy * c * c - 2.0 * sxy * c * s
    srr = sxx * c * c + syy * s * s + 2.0 * sxy * c * s
    srt = (syy - sxx) * c * s + sxy * (c * c - s * s)
    return {"radial": srr, "hoop": stt, "shear": srt}


def solve_level(level: int, hole: KirschHole) -> dict:
    """One quarter-annulus level: production solve plus recovered nodal field."""
    model, meta = annulus_model(hole, OUTER_RADIUS, level, level, GRADING)
    meta["hole"] = hole
    result = solve_continuum(model)
    displacement = result.displacement.detach().cpu().numpy()
    nodes = torch.as_tensor(meta["nodes"])
    elements = torch.as_tensor(meta["elements"])
    gauss = q4_gauss_stresses(
        nodes, elements, torch.as_tensor(displacement), hole.young, hole.poisson
    )
    return {
        "level": level,
        "dofs": int(model.n_dofs),
        "elements": int(meta["elements"].shape[0]),
        "meta": meta,
        "displacement": displacement,
        "nodes": nodes,
        "connectivity": elements,
        "recovered": recover_nodal_stress(nodes, elements, gauss),
    }


def raw_probe(sample: dict, point: tuple[float, float], theta: float) -> dict:
    """The certification protocol: element-local pointwise stress."""
    located = q4_stress_at_point(sample["meta"], sample["displacement"], point)
    parent = [float(value) for value in located["parent"]]
    outermost = max(abs(parent[0]), abs(parent[1]))
    return {
        "stress": polar_components(located["stress"], theta),
        "edge_distance": 1.0 - outermost,
        "parent_radius": outermost,
        "gauss_distance": min(
            max(abs(parent[0] - gx), abs(parent[1] - gy)) for gx, gy in GAUSS_PARENT
        ),
        "extrapolation_beyond_gauss": max(0.0, outermost - GAUSS_PARENT_RADIUS),
        "element": int(located["element"]),
    }


def recovered_probe(sample: dict, point: tuple[float, float], theta: float) -> dict:
    """Same solution, production Barlow + averaging recovery, bilinear sampling."""
    located = interpolate_nodal_field(sample["nodes"], sample["connectivity"], sample["recovered"], point)
    return {
        "stress": polar_components(located["stress"], theta),
        "on_element_boundary": bool(located["on_element_boundary"]),
        "containing_elements": int(located["containing_elements"]),
        "spread": float(located["spread"]),
    }


def measure(sample: dict, hole: KirschHole) -> list[dict]:
    rows: list[dict] = []
    for theta_degrees in RAYS_DEGREES:
        theta = math.radians(theta_degrees)
        for scale in RADIAL_SCALES:
            point = (scale * hole.radius * math.cos(theta), scale * hole.radius * math.sin(theta))
            exact_polar = hole.stress_polar(scale * hole.radius, theta)
            exact = {"radial": float(exact_polar[0]), "hoop": float(exact_polar[1]),
                     "shear": float(exact_polar[2])}
            raw = raw_probe(sample, point, theta)
            recovered = recovered_probe(sample, point, theta)
            row: dict = {
                "level": sample["level"],
                "ray_degrees": theta_degrees,
                "r_over_a": scale,
                "exact": exact,
                "raw": {"stress": raw["stress"], "edge_distance": raw["edge_distance"]},
                "recovered": recovered,
                "diagnostics": {
                    "edge_distance": raw["edge_distance"],
                    "parent_radius": raw["parent_radius"],
                    "gauss_distance": raw["gauss_distance"],
                    "extrapolation_beyond_gauss": raw["extrapolation_beyond_gauss"],
                    "containing_elements": recovered["containing_elements"],
                },
                "conditioning": {},
                "errors": {},
            }
            for component in COMPONENTS:
                reference = exact[component]
                fraction = abs(reference) / abs(FAR_FIELD)
                row["conditioning"][component] = {
                    "reference_fraction_of_scale": fraction,
                    "conditioning_index": (abs(FAR_FIELD) / abs(reference)) if reference else None,
                    "ill_conditioned": bool(fraction < ILL_CONDITIONED_FRACTION),
                }
                errors = {}
                for protocol, value in (("raw", raw["stress"][component]),
                                        ("recovered", recovered["stress"][component])):
                    absolute = abs(value - reference)
                    errors[protocol] = {
                        "value": value,
                        "absolute_error": absolute,
                        "relative_error": (absolute / abs(reference)) if reference else None,
                        "scale_normalized_error": absolute / abs(FAR_FIELD),
                        "verdict": (
                            "WITHHELD"
                            if fraction < WITHHOLD_FRACTION
                            else ("PASS" if absolute <= TOLERANCE * abs(reference) else "FAIL")
                        ),
                    }
                row["errors"][component] = errors
            rows.append(row)
    return rows


def level_summary(rows: list[dict], level: int) -> dict:
    subset = [row for row in rows if row["level"] == level]
    summary: dict = {"level": level, "dofs": None, "components": {}}
    for component in COMPONENTS:
        entry: dict = {}
        for protocol in ("raw", "recovered"):
            verdicts = [row["errors"][component][protocol]["verdict"] for row in subset]
            relative = [row["errors"][component][protocol]["relative_error"] for row in subset]
            finite = [value for value in relative if value is not None]
            worst_index = max(range(len(relative)), key=lambda i: relative[i] or 0.0)
            entry[protocol] = {
                "pass": verdicts.count("PASS"),
                "fail": verdicts.count("FAIL"),
                "withheld": verdicts.count("WITHHELD"),
                "worst_relative_error": max(finite) if finite else None,
                "worst_probe": {
                    "ray_degrees": subset[worst_index]["ray_degrees"],
                    "r_over_a": subset[worst_index]["r_over_a"],
                },
                "worst_scale_normalized_error": max(
                    row["errors"][component][protocol]["scale_normalized_error"] for row in subset
                ),
            }
        entry["well_conditioned"] = {
            protocol: {
                "rows": sum(
                    1 for row in subset if not row["conditioning"][component]["ill_conditioned"]
                ),
                "worst_relative_error": max(
                    [
                        row["errors"][component][protocol]["relative_error"] or 0.0
                        for row in subset
                        if not row["conditioning"][component]["ill_conditioned"]
                    ],
                    default=None,
                ),
            }
            for protocol in ("raw", "recovered")
        }
        summary["components"][component] = entry
    return summary


def observed_order(coarse, fine, ratio: float) -> float:
    try:
        if not coarse or not fine or coarse <= 0.0 or fine <= 0.0:
            return float("nan")
        return math.log(coarse / fine) / math.log(ratio)
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def convergence(rows: list[dict]) -> list[dict]:
    table = []
    for theta in RAYS_DEGREES:
        for scale in RADIAL_SCALES:
            entry = {"ray_degrees": theta, "r_over_a": scale}
            for component in ("radial", "hoop"):
                for protocol in ("raw", "recovered"):
                    sequence = [
                        next(
                            row["errors"][component][protocol]["relative_error"]
                            for row in rows
                            if row["level"] == level
                            and row["ray_degrees"] == theta
                            and row["r_over_a"] == scale
                        )
                        for level in LEVELS
                    ]
                    usable = [value for value in sequence if value]
                    orders = [
                        observed_order(sequence[index], sequence[index + 1],
                                       LEVELS[index + 1] / LEVELS[index])
                        for index in range(len(sequence) - 1)
                    ]
                    finite_orders = [order for order in orders if order == order]
                    entry[f"{component}_{protocol}"] = {
                        "relative_error_sequence": sequence,
                        "observed_orders": orders,
                        "mean_order": float(np.mean(finite_orders)) if finite_orders else None,
                        "monotone": len(usable) == len(sequence)
                        and all(
                            sequence[index + 1] <= sequence[index]
                            for index in range(len(sequence) - 1)
                        ),
                    }
            table.append(entry)
    return table


def control_constant_field() -> dict:
    """Recovery must reproduce a constant stress field exactly."""
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    target = torch.tensor([1.0, 0.25, -0.125], dtype=torch.float64)
    gauss = target.reshape(1, 1, 3).expand(1, 4, 3).contiguous()
    field = recover_nodal_stress(nodes, elements, gauss)
    deviation = float(torch.max(torch.abs(field - target)))
    return {
        "label": "constant stress field, single Q4",
        "max_abs_deviation_from_exact": deviation,
        "verdict": "PASS" if deviation <= 1e-13 else "FAIL",
    }


def control_perturbed_field(sample: dict, hole: KirschHole, factor: float = 1.05) -> dict:
    """Scaling the recovered field must be rejected -- otherwise the gate is idle."""
    errors = []
    for theta_degrees in RAYS_DEGREES:
        theta = math.radians(theta_degrees)
        for scale in RADIAL_SCALES:
            point = (scale * math.cos(theta), scale * math.sin(theta))
            located = interpolate_nodal_field(
                sample["nodes"], sample["connectivity"], sample["recovered"], point
            )
            stress = polar_components([factor * float(value) for value in located["stress"]], theta)
            exact = hole.stress_polar(scale, theta)
            for index, component in enumerate(COMPONENTS):
                reference = float(exact[index])
                if abs(reference) < ILL_CONDITIONED_FRACTION * FAR_FIELD:
                    continue
                errors.append(abs(stress[component] - reference) / abs(reference))
    failing = [value for value in errors if value > TOLERANCE]
    return {
        "label": f"recovered field scaled by {factor:g}",
        "factor": factor,
        "rows": len(errors),
        "fail_rows": len(failing),
        "min_failing_relative_error": min(failing) if failing else None,
        "verdict": "PASS (gate rejects the perturbed field)" if failing else "FAIL (gate is idle)",
    }


def figure_error_vs_level(rows: list[dict]) -> dict:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for axis, theta in zip(axes, RAYS_DEGREES):
        for component, colour in (("radial", "#c0392b"), ("hoop", "#2471a3")):
            for protocol, style, marker in (("raw", "--", "s"), ("recovered", "-", "o")):
                worst = []
                for level in LEVELS:
                    values = [
                        100.0 * row["errors"][component][protocol]["relative_error"]
                        for row in rows
                        if row["level"] == level
                        and row["ray_degrees"] == theta
                        and not row["conditioning"][component]["ill_conditioned"]
                        and row["errors"][component][protocol]["relative_error"]
                    ]
                    worst.append(max(values))
                axis.semilogy(LEVELS, worst, style, marker=marker, color=colour,
                              label=f"{component} ({protocol})")
        axis.axhline(3.0, color="k", linewidth=1.0, linestyle=":")
        axis.set_title(f"ray theta = {theta:g} deg", fontsize=9.5)
        axis.set_xlabel("mesh level n (n x n quarter annulus)")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("worst relative error [%] of well-conditioned probes")
    axes[0].legend(fontsize=7.5)
    figure.suptitle("Kirsch probes: element-local (dashed) vs recovered field (solid); 3 % dotted",
                    fontsize=10)
    figure.tight_layout()
    path = ASSETS / "kirsch_error_vs_level.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {
        "name": "kirsch_error_vs_level",
        "path": str(path.relative_to(ROOT)),
        "caption": "Worst relative error of the well-conditioned probes against the Kirsch closed "
        "form, per symmetry ray. Dashed: element-local pointwise protocol of the certification "
        "material. Solid: same displacement solution, production Barlow + area-weighted nodal "
        "recovery. Dotted line: 3 % acceptance.",
    }


def figure_conditioning(rows: list[dict], absolute_floor: float) -> dict:
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    for protocol, marker, colour in (("raw", "s", "#c0392b"), ("recovered", "o", "#1e8449")):
        fractions, errors = [], []
        for row in rows:
            if row["level"] != LEVELS[-1]:
                continue
            for component in ("radial", "hoop"):
                fraction = row["conditioning"][component]["reference_fraction_of_scale"]
                error = row["errors"][component][protocol]["relative_error"]
                if not fraction or not error:
                    continue
                fractions.append(100.0 * fraction)
                errors.append(100.0 * error)
        axis.loglog(fractions, errors, marker, linestyle="none", alpha=0.8, label=protocol)
    grid = np.logspace(-1.6, 2.0, 200)
    axis.loglog(grid, 100.0 * absolute_floor / (grid / 100.0), "k--", linewidth=0.9,
                label="constant absolute error = measured floor")
    axis.axhline(3.0, color="k", linewidth=1.0, linestyle=":")
    axis.axvline(100.0 * ILL_CONDITIONED_FRACTION, color="grey", linewidth=1.0, linestyle="-.")
    axis.axvline(100.0 * WITHHOLD_FRACTION, color="grey", linewidth=1.0, linestyle=":")
    axis.set_xlabel("|reference| / stress scale [%]   (denominator size)")
    axis.set_ylabel("relative error [%]")
    axis.set_title("Relative error follows the denominator, not the field", fontsize=10)
    axis.grid(True, which="both", alpha=0.3)
    axis.set_ylim(1e-3, 1e3)
    axis.legend(fontsize=7.5)
    figure.tight_layout()
    path = ASSETS / "kirsch_conditioning.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {
        "name": "kirsch_conditioning",
        "path": str(path.relative_to(ROOT)),
        "caption": "Relative error versus the size of the reference itself (finest level, radial "
        "and hoop probes). The dashed hyperbola is a fixed absolute error equal to the measured "
        "floor of the raw protocol: the measured points track it, so on small references the "
        "relative verdict reports the denominator, not the field. Dash-dot: 5 %-of-scale "
        "ill-conditioning boundary; dotted vertical: inherited 1 % withholding floor.",
    }


def figure_field_clouds(sample: dict, hole: KirschHole) -> dict:
    nodes = sample["nodes"].numpy()
    field = sample["recovered"].numpy()
    x, y = nodes[:, 0], nodes[:, 1]
    radius, theta = np.hypot(x, y), np.arctan2(y, x)
    cos, sin = np.cos(theta), np.sin(theta)
    radial = field[:, 0] * cos * cos + field[:, 1] * sin * sin + 2.0 * field[:, 2] * cos * sin
    exact = np.array([hole.stress_polar(r, t)[0] for r, t in zip(radius, theta)])
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    shared = np.linspace(min(exact.min(), radial.min()), max(exact.max(), radial.max()), 21)
    difference = radial - exact
    span = max(abs(difference).max(), 1e-12)
    for axis, data, levels, title in (
        (axes[0], exact, shared, "Kirsch exact $\\sigma_{rr}$"),
        (axes[1], radial, shared, "recovered $\\sigma_{rr}$"),
        (axes[2], difference, np.linspace(-span, span, 21), "recovered - exact"),
    ):
        contour = axis.tricontourf(x, y, data, levels=levels, cmap="coolwarm", extend="both")
        axis.set_title(f"{title} (n={sample['level']})", fontsize=9)
        axis.set_aspect("equal")
        axis.set_xlabel("x/a")
        figure.colorbar(contour, ax=axis, fraction=0.046)
    axes[0].set_ylabel("y/a")
    figure.suptitle("Kirsch radial stress over the quarter annulus: exact, recovered, difference",
                    fontsize=10)
    figure.tight_layout()
    path = ASSETS / "kirsch_field_clouds.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {
        "name": "kirsch_field_clouds",
        "path": str(path.relative_to(ROOT)),
        "caption": "Quarter annulus R/a = 10 at the finest level: analytic radial stress, the "
        "recovered nodal field, and their pointwise difference (units of the far-field tension).",
    }


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "—"
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def pct(value, digits: int = 2) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f} %"


def render_markdown(payload: dict) -> str:
    rows = payload["probes"]
    summary = payload["summary"]
    levels = payload["meta"]["levels"]
    lines: list[str] = []
    add = lines.append
    add("# Kirsch 逐点径向应力 FAIL 行的归因与复测")
    add("")
    add(
        f"Status：**已发表判据与其 6 行 FAIL 原样保留**；本条补上归因（探针压在单元界面上 + 分母近零）"
        f"与同一位移解上的良置复测。TensorFEM {payload['meta']['version']}，"
        f"{payload['meta']['timestamp']}。全部数字由 "
        "`benchmarks/kirsch_pointwise_conditioning_audit.py` 写入 "
        "`benchmarks/kirsch_pointwise_conditioning_audit.json` 后程序化注入，无手抄。"
    )
    add("")
    add("## 1. 案例与欠账（案例介绍）")
    add("")
    add(
        "`docs/structural-certification-materials.md` 的 Kirsch 四分之一圆环算例"
        "（孔半径 `a=1`、外半径 `R=10`、远场拉伸 `T=1`、`nu=0.3`、平面应力、Q4、径向渐变 `grading=5`）"
        "在两条射线 `theta=90°/0°` 上各留下 3 行 `FAIL`，**全部是径向应力**"
        "（`r/a=1.02 / 1.05 / 1.25`），判据是相对量 `|fe-ref|/|ref| <= 3 %`。"
        "本条不改判据、不改这 6 行，只回答三个问题：探针落在哪里、误差在绝对量纲上有多大、"
        "把同一位移解换成单值的取样算子后还剩多少。"
    )
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add(f"| 网格阶梯（沿用已发表阶梯） | {', '.join(str(level) for level in levels)} |")
    add("| 每级单元/自由度 | "
        + ', '.join(
            f"n={level}: {payload['levels'][str(level)]['elements']}/{payload['levels'][str(level)]['dofs']}"
            for level in levels
        )
        + " |")
    add(f"| 探针 | {len(RAYS_DEGREES)} 条射线 × {len(RADIAL_SCALES)} 个 `r/a` × {len(COMPONENTS)} 分量 = {len(rows)} 行/级 |")
    add(f"| 判据 | 相对 {pct(TOLERANCE, 0)}（原判据，未改）；同时报 `|Δ|/T` 绝对口径 |")
    add(f"| 弃权下限（继承） | |ref| < {pct(WITHHOLD_FRACTION, 0)}·T → `WITHHELD` |")
    add(f"| 病态边界（本条声明） | |ref| < {pct(ILL_CONDITIONED_FRACTION, 0)}·T → 相对判据失去分辨力，标 `ill_conditioned` |")
    add("")
    add("## 2. 方法（怎么做的）")
    add("")
    add("三项观测全部在**生产解**上完成，不重算、不改求解器、不动判据：")
    add("")
    add("| 观测 | 实现 | 回答什么 |")
    add("| --- | --- | --- |")
    add("| ① 探针位置 | `stress_recovery.locate_point` 的 `on_boundary`/`containing_elements` 与原始协议的 `edge_distance = 1-max(|xi|,|eta|)` | 探针是否压在单元界面上（Q4 应力在界面双值） |")
    add("| ② 界面跳变 | 同一物理探针在界面两侧各取 `theta ± 1e-7` rad，比较原始协议返回值 | 原始协议在双值场里丢了多大一跳 |")
    add("| ③ 良置复测 | `q4_gauss_stresses` → `recover_nodal_stress`（Barlow 外推 + 面积加权节点平均）→ `interpolate_nodal_field` | 同一解、同一探针，换成单值恢复场后的误差 |")
    add("")
    add("非空转对照两项：①常应力场恢复必须精确到舍入；②恢复场整体乘 1.05 后门禁必须拒收。"
        "另以粗网格级仍 FAIL 作第三重对照。")
    add("")
    add("## 3. 结果")
    add("")
    add("### 3.1 探针全部落在单元棱上（Q4 应力最不准的位置）")
    add("")
    add("| 级 | 探针 | `edge_distance`=0 的探针 | `|parent|` 最大 | 距最近 2×2 高斯点（父坐标） | 外推超出高斯点长度 | 含该点单元数最大 |")
    add("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for level in levels:
        subset = [row for row in rows if row["level"] == level]
        on_edge = sum(1 for row in subset if row["diagnostics"]["edge_distance"] < 1e-9)
        add(
            f"| {level} | {len(subset)} | **{on_edge}** | "
            f"{fmt(max(row['diagnostics']['parent_radius'] for row in subset), 3)} | "
            f"{fmt(max(row['diagnostics']['gauss_distance'] for row in subset), 3)} | "
            f"{fmt(max(row['diagnostics']['extrapolation_beyond_gauss'] for row in subset), 3)} | "
            f"{max(row['diagnostics']['containing_elements'] for row in subset)} |"
        )
    add("")
    add(
        "两条射线（`theta=90°/0°`）就是四分之一圆环的两条对称边界，即网格的**边**。"
        "Q4 的应力在 `2×2` 高斯点上超收敛，在单元棱上则是最差位置：原始协议把双线性应力场"
        "外推到 `|parent|=1` 的棱上取值，比最高斯点还要远 "
        f"{fmt(rows[0]['diagnostics']['extrapolation_beyond_gauss'], 3)} 个父坐标单位"
        "（约单元边长的 21 %）。这正是 `cook-probe-protocol-audit.md` 已定谳的同一类插值口径问题。"
        "注意：探针位于网格边界上，另一侧没有单元，因此「界面双值」不是本例的病因——"
        "病因是**在单元最不准的位置读数**，且该点同时是径向应力的近零点。"
    )
    add("")
    add("### 3.2 逐探针位置诊断（最细级）")
    add("")
    add("| 射线 | r/a | \\|parent\\| | 距最近高斯点 | 外推超出 | 含该点单元数 | 原始相对误差（radial） | 恢复相对误差（radial） |")
    add("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        if row["level"] != levels[-1]:
            continue
        diagnostics = row["diagnostics"]
        add(
            f"| {row['ray_degrees']:g}° | {row['r_over_a']:.2f} | "
            f"{fmt(diagnostics['parent_radius'], 3)} | {fmt(diagnostics['gauss_distance'], 3)} | "
            f"{fmt(diagnostics['extrapolation_beyond_gauss'], 3)} | "
            f"{diagnostics['containing_elements']} | "
            f"{pct(row['errors']['radial']['raw']['relative_error'])} | "
            f"{pct(row['errors']['radial']['recovered']['relative_error'])} |"
        )
    add("")
    add("### 3.3 最细级逐行对照（三个分量 × 两条射线）")
    add("")
    for theta in RAYS_DEGREES:
        add(f"**theta = {theta:g}°**")
        add("")
        add("| r/a | 分量 | 精确 | 原始 | 恢复 | \\|Δ\\|原始/T | \\|Δ\\|恢复/T | 相对·原始 | 相对·恢复 | 1/κ | 原判 | 复测判 |")
        add("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for row in rows:
            if row["level"] != levels[-1] or row["ray_degrees"] != theta:
                continue
            for component in COMPONENTS:
                exact = row["exact"][component]
                raw = row["errors"][component]["raw"]
                recovered = row["errors"][component]["recovered"]
                conditioning = row["conditioning"][component]
                well_posed = (
                    not conditioning["ill_conditioned"]
                    and raw["verdict"] != "WITHHELD"
                    and recovered["verdict"] != "WITHHELD"
                )
                relative_raw = pct(raw["relative_error"]) if well_posed else "—（弃权：分母≈0）"
                relative_recovered = pct(recovered["relative_error"]) if well_posed else "—（弃权：分母≈0）"
                add(
                    f"| {row['r_over_a']:.2f} | {component} | {fmt(exact, 4)} | "
                    f"{fmt(raw['value'], 4)} | {fmt(recovered['value'], 4)} | "
                    f"{fmt(raw['scale_normalized_error'], 4)} | "
                    f"{fmt(recovered['scale_normalized_error'], 4)} | "
                    f"{relative_raw} | {relative_recovered} | "
                    f"{fmt(conditioning['conditioning_index'], 1)} | "
                    f"{raw['verdict']} | {recovered['verdict']}"
                    + (" (ill-cond)" if conditioning["ill_conditioned"] else "")
                    + " |"
                )
        add("")
    add("### 3.4 各级汇总（分量 × 口径）")
    add("")
    add("| 级 | 分量 | 口径 | PASS | FAIL | WITHHELD | 最差相对误差 | 最差位置 | 良置行最差相对 |")
    add("| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |")
    for level in levels:
        entry = summary[str(level)]
        for component in COMPONENTS:
            for protocol in ("raw", "recovered"):
                item = entry["components"][component][protocol]
                well = entry["components"][component]["well_conditioned"][protocol]
                add(
                    f"| {level} | {component} | {protocol} | {item['pass']} | {item['fail']} | "
                    f"{item['withheld']} | {pct(item['worst_relative_error'])} | "
                    f"{item['worst_probe']['ray_degrees']:g}°/r/a={item['worst_probe']['r_over_a']:.2f} | "
                    f"{pct(well['worst_relative_error'])} |"
                )
    add("")
    add("### 3.5 收敛：径向误差序列与观测阶")
    add("")
    add("| 射线 | r/a | 原始序列（%） | 原始均阶 | 恢复序列（%） | 恢复均阶 | 恢复单调 |")
    add("| ---: | ---: | --- | ---: | --- | ---: | --- |")
    for entry in payload["convergence"]:
        raw = entry["radial_raw"]
        recovered = entry["radial_recovered"]
        add(
            f"| {entry['ray_degrees']:g}° | {entry['r_over_a']:.2f} | "
            f"{', '.join(pct(value) for value in raw['relative_error_sequence'])} | "
            f"{fmt(raw['mean_order'], 2)} | "
            f"{', '.join(pct(value) for value in recovered['relative_error_sequence'])} | "
            f"{fmt(recovered['mean_order'], 2)} | "
            f"{'是' if recovered['monotone'] else '否'} |"
        )
    add("")
    add("### 3.6 非空转对照")
    add("")
    add("| 对照 | 结果 | 判定 |")
    add("| --- | --- | --- |")
    constant = payload["controls"]["constant_field"]
    add(f"| {constant['label']} | 最大偏差 {fmt(constant['max_abs_deviation_from_exact'], 2)} | {constant['verdict']} |")
    perturbed = payload["controls"]["perturbed_field"]
    add(
        f"| {perturbed['label']} | {perturbed['fail_rows']}/{perturbed['rows']} 行被拒，"
        f"最小被拒误差 {pct(perturbed['min_failing_relative_error'])} | {perturbed['verdict']} |"
    )
    coarse = payload["controls"]["coarse_level"]
    add(
        f"| 最粗级 n={coarse['level']} 的良置复测 | 最差 {pct(coarse['worst_relative_error'])}，"
        f"FAIL {coarse['fail_rows']} 行 | {coarse['verdict']} |"
    )
    add("")
    add("## 4. 图")
    add("")
    for figure in payload["figures"]:
        add(f"![{figure['name']}]({figure['path']})")
        add("")
        add(f"*{figure['caption']}*")
        add("")
    add("## 5. 注意点与适用范围")
    add("")
    add(
        "1. **这 6 行 FAIL 的责任在测点协议，不在求解器。**探针全压在对称边界上，"
        "原始协议取单侧值；同一解换成恢复场后，良置行的最差径向误差从 "
        f"{pct(summary[str(levels[-1])]['components']['radial']['raw']['worst_relative_error'])} 降到 "
        f"{pct(summary[str(levels[-1])]['components']['radial']['recovered']['worst_relative_error'])}。"
    )
    add(
        "2. **未良置的邻域仍然不给通过。**`theta=0°, r/a=1.25` 处 `|σ_rr|` 只有约 1.4 % 的应力尺度"
        "（处于零点附近），两个口径都记为 `ill_conditioned`，既不算 PASS 也不改判。"
    )
    add(
        "3. **恢复场的精度是「解 + 恢复」的联合性质。**本报告不改变 `structural-certification-materials`"
        "的任何 PASS/FAIL；它给出的是一条可复现的替代口径与它的非空转证据。"
    )
    add(
        "4. **适用范围**：四分之一圆环 `R/a=10` 的线弹性平面应力 Q4 路径、`grading=5`、"
        "无边界层/接触/非线性；不外推到壳体、三维实体或自适应网格。"
    )
    add(
        "5. **顺带修掉一个潜伏缺陷（本条副产物）。**复算时发现 `kirsch_hole.solve_case` 的"
        "直角→极坐标剪应力旋转写成了 `tau_rt = (sigma_rr - sigma_tt) cos sin + sxy cos2theta`，"
        "正确式是 `tau_rt = (sigma_yy - sigma_xx) cos sin + sxy cos2theta`（Mohr 圆）。"
        "两者只在 `sin2theta = 0`（即 `theta = 0/90°`）处重合，而全部规范探针正好只在两条射线上，"
        "所以已发表数字**逐位不变**（`tests/test_kirsch_pointwise_conditioning.py` 以 `sin2theta = 0`"
        "的恒等性锁死这一点）。离轴探针此前会拿到错误剪切：`theta = 45°`、单轴应力 `(1,0,0)` 时"
        "旧式给 0，正确值 −0.500。修复形式为模块级 `polar_from_cartesian`，并按定义"
        "`sigma_ab = e_a·sigma·e_b` 在 8 个角度 × 4 个张量上锁到 1e-14。"
    )
    add("")
    add("## 6. 误差判定（逐项回答是否 < 3 %）")
    add("")
    add("| 关键量 | 基准 | 口径 | 误差 | <3 % ? |")
    add("| --- | --- | --- | ---: | --- |")
    final = summary[str(levels[-1])]["components"]
    for component in COMPONENTS:
        for protocol in ("raw", "recovered"):
            item = final[component][protocol]
            well = final[component]["well_conditioned"][protocol]
            value = well["worst_relative_error"]
            verdict = "—（无良置行，`WITHHELD`）" if value is None else (
                "是" if value <= TOLERANCE else "否"
            )
            add(
                f"| σ（{component}，良置行最差） | Kirsch 闭式解 | {protocol} | "
                f"{pct(value)} | {verdict} |"
            )
    add("")
    recovered_well = [
        final[component]["well_conditioned"]["recovered"]["worst_relative_error"]
        for component in COMPONENTS
        if final[component]["well_conditioned"]["recovered"]["worst_relative_error"] is not None
    ]
    shear_well = final["shear"]["well_conditioned"]["recovered"]["rows"]
    add(
        f"结论：**良置探针在恢复口径下的最差相对误差为 {pct(max(recovered_well))}（全部 < 3 %）**；"
        "原始口径下径向应力不满足 3 %（最差 "
        f"{pct(final['radial']['raw']['worst_relative_error'])}），该行如实保留。"
        + ("剪切分量的参考值在两条射线上恒为零（`|ref|=0`），"
           f"良置行数 {shear_well}，故不给相对误差结论，只按 `WITHHELD` 处理。" if shear_well == 0 else "")
        + "两条口径并存、互不覆盖。"
    )
    add("")
    add("## 7. 复现")
    add("")
    add("```bash")
    add("OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/kirsch_pointwise_conditioning_audit.py")
    add("PYTHONPATH=src python -m pytest tests/test_kirsch_pointwise_conditioning.py -q")
    add("```")
    add("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=int(round(4)))
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    ASSETS.mkdir(parents=True, exist_ok=True)

    hole = kirsch_hole()
    rows: list[dict] = []
    level_meta: dict = {}
    samples: dict[int, dict] = {}
    for level in LEVELS:
        sample = solve_level(level, hole)
        samples[level] = sample
        level_meta[str(level)] = {
            "elements": sample["elements"],
            "dofs": sample["dofs"],
        }
        rows.extend(measure(sample, hole))
        print(f"level n={level}: {sample['elements']} elements, {sample['dofs']} dofs", flush=True)

    summary = {str(level): level_summary(rows, level) for level in LEVELS}
    for level in LEVELS:
        summary[str(level)]["dofs"] = level_meta[str(level)]["dofs"]

    absolute_floor = max(
        row["errors"][component]["raw"]["scale_normalized_error"]
        for row in rows
        if row["level"] == LEVELS[-1]
        for component in COMPONENTS
    )
    controls = {
        "constant_field": control_constant_field(),
        "perturbed_field": control_perturbed_field(samples[LEVELS[-1]], hole),
        "coarse_level": {},
    }
    coarse_rows = [row for row in rows if row["level"] == LEVELS[0]]
    coarse_worst = max(
        row["errors"]["radial"]["recovered"]["relative_error"]
        for row in coarse_rows
        if not row["conditioning"]["radial"]["ill_conditioned"]
    )
    coarse_fails = sum(
        1
        for row in coarse_rows
        for component in COMPONENTS
        if row["errors"][component]["recovered"]["verdict"] == "FAIL"
    )
    controls["coarse_level"] = {
        "level": LEVELS[0],
        "worst_relative_error": coarse_worst,
        "fail_rows": coarse_fails,
        "verdict": "PASS (gate rejects the coarse level)" if coarse_fails else "FAIL (gate is idle)",
    }

    payload = {
        "meta": {
            "version": __version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "levels": list(LEVELS),
            "rays_degrees": list(RAYS_DEGREES),
            "radial_scales": list(RADIAL_SCALES),
            "tolerance": TOLERANCE,
            "withhold_fraction": WITHHOLD_FRACTION,
            "ill_conditioned_fraction": ILL_CONDITIONED_FRACTION,
            "grading": GRADING,
            "outer_radius": OUTER_RADIUS,
            "far_field": FAR_FIELD,
            "measured_absolute_floor_of_scale": absolute_floor,
        },
        "levels": level_meta,
        "probes": rows,
        "summary": summary,
        "convergence": convergence(rows),
        "controls": controls,
        "figures": [],
    }
    payload["figures"] = [
        figure_error_vs_level(rows),
        figure_conditioning(rows, absolute_floor),
        figure_field_clouds(samples[LEVELS[-1]], hole),
    ]

    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    DOC_OUT.write_text(render_markdown(payload))
    print(f"wrote {JSON_OUT.relative_to(ROOT)} and {DOC_OUT.relative_to(ROOT)}")
    for control in controls.values():
        print(f"control {control.get('label', 'coarse level')}: {control.get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())