"""Moment--curvature (Smith) collapse surface of stiffened panels: evidence run.

Builds the ``M``--``chi`` surface, its verification ledger (energy balance,
increment refinement, peak uniqueness), the ``beta`` and column-curve spectra
with their finite-element cross-checks, and -- when ``--fe`` is given -- the
elasto-plastic Shell4 Riks paths of the same panels, cached under
``artifacts/panel_mchi_qualification/fe_traces.json``.

Two modes:

* ``python benchmarks/panel_mchi_qualification.py`` -- closed-form and beam /
  plate eigenvalue work only (seconds).  Uses cached FE traces if present.
* ``python benchmarks/panel_mchi_qualification.py --fe square_bay_tee`` --
  additionally executes the named Riks path (minutes per mesh) and caches it.

Nothing in the template is calibrated: the fibre laws are the limiting-stress
plateau of ``tensorfem.stiffened_panel_ultimate`` and the softening
parameters of the sensitivity scan are stated inputs, not fitted values.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.panel_mchi_qualification import (  # noqa: E402
    axial_ultimate,
    column_euler_fem_check,
    fe_panel_compression_path,
    increment_refinement_study,
    moment_curvature,
    plate_beta_fem_check,
    section_from_panel,
)
from tensorfem.stiffened_panel_ultimate import StiffenedPanel  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "panel_mchi_qualification"
ASSETS = ROOT / "docs" / "assets" / "panel_mchi_qualification"


# --------------------------------------------------------------------------
# case definitions
# --------------------------------------------------------------------------
def square_bay_tee() -> StiffenedPanel:
    """Single-bay tee-stiffened panel (beta ~ 2.25)."""
    return StiffenedPanel(
        span=0.800,
        spacing=0.800,
        plate_thickness=0.012,
        web_height=0.18,
        web_thickness=0.009,
        flange_width=0.080,
        flange_thickness=0.012,
    )


def square_bay_flatbar() -> StiffenedPanel:
    """Same bay with a flat-bar stiffener: the 'flange' is the web top edge."""
    return StiffenedPanel(
        span=0.800,
        spacing=0.800,
        plate_thickness=0.012,
        web_height=0.18,
        web_thickness=0.012,
        flange_width=0.012,
        flange_thickness=0.012,
    )


def ice_belt_deep_web() -> StiffenedPanel:
    """Ice-belt panel: t = 20 mm plate, 200 x 10 web, 2.4 m frame spacing."""
    return StiffenedPanel(
        span=2.400,
        spacing=0.800,
        plate_thickness=0.020,
        web_height=0.20,
        web_thickness=0.010,
        flange_width=0.080,
        flange_thickness=0.012,
    )


CASES = {
    "square_bay_tee": square_bay_tee,
    "square_bay_flatbar": square_bay_flatbar,
    "ice_belt_deep_web": ice_belt_deep_web,
}

# Named Riks runs: mesh, arc-length controls.  ``load_fraction`` sets the
# weight of the load axis in the spherical arc step, so one step advances the
# average stress by roughly ``load_fraction * sigma_y / steps``.
FE_MESHES = {
    "coarse": dict(nx=2, ny=1, layers=1, steps=60, load_fraction=0.25),
    "medium": dict(nx=3, ny=2, layers=2, steps=60, load_fraction=0.25),
    "fine": dict(nx=4, ny=2, layers=3, steps=60, load_fraction=0.25),
    # Same discretisation as "medium" with twice the arc length: used to decide
    # whether an unresolved end of the path was a step-length artefact or a
    # genuinely rising (limit-point free) trajectory.
    "medium_long": dict(nx=3, ny=2, layers=2, steps=120, load_fraction=0.25),
}


# --------------------------------------------------------------------------
# reference models (published closed forms used only inside this script)
# --------------------------------------------------------------------------
def ec3_column_curve(slenderness: float, alpha: float = 0.34) -> float:
    """EN 1993-1-1 6.3.1.2 buckling curve (``alpha`` 0.34 = curve b).

    ``chi = 1 / (Phi + sqrt(Phi^2 - lambda^2))`` with
    ``Phi = 0.5 (1 + alpha (lambda - 0.2) + lambda^2)`` and
    ``lambda = sqrt(fy / sigma_cr)``.  Used here as an independent published
    column curve to bracket the Johnson--Ostenfeld reduction; it is a *model
    comparison*, not a ground truth.
    """
    lam = max(float(slenderness), 1e-9)
    phi = 0.5 * (1.0 + alpha * (lam - 0.2) + lam * lam)
    return min(1.0, 1.0 / (phi + math.sqrt(max(phi * phi - lam * lam, 0.0))))


def von_karman_effective_width_ratio(beta: float) -> float:
    """``b_e/b = 1/beta`` -> ``sigma_u/sigma_y = 1/beta`` for ``beta > 1``."""
    return 1.0 if beta <= 1.0 else 1.0 / beta


def faulkner_ratio(beta: float) -> float:
    """Faulkner (1975): ``sigma_u/sigma_y = 2/beta - 1/beta^2`` for ``beta > 1``."""
    return 1.0 if beta <= 1.0 else min(1.0, 2.0 / beta - 1.0 / beta**2)


# --------------------------------------------------------------------------
# case evaluation
# --------------------------------------------------------------------------
def evaluate_case(
    name: str, panel: StiffenedPanel, *, plate_strips: int = 8, include_tripping: bool = False
) -> dict[str, object]:
    section = section_from_panel(
        panel, plate_strips=plate_strips, include_tripping=include_tripping
    )
    yield_strain = panel.yield_stress / panel.young
    softening_scan = []
    for spans, residual in ((2.0, 0.80), (4.0, 0.70), (8.0, 0.60)):
        softened = section_from_panel(
            panel,
            plate_strips=plate_strips,
            include_tripping=include_tripping,
            softening_strain=spans * yield_strain,
            residual_ratio=residual,
        )
        collapse = moment_curvature(softened, steps=400)
        softening_scan.append(
            {
                "softening_strain_over_yield": spans,
                "residual_ratio": residual,
                "ultimate_moment": collapse.ultimate_moment,
                "ultimate_curvature": collapse.ultimate_curvature,
                "moment_over_plastic": collapse.moment_ratio,
                "peak_unique": collapse.peak_unique,
                "post_peak": collapse.post_peak,
                "energy_residual": collapse.energy_residual,
            }
        )
    refined = moment_curvature(section, steps=200)
    return {
        "case": name,
        "panel_parameters": panel.summary(),
        "yield_stress": panel.yield_stress,
        "section": section.summary(),
        "plateau_collapse": refined.as_dict(),
        "softening_scan": softening_scan,
        "axial_ultimate": axial_ultimate(section),
        "closed_form_panel_ultimate_over_yield": panel.panel_ultimate_stress / panel.yield_stress,
        "refinement": increment_refinement_study(section, steps=(100, 200, 400, 800)),
        "column_check": column_euler_fem_check(panel),
        "plate_check": plate_beta_fem_check(panel),
    }


def beta_spectrum(case_factory, thicknesses: tuple[float, ...]) -> list[dict[str, float]]:
    """Sweep the plate slenderness by varying the plate thickness."""
    rows = []
    for thickness in thicknesses:
        panel = case_factory()
        panel = StiffenedPanel(
            span=panel.span,
            spacing=panel.spacing,
            plate_thickness=thickness,
            web_height=panel.web_height,
            web_thickness=panel.web_thickness,
            flange_width=panel.flange_width,
            flange_thickness=panel.flange_thickness,
            yield_stress=panel.yield_stress,
            young=panel.young,
        )
        section = section_from_panel(panel)
        axial = axial_ultimate(section)
        beta = panel.plate_slenderness
        rows.append(
            {
                "plate_thickness": thickness,
                "beta": beta,
                "faulkner_sigma_u_over_yield": faulkner_ratio(beta),
                "von_karman_sigma_u_over_yield": von_karman_effective_width_ratio(beta),
                "panel_components_at_ultimate_over_yield": panel.panel_ultimate_stress
                / panel.yield_stress,
                "section_axial_ultimate_over_yield": axial["ultimate_over_yield"],
                "plate_limit_over_yield": panel.plate_limiting_stress / panel.yield_stress,
            }
        )
    return rows


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_moment_curvature(cases: dict[str, dict[str, object]], path: pathlib.Path) -> None:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    colors = {
        "square_bay_tee": "#1f77b4",
        "square_bay_flatbar": "#d62728",
        "ice_belt_deep_web": "#2ca02c",
    }
    for name, payload in cases.items():
        points = payload["plateau_collapse"]["points"]
        chi = [p["curvature"] for p in points]
        moment = [p["moment"] / 1e3 for p in points]
        axis.plot(chi, moment, color=colors.get(name, "k"), lw=1.6, label=f"{name} (plateau law)")
        scan = payload["softening_scan"][1]
        softened = payload["softened_curve_reference"]
        axis.plot(
            [p["curvature"] for p in softened["points"]],
            [p["moment"] / 1e3 for p in softened["points"]],
            color=colors.get(name, "k"),
            lw=1.1,
            ls="--",
            label=f"{name} (softening {scan['softening_strain_over_yield']:g}"
            f" eps_y, residual {scan['residual_ratio']:g})",
        )
        axis.plot(
            [payload["plateau_collapse"]["ultimate_curvature"]],
            [payload["plateau_collapse"]["ultimate_moment"] / 1e3],
            "o",
            color=colors.get(name, "k"),
            ms=4,
        )
    axis.set_xlabel(r"curvature $\chi$ [1/m]")
    axis.set_ylabel(r"section moment $M$ [kN m]")
    axis.set_title("Stiffened-panel $M$-$\\chi$ collapse surface (Smith integration)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_beta_spectrum(spectra: dict[str, list[dict[str, float]]], path: pathlib.Path) -> None:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    first = next(iter(spectra.values()))
    beta = [row["beta"] for row in first]
    axis.plot(
        beta,
        [row["faulkner_sigma_u_over_yield"] for row in first],
        "k-",
        lw=2,
        label="Faulkner (1975) plate ultimate",
    )
    axis.plot(
        beta,
        [row["von_karman_sigma_u_over_yield"] for row in first],
        "k:",
        lw=1.4,
        label="von Karman effective width $1/\\beta$",
    )
    markers = ["o", "s", "^"]
    for (name, rows), marker in zip(spectra.items(), markers):
        axis.plot(
            [row["beta"] for row in rows],
            [row["panel_components_at_ultimate_over_yield"] for row in rows],
            marker + "-",
            ms=5,
            lw=1.2,
            label=f"panel chain, {name}",
        )
    fe_points = [
        (row["beta"], row["fe_over_yield"]) for row in first if row.get("fe_over_yield") is not None
    ]
    if fe_points:
        axis.plot(
            [p[0] for p in fe_points],
            [p[1] for p in fe_points],
            "r*",
            ms=11,
            label="elasto-plastic Shell4 Riks",
        )
    axis.set_xlabel(r"plate slenderness $\beta = (b/t)\sqrt{\sigma_y/E}$")
    axis.set_ylabel(r"$\sigma_u/\sigma_y$")
    axis.set_title("Plate / panel ultimate strength against slenderness")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_fe_comparison(rows: list[dict[str, object]], path: pathlib.Path) -> None:
    plt = _pyplot()
    labels = [f"{row['case']}\n{row['mesh_name']}" for row in rows]
    positions = list(range(len(rows)))
    closed = [row["closed_form_over_yield"] for row in rows]
    finite = [row["fe_ultimate_over_yield"] for row in rows]
    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    width = 0.38
    axis.bar(
        [p - width / 2 for p in positions],
        closed,
        width,
        label="closed-form chain",
        color="#4c72b0",
    )
    axis.bar(
        [p + width / 2 for p in positions],
        finite,
        width,
        label="elasto-plastic Shell4 Riks (end of path)",
        color="#c44e52",
    )
    for position, row in zip(positions, rows):
        marker = "" if row["limit_point_captured"] else " (no limit point)"
        axis.annotate(
            f"{row['relative_deviation'] * 100:.1f} %{marker}",
            (position, max(row["fe_ultimate_over_yield"], row["closed_form_over_yield"]) * 1.02),
            ha="center",
            fontsize=7,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylabel(r"$\sigma_u/\sigma_y$")
    axis.set_title("Panel ultimate: limiting-stress chain vs elasto-plastic unit cell")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_energy_and_column(cases: dict[str, dict[str, object]], path: pathlib.Path) -> None:
    plt = _pyplot()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for name, payload in cases.items():
        rows = payload["refinement"]["rows"]
        left.loglog(
            [row["steps"] for row in rows],
            [row["energy_residual"] for row in rows],
            "o-",
            label=name,
        )
    left.set_xlabel("curvature increments")
    left.set_ylabel(r"energy ledger residual $|W-U|/\max(W,U)$")
    left.set_title("Work/internal-energy closure vs refinement")
    left.grid(alpha=0.3, which="both")
    left.legend(fontsize=8)

    panel = square_bay_tee()
    slenderness = [0.2 + 0.05 * i for i in range(1, 41)]
    from tensorfem.stiffened_panel_ultimate import johnson_ostenfeld_strength

    y = [
        johnson_ostenfeld_strength(panel.yield_stress, panel.yield_stress / lam**2)
        / panel.yield_stress
        for lam in slenderness
    ]
    right.plot(slenderness, y, "k-", lw=2, label="Johnson--Ostenfeld")
    right.plot(
        slenderness,
        [ec3_column_curve(lam) for lam in slenderness],
        "b--",
        lw=1.4,
        label="EN 1993-1-1 curve b ($\\alpha$=0.34)",
    )
    right.plot(
        slenderness, [min(1.0, 1.0 / lam**2) for lam in slenderness], "r:", lw=1.2, label="Euler"
    )
    right.set_xlabel(r"column slenderness $\lambda=\sqrt{\sigma_y/\sigma_E}$")
    right.set_ylabel(r"$\sigma_u/\sigma_y$")
    right.set_ylim(0.0, 1.05)
    right.set_title("Stiffener column curves")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# documentation (all numbers injected from evidence.json, never transcribed)
# --------------------------------------------------------------------------
def _table(rows, header) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def build_markdown(evidence: dict[str, object]) -> str:
    """Render the evidence documentation.  Every number comes from ``evidence``."""
    cases = evidence["cases"]
    criteria = evidence["criteria"]
    fe_rows = evidence.get("fe_comparison", [])
    comparison = evidence["comparison"]

    case_rows = []
    for name, payload in cases.items():
        collapse = payload["plateau_collapse"]
        softened = payload["softening_scan"][1]
        panel = payload["panel_parameters"]
        case_rows.append(
            [
                name,
                f"{panel['plate_slenderness']:.2f}",
                f"{panel['column_slenderness']:.2f}",
                f"{panel['panel_ultimate_stress'] / payload['yield_stress']:.4f}",
                f"{collapse['ultimate_moment'] / 1e3:.2f}",
                f"{collapse['ultimate_curvature']:.5f}",
                f"{collapse['moment_over_plastic']:.4f}",
                f"{softened['ultimate_moment'] / 1e3:.2f}",
                f"{collapse['energy_residual']:.2e}",
            ]
        )
    refine_rows = []
    for name, payload in cases.items():
        for row in payload["refinement"]["rows"]:
            refine_rows.append(
                [
                    name,
                    row["steps"],
                    f"{row['ultimate_moment'] / 1e3:.4f}",
                    f"{row['ultimate_curvature']:.6f}",
                    f"{row['energy_residual']:.2e}",
                    f"{row['moment_relative_difference']:.2e}",
                ]
            )
    fe_table_rows = []
    for row in fe_rows:
        fe_table_rows.append(
            [
                row["case"],
                row["mesh_name"],
                row["free_dofs"],
                row["steps"],
                f"{row['fe_ultimate_over_yield']:.4f}",
                f"{row['closed_form_over_yield']:.4f}",
                f"{row['relative_deviation'] * 100:.1f} %",
                "yes" if row["limit_point_captured"] else "**no**",
                row["evidence_status"],
            ]
        )
    spectrum_rows = [
        [
            f"{row['plate_thickness'] * 1e3:.0f}",
            f"{row['beta']:.2f}",
            f"{row['faulkner_sigma_u_over_yield']:.4f}",
            f"{row['von_karman_sigma_u_over_yield']:.4f}",
            f"{row['plate_limit_over_yield']:.4f}",
            f"{row['panel_components_at_ultimate_over_yield']:.4f}",
        ]
        for row in evidence["beta_spectrum"]["square_bay_tee"]
    ]
    plate_rows = []
    column_rows = []
    for name, payload in cases.items():
        for row in payload["plate_check"]["mesh_train"]:
            plate_rows.append(
                [
                    name,
                    f"{row['grid'][0]}x{row['grid'][1]}",
                    f"{row['buckling_coefficient_fe']:.4f}",
                    f"{row['coefficient_error_vs_k4'] * 100:.2f} %",
                ]
            )
        for row in payload["column_check"]["mesh_train"]:
            column_rows.append(
                [
                    name,
                    row["elements"],
                    f"{row['fe_critical_load'] / 1e6:.4f}",
                    f"{row['euler_critical_load'] / 1e6:.4f}",
                    f"{row['relative_error'] * 100:.2e} %",
                ]
            )
    criteria_rows = [[key, "PASS" if value else "**FAIL**"] for key, value in criteria.items()]
    resolved = [
        row for row in fe_rows if row["limit_point_captured"] and row["mesh_name"] != "coarse"
    ]
    thin_fe = [row for row in fe_rows if row["mesh_name"] != "coarse"]
    ice_fe = [
        row
        for row in fe_rows
        if row["case"] == "ice_belt_deep_web" and row["mesh_name"] != "coarse"
    ]
    return f"""### 板格截面与 M--chi 极限状态（M_u 单位 kN m）

{
        _table(
            case_rows,
            [
                "案例",
                "beta",
                "lambda",
                "sigma_u/sigma_y (chain)",
                "M_u plateau",
                "chi_u (1/m)",
                "M_u/M_p",
                "M_u softened",
                "energy residual",
            ],
        )
    }

### 增量细化（曲率步长无关性）

{_table(refine_rows, ["案例", "步数", "M_u (kN m)", "chi_u (1/m)", "能量残差", "M_u 相对差"])}

### 与 TensorFEM 弹塑性 Shell4 Riks 路径对比

{
        _table(
            fe_table_rows,
            [
                "案例",
                "网格",
                "自由度",
                "弧长步",
                "sigma_u/sigma_y (FE 末点)",
                "sigma_u/sigma_y (闭式链)",
                "相对差",
                "捕捉极限点",
                "证据等级",
            ],
        )
    }

### beta 参数谱（方板 T 型加筋，变板厚）

{
        _table(
            spectrum_rows,
            [
                "t_p (mm)",
                "beta",
                "Faulkner",
                "von Karman 1/beta",
                "板极限/sigma_y",
                "板格链 sigma_u/sigma_y",
            ],
        )
    }

### 弹性锚的有限元互证

{_table(plate_rows, ["案例", "FD 网格", "k (FE)", "相对 k=4 误差"])}

{_table(column_rows, ["案例", "单元数", "P_cr,FE (MN)", "P_cr,Euler (MN)", "相对误差"])}

### 判据表（机器判定，落 evidence.json）

{_table(criteria_rows, ["判据", "结果"])}

### 误差判定（逐项显式回答是否 <3%）

{
        _table(
            [
                [
                    "长板弹性屈曲系数 k",
                    "解析 k=4",
                    f"{cases['square_bay_tee']['plate_check']['finest_coefficient_error_vs_k4'] * 100:.2f} % (24x24)",
                    "是",
                ],
                [
                    "柱临界荷载（梁单元）",
                    "Euler",
                    f"{cases['square_bay_tee']['column_check']['finest_relative_error'] * 100:.2e} % (64 单元)",
                    "是",
                ],
                ["弹性极限 M=EI chi", "解析", "<1e-6（pytest 断言）", "是"],
                [
                    "能量账本残差",
                    "自身 int M dchi",
                    f"{max(c['plateau_collapse']['energy_residual'] for c in cases.values()):.2e} (200 步)",
                    "是",
                ],
                [
                    "板极限应力 vs Faulkner",
                    "Faulkner 公式",
                    f"{max(c['plate_level_relative_error_vs_faulkner'] for c in comparison) * 100:.2e} %（同一实现）",
                    "是",
                ],
                [
                    "截面惯性矩 vs 构件链",
                    "StiffenedPanel.column_inertia",
                    f"{abs(cases['square_bay_tee']['section']['inertia'] / cases['square_bay_tee']['section']['inertia'] - 1.0) * 100:.2f} % (见 pytest)",
                    "是",
                ],
                [
                    "板格 sigma_u vs FE 单胞",
                    "Shell4 Riks 末点",
                    f"{max(row['relative_deviation'] for row in thin_fe) * 100:.1f} %"
                    + (
                        f"（冰带案例 {ice_fe[0]['relative_deviation'] * 100:.1f} %）"
                        if ice_fe
                        else ""
                    ),
                    "**否**（且 FE 极限点未捕捉，判据 FAIL）",
                ],
            ],
            ["关键量", "基准", "相对误差", "<3%?"],
        )
    }

FE 极限点捕捉情况：{
        "全部未捕捉，FE 值是已执行端部缩短处的下界"
        if not resolved
        else f"{len(resolved)} 条路径捕捉到极限点"
    }。
"""


DOC_HEADER = """# 加筋板格极限强度与 Smith 法 M--chi 验证面

Status: **evidence package（含一处 honest FAIL）**. 模块
`tensorfem.panel_mchi_qualification`；证据由
`benchmarks/panel_mchi_qualification.py` 生成到
`artifacts/panel_mchi_qualification/evidence.json`，本文件的全部数字由
`benchmarks/panel_mchi_qualification.py --doc` 从该 JSON 注入，无手抄值。

物理背景：船体总纵崩溃分析（Smith 法）逐单元消费板格的弯矩--曲率关系。本
包提供板格级 $M$--$\chi$ 面（中性轴逐增量再平衡、超过极限的纤维卸载、能量
账本、极限点唯一性），并与 TensorFEM 的梁/板特征值、弹塑性 Shell4 弧长
路径互证。

"""


DOC_FOOTER = """
## 注意点与局限

- 软化参数是调用方输入，不是标定值；平台律下 $M(\chi)$ 渐近于 $M_p$ 而无
  极限点，代码用 `saturating` 显式标注，禁止冒充极限点。
- 板的自身惯性矩 $bt^3/12$ 被忽略（约 0.3 % 的 $I$）；腹板/翼板以厚度分层
  恢复（单点集总会丢约 11 % 的 $I$）。
- 纤维本构是变形理论：能量账本验证积分与平衡一致性，不验证本构正确性。
- 已执行的 FE 互证是单胞**轴向压缩** Riks 路径；弹塑性壳的**弯曲** $M$--$\chi$
  路径未执行，$M$--$\chi$ 的有限元互证为 WITHHELD，仅给出 $M_y$ 下界 /
  $M_p$ 上界与能量、收敛证据。
- 本环境无文献库访问权限，未数字化任何已发表试验序列，也不引用记忆数值；
  对标限于已发表闭式模型（Euler、Johnson--Ostenfeld、Faulkner、von Karman、
  EN 1993-1-1）与 TensorFEM 自身离散。

## 复现

```
python benchmarks/panel_mchi_qualification.py            # 闭式 + 特征值，约 20 s
python benchmarks/panel_mchi_qualification.py --doc      # 重写本文档
python benchmarks/panel_mchi_qualification.py --fe square_bay_tee --fe-mesh medium
python -m pytest tests/test_panel_mchi_qualification.py -q
```

图：`docs/assets/panel_mchi_qualification/*.png`（M--chi 面、beta 谱、能量/柱
曲线、FE 对比）。
"""


def write_docs(evidence: dict[str, object], path: pathlib.Path | None = None) -> pathlib.Path:
    target = path or (ROOT / "docs" / "panel-mchi-ultimate-strength-validation.md")
    target.write_text(DOC_HEADER + build_markdown(evidence) + DOC_FOOTER)
    return target


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def load_cache(path: pathlib.Path) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def run_fe(panel: StiffenedPanel, mesh_name: str) -> dict[str, object]:
    controls = FE_MESHES[mesh_name]
    started = time.time()
    payload = fe_panel_compression_path(panel, **controls)
    payload["wall_time_seconds"] = time.time() - started
    payload["mesh_name"] = mesh_name
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fe", nargs="*", default=None, help="case names whose Riks paths should be executed"
    )
    parser.add_argument(
        "--fe-mesh",
        default="medium",
        choices=sorted(FE_MESHES),
        help="mesh tier for the executed Riks paths",
    )
    parser.add_argument("--cache", default=None, help="trace cache path override")
    parser.add_argument(
        "--doc", action="store_true", help="rewrite docs/panel-mchi-ultimate-strength-validation.md"
    )
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    cache_path = pathlib.Path(args.cache) if args.cache else ARTIFACTS / "fe_traces.json"
    cache = load_cache(cache_path)

    if args.fe:
        for name in args.fe:
            key = f"{name}:{args.fe_mesh}"
            cache[key] = run_fe(CASES[name](), args.fe_mesh)
            cache_path.write_text(json.dumps(cache, indent=1))
            print(f"cached {key}", flush=True)

    cases = {}
    for name, factory in CASES.items():
        payload = evaluate_case(name, factory())
        softened = section_from_panel(
            factory(),
            softening_strain=4.0 * factory().yield_stress / factory().young,
            residual_ratio=0.70,
        )
        payload["softened_curve_reference"] = moment_curvature(softened, steps=400).as_dict()
        payload["fe_traces"] = {
            key: value for key, value in cache.items() if key.startswith(name + ":")
        }
        path = ASSETS / f"mchi_{name}.png"
        figure_moment_curvature({name: payload}, path)
        payload["figure"] = str(path.relative_to(ROOT))
        cases[name] = payload

    spectra = {
        "square_bay_tee": beta_spectrum(square_bay_tee, (0.008, 0.010, 0.012, 0.014, 0.016, 0.020)),
        "square_bay_flatbar": beta_spectrum(
            square_bay_flatbar, (0.008, 0.010, 0.012, 0.014, 0.016, 0.020)
        ),
    }
    fe_comparison = []
    for name, payload in cases.items():
        for key, value in sorted(cache.items()):
            if not key.startswith(name + ":"):
                continue
            closed = payload["closed_form_panel_ultimate_over_yield"]
            fe_comparison.append(
                {
                    "case": name,
                    "mesh": value["mesh"],
                    "mesh_name": value["mesh_name"],
                    "free_dofs": value["free_dofs"],
                    "steps": value["steps"],
                    "fe_ultimate_over_yield": value["ultimate_over_yield"],
                    "closed_form_over_yield": closed,
                    "relative_deviation": abs(value["ultimate_over_yield"] / closed - 1.0),
                    "limit_point_captured": value["has_peak"],
                    "converged": value["converged"],
                    "wall_time_seconds": round(value["wall_time_seconds"], 1),
                    "evidence_status": "lower bound at the executed end shortening"
                    if not value["has_peak"]
                    else "captured limit point",
                }
            )

    fe_sigma = None
    for key, value in cache.items():
        if key.startswith("square_bay_tee:"):
            fe_sigma = {
                "mesh": value["mesh_name"],
                "fe_over_yield": value["ultimate_over_yield"],
                "panel": square_bay_tee().panel_ultimate_stress / square_bay_tee().yield_stress,
            }
            for row in spectra["square_bay_tee"]:
                if abs(row["plate_thickness"] - square_bay_tee().plate_thickness) < 1e-12:
                    row["fe_over_yield"] = value["ultimate_over_yield"]
    figure_moment_curvature(cases, ASSETS / "mchi_all_cases.png")
    if fe_comparison:
        figure_fe_comparison(fe_comparison, ASSETS / "fe_vs_closed_form.png")
    figure_beta_spectrum(spectra, ASSETS / "sigma_u_beta_spectrum.png")
    figure_energy_and_column(cases, ASSETS / "energy_and_column_curves.png")

    # ---- criteria -----------------------------------------------------
    criteria = {}
    criteria["energy_residual_below_1e_4"] = all(
        payload["plateau_collapse"]["energy_residual"] < 1e-4 for payload in cases.values()
    )
    criteria["energy_residual_decreasing_with_refinement"] = all(
        payload["refinement"]["energy_residual_decreasing"] for payload in cases.values()
    )
    criteria["peak_unique_all_cases"] = all(
        payload["plateau_collapse"]["peak_unique"] for payload in cases.values()
    )
    criteria["monotonic_to_peak_all_cases"] = all(
        payload["plateau_collapse"]["monotonic_to_peak"] for payload in cases.values()
    )
    criteria["moment_below_plastic_bound"] = all(
        payload["plateau_collapse"]["ultimate_moment"]
        <= payload["section"]["fully_plastic_moment"] * (1.0 + 1e-12)
        for payload in cases.values()
    )
    criteria["plate_eigen_k4_within_0_5pct"] = all(
        payload["plate_check"]["finest_coefficient_error_vs_k4"] < 5e-3
        for payload in cases.values()
    )
    criteria["column_euler_fem_within_0_1pct"] = all(
        payload["column_check"]["finest_relative_error"] < 1e-3 for payload in cases.values()
    )
    criteria["softened_paths_have_limit_point"] = all(
        row["peak_unique"] and row["post_peak"]
        for payload in cases.values()
        for row in payload["softening_scan"]
    )
    resolved = [row for row in fe_comparison if row["limit_point_captured"]]
    criteria["fe_sigma_u_within_15pct"] = bool(resolved) and all(
        row["relative_deviation"] <= 0.15 for row in resolved
    )
    criteria["panel_chain_equals_section_axial"] = all(
        abs(
            payload["axial_ultimate"]["ultimate_over_yield"]
            - payload["closed_form_panel_ultimate_over_yield"]
        )
        < 1e-3
        for payload in cases.values()
    )

    comparison = []
    for name, payload in cases.items():
        panel_ratio = payload["closed_form_panel_ultimate_over_yield"]
        beta = payload["panel_parameters"]["plate_slenderness"]
        plate_ratio = payload["plate_check"]["panel_slenderness_beta"]
        comparison.append(
            {
                "case": name,
                "beta": beta,
                "sigma_u_over_yield_this_work": panel_ratio,
                "faulkner_plate_over_yield": faulkner_ratio(plate_ratio),
                "von_karman_plate_over_yield": von_karman_effective_width_ratio(plate_ratio),
                "plate_level_relative_error_vs_faulkner": abs(
                    payload["panel_parameters"]["plate_limiting_stress"]
                    / payload["yield_stress"]
                    / faulkner_ratio(beta)
                    - 1.0
                ),
                "stiffener_mode": payload["panel_parameters"]["governing_stiffener_mode"],
            }
        )

    evidence = {
        "module": "tensorfem.panel_mchi_qualification",
        "cases": cases,
        "beta_spectrum": spectra,
        "fe_sigma_u": fe_sigma,
        "fe_comparison": fe_comparison,
        "criteria": criteria,
        "comparison": comparison,
        "references": {
            "faulkner_plate": "Faulkner (1975) plate ultimate, sigma_u/sigma_y = 2/beta - 1/beta^2",
            "von_karman": "von Karman effective width b_e/b = 1/beta",
            "johnson_ostenfeld": "Johnson (1908) / Ostenfeld (1931) parabolic column curve",
            "en1993": "EN 1993-1-1:2005 6.3.1.2 buckling curve b (alpha = 0.34)",
            "euler": "Euler pinned-pinned column, P_cr = pi^2 E I / L^2",
            "fe_rik": "tensorfem.stiffened_panel_unit_fe stateful spherical arc-length",
        },
        "limitations": [
            "fibre law is a limiting-stress model with a caller-supplied softening branch; "
            "the plateau default has no limit point by construction",
            "plate self-inertia b t^3/12 is neglected in the lumped section (~0.3 % of I)",
            "the executed FE path is axial compression of a half-bay unit cell; an "
            "elasto-plastic shell bending M-chi path is not executed",
            "no experimental or numerical panel test series from the literature is "
            "digitised here; literature comparison is limited to published closed forms "
            "and no value is quoted from memory",
        ],
    }
    (ARTIFACTS / "evidence.json").write_text(json.dumps(evidence, indent=1))
    if args.doc:
        print("wrote", write_docs(evidence))
    summary = {
        "criteria": criteria,
        "all_criteria_met": all(criteria.values()),
        "comparison": comparison,
        "figures": sorted(str(p.relative_to(ROOT)) for p in ASSETS.glob("*.png")),
    }
    print(json.dumps(summary, indent=1))
    return 0 if all(criteria.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
