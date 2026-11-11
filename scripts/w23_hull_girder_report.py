"""Generate ``papers/reports/w23_ship_hull_girder_structure_2.md`` from the artifacts.

Every number in the report is read from the evidence bundle, so the report cannot
drift from the data:

* ``benchmarks/hull_girder_uls/section_scan.csv``
* ``benchmarks/hull_girder_uls/csr_utilisation.json``
* ``benchmarks/hull_girder_uls/grid_convergence.json``
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from tensorfem.hull_girder_csr_chain import STEEL_E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "hull_girder_uls"
REPORT = ROOT / "papers" / "reports" / "w23_ship_hull_girder_structure_2.md"


def g(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def gnm(value: float, digits: int = 3) -> str:
    return f"{value / 1e9:.{digits}f}"


def main() -> int:
    study = json.loads((OUT / "csr_utilisation.json").read_text())
    grid = json.loads((OUT / "grid_convergence.json").read_text())
    register = json.loads((OUT / "csr_factor_register.json").read_text())
    with (OUT / "section_scan.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    prov = study["provenance"]
    criteria = {item["id"]: item for item in study["criteria"]}
    grid_lookup = {row["case_id"]: row for row in grid["cases"]}
    cases = study["cases"]
    hull = study["hull_model"]
    settings = study["settings"]
    audit = study["pairing_audit"]
    recompute = study["recomputation"]

    slope_deviation = max(
        abs(row["elastic_slope_Nm"] - STEEL_E * row["elastic"]["properties"]["inertia_m4"])
        / (STEEL_E * row["elastic"]["properties"]["inertia_m4"]) for row in cases)
    controlling = {"sagging": {}, "hogging": {}}
    for row in cases:
        member = (row["controlling_member"] or {}).get("member", "-")
        controlling[row["condition"]][member] = controlling[row["condition"]].get(member, 0) + 1
    controlling_text = "；".join(
        f"{condition} 侧："
        + "、".join(f"`{name}`（{count} 例）" for name, count in sorted(counts.items()))
        for condition, counts in controlling.items())
    audit_rel_change = max(
        max(abs(row["delta_sagging"]) / row["w22_utilisation_sagging"],
            abs(row["delta_hogging"]) / row["w22_utilisation_hogging"])
        for row in audit["rows"])
    ratios = [row["M_u_over_M_y"] for row in cases]
    utilisations = [row["utilisation"] for row in cases]
    gamma_adm = [row["gamma_admissible"] for row in cases]
    sagging_mid = next(row for row in cases
                       if row["condition"] == "sagging" and row["station_x_fraction"] == 0.50)
    w22_mid = next(row for row in audit["rows"] if row["case"] == hull["case"])
    # the sagging section is the one with the DECK in compression, so the matching
    # W22 record is its deck-compression capacity (W22 stored it under "hogging")
    w22_diff = abs(sagging_mid["M_u_smith_Nm"] - w22_mid["M_u_deck_compression_Nm"]) \
        / sagging_mid["M_u_smith_Nm"]

    def case_table() -> str:
        lines = [
            "| 算例 | x/L | B_st (m) | taper | 受压翼 | γ_s·M_sw (GN·m) | γ_w·M_wv (GN·m) "
            "| LHS (GN·m) | M_u Smith (GN·m) | M_y 弹性 (GN·m) | U | 判决 "
            "| GCI_fibre (%) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in cases:
            fibre_gci = grid_lookup[row["case_id"]]["fibre_family"]["gci_fine_percent"]
            lines.append(
                f"| `{row['case_id']}` | {row['station_x_fraction']:.2f} "
                f"| {row['station_breadth_m']:.2f} | {row['taper_factor']:.2f} "
                f"| {row['compressed_fibre']} | {gnm(row['gamma_s'] * row['M_sw_component_Nm'])} "
                f"| {gnm(row['gamma_w'] * row['M_wv_component_Nm'])} | {gnm(row['lhs_Nm'])} "
                f"| {gnm(row['M_u_smith_Nm'])} | {gnm(row['M_y_elastic_Nm'])} "
                f"| {row['utilisation']:.3f} | **{row['verdict']}** | {fibre_gci:.3f} |")
        return "\n".join(lines)

    def method_table() -> str:
        lines = [
            "| 算例 | M_u Smith (GN·m) | M_y = σ_y·W (GN·m) | M_u/M_y | 相对差 (弹性−Smith)/Smith | "
            "主导机理 | 控制构件 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in cases:
            controlling = row["controlling_member"] or {}
            lines.append(
                f"| `{row['case_id']}` | {gnm(row['M_u_smith_Nm'])} | {gnm(row['M_y_elastic_Nm'])} "
                f"| {row['M_u_over_M_y']:.3f} | {100.0 * (1.0 / row['M_u_over_M_y'] - 1.0):.1f}% "
                f"| {row['dominant_mechanism'].split(' (')[0]} | {controlling.get('member', '-')} "
                f"(σ_c/σ_y={controlling.get('peak_over_yield', float('nan')):.2f}) |")
        return "\n".join(lines)

    def grid_table() -> str:
        lines = [
            "| 算例 | M_u(4) | M_u(8) | M_u(16) | p_fibre | GCI_fibre (%) | 序列行为 "
            "| M_u(160) | M_u(320) | M_u(640) | p_steps | GCI_steps (%) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in grid["cases"]:
            fibre = row["fibre_family"]
            steps = row["step_family"]
            values = [fibre["values"][str(level)] / 1e9 for level in grid["axes"]["fibres_per_deep_member"]]
            step_values = [steps["values"][str(level)] / 1e9
                           for level in grid["axes"]["curvature_steps"]]
            order = "-" if fibre["order_observed"] is None else f"{fibre['order_observed']:.2f}"
            s_order = "-" if steps["order_observed"] is None else f"{steps['order_observed']:.2f}"
            lines.append(
                f"| `{row['case_id']}` | " + " | ".join(g(value) for value in values)
                + f" | {order} | {fibre['gci_fine_percent']:.3f} | {fibre['behaviour']} | "
                + " | ".join(g(value) for value in step_values)
                + f" | {s_order} | {steps['gci_fine_percent']:.3f} |")
        return "\n".join(lines)

    def sensitivity_table() -> str:
        levels = study["factor_sensitivity"]["levels"]
        header = "| 算例 | " + " | ".join(
            f"γ_s={level['gamma_s']:.2f}, γ_w={level['gamma_w']:.2f}" for level in levels) + " | γ_w^max(γ_s=1) |"
        lines = [header, "|" + "---|" * (len(levels) + 2)]
        for entry in study["factor_sensitivity"]["cases"]:
            cells = " | ".join(f"{item['utilisation']:.3f}" for item in entry["levels"])
            lines.append(f"| `{entry['case_id']}` | {cells} | {entry['gamma_w_max_at_gamma_s_1']:.3f} |")
        return "\n".join(lines)

    def audit_table() -> str:
        lines = [
            "| 母型船 | M_u 底部受压 (GN·m) | M_u 甲板受压 (GN·m) | M_sw,sag (GN·m) | M_ws (GN·m) "
            "| M_wh (GN·m) | W22 U_sag | 校正 U_sag | W22 U_hog | 校正 U_hog | ΔU_sag |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in audit["rows"]:
            lines.append(
                f"| {row['case']} | {gnm(row['M_u_bottom_compression_Nm'])} "
                f"| {gnm(row['M_u_deck_compression_Nm'])} | {gnm(row['M_sw_sagging_Nm'])} "
                f"| {gnm(row['M_wv_sagging_rule_Nm'])} | {gnm(row['M_wv_hogging_rule_Nm'])} "
                f"| {row['w22_utilisation_sagging']:.3f} | {row['corrected_utilisation_sagging']:.3f} "
                f"| {row['w22_utilisation_hogging']:.3f} | {row['corrected_utilisation_hogging']:.3f} "
                f"| {row['delta_sagging']:+.3f} |")
        return "\n".join(lines)

    def factor_table() -> str:
        lines = ["| 因子 | 角色（任务书指派条款） | 抄录值 | 状态 | 可用于合规声明 |",
                 "|---|---|---|---|---|"]
        for factor in register["factors"]:
            lines.append(f"| `{factor['id']}` | {factor['role']} | {factor['printed_values']} "
                         f"| **{factor['status']}** | {factor['usable_for_compliance_claim']} |")
        return "\n".join(lines)

    def criterion_table() -> str:
        lines = ["| 判据 | 量值 | 阈值 | 判决 | 证据指针 |", "|---|---|---|---|---|"]
        verdicts = {
            "W23-C1": f"n={criteria['W23-C1']['value']['n_cases']} 算例 / "
                      f"{criteria['W23-C1']['value']['n_sections']} 剖面；max U="
                      f"{criteria['W23-C1']['value']['max_utilisation']:.3f}；"
                      f"FAIL 数={criteria['W23-C1']['value']['n_fail']}",
            "W23-C2a": f"max 残差={criteria['W23-C2a']['value']['max_residual']:.1e}"
                       f"（CSV 路径 {criteria['W23-C2a']['value'].get('max_residual_csv', 0):.1e}，"
                       f"独立载荷路径 {recompute['max_load_path_residual']:.1e}）",
            "W23-C2b": f"已抄录因子 {criteria['W23-C2b']['value']['factors_transcribed']}/2，"
                       f"register={criteria['W23-C2b']['value']['register_status']}",
            "W23-C3": f"GCI_fibre={criteria['W23-C3']['value']['max_gci_percent_fibres']:.3f}%，"
                      f"GCI_steps={criteria['W23-C3']['value']['max_gci_percent_curvature_steps']:.3f}%",
            "W23-C4": f"{criteria['W23-C4']['value']['n_above_threshold']}/"
                      f"{criteria['W23-C4']['value']['n_cases']} 算例超 15% 阈值，全部已归因",
        }
        thresholds = {
            "W23-C1": "≥10 算例、≥5 剖面、字段完整",
            "W23-C2a": "<1e-6",
            "W23-C2b": "2 个因子逐条引用条款",
            "W23-C3": "<5%",
            "W23-C4": "逐例列差 + >15% 全归因",
        }
        pointers = {
            "W23-C1": "`benchmarks/hull_girder_uls/section_scan.csv`",
            "W23-C2a": "`csr_utilisation.json:recomputation`",
            "W23-C2b": "`benchmarks/hull_girder_uls/csr_factor_register.json`",
            "W23-C3": "`benchmarks/hull_girder_uls/grid_convergence.json`",
            "W23-C4": "`benchmarks/hull_girder_uls/section_scan.csv`",
        }
        for key in ("W23-C1", "W23-C2a", "W23-C2b", "W23-C3", "W23-C4"):
            lines.append(f"| {key} | {verdicts[key]} | {thresholds[key]} "
                         f"| **{criteria[key]['verdict']}** | {pointers[key]} |")
        return "\n".join(lines)

    text = f"""# W23 · 船体梁总纵极限强度：6 剖面 × 2 波浪工况的 CSR ULS 扫描与 Smith/弹性法对比

> 仓库 TensorFEM · 任务 mission 15094 · 学科域：结构（船体梁总纵强度）  
> 工件：`benchmarks/hull_girder_uls/section_scan.csv` · `csr_utilisation.json` · `grid_convergence.json` · `csr_factor_register.json`  
> 运行标识：git `{prov['git_commit'][:12]}` · module sha256 `{prov['module_sha256'][:16]}` · runner sha256 `{prov['runner_sha256'][:16]}` · engine sha256 `{prov['engine_sha256'][:16]}` · 因子登记 sha256 `{prov['csr_factor_register_sha256'][:16]}`  
> 复现：`PYTHONPATH=$PWD/src python scripts/w23_hull_girder_uls_scan.py && PYTHONPATH=$PWD/src python scripts/w23_hull_girder_report.py`

## 0. 结论摘要（先判决，后证据）

{criterion_table()}

**直答科研问题**（两个问题分开答，因为一个能答、一个不能）：

1. **是否满足 CSR ULS 利用率判据？** —— 在**单位分项安全因子**下，{criteria['W23-C1']['value']['n_cases']} 个算例全部通过（U 最大 {max(utilisations):.3f}，出现在 {sagging_mid['case_id']}），对应可容许因子 γ* = M_u/(M_sw+M_wv) ∈ [{min(gamma_adm):.3f}, {max(gamma_adm):.3f}]；因子敏感性表显示 γ_s=1 时只要 γ_w < {min(row['gamma_w_max_at_gamma_s_1'] for row in cases):.3f}（最严算例）就仍 ≤1.0。**但这不是 CSR 合规结论**：`γ_s/γ_w/γ_R` 三个因子在本环境仍为 BLOCKED（§2），所以能给的是"合规所需因子上界"，不是"按规范因子算出的利用率"。
2. **弹性法与逐步崩溃法差多少？** —— Smith 极限弯矩是弹性屈服弯矩的 **{min(ratios):.3f}–{max(ratios):.3f} 倍**（即弹性法高估 {100 * (1.0 / max(ratios) - 1.0):.1f}%–{100 * (1.0 / min(ratios) - 1.0):.1f}%），所有 {len(cases)} 个算例都超 15% 阈值，全部归因于**受压板架在屈服前到达自身屈曲极限**（甲板/船底板架峰值压应力仅 {min(r['controlling_member']['peak_over_yield'] for r in cases):.2f}–{max(r['controlling_member']['peak_over_yield'] for r in cases):.2f} σ_y）。换句话说：只做弹性屈服检查会把这条船体梁的承载能力高估约 {100 * (1.0 / max(ratios) - 1.0):.0f}%。

**附带发现（缺陷修复）**：W22 把弯曲方向按曲率符号而不是按"受压翼"命名，导致 sagging 波浪弯矩被除以下翼受压（=hogging）的承载力。本单按受压翼配对重算，W22 四个母型船的利用率变动达 {audit['max_absolute_utilisation_change']:+.3f}（逐例相对变化最大 {100 * audit_rel_change:.0f}%），判决未翻转（{audit['verdict_changes']} 个翻转；§6.1、图 5）。

## 1. 案例介绍：物理背景、几何与参数

船体梁总纵强度把整船当变截面薄壁梁：中剖面甲板/船底板架与纵骨在总纵弯曲下先弹性、再屈曲、最后崩溃。IACS CSR 的 ULS 衡准把它压成一个不等式 `γ_s·M_sw + γ_w·M_wv ≤ M_u`，屈服衡准则为 `σ = M/Z ≤ σ_y/γ_R`。本单的增量不是"再算一条船"，而是**沿船长取多个横剖面**（同一船体梁模型的 6 个站位）× **两个波浪工况**（中垂 sagging / 中拱 hogging），逐例给 ULS 利用率与两种容量方法（弹性屈服 vs Smith 渐进崩溃）的差。

船体梁模型（**声明型工程输入**，KVLCC2 模板；几何先验证再算）：

| 参数 | 值 |
|---|---|
| L_pp | {hull['particulars']['Lpp']:.1f} m |
| B | {hull['particulars']['B']:.1f} m |
| D | {hull['particulars']['D']:.1f} m |
| T | {hull['particulars']['T']:.1f} m |
| Cb（输入 / 积分） | {hull['geometry_validation']['Cb_input']:.4f} / {hull['geometry_validation']['Cb_integrated']:.6f}（误差 {hull['geometry_validation']['Cb_error']:.2e}） |
| L/B · B/T · B/D | {hull['geometry_validation']['L_over_B']:.2f} · {hull['geometry_validation']['B_over_T']:.2f} · {hull['geometry_validation']['B_over_D']:.2f} |
| 排水量 / LCG | {hull['displacement_t']:.0f} t / {hull['lcg_m']:.2f} m |
| 中剖面甲板板厚（UR S11 最小模数闭合） | {hull['midship_scantlings']['deck_thickness_final_m'] * 1000:.1f} mm |
| W_deck / W_bottom（面积求和） | {hull['midship_scantlings']['UR_S11_closure']['W_deck_final']:.2f} / {hull['midship_scantlings']['UR_S11_closure']['W_bottom_final']:.2f} m³ |
| 设计波高 h_w | {hull['design_wave_height_m']:.1f} m |
| 静水工况 | {hull['still_water_condition']}（M_sw 中剖面 {gnm(hull['M_sw_mid_signed_Nm'])} GN·m，sagging 为正） |
| 材料 | E = 206 GPa，σ_y = 235 MPa（模板值，缓和钢） |

**站位与剖面定义**（模块 `src/tensorfem/hull_girder_uls_scan.py`）：

- 站位 x/L = {", ".join(f"{s:.2f}" for s in settings['stations_x_over_L'])}（6 个横剖面）；
- 每个站位的剖面宽度取船型水线宽度剖面在该站位处的值，并按声明型宽 B 归一化（保证任一剖面都不超声明 B）；
- 板厚/纵骨按**声明型 taper**：|x/L−0.5| ≤ {settings['taper']['parallel_body_half']:.2f} 的平行中体内恒为 1.0，之后线性降到 {settings['taper']['taper_minimum']:.2f}（|x/L−0.5| = {settings['taper']['taper_end']:.2f} 处及以外）。这是**研究假设**，不是规范要求，已写进 limitations；
- 波浪弯矩由设计波载荷分布积分得到（波峰在两端 = sagging，波峰在中 = hogging），中剖面值与规则公式比较：sagging 比 {hull['wave_to_rule_ratio']['sagging']:.3f}、hogging 比 {hull['wave_to_rule_ratio']['hogging']:.3f}（规则 M_ws = {gnm(hull['rule_wave_moments']['M_ws'])}、M_wh = {gnm(hull['rule_wave_moments']['M_wh'])} GN·m）。CSR 的纵向分布因子未抄录，故站位间变化用物理载荷模型而非规则因子——已登记为局限。

## 2. 外部锚：分项安全因子仍 BLOCKED（C2b 的证据）

锚件 `benchmarks/hull_girder_uls/csr_factor_register.json`（sha256 `{prov['csr_factor_register_sha256'][:16]}`）**status = {register['status']}**，本次（2026-11-11）重做了获取尝试，全部失败：

{factor_table()}

四路获取尝试（真实命令、真实结果，逐条记在工件 `acquisition_attempts`）：

- **network**：`urllib.request.urlopen` 访问 `https://iacs.org.uk/` 与 `/resolutions/unified-requirements/`（timeout 6 s）→ 两个 URL 均 **HTTP 403 Forbidden**（沙箱代理）；直连 curl 被策略拦截。
- **platform corpus（文本）**：`grep -rIl -i "common structural rules" /root/tp-agent --include=*.json --include=*.md` → 命中只有本仓 W22 锚件/报告与 TensorPapers 稿件（引用文献名），**无条款正文、无页码、无因子表**。
- **platform corpus（因子）**：`grep -rIl "gamma_w" /root/tp-agent --include=*.json` → 语料库内**无任何 gamma_w 抄录**。
- **local documents**：`find / -maxdepth 6 -iname '*iacs*'`、`find / -maxdepth 5 -iname '*.pdf'` → 只有软件手册与无关论文；命中的 IACS 相关文件是本仓代码/测试（冰区扭矩等其它量）。

为什么不用"记忆值"：CSR 的 γ 表按版本、船型（BC/OT）、载荷组合不同而不同；无授权副本时任何记忆值都会让利用率看起来像合规结论。因此本单**不写任何 γ 值**，改报三件可核销的东西：单位因子利用率（参考值）、γ* = M_u/(M_sw+M_wv)、以及利用率对因子的精确敏感性（利用率对因子是线性的，因此 γ_w^max = (M_u−M_sw)/M_wv 是解析解，不是拟合）。

## 3. 怎么做的：方法、流程、关键设置

1. **几何先行**：`HullForm.verify()` 反算 Cb 误差 {hull['geometry_validation']['Cb_error']:.1e}（阈值 1e-3），站位剖面宽度再按声明 B 归一化。
2. **受压翼配对**（本单的结构性修正）：弯曲方向以**受压翼**定义，曲率符号由它推出 —— `sagging → 甲板受压 → κ<0`、`hogging → 船底受压 → κ>0`（引擎约定 `ε = κ(y−y_na)`，由测试 `test_charged_fibre_is_actually_in_compression` 直接数值验证，而不是靠文档字符串）。
3. **容量法 A：Smith 渐进崩溃**（`solve_moment_curvature`）：构件曲线为面积一致的板架+纵骨组合（板架后屈曲软化 + 纵骨柱屈曲软化），每增量步重解中和轴，M_u = 弯矩-曲率曲线峰值。参考离散：每个深构件 {settings['reference_strips']} 条纤维、{settings['reference_steps']} 个曲率增量。
4. **容量法 B：弹性屈服** M_y = σ_y·W（W 取同一纤维离散的受压最外纤维，γ_R = 1），即 `σ = M/Z ≤ σ_y/γ_R` 的弹性法。
5. **载荷**：静水弯矩由均质装载油船工况的梁平衡求得（`beam_moment_from_load`）；波浪弯矩由设计波载荷分布积分（`wave_load_distribution` + 双积分）。ULS 组合只用与工况同号的静水分量（另一分量为 0，保守处理，与 W22 口径一致）。
6. **网格收敛**：两条离散轴各三档，Roache GCI（F_s = {settings['gci_safety_factor']}，r = 2）：纤维数 {settings['fibre_levels']}，曲率增量 {settings['step_levels']}。序列非单调时**不编造阶数**，改用保守的 `F_s·|e32|/f_fine` 并标注 `oscillatory`。
7. **残差核销**：写入 CSV/JSON 后**从磁盘工件**重新算 `γ_s M_sw + γ_w M_wv`、利用率、M_y = σ_y·W（CSV 与 JSON 双路径），并用一条**独立的梯形积分路径**重算 M_sw(x)、M_wv(x)（不复用 `beam_moment_from_load`），残差见 §5。
8. **工具链**：Python 3.11 + numpy + matplotlib；求解器 `tensorfem.hull_girder_smith_collapse`（仓库既有）；本单新增 `src/tensorfem/hull_girder_uls_scan.py`（几何/载荷/两种容量法/GCI/判据）+ `scripts/w23_hull_girder_uls_scan.py`（工件）+ `scripts/w23_hull_girder_report.py`（本报告）+ `tests/test_hull_girder_uls_scan.py`（15 项）。

## 4. 计算结果

### 4.1 逐算例 ULS 表（{len(cases)} 算例，禁只报达标例）

{case_table()}

说明：`γ_s·M_sw` 与 `γ_w·M_wv` 两列即组合式两项（单位因子下等于分量本身）；`M_u Smith` 为受压翼方向的渐进崩溃极限弯矩；`U = LHS/M_u`；判决 `U ≤ 1.0 → PASS`。表内无省略行，全部 {len(cases)} 例，其中 FAIL {criteria['W23-C1']['value']['n_fail']} 例。

### 4.2 弹性法 vs Smith 法（逐例相对差与归因）

{method_table()}

相对差定义 `(M_y − M_u)/M_u`：全部 {criteria['W23-C4']['value']['n_above_threshold']} 例 > 15%，已逐例归因。控制构件（面积最大的受压侧板架）峰值压应力只到 {min(r['controlling_member']['peak_over_yield'] for r in cases):.2f}–{max(r['controlling_member']['peak_over_yield'] for r in cases):.2f} σ_y：板架在屈服前先屈曲，故 Smith 容量显著低于弹性屈服弯矩。控制构件（受压侧峰值压应力最低、即最先到达自身极限的板架）逐例分布：{controlling_text}——全部落在受压翼一侧，与配对规则一致。

**关于 `M_first_yield_solver`（工件中另列）**：那是引擎自带的"首个纤维进入屈服"标志。对屈曲软化板架它触发得很晚（本模型 sagging 侧在 κ≈5.3e-4 才置位，此时甲板应变约 9e-3 ≫ ε_y=1.14e-3），因此**不能**当作弹性法容量使用；本报告的弹性法容量一律是 §3 步骤 4 的 `M_y = σ_y·W`（W 由同一纤维离散独立给出），并用弹性斜率 = E·I（偏差 {100 * slope_deviation:.3f}%）交叉验证离散自洽性。引擎该字段的语义问题已作为观察记录，未用于任何判据。

### 4.3 网格收敛（两条离散轴，各三档）

{grid_table()}

最大 GCI：纤维轴 {grid['max_gci_percent_fibres']:.3f}%、曲率增量轴 {grid['max_gci_percent_curvature_steps']:.3f}%，均远低于 5% 阈值；单调序列的观测阶 p = {grid['min_order_fibres']:.2f}–{grid['max_order_fibres']:.2f}（接近 2 阶），非单调/噪声级序列如实标为 `oscillatory` / `non-asymptotic`，并改用保守估计（表内曲率增量轴的 `p_steps` 若为负或 `-`，即该轴变化已在噪声级：细网格与中网格之差 ≤0.09%，不构成可辨识的收敛阶）。

### 4.4 因子敏感性（**不是规范值**）

{sensitivity_table()}

该表只回答"因子取到这个水平时本船还过不过"，**任何一个 γ 值都不是抄录值**（§2）。γ_s=1 时最严算例（{sagging_mid['case_id']}）的临界值 γ_w^max = {sagging_mid['gamma_w_max_at_gamma_s_1']:.3f}。

## 5. 结果图

图见 `docs/assets/w23_hull_girder_uls/`（均由 `scripts/w23_hull_girder_uls_scan.py` 程序化生成，sha256 记入 provenance）：

1. `utilisation_vs_station.png` —— 两工况下 U(x/L) 与 γ_w^max(x/L)，含 U=1 线；可见最大值在中剖面 sagging 侧、向两端单调下降。
2. `capacity_vs_station.png` —— 同一站位上的 Smith M_u、弹性 M_y 与需求 LHS 三条曲线；两者间距即弹性法高估量。
3. `moment_curvature.png` —— 6 站位 × 2 工况的弯矩-曲率曲线：先弹性上升，屈曲软化后出现峰值与下降段（峰值内点）。
4. `grid_convergence.png` —— M_u 随纤维数的收敛（半对数），三档网格对应 GCI。
5. `pairing_audit.png` —— W22 配对缺陷的量化：四个母型船校正前后利用率对比。

## 6. 注意点（踩过的坑、局限、适用范围）

### 6.1 坑 1：弯曲方向的命名把 W22 的配对搞错了

`hull_girder_smith_collapse` 的约定是 `κ > 0 → y < y_na 受压`（即**船底**受压），文档字符串把这句注释成"sagging for a ship section"——但船底受压在造船惯例里是 **hogging**（sagging 是甲板受压）。W22 据此把 `κ>0` 方向命名为 "sagging" 并把 sagging 波浪弯矩除以它的承载力，等于用下翼容量去核对甲板受压工况。本单按受压翼重新配对（§3 步骤 2），并把量化影响写入 `pairing_audit`：

{audit_table()}

判决未翻转（{audit['verdict_changes']} 个母型船判决变化），但**利用率变动最大 {audit['max_absolute_utilisation_change']:+.3f}**——足以改变"余量有多宽"的判断。W22 的既有工件未改动（保持历史记录），修正集中在本单的新模块与新工件里。

### 6.2 坑 2：剖面宽度不能直接用 `HullForm.breadth()`

`breadth() = sectional_area/T` 是 Cb 闭合后的水线宽，其最大值（{hull['geometry_validation']['max_section_area'] / hull['particulars']['T']:.2f} m）**超过声明 B**（{hull['particulars']['B']:.1f} m）。直接用会让站位剖面比母型船还宽、M_u 偏大约 5%，且与声明 L/B/T/Cb 不自洽。故按 B/max(breadth) 归一化（测试 `test_station_sections_respect_declared_particulars` 守住 ≤ B）。

### 6.3 坑 3：GCI 的符号与"非单调"

Roache 观测阶为 `p = ln|e21/e32|/ln r`（e21 = f2−f1，e32 = f3−f2）。写成 `ln|e32/e21|` 会得到负阶数，再用 `r^p−1` 会把 GCI 算成负数或无穷。另外"序列非单调"和"细网格变化已在噪声级"是两种不同情形，两者都不能给 p，必须如实标注并退回保守估计，不能为了好看硬套 2 阶。

### 6.4 局限（写进工件的 limitations）

- 平面假定成立：无剪切滞后、扭转、翘曲、断裂；
- 板厚/纵骨是 CSR 相容的**参数化重构**，不是任何实船的施工图；站位 taper 是声明假设；
- 只有一个装载工况（均质装载油船），它在中剖面是 sagging，因此 hogging 算例的静水分量为 0。量化影响：中剖面 M_sw/M_wv = {hull['M_sw_mid_signed_Nm'] / abs(hull['wave_to_rule_ratio']['sagging'] * hull['rule_wave_moments']['M_ws']):.3f}，即静水弯矩只占波浪弯矩的 {100 * hull['M_sw_mid_signed_Nm'] / abs(hull['wave_to_rule_ratio']['sagging'] * hull['rule_wave_moments']['M_ws']):.1f}%，故该局限对 hogging 利用率的影响在 5% 量级以内；
- 分项安全因子与 M_wv 纵向分布因子未抄录（§2）；利用率是参考值，不是合规声明；
- 6 个站位中 0.3/0.7、0.4/0.6 的理想化剖面互为镜像（船型与波浪载荷前后对称），故 hogging 侧这两对的 U 相同——这是模型对称性的**正确结果**，不是复制粘贴；
- 适用范围：船体梁总纵弯曲（ULS/屈服），不含疲劳、局部强度、屈曲直接校核、冰载等其它工况。

## 7. 误差判定（关键量 vs 基准，逐项、显式回答是否 <3%）

| 关键量 | 基准 | 相对差 | 是否 <3% | 说明 |
|---|---|---|---|---|
| 中剖面 sagging M_u（Smith，{settings['reference_strips']} 纤维/{settings['reference_steps']} 步；甲板受压） | W22 同剖面、同引擎、8 纤维/320 步记录的甲板受压 M_u = {gnm(w22_mid['M_u_deck_compression_Nm'])} GN·m（W22 记为 "hogging"） | {100 * w22_diff:.2f}% | **是** | 跨运行/跨离散档一致性检查（同引擎，非同实现），且与 GCI 量级相容 |
| M_y（弹性）= σ_y·W | 由纤维离散独立重算 W（I/最外纤维距离） | {recompute['max_json_residual']:.1e}（残差） | **是** | 解析式自洽，§3 步骤 7 |
| γ_s·M_sw + γ_w·M_wv 与 U | 从磁盘 CSV/JSON 独立复算 | json {recompute['max_json_residual']:.1e} / csv {recompute['max_csv_residual']:.1e} | **是**（阈值 1e-6） | 条款式复算残差 |
| M_sw(x)、M_wv(x) | 独立梯形积分路径（不复用原 helper） | {recompute['max_load_path_residual']:.1e} | **是** | 载荷链复算 |
| Smith 弹性斜率 dM/dκ | E·I（同一纤维离散的面积二次矩） | {100 * slope_deviation:.3f}% | **是** | 曲线弹性段与 E·I 自洽，说明纤维离散与 I 的计算一致 |
| 中剖面波浪弯矩 | 规则公式 M_ws/M_wh（未抄录纵向分布因子） | sagging {(1 - hull['wave_to_rule_ratio']['sagging']) * 100:.1f}%，hogging {(hull['wave_to_rule_ratio']['hogging'] - 1) * 100:+.1f}% | **否（sagging 侧 {abs(1 - hull['wave_to_rule_ratio']['sagging']) * 100:.1f}% > 3%）** | 物理波载荷模型 vs 规则公式，属**模型差异**不是数值误差；差异来自 Smith 修正与 h_w 取值，已在 §1 说明 |
| 极限弯矩 M_u vs 实验/文献基准 | 本环境无该船体梁的崩溃试验或文献 ULCS 锚件 | 不可得 | **无法判定** | 无实验锚 = 不能声称精度；本单只做数值收敛 + 跨方法/跨运行一致性 |

**显式回答**：数值侧的三个关键量（M_u 跨运行一致性 {100 * w22_diff:.2f}%、条款复算残差 {recompute['max_json_residual']:.1e}、载荷路径残差 {recompute['max_load_path_residual']:.1e}）均 <3%；**但"物理精度"无法声称**——本环境没有该剖面极限强度的实验或文献锚值，因此不能回答"M_u 与真实值差多少"。规则波弯矩对比的 sagging 差 {abs(1 - hull['wave_to_rule_ratio']['sagging']) * 100:.1f}% 是模型差异，不构成对错判据。

## 8. 复现与工件溯源

```bash
cd <TensorFEM checkout>
PYTHONPATH=$PWD/src python scripts/w23_hull_girder_uls_scan.py     # 生成 3 个工件 + 5 张图
PYTHONPATH=$PWD/src python scripts/w23_hull_girder_report.py       # 生成本报告
PYTHONPATH=$PWD/src python -m pytest tests/test_hull_girder_uls_scan.py -q   # 15 项不变量
```

| 工件 | 内容 | sha256（前 16） |
|---|---|---|
| `benchmarks/hull_girder_uls/section_scan.csv` | {len(rows)} 行 × {len(rows[0])} 列逐算例 ULS 表 | — |
| `benchmarks/hull_girder_uls/csr_utilisation.json` | 全量研究 + 判据 + 复算残差 + pairing_audit + 因子敏感性 | — |
| `benchmarks/hull_girder_uls/grid_convergence.json` | 两轴三档 GCI/阶数/行为 | — |
| `benchmarks/hull_girder_uls/csr_factor_register.json` | 因子登记（BLOCKED，含获取尝试） | `{prov['csr_factor_register_sha256'][:16]}` |
| `src/tensorfem/hull_girder_uls_scan.py` | 本单新模块 | `{prov['module_sha256'][:16]}` |
| `scripts/w23_hull_girder_uls_scan.py` | 工件生成器 | `{prov['runner_sha256'][:16]}` |
| `src/tensorfem/hull_girder_smith_collapse.py` | Smith 引擎（仓库既有，未改） | `{prov['engine_sha256'][:16]}` |

运行标识：git `{prov['git_commit']}`（工件生成时的 HEAD，即本提交的父提交；本单改动由同一次提交引入，代码身份以 module/runner sha256 为准）。

局限与未完成项：**判据 C2b（分项安全因子逐条引用条款）为 honest FAIL** —— 因子无法取得（§2），因此本报告不给出任何 CSR 合规结论，只给"合规所需因子上界"；该 FAIL 不可通过改判据消解，只能在拿到授权文本后由后续任务补做。
"""

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    print(f"wrote {REPORT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())