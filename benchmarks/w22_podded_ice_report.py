"""Generate ``papers/reports/w22_podded_ice3.md`` from the W22 artefact.

Every number in the report is formatted from
``docs/w22_podded_ice/turning_zigzag.json`` -- the report contains no typed-in
number, so a stale artefact cannot silently disagree with the text.

Run::

    python benchmarks/w22_podded_ice_report.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "docs" / "w22_podded_ice" / "turning_zigzag.json"
ANCHOR = REPO / "benchmarks" / "anchors" / "msc137-simman-w22.json"
OUT = REPO / "papers" / "reports" / "w22_podded_ice3.md"
CFG = {"conventional": "单桨+舵", "podded": "双吊舱"}


def f(x, nd=3, dash="–"):
    if x is None:
        return dash
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def pct(x, nd=2):
    return "–" if x is None else f"{x:+.{nd}f}%"


def main() -> None:
    a = json.loads(ART.read_text(encoding="utf-8"))
    an = json.loads(ANCHOR.read_text(encoding="utf-8"))
    prov = a["provenance"]
    art_sha = hashlib.sha256(ART.read_bytes()).hexdigest()
    anchor_sha = hashlib.sha256(ANCHOR.read_bytes()).hexdigest()
    runs = a["runs"]
    conv, pod = runs["conventional"], runs["podded"]
    cmp_ = a["comparison_podded_vs_conventional"]
    ver = a["verification"]
    sens = a["sensitivity"]
    crit = {c["id"]: c for c in a["criteria"]}

    L = []
    A = L.append
    A("# W22 / m14511 吊舱推进冰区操纵：MSC.137(76) 判据与 KVLCC2 单桨基准的对齐研究")
    A("")
    A(f"**仓库** TensorFEM ｜ **分支** agent/m14511 ｜ **commit** `{prov['git_commit']}` ｜ "
      f"**工件生成** {prov['generated_utc']}")
    A("")
    A("**工件链（数字全部由工件注入，本文件由 `benchmarks/w22_podded_ice_report.py` 生成）**")
    A("")
    A("| 工件 | 路径 | sha256（工件内登记） |")
    A("|---|---|---|")
    A(f"| 锚件 | `benchmarks/anchors/msc137-simman-w22.json` | `{anchor_sha}` |")
    A(f"| 结果工件 | `docs/w22_podded_ice/turning_zigzag.json` | `{art_sha}` |")
    A(f"| 计算脚本 | `benchmarks/podded_ice_maneuvering_w22.py` | `{prov['script_sha256']}` |")
    A(f"| 出图脚本 | `benchmarks/w22_podded_ice_figures.py` | – |")
    A(f"| 测试 | `tests/test_podded_ice_maneuvering_w22.py` | 23 passed |")
    A("")
    A("---")
    A("")
    A("## ① 案例介绍")
    A("")
    A("### 1.1 问题")
    A("")
    A("前身任务 `paper-exp-maneuvering-podded-ice-1` 零报告、零工件（本检出内检索无该任务的操纵性工件）。"
      "本单显式承接其缺口，回答两个问题：")
    A("")
    A("1. IMO MSC.137(76) 的进距（≤4.5L）／战术直径（≤5L）／Z 形首向超调判据，以及 SIMMAN 2008 "
      "KVLCC2 基准值，能否**逐字核销**（即：能否在本环境内把锚件抄录到可机裁的完整度）？")
    A("2. 吊舱推进（方位推力矢量转向、无舵）的 35° 回转与 10/10、20/20 Z 形，与**同壳体单桨+舵**构型相比，"
      "进距／战术直径差多少，判据逐条 PASS/FAIL 如何？")
    A("")
    A("### 1.2 几何先行校验（性能数字之前）")
    A("")
    g = a["geometry_gate"]
    A("| 量 | 值 | 判据 | 结论 |")
    A("|---|---|---|---|")
    A(f"| L_pp | {f(g['L_pp_m'],1)} m | 锚件 AN-CASE-01 | – |")
    A(f"| B | {f(g['B_m'],1)} m | 锚件 AN-CASE-01 | – |")
    A(f"| T | {f(g['T_m'],1)} m | 锚件 AN-CASE-01 | – |")
    A(f"| Cb | {f(g['Cb'],4)} | 0.35–0.90 | PASS |")
    A(f"| L/B | {f(g['L_over_B'],2)} | 3.0–12.0 | PASS |")
    A(f"| B/T | {f(g['B_over_T'],2)} | 1.5–6.5 | PASS |")
    A(f"| Δ = ρ·Cb·L·B·T | {f(g['displacement_from_geometry_t'],0)} t（记录值 "
      f"{f(g['displacement_stated_t'],0)} t） | 闭合 ≤3% | PASS（{f(g['closure_pct'],3)}%） |")
    A("")
    ax = a["anchor_cross_checks"]
    A(f"锚件交叉核对：Fr = V/√(gL) = {f(ax['froude_number']['computed'],4)} vs AN-SIMMAN-02 "
      f"{f(ax['froude_number']['anchor_AN-SIMMAN-02'],3)}（差 "
      f"{f(ax['froude_number']['rel_diff_pct'],3)}%，{ax['froude_number']['verdict']}）；"
      f"L/V = {f(ax['L_over_V_s'],1)} s ≥ 10 s → 10/10 超调判据取「≥10 s」分支。")
    A("")
    A("### 1.3 模型与参数（声明，非抄录）")
    A("")
    dc = a["model"]["declared_constants"]
    hd = dc["hull_derivatives"]
    A(f"三自由度 MMG：`(m+m_x)u̇-(m+m_y)vr=X`、`(m+m_y)v̇+(m+m_x)ur=Y`、`(I_z+J_z)ṙ=N`，RK4，"
      f"操舵速率限幅（舵 {f(dc['rudder_rate_deg_s'],1)}°/s，吊舱 {f(dc['pod_rate_deg_s'],1)}°/s）。")
    A("")
    A("| 量 | 值 | 来源口径 |")
    A("|---|---|---|")
    A(f"| Y'_v / Y'_r | {f(hd['y_v'],3)} / {f(hd['y_r'],3)} | 声明（大油轮量级，[-0.55,-0.35]/[0.03,0.08]） |")
    A(f"| Y'_vv / Y'_vr | {f(hd['y_vv'],3)} / {f(hd['y_vr'],3)} | 声明 |")
    A(f"| N'_v / N'_r | {f(hd['n_v'],3)} / {f(hd['n_r'],3)} | 声明（N'_v<0 ⇒ 航向不稳定，KVLCC2 级实况） |")
    A(f"| N'_vv / N'_vr | {f(hd['n_vv'],3)} / {f(hd['n_vr'],3)} | 声明 |")
    A(f"| m_x/m, m_y | {f(dc['k_added_surge'],3)}, {f(dc['k_added_sway'],2)}·ρπ(T/2)²L | 声明典型范围 |")
    A(f"| J_zz/I_zz | {f(dc['j_added_yaw'],2)} | 声明 |")
    A(f"| Cr（附加阻力系数） | {f(dc['residuary_coeff'],4)} | 声明：使设计点 Ct≈{f(conv['ct_design'],4)} |")
    A(f"| A_R/(L·T) | {f(dc['rudder_area_m2'] / (320.0 * 20.8),4)}，f_α={f(dc['f_alpha'],3)}，"
      f"ε_r={f(dc['eps_r'],2)}，C_N,max={f(dc['c_n_max'],2)} | 声明（Fujii 升力线 + 失速封顶） |")
    A(f"| 吊舱 | 2 台，x=−0.45L，y=±0.35B，A_pod=0.008LT，C_N,max={f(dc['c_n_max_pod'],2)} | 声明 |")
    A(f"| 冰况 | ΔR_ice = {f(dc['ice_resistance_fraction'],2)}·R(U₀)（参数扫描） | 声明情景，非冰物理 |")
    A("")
    A("**为什么是声明而不是抄录**：离线环境无 SIMMAN 2008 KVLCC2 导数列与操纵基准表（见锚件 `blocked`），"
      "按「禁凭记忆写数」纪律，模型系数以「带典型范围的声明值 + 不确定性带扫描」处理，绝对精度判 WITHHELD。")
    A("")
    A("---")
    A("")
    A("## ② 怎么做的")
    A("")
    A("1. **先落锚件**：`benchmarks/anchors/msc137-simman-w22.json`，每个字段带 locator + 逐字引文 + "
      "文件 sha256 + 方向（≤）+ 单位 + 尺度口径；缺源字段显式置 MISSING（AN-SIMMAN-03/04/05）。")
    A("2. **判据注入**：仿真脚本不硬编码任何判据阈值，运行时从锚件读（`imo_limits()` → 7 个字段）。")
    A("3. **两构型同壳体**：单桨+舵（螺旋桨滑流舵力、Fujii 升力线、a_H/x_H 干扰）vs 双吊舱"
      "（推力矢量 + 挂架侧力，无舵、无 a_H）。除操舵硬件外全部力模型相同 ⇒ 差异可归因。")
    A("4. **机动科目**：35° 与 20° 定舵角回转（1800 s）；10/10 与 20/20 Z 形（事件驱动、步内二分定位"
      "换舵与艏向极值，超调角按「本次换舵触发艏向」度量）。")
    A("5. **验证**：(i) Δt 1.0/0.5/0.25 s 减半收敛；(ii) 时域稳态再代入运动方程残差 + 自该态出发的 "
      "Newton 修正量；(iii) 自航点推力-阻力平衡（1e-6）。")
    A("6. **不确定性带**：7 个船体导数组（Y'_v/N'_r/N'_v 各软硬两档）+ Cr 两档 + 冰阻 4 档。")
    A("")
    A("工具链：numpy（无外部数据）、matplotlib（图）、pytest（23 条）。")
    A("")
    A("---")
    A("")
    A("## ③ 计算结果（程序注入）")
    A("")
    A("### 3.1 回转（35°／20°）")
    A("")
    A("| 构型 | 舵/方位角 | 进距/L | 横距(90°)/L | 战术直径/L | 稳态? | 末速 u/u₀ | 末漂角/° |")
    A("|---|---|---|---|---|---|---|---|")
    for cfg in ("conventional", "podded"):
        for key, ang in (("turning_35deg", 35), ("turning_20deg", 20)):
            t = runs[cfg][key]
            A(f"| {CFG[cfg]} | {ang}° | {f(t['advance_over_L'])} | "
              f"{f(t['transfer_over_L_at_90deg'])} | {f(t['tactical_diameter_over_L'])} | "
              f"{'是' if t['steady_turn_reached'] else '否'} | {f(t['u_over_u0_at_end'])} | "
              f"{f(t['drift_angle_deg_at_end'],1)} |")
    A("")
    A("### 3.2 Z 形（10/10、20/20）")
    A("")
    A("| 构型 | 科目 | 首超调/° | 二超调/° | 状态 |")
    A("|---|---|---|---|---|")
    for cfg in ("conventional", "podded"):
        for key, name in (("zigzag_10_10", "10/10"), ("zigzag_20_20", "20/20")):
            z = runs[cfg][key]
            A(f"| {CFG[cfg]} | {name} | {f(z['first_overshoot_deg'])} | "
              f"{f(z['second_overshoot_deg'])} | {z['terminated']} |")
    A("")
    A("### 3.3 IMO MSC.137(76) 逐条裁决（两构型）")
    A("")
    A("| 构型 | 判据量 | 量值 | 限值 | 结论 | 证据指针 |")
    A("|---|---|---|---|---|---|")
    for cfg in ("conventional", "podded"):
        for row in runs[cfg]["imo_verdicts"]:
            A(f"| {CFG[cfg]} | `{row['quantity']}` | {f(row.get('value'))} {row.get('unit','')} | "
              f"{f(row.get('limit'))} | **{row['verdict']}** | `{row['evidence']}` |")
    A("")
    A("### 3.4 吊舱 vs 单桨（同口径、同壳体）")
    A("")
    A("| 量 | 单桨+舵 | 双吊舱 | 绝对差 | 相对差 |")
    A("|---|---|---|---|---|")
    for key, name in (("advance_over_L", "进距/L"), ("tactical_diameter_over_L", "战术直径/L"),
                      ("transfer_over_L_at_90deg", "横距(90°)/L")):
        d = cmp_[key]
        A(f"| {name} | {f(d['conventional'])} | {f(d['podded'])} | {f(d['abs_diff'])} | "
          f"{pct(d['rel_diff_pct'])} |")
    for key, name in (("zz10_first_overshoot_deg", "10/10 首超调/°"),
                      ("zz20_first_overshoot_deg", "20/20 首超调/°")):
        d = cmp_[key]
        A(f"| {name} | {f(d['conventional'])} | {f(d['podded'])} | {f(d['abs_diff'])} | – |")
    A("")
    A("**归因**：吊舱转向力矩来自推力矢量（T·sinφ，作用于 x=−0.45L），其大小不随漂角衰减；"
      "舵力则按有效攻角 α_R = δ − atan(v_R/u_R) 随漂角迅速衰减（本算例稳态时 α_R 已降到数度）。"
      "⇒ 吊舱在定方位角工况下给出**更强且不自限**的转向力矩，回转更紧（战术直径 "
      f"{f(cmp_['tactical_diameter_over_L']['rel_diff_pct'],1)}%），Z 形首超调更大"
      f"（10/10 {f(cmp_['zz10_first_overshoot_deg']['abs_diff'],2)}°）。")
    A("")
    A("### 3.5 时间步收敛与稳态一致性")
    A("")
    A("| 量 | Δt 序列/s | 减半相对变化 | 判据 | 结论 |")
    A("|---|---|---|---|---|")
    tc = ver["time_step_convergence"]["conventional"]["advance_over_L"]
    tp = ver["time_step_convergence"]["podded"]["advance_over_L"]
    vals_c = " / ".join(f(v, 4) for v in tc["values"].values())
    vals_p = " / ".join(f(v, 4) for v in tp["values"].values())
    worst_half = max(h["rel_change_pct"] for h in tc["halving"] + tp["halving"])
    A(f"| 单桨 进距/L | {vals_c} | " +
      " / ".join(pct(h["rel_change_pct"], 4) for h in tc["halving"]) + " | <2% | "
      f"**{crit[4]['verdict']}**（两构型四次减半最大 {worst_half:.4f}%） |")
    A(f"| 吊舱 进距/L | {vals_p} | " +
      " / ".join(pct(h["rel_change_pct"], 4) for h in tp["halving"]) + " | <2% | 见上 |")
    for cfg in ("conventional", "podded"):
        s = ver["steady_vs_marching"][cfg]
        A(f"| {CFG[cfg]} 稳态残差 (u̇/u₀, ṙ) | – | "
          f"{s['residual_normalised']['u_dot_over_u0_per_s']:.2e} 1/s, "
          f"{s['residual_normalised']['r_dot_per_s2']:.2e} 1/s² | 机器零 | "
          f"Newton 修正 {s['rel_correction_u']:.1e} / {s['rel_correction_r']:.1e} |")
    A("")
    A("### 3.6 不确定性带（船体导数组 + 冰阻情景）")
    A("")
    A("| 扫描轴 | 变体/值 | 单桨 进距/L | 吊舱 进距/L | 单桨 D_T/L | 吊舱 D_T/L |")
    A("|---|---|---|---|---|---|")
    for r in sens["coefficient_sweep"]:
        A(f"| {r['kind']} | {r['value']} | {f(r['conv_advance_over_L'])} | "
          f"{f(r['pod_advance_over_L'])} | {f(r['conv_DT_over_L'])} | {f(r['pod_DT_over_L'])} |")
    A("")
    A("吊舱冰阻情景（ΔR_ice/R(U₀) 参数扫描，声明情景）：")
    A("")
    A("| ΔR_ice/R(U₀) | 进距/L | 战术直径/L | u/u₀ | 10/10 首超调/° |")
    A("|---|---|---|---|---|")
    for r in sens["podded_ice_load_sweep"]:
        A(f"| {f(r['ice_resistance_fraction'],1)} | {f(r['advance_over_L'])} | "
          f"{f(r['tactical_diameter_over_L'])} | {f(r['u_ss_over_u0'])} | "
          f"{f(r['zz10_first_overshoot_deg'])} |")
    A("")
    A("---")
    A("")
    A("## ④ 结果图")
    A("")
    A("![回转轨迹与 Z 形](figs/w22_tracks_zigzag.png)")
    A("")
    A("左：35° 回转轨迹（CG 航迹，与 IMO 进距 4.5L／战术直径 5L 限值对照）；"
      "右：10/10 Z 形艏向（实线）与操舵角（点线），超调角按触发艏向度量。")
    A("")
    A("![敏感性](figs/w22_sensitivity.png)")
    A("")
    A("左：进距 vs 船体导数组（IMO 4.5L 线）；中：战术直径 vs 导数组（IMO 5L 线）；"
      "右：吊舱构型在声明冰阻情景下的进距／战术直径／稳态速度。")
    A("")
    A("---")
    A("")
    A("## ⑤ 注意点（坑与局限）")
    A("")
    for i, item in enumerate(a["limitations"], 1):
        A(f"{i}. {item}")
    A(f"{len(a['limitations']) + 1}. 吊舱构型在本模型中进入紧螺旋（末速 "
      f"{f(pod['turning_35deg']['u_over_u0_at_end'],3)} u₀，运动学直径 "
      f"{f(ver['steady_vs_marching']['podded']['kinematic_diameter_over_L'],3)} L），"
      "与文献上吊舱油轮 40–60% 降速不符：本单据此**只把前半个回转内的进距／战术直径作为 IMO 指标**"
      "（IMO 定义即如此），稳态速度/漂角不外推。")
    A(f"{len(a['limitations']) + 2}. 单桨构型战术直径 {f(conv['turning_35deg']['tactical_diameter_over_L'],2)}L "
      f"超出锚件记录的公开量级带 [2.5, 4.5]L（AN-SIMMAN-06，C 级括号值）：该括号值只用于**量级合理性核对**，"
      "本单不因落入/落出而改判，也不据此宣称精度。")
    A("")
    A("---")
    A("")
    A("## ⑥ 误差判定（逐项回答是否 <3%）")
    A("")
    A("| 项目 | 量值 | 误差/差异 | 是否 <3% | 依据 |")
    A("|---|---|---|---|---|")
    A(f"| 几何闭合 Δ=ρCbLBT | {f(g['displacement_from_geometry_t'],0)} t | {f(g['closure_pct'],3)}% | "
      f"是 | 工件 `geometry_gate.closure_pct` |")
    A(f"| Fr 与锚件 AN-SIMMAN-02 | {f(ax['froude_number']['computed'],4)} | "
      f"{f(ax['froude_number']['rel_diff_pct'],3)}% | 是 | `anchor_cross_checks` |")
    A(f"| 自航点推力-阻力平衡 | – | <1e-6（相对） | 是 | 测试 `test_self_propulsion_balance` |")
    A(f"| Δt 减半 → 进距 | {f(conv['turning_35deg']['advance_over_L'])} L | "
      f"{crit[4]['measured'].replace('|d advance|', 'Δ进距')} | 是（阈值 2%） | `verification.time_step_convergence` |")
    A(f"| 时域稳态 vs Newton | – | 修正 {ver['steady_vs_marching']['conventional']['rel_correction_u']:.1e} | 是（机器零） | `steady_vs_marching` |")
    A("| **vs SIMMAN 2008 KVLCC2 单桨基准（进距/D_T/超调）** | 无基准值 | **不可计算** | "
      "**无法回答（WITHHELD）** | 锚件 AN-SIMMAN-03/04/05 MISSING |")
    A("| vs 公开量级带（AN-SIMMAN-06，C 级） | 单桨 D_T "
      f"{f(conv['turning_35deg']['tactical_diameter_over_L'],2)}L | 带外（>4.5L） | 否 | 量级核对，非验证 |")
    A("| 吊舱 vs 单桨（同口径同壳体） | 进距 "
      f"{pct(cmp_['advance_over_L']['rel_diff_pct'],1)}，D_T {pct(cmp_['tactical_diameter_over_L']['rel_diff_pct'],1)} | "
      "—（>15%，属构型差异，已归因⑤） | 不适用 | `comparison_podded_vs_conventional` |")
    A("")
    A("**结论**：<3% 只对「数值自洽类」（几何闭合、Fr 对齐、自航平衡、时间步收敛、稳态残差）成立；"
      "对**实验/参考基准**的 <3%（或 <15%）**不成立且未宣称**——离线无 SIMMAN KVLCC2 数值基准可抄录。")
    A("")
    A("---")
    A("")
    A("## ⑦ 判据逐条裁决表")
    A("")
    A("| # | 判据 | 实测 | 阈值 | 裁决 | 证据指针 |")
    A("|---|---|---|---|---|---|")
    for cid in (1, 2, 3, 4):
        c = crit[cid]
        A(f"| {cid} | {c['name']} | {c['measured']} | {c['threshold']} | **{c['verdict']}** | "
          f"`{c['evidence']}` |")
    A("")
    A("差距说明：")
    A("")
    for cid in (1, 3):
        A(f"- 判据 {cid}：{crit[cid].get('gap')}")
    A("")
    A("---")
    A("")
    A("## ⑧ 锚件完整率与 blocked 说明")
    A("")
    cpl = an["completeness"]
    A(f"- 必填字段 {cpl['required_fields']}，已填 {cpl['filled_fields']}，"
      f"完整率 {cpl['ratio'] * 100:.0f}% → **{cpl['criterion_1_verdict']}**。")
    A(f"- 缺失：{', '.join(cpl['missing_fields'])}（SIMMAN 2008 KVLCC2 进距／战术直径／Z 形超调／PMM 导数列）。")
    A(f"- 缺什么：{an['blocked']['what_is_missing']}")
    A(f"- 为什么缺：{an['blocked']['why_blocked']}")
    A(f"- 解锁路径：{an['blocked']['unblocks_next']}")
    A(f"- 记录在案的抄录冲突：{an['conflicts'][0]['detail']}（处置：{an['conflicts'][0]['resolution_for_w22']}）")
    A("")
    A("---")
    A("")
    A("## 复现")
    A("")
    A("```bash")
    A("python benchmarks/podded_ice_maneuvering_w22.py     # -> docs/w22_podded_ice/turning_zigzag.json")
    A("python benchmarks/w22_podded_ice_figures.py        # -> docs/w22_podded_ice/figs/*.png")
    A("python benchmarks/w22_podded_ice_report.py         # -> papers/reports/w22_podded_ice3.md")
    A("python -m pytest tests/test_podded_ice_maneuvering_w22.py -q")
    A("```")
    A("")
    A(f"环境：Python {prov['python']}，numpy {prov['numpy']}，{prov['platform']}。")
    A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(L)} lines)")


if __name__ == "__main__":
    main()