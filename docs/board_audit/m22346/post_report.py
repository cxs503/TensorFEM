"""Post the m22346 audit report (paper-grade, six sections, 3 embedded figures)."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")

V = json.loads((HERE / "verify_output_m22346.json").read_text())
P, FV, BR, RULE = V["published"], V["P2_fv_route"], V["P1_ellipse_bracket"], V["P3_rule_reprobe"]
CC = V["cross_checks"]

MARKDOWN = f"""# m22346 交差核验：m22319 / GAP-ICE-GATE-POWER 信息量闸（对抗式复核，判级 L2）

## ① 案例介绍
审计对象=TensorFEM 分支 `agent/m22319` 提交 `ee385cf3`（11 文件 +2277 行，mission 22319，
报告 20090）。被判对象是一条**判据**而非求解器：冰区耦合 6DOF 链声明合成辐射模型
`A_inf = 1.00·m`（L=100 m、B=20 m、T=6 m、C_b=0.70、ρ=1025 kg/m³、m=8.610e6 kg），折算
`A_inf/L = {P['declared_t_per_m']:.3f} t/m`；报告 #19234 §Q5 的独立势流锚为 228.6 t/m。
**基准裁判**：`tensor_benchmark_cases WHERE anchor_slug='GAP-ICE-GATE-POWER'` → **0 行**（本席亲查）。

## ② 怎么做的
1. 取证：`git show --stat ee385cf3`；读回 mission 22319 的 result 与 `server_quality_audit`；报告 20090（3 图）。
2. 授权复现：`pytest tests/test_ice_gate_power.py -q` → 19 passed / 27.31 s（**零 deselect**）；
   `run_gate_evidence.py` → 41.2 s，重生成 `gate_evidence.json` 与提交版**逐键全等**；
   `make_doc.py` 重生成后 `git status` **干净**（文档数字确由代码注入）。
3. **独立重算**（不复用被审计 BEM 算子，`docs/board_audit/m22346/verify_gate_power.py`，57 s）：
   (P1) 闭式解析界：椭圆附加质量 `m_yy=ρπa²`（Lamb art.71/73）+ 集合包含单调性；
   (P2) 独立离散族：单元中心**有限体积**（体面与 cut cell 精确对齐），加倍截面从盒中挖去，
        h 三层 Richardson + R 三层外推，两个泛函（力积分/动能）互校；
   (P3) 判据公式重算（护带限、检出率、δ₉₅、判决稳健域）；
   (P4) 判定分辨率（per_edge 160→640）漂移。
4. 工具链：pytest/numpy/scipy.sparse/matplotlib + psql 直查平台库；未递交集群（全部 ≤57 s 短测试）。

## ③ 计算结果（程序注入）
| 量 | 参考 ref | 声明/计算 got | 偏差 | 判定 |
|---|---|---|---|---|
| 垂荡附加质量/长度 | {P['reference_t_per_m']:.3f} t/m（两镜像路线均值） | {P['declared_t_per_m']:.3f} t/m（链路程序读取） | **{P['declared_t_per_m']/P['reference_t_per_m']-1:+.2%}** | **reject** |
| 路线 ω→0 / ω→∞ | 同上 | {P['routes_t_per_m']['omega0']:.3f} / {P['routes_t_per_m']['infinity']:.3f} t/m | ±{P['band_half_width_rel']:.2%} | accept |
| 独立锚 #19234 | 同上 | 228.600 t/m | {P['anchor_deviation']:+.3%} | accept |
| **本席独立 FV 路线（ω→0 变体）** | 被审计 225.197 t/m | **{FV['combined_estimate_t_per_m']:.3f} t/m** | **{CC['fv_vs_audited_omega0_rel']:+.3%}** | 一致 |
| 闭式解析界 | Lamb 解析 | [{BR['a_omega0_lower_t_per_m']:.2f}, {BR['a_omega0_upper_t_per_m']:.2f}] t/m | 主参考落在界内 | ✓ |
| 声明 86.1 对界下界 | {BR['a_omega0_lower_t_per_m']:.2f} t/m | 86.100 t/m | **{BR['declared_vs_lower_bound_rel']:+.2%}** | 全域 reject |

判据公式重算（本席）：接受限 {RULE['acceptance_limit']:.4f}、拒绝限 {RULE['rejection_limit']:.4f}、
被审偏差处检出率 {RULE['power_at_deviation']:.2f}、δ₉₅ = {RULE['critical_deviation_95']:.4f}、
tol 处误接受率 {RULE['false_accept_at_tolerance']:.4f}（裸规则 tol 处检出率 {RULE['naive_power_at_tolerance']:.2f}）；
**判决稳健域**：u ≤ {RULE['u_max_still_rejecting']:.4f} 时仍 reject（实际 u={RULE['u']:.3f}），
偏差/拒绝限 = {RULE['deviation_over_rejection_limit']:.2f}×。
判定分辨率漂移：ω→0 {P['discretisation_shift_160_to_640']['omega0']:+.3%}、ω→∞ {P['discretisation_shift_160_to_640']['infinity']:+.3%}。

## ④ 结果图
图 1：本席独立 FV 路线（h 三层 × 三个盒尺寸）与闭式上下界、被审计两条路线、链路声明同尺度；
右侧 1/R² 截断外推（增量比 {FV['R_series_h1']['increment_ratio']:.2f} ⇒ 截断误差 O(1/R²)）。
图 2：护带规则 vs 裸规则的检出率曲线（本席重算），标出 tol / 接受限 / 拒绝限 / δ₉₅ / 被审偏差。
图 3：8 个参考与控制值在三段判定带中的位置（got vs ref）。

## ⑤ 注意点
1. **本席 FV 与被审计 BEM 共享"自由面镜像 + 折算单位长度"的模型**，独立性只在离散族；
   0.28% 的一致证明两条实现都没写错，**不**证明镜像近似本身正确。
2. FV 截断（R=120 m）残余约 +1.5%，组合估计按 ±1.5% 记；未做 R=240 & h≤0.5（超本机预算）。
3. ω→∞（奇镜像，234.042 t/m）路线本席**未**独立复核（not_run）；主参考的独立支持只有一半。
4. MFS（点源、零净通量）在本席手里不收敛（BC 残差 ~2、cond≈1e18），如实记 `not_run`。
5. 文献表值（Vugts 1968）本席亦未取得（无外网）；`external_anchor` 仍为生态内 #19234。
6. 原单 `server_quality_audit={{gate_valid:false, key_present:false, gate_missing_warning:true}}`，
   `succeeded` 由 dev_acceptance 自动批准，不代表门禁被平台认过。

## ⑥ 误差判定（逐项回答是否 <3%）
| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 深浸圆柱 偶/奇镜像（8a） | ρπa² | -0.59% / +0.98% | **是** |
| 深浸圆柱（2a） | ρπa² | -11.51% / +13.65% | 否（物理自由面干涉，已声明不计数值验证） |
| 判定分辨率 160→640 漂移 | 自身 | {P['discretisation_shift_160_to_640']['omega0']:+.3%} / {P['discretisation_shift_160_to_640']['infinity']:+.3%} | 是 |
| 主参考 vs 独立锚 #19234 | 228.600 t/m | {P['anchor_deviation']:+.3%} | 是（生态内锚） |
| **ω→0 路线 vs 本席 FV** | {FV['combined_estimate_t_per_m']:.3f} t/m | **{CC['fv_vs_audited_omega0_rel']:+.3%}** | **是（本单新增独立复核）** |
| 主参考 vs 本席 FV | 同上 | {P['reference_t_per_m']/FV['combined_estimate_t_per_m']-1:+.3%}（自由面模型带） | 是 |
| **链路声明 vs 参考（全域）** | 161.0 … 234.0 t/m | **-46.5% … -63.2%** | **否（被测链路诚实 FAIL）** |
| 判据可决性（tol=0.15, u=0.029） | δ₉₅=0.2557 | 15% 容差本身不可 95% 检出 | 否（原单已如实记录） |

## 判级与裁定（L0–L5）
**L2（独立功能实现）**：真实现（615 行，输入=链路模块→计算→结构化判决）+ 可执行入口 +
工件可重生成到逐键全等 + 19 项含变异非恒等测试。
**L3 不成立**：`grep -rn ice_gate_power` 只命中它自己的 benchmark 与文档，**无主求解器/平台调用方**
—— 判据拦不住任何真实交付（F2）。
**L4 不成立**：`tensor_benchmark_cases` 该锚 **0 行**，无登记裁判；got-vs-ref 由被审对象自算（F1）。
**L5 不成立**：无文献/实验外部对标与工程闭环。
**处置：approve with notes（L2）**，verdict 已落表
`tensor_maturity_verdicts` id=901（repo=TensorFEM, scope='m22319 GAP-ICE-GATE-POWER',
mission_id=22346, level=2, verdict_by='board-audit'）。

**发现清单（按严重度）**
- **F1 blocking-for-L4（平台侧）**：基准裁判缺行（0 行）。返修=登记 `(anchor_slug, case_name='ice_chain_heave_added_mass_2d', tolerance=0.15, ref_source='#19234 Q5 228.6 t/m')`。
- **F2 major（集成）**：判据无调用方，"可机检"是抱负非事实。返修=在冰区链 run/CI 里 fail-fast 调用 `gate_verdict`。
- **F3 minor（证据完整性）**：`make_doc.py:191` 硬编码 `accept（0.38%）`，而注入偏差是 -0.44%，JSON 无 0.38% 该量。
- **F4 minor（可审计性）**：④-4、⑤-1（×2）、⑤-2、⑤-3 生成表**缺表头行**。
- **F5 minor（方法表述）**：⑤-1 把 u=2.9% 归因于面板亚线性收敛；实测判定分辨率漂移仅 0.04–0.1%，u 来自 #19234 路线残差与 ±1.93% 自由面模型带（u 偏保守，非虚标）。
- **F6 信息项**：MFS 与 ω→∞ 复核 `not_run`。
- **成立且不虚标**：控制 N1–N5 全是真实来源（链路通道/另一路线/被否的 m22156 参考），
  不是 `ref×常数`（m22156 失分模式已修）；元闸双向拒绝退化闸；文档可重生成；19 项全跑。

**未做的检查（unknown/not_run）**：现成基准行逐项比对（无对象）、调用方 grep（无调用方）、
ω→∞ 独立复核、R=240 & h≤0.5、Vugts 1968 文献表、MFS、原单集群作业 cj4035。
"""


def main() -> int:
    try:
        tok = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{BASE}/auth",
        data=json.dumps({"user": USER, "pass": PASS}).encode(),
        headers={"Content-Type": "application/json"})).read())["token"]
    except Exception as exc:
        print("AUTH FAILED (payload kept on disk):", exc)
        return 1

    figs = [
        ("fig1_independent_reference.png",
         "（a）本席独立有限体积路线的 h 收敛（盒子 60/120 m）与力积分/动能双泛函，"
         "叠加闭式上下界 161.0/322.0 t/m、被审计 ω→0（225.197）与 ω→∞（234.042）路线、"
         "主参考 229.620 及链路声明 86.1 t/m；"
         "（b）盒尺寸 R 外推：增量比 3.90≈2² ⇒ 截断误差 O(1/R²)，合成估计 225.83 t/m。"),
        ("fig2_power_recomputed.png",
         "本席用被审计判据的公开公式重算：护带规则（tol=0.15,u=0.029,k=2）与裸规则 |δ|≤tol 的检出率、"
         "以及护带误接受率；标出接受限 0.092、拒绝限 0.208、δ₉₅=0.2557 与被审偏差 0.625（检出率 1.00）。"),
        ("fig3_got_vs_ref_bands.png",
         "三个参考值（P1–P3）与五个真实负控（N1–N5）相对被审计参考的偏差，落在护带判定的"
         "接受/不确定/拒绝三段带中的位置（本席按发布数值逐条重算）。"),
    ]
    payload_figs = []
    for name, cap in figs:
        png = (HERE / name).read_bytes()
        payload_figs.append({"name": name, "caption": cap,
                             "png_base64": base64.b64encode(png).decode()})

    title = "m22346 交差核验：GAP-ICE-GATE-POWER 信息量闸判级 L2（独立 FV + 闭式界复核）"
    summary = (f"对抗式复核 m22319 判据：参考 {P['reference_t_per_m']:.2f} t/m 由本席独立有限体积路线"
                    f"（{FV['combined_estimate_t_per_m']:.2f} t/m）与闭式椭圆界 [161.0, 322.0] 双重支持，"
                    f"声明 86.1 t/m 全域 reject（最有利参考下仍 -46.5%）；基准行 0 行 ⇒ L4 不可达，判级 L2（approve with notes）")
    (HERE / "report_m22346.json").write_text(json.dumps({
        "title": title, "summary": summary,
        "scenario_key": "ice-structure", "mission_id": 22346,
        "tags": ["GAP-ICE-GATE-POWER", "board-audit", "maturity-L2"],
        "markdown": MARKDOWN, "figures": payload_figs}, ensure_ascii=False))
    body = json.dumps({
        "title": title, "summary": summary,
        "scenario_key": "ice-structure",
        "mission_id": 22346,
        "tags": ["GAP-ICE-GATE-POWER", "TensorFEM", "board-audit", "adversarial-review",
                 "added-mass", "finite-volume", "closed-form-bound", "maturity-L2"],
        "markdown": MARKDOWN,
        "figures": payload_figs,
    }, ensure_ascii=False).encode()

    req = urllib.request.Request(f"{BASE}/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {tok}"})
    try:
        r = json.loads(urllib.request.urlopen(req).read())
        print("REPORT_ID:", r.get("id") or r)
        return 0
    except Exception as exc:                      # keep the payload on disk either way
        (HERE / "report_m22346.json").write_text(json.dumps(
            {"title": title, "summary": summary, "scenario_key": "ice-structure",
             "mission_id": 22346, "tags": ["GAP-ICE-GATE-POWER", "board-audit", "maturity-L2"],
             "markdown": MARKDOWN, "figures": payload_figs}, ensure_ascii=False))
        print("POST FAILED:", exc, "-> payload saved to", HERE / "report_m22346.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())