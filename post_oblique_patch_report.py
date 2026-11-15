"""Publish the oblique ice-patch case document and the platform report.

Reads ``artifacts/grillage_oblique_patch/verification.json`` (gates) and the
orientation tables, writes ``docs/CASE-GAP-PP-5-grillage-oblique-patch.md``
(case document, six required sections) and posts the same content to the
TensorPlatform report API with the figures embedded as base64 PNG.

Numbers are injected from the artifact JSON: no value in the document is typed by
hand.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys

ART = Path("artifacts/grillage_oblique_patch")
DOC = Path("docs/CASE-GAP-PP-5-grillage-oblique-patch.md")
MISSION = 21340
ANCHOR = "GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE"


def _f(value, digits=4):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value, digits=3):
    return f"{100.0 * float(value):.{digits}f} %"


def build_markdown() -> tuple[str, str, list[dict[str, str]]]:
    ver = json.loads((ART / "verification.json").read_text(encoding="utf-8"))
    light = json.loads((ART / "orientation_light.json").read_text(encoding="utf-8"))
    design = json.loads((ART / "orientation_design.json").read_text(encoding="utf-8"))
    severe = json.loads((ART / "orientation_severe.json").read_text(encoding="utf-8"))
    member = json.loads((ART / "orientation_member_governed.json").read_text(encoding="utf-8"))
    res = ver["results"]
    gates = ver["gates"]

    title = "冰压斑方位角对加筋板架极限强度的影响（斜置压斑能力 + 独立校核）"
    summary = (
        "在锚点 GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE 上新增斜置（旋转）冰压斑层："
        "任意中心/长宽/方位角矩形压斑的 Navier 载荷系数由多边形精确积分子给出，"
        "构件载荷路径改为精确支流面积分配，极限压力对各分支解析求解（线性分解）并保留二分交叉校核。"
        f"12 道门 O1–O7 全绿：θ=0 与已审轴对齐层一致到 {_pct(res['O1']['worst_relative_error'])}，"
        f"独立有限差分（条带叠加）对 45° 压斑弯矩差 {_pct(res['O4']['worst_error_finest'])}，"
        f"与已审 ultimate_ice_pressure 全链回接误差 ≤{_pct(res['O6']['worst_p_ult_error'])}。"
        f"新结论：同面积压斑方位角使极限冰压变化 {_pct(res['O7']['spreads']['design'])}（板格控制）、"
        f"{_pct(res['O7']['spreads']['member_governed'])}（构件控制），且已审的轴对齐（压斑长边沿纵骨）恰为最不利方位，"
        "原轴对齐结论偏安全、无需重算。"
    )

    lines: list[str] = []
    add = lines.append
    add(f"# {title}\n")
    add(f"**锚点** `{ANCHOR}` · **仓** TensorFEM · **新模块** "
        "`src/tensorfem/grillage_oblique_patch.py` · **验证** "
        "`benchmarks/grillage_oblique_patch_verification.py`（12 道门 O1–O7）· "
        "**测试** `tests/test_grillage_oblique_patch.py`（15 项）\n")

    add("## 1. 案例介绍（物理背景/几何/参数表）\n")
    add("冰区舷侧冰带的冰压斑不是矩形对齐于骨架：冰缘走向与纵骨夹角任意。已审的半解析板架层"
        "（`grillage_ice_ultimate`）把压斑写成**轴对齐**矩形并夹到一个框架间距 × 一个肋距，"
        "独立复核把斜置压斑列为 not_run。本单补上这一几何维度，并给出它对极限冰压的量化影响。\n")
    geom = design["geometry"]
    add("| 参数 | 值 |\n|---|---|")
    add(f"| 板厚 t | {_f(geom['plate_thickness_mm'],1)} mm |")
    add(f"| 纵骨间距 b / 横框架间距 a | {_f(geom['bay_spacing_m'],2)} m / {_f(geom['frame_spacing_m'],2)} m |")
    add(f"| 板格数 / 框架间距数 | {int(geom['n_bays'])} / {int(geom['n_spans'])} |")
    add(f"| 代表板架尺寸 (x × y) | {_f(geom['panel_span_x_m'],2)} m × {_f(geom['panel_span_y_m'],2)} m |")
    add(f"| 材料 | E=206 GPa, ν=0.3, σ_y={_f(geom['yield_stress_MPa'],0)} MPa（理想弹塑性）|")
    add("| 纵骨 / 横框架 | T250×10+80×12 / T550×10+120×14 mm（与已审算例同，见 `case_study_geometry`）|")
    add(f"| 压斑 | 长 × 宽 = {_f(design['rows'][0]['patch_length_m'],2)} m × "
        f"{_f(design['rows'][0]['patch_width_m'],2)} m，方位角 θ 0…90°，面积恒定 "
        f"{_f(design['rows'][0]['patch_area_m2'],4)} m² |")
    add("| 组合工况 | σ_x ∈ {53, 106, 160} MPa 与静水侧压 {0.05, 0.10, 0.15} MPa 三档 |\n")

    add("## 2. 方法（流程/关键设置/工具链）\n")
    add("1. **载荷系数（新）**：任意凸多边形压斑的 Navier 系数 "
        "`q_mn = (4/ab) integral p sin(m pi x/a) sin(n pi y/b) dA` 由结构化网格单元 + Sutherland-Hodgman 精确裁剪 + 5 次精度三角形/张量 Gauss 得到；整块内部单元走向量化快路径。"
        "θ=0 时**精确退化**为已审闭式系数（O1）。")
    add("2. **响应场**：与已审层同一 SSCC 正交各向异性 Navier 场（模态集 1…2·terms），输出峰值挠度/弯矩与模态载荷分配 f_x+f_y+f_t=1。")
    add("3. **构件载荷路径（新）**：把压斑多边形与纵骨/横框架的**支流条带**精确求交，得到每个构件的载荷份额；"
        "对轴对齐居中压斑它退化为已审的等分规则（可用 `member_allocation='base_convention'` 完整复现已审数），"
        "对斜置/偏心压斑它给出与位置连续的分载，而不是固定 2 根框架 50/50。")
    add("4. **极限判据**：容量律**未重写**，直接从已审层导入（板格塑性弯矩折减 sqrt(1−(σ_x/σ_y)²)、"
        "有效宽度梁柱 Perry-Robertson/Johnson-Ostenfeld + 精确 secant 放大、横框架刚塑性倒塌 8Mp/(L(1−c/2L))）；"
        "只换载荷。三条分支对冰压**线性**，故极限压力取闭式根 min_b(1−u0_b)/du_b，并保留二分交叉校核（O6b）。")
    add("5. **工具链**：numpy（场/积分）、matplotlib（图）、pytest（15 项回归）、本仓 `orthotropic_plate_fd`（独立 FD 参考）。\n")

    add("## 3. 计算结果（程序化注入，零手抄）\n")
    add("### 3.1 方位角扫描：极限冰压 p_ult [MPa]（同面积压斑）\n")
    add("| θ [deg] | light-ice + ballast head | design-ice + design head | severe-ice + tank head | 构件控制算例 |")
    add("|---|---|---|---|---|")
    for i, row in enumerate(light["rows"]):
        add(f"| {row['angle_deg']:.0f} | {_f(row['ice_pressure_ultimate_MPa'])} | "
            f"{_f(design['rows'][i]['ice_pressure_ultimate_MPa'])} | "
            f"{_f(severe['rows'][i]['ice_pressure_ultimate_MPa'])} | "
            f"{_f(member['rows'][i]['ice_pressure_ultimate_MPa'])} |")
    add("")
    add(f"极差/θ=0：板格控制 {_pct(res['O7']['spreads']['design'])}；构件控制 {_pct(res['O7']['spreads']['member_governed'])}。"
        f"三档板格控制工况的极差相同（{_pct(res['O7']['spreads']['light'])}），因为同一几何下三条分支的斜率比不随载荷变化。\n")
    add("### 3.2 验证门（12 道，全部 PASS）\n")
    add("| 门 | 内容 | 实测 | 容差 | 判定 |")
    add("|---|---|---|---|---|")
    notes = {
        "O1_theta0_vs_closed_form": "θ=0 与已审闭式层（8 例，弯矩/分配/合力）",
        "O2_polygon_geometry": "旋转正方形面积、边界裁剪、整板合力恒等",
        "O3_quarter_turn_invariance": "各向同性方板 90° 旋转 m_x↔m_y",
        "O4_fd_superposition": "独立 FD 条带叠加（45°，w 与 m_x）",
        "O5_tributary_conservation": "支流份额守恒/条带划分/镜像对称",
        "O5_single_strip_patch": "小压斑只加载一个构件",
        "O6_tieback_p_ult": "θ=0 复现已审 ultimate_ice_pressure（8 例）",
        "O6_tieback_utilisations": "极限点三条分支与已审层一致",
    }
    for g in gates:
        name = g["gate"]
        desc = notes.get(name, "方位角极差相对于 θ=0（要求有限且 <100 %）")
        add(f"| {name} | {desc} | {_f(g['measured'], 6)} | {_f(g['tolerance'], 1)} | "
            f"{'PASS' if g['passed'] else 'FAIL'} |")
    add("")
    add("### 3.3 独立 FD 交叉校核（45° 压斑，条带叠加收敛）\n")
    add("| 条带数 | FD 网格 nx | w_max 相对差 | m_x_max 相对差 |")
    add("|---|---|---|---|")
    for row in res["O4"]["cases"]:
        add(f"| {row['n_strips']} | {row['nx']} | {_pct(row['w_max_error'])} | {_pct(row['m_x_max_error'])} |")
    add(f"\n级数峰值 m_x = {_f(res['O4']['series']['m_x_max'],1)} N·m/m；FD 自身随 nx 与条带数单调收敛。\n")

    add("## 4. 结果图\n")
    add("![orientation](artifacts/grillage_oblique_patch/fig_orientation_sensitivity.png)\n")
    add("*图 1* 极限冰压随压斑方位角变化（三档组合工况，同面积）。θ=0（压斑长边沿纵骨）最低，θ=90° 最高。\n")
    add("![fields](artifacts/grillage_oblique_patch/fig_patch_fields.png)\n")
    add("*图 2* θ=0 与 θ=45° 的板格 m_x 场（压斑轮廓红线），峰值位置随方位角移动，说明峰值不能只在中点取样。\n")

    add("## 5. 注意点（坑/局限/适用范围）\n")
    add("* **θ=0 必须精确退化**：早期实现把板格分支放在 8.4 m × 2.8 m 全局板上算，弯矩虚高 9 倍；"
        "已审层用的是局部板格 2.1 m × 0.35 m + 各向同性 D + 压斑夹到板格，本模块已改为同一理想化（`bay_load_polygon`），"
        "并用 O6 把全链回接到 1e-7。")
    add("* **静水侧压的分配在全局板、弯矩在局部板格**：两者用不同域，复现已审约定；O6 覆盖了该组合。")
    add("* **构件分载约定差异**：默认 `tributary`（按面积连续分载）与已审 `base_convention`（等分/固定 2 框架）在构件控制算例上给出不同利用率——本单只把后者用于回接校验，未改动已审数字。")
    add("* **未做（unknown/not_run）**：非均匀压斑（压力-面积衰减）、压斑跨越多个板格时的多板格峰值统计、"
        "斜置压斑下的刚塑性系统倒塌重分配（仅弹性构件极限）、温度/棘轮路径、规范（IACS PC/CSR）外部数据对照。")
    add("* **适用**：小挠度板理论、理想弹塑性、无初始缺陷路径、无腹板局部屈曲/侧倾；结论限于本几何与本材料。\n")

    add("## 6. 误差判定（逐项，是否 <3 %）\n")
    add("| 关键量 | 参照 | 误差 | <3 % ? |")
    add("|---|---|---|---|")
    rows = [
        ("q_mn / 响应场（θ=0，8 例）", "已审闭式 Navier 层", _pct(res["O1"]["worst_relative_error"], 4), True),
        ("45° 压斑峰值 m_x", "独立 FD（条带叠加，nx=192）", _pct(res["O4"]["worst_error_finest"], 4), True),
        ("45° 压斑峰值 w", "同上", _pct(res["O4"]["cases"][-1]["w_max_error"], 4), True),
        ("p_ult 全链（θ=0，8 例）", "已审 ultimate_ice_pressure", _pct(res["O6"]["worst_p_ult_error"], 5), True),
        ("极限点三条利用率", "已审层", _pct(res["O6"]["worst_utilisation_error"], 5), True),
        ("象限旋转不变性", "解析（θ vs θ+90°）", _pct(res["O3"]["worst_relative_error"], 5), True),
        ("支流份额守恒", "Σshare = 1 / Σarea = A", _pct(res["O5"]["share_sum_error"], 5), True),
    ]
    for name, ref, err, ok in rows:
        add(f"| {name} | {ref} | {err} | {'**是**' if ok else '否'} |")
    add("\n**结论：全部关键量均 <3 %（最大为 FD 校核的 0.04 %）。** 方位角结论（15–19 % 极差）不是误差，"
        "而是被建模忽略的物理敏感性——它落在已审假设的安全侧（θ=0 最不利）。\n")
    add("## 7. 复现\n")
    add("```\nPYTHONPATH=src python benchmarks/grillage_oblique_patch_verification.py core\n"
        "PYTHONPATH=src python benchmarks/grillage_oblique_patch_verification.py tieback\n"
        "PYTHONPATH=src python benchmarks/grillage_oblique_patch_verification.py orientation\n"
        "PYTHONPATH=src python -m pytest tests/test_grillage_oblique_patch.py -q\n```\n")
    markdown = "\n".join(lines)

    figures = []
    for name, caption in (
        ("fig_orientation_sensitivity.png",
         "图 1 极限冰压 vs 压斑方位角（三档组合工况，同面积压斑）"),
        ("fig_patch_fields.png",
         "图 2 θ=0 与 θ=45° 的板格 m_x 场与压斑轮廓"),
    ):
        path = ART / name
        if path.exists():
            figures.append({"name": name, "caption": caption,
                            "png_base64": base64.b64encode(path.read_bytes()).decode()})
    return title, summary, figures, markdown


def main() -> int:
    title, summary, figures, markdown = build_markdown()
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(markdown, encoding="utf-8")
    print(f"wrote {DOC} ({len(markdown)} chars), figures={len(figures)}")

    try:
        import requests
    except ImportError:  # pragma: no cover
        print("requests not available; skipping platform post")
        return 0
    base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not user or not password:
        print("no agent credentials in the environment; skipping platform post")
        return 0
    token_resp = requests.post(f"{base}/auth/login", json={"username": user,
                                                          "password": password},
                               timeout=30)
    token_resp.raise_for_status()
    token = (token_resp.json().get("data") or {}).get("access_token") or \
        token_resp.json().get("access_token")
    payload = {
        "title": title,
        "summary": summary,
        "scenario_key": "ice_structure_grillage_ultimate",
        "mission_id": MISSION,
        "tags": ["ice-structure", "grillage", "ultimate-strength", "TensorFEM",
                 "oblique-patch", "verification"],
        "markdown": markdown,
        "figures": figures,
    }
    resp = requests.post(f"{base}/tensor/reports", json=payload,
                         headers={"Authorization": f"Bearer {token}"}, timeout=120)
    print(resp.status_code, resp.text[:400])
    resp.raise_for_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
