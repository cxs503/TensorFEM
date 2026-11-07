"""Publish the stiffened-panel ultimate-strength report to the platform.

The markdown body is generated from ``artifacts/stiffened_panel_ultimate/evidence.json``
so that every number in the text is the value the benchmark wrote (no manual
transcription).  Figures are downscaled before embedding to keep the request
payload small.
"""
from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "artifacts" / "stiffened_panel_ultimate" / "evidence.json"
FIGURES = ROOT / "docs" / "assets" / "stiffened_panel_ultimate"
MISSION_ID = 5097


def _api_base() -> str:
    return os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")


def _login() -> str:
    payload = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    request = urllib.request.Request(f"{_api_base()}/auth/login", data=payload,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read())
    for key in ("token", "access_token"):
        if key in body:
            return body[key]
    raise RuntimeError(f"login response has no token: {sorted(body)}")


def _png(path: pathlib.Path, width: int = 900, limit: int = 55000) -> str:
    from PIL import Image
    image = Image.open(path).convert("RGB")
    if image.width > width:
        ratio = width / image.width
        image = image.resize((width, int(image.height * ratio)))
    while True:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
        if len(data) <= limit or image.width < 420:
            break
        ratio = 0.85
        image = image.resize((int(image.width * ratio), int(image.height * ratio)))
    return base64.b64encode(data).decode()


def _fmt(value, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def build_markdown(evidence: dict) -> str:
    closed = evidence["closed_form"]
    ice = evidence["iso19906"]
    hull = evidence["hull_girder"]
    errors = evidence["fe_vs_closed_form"]
    convergence = evidence["step_convergence"]
    criteria = evidence["criteria"]
    fe = evidence["finite_element"]

    lines: list[str] = []
    lines.append("# 加筋板极限强度：壳体弹塑性弧长路径 + ISO 19906 / CSR 校核链\n")
    lines.append("## 1. 案例介绍\n")
    lines.append(
        "胜负几何采用任务书给定的冰带板格（板厚 t = 20 mm、纵骨间距 s = 800 mm、"
        "横框架间距 a = 2.4 m、腹板 200×10 mm、面板 80×12 mm、E = 206 GPa、"
        "σ_y = 235 MPa），另加一个可在现有算力下解析失稳模态的单跨板格"
        "（a = 0.8 m、t = 12 mm、腹板 180×9 mm）用于追踪完整后屈曲路径。"
        "两套几何的板格柔度分别为 β = "
        f"{_fmt(closed['ice_belt']['plate_slenderness'],3)} 与 β = "
        f"{_fmt(closed['square_bay']['plate_slenderness'],3)}。\n")
    lines.append(
        "| 量 | 冰带板格 | 单跨板格 |\n|---|---|---|\n"
        f"| 板柔度 β | {_fmt(closed['ice_belt']['plate_slenderness'],3)} | "
        f"{_fmt(closed['square_bay']['plate_slenderness'],3)} |\n"
        f"| 柱柔度 λ | {_fmt(closed['ice_belt']['column_slenderness'],3)} | "
        f"{_fmt(closed['square_bay']['column_slenderness'],3)} |\n"
        f"| 板极限 σ_u/σ_y (Faulkner) | "
        f"{_fmt(closed['ice_belt']['plate_limiting_stress'] / 235e6,3)} | "
        f"{_fmt(closed['square_bay']['plate_limiting_stress'] / 235e6,3)} |\n"
        f"| 纵骨极限 σ/σ_y (Johnson–Ostenfeld) | "
        f"{_fmt(closed['ice_belt']['stiffener_limiting_stress'] / 235e6,3)} | "
        f"{_fmt(closed['square_bay']['stiffener_limiting_stress'] / 235e6,3)} |\n"
        f"| 板格极限 σ_u/σ_y (面积加权) | {_fmt(closed['ice_belt']['panel_ultimate_over_yield'],3)} | "
        f"{_fmt(closed['square_bay']['panel_ultimate_over_yield'],3)} |\n")

    lines.append("\n## 2. 方法与工具链\n")
    lines.append(
        "**闭式链**（`tensorfem.stiffened_panel_ultimate`）：Faulkner/von Kármán 有效宽度板极限、"
        "Johnson–Ostenfeld 非弹性柱曲线（含 Euler 应力 π²EI/(A a²)）、St Venant 扭转屈曲筛选、"
        "面板局部屈曲（k=0.425）、面积加权板格极限、Smith(1977) 渐进崩溃（逐曲率积分、"
        "中和轴取弹性值）、Navier 双级数精确解（四边简支板受均布压）、"
        "刚性塑性屈服线上限解（4 铰金字塔机制 + 单向条带机制）。\n")
    lines.append(
        "**数值链**（`tensorfem.stiffened_panel_unit_fe`）：半跨加筋板单元胞，板=分层 Shell4 "
        "（沿厚分层 J2 塑性），腹板/面板为独立厚度的壳条带与板共节点刚性连接；"
        "x=0 为框架线（u_x=0, w=0），x=a 为加载边（w=0），y=0 为纵骨中心对称面"
        "（u_y=θ_x=θ_z=0），y=b/2 只保留转动对称。缺陷取柱型 a/1000·sin(πx/a) 与板型局部模态。"
        "载荷按节点 tributary 面积乘 1 Pa 归一，因此弧长载荷因子直接等于施加的平均应力（Pa）。"
        "求解器为球面弧长（Crisfield）配一致的 J2 返回映射切线，且**每个收敛点提交一次材料状态**"
        "（把状态整条路径冻结会退化为单一塑性增量，这是本模块必须自带驱动器的原因）。\n")
    lines.append(
        "**规则校核**：ISO 19906 冰带板格链 = 压力-面积律 p_ice → 板格弯曲应力（Navier 级数）"
        "→ 许用应力利用率，并给出屈服线极限压力作为 ULS 对照；CSR 链 = 波/冰垂向弯矩 → Smith "
        "渐进崩溃 M_u → 利用率（阻力侧分项系数由调用方给出，基准取 1.05）；"
        "冰-波组合用二次利用面 sqrt((σ_c/σ_u)² + (σ_b/σ_y)²) ≤ 1。\n")

    lines.append("\n## 3. 计算结果\n")
    lines.append("### 3.1 有限元弧长路径（程序化输出，见 evidence.json）\n")
    lines.append("| 算例 | 网格 | 平均应力峰值 σ_u/σ_y | 是否出现峰值 | 峰后下降 | 完成步数 | 闭式 σ_u/σ_y | 相对偏差 |\n|---|---|---|---|---|---|---|---|\n")
    for name, trace in fe.items():
        if name not in errors:
            continue
        error = errors[name]["relative_error"]
        lines.append(
            f"| {name} | {trace['mesh']['nx']}×{trace['mesh']['ny']} | "
            f"{_fmt(trace['ultimate_over_yield'],4)} | {_fmt(trace['has_peak'])} | "
            f"{_fmt(trace['post_peak_drop'],3)} | {trace['points_completed']} | "
            f"{_fmt(errors[name]['closed_form_over_yield'],4)} | {_fmt(error,3)} |\n")
    if convergence:
        lines.append(
            f"\n步长收敛：Δs = {_fmt(convergence['coarse_step'],3)} 与 Δs/2 = "
            f"{_fmt(convergence['fine_step'],3)} 两次路径的峰值应力相对差 "
            f"**{_fmt(convergence['relative_change'],3)}**（判据 < 15%）。\n")

    lines.append("\n### 3.2 ISO 19906 冰带板格\n")
    lines.append(
        f"- 冰压力 p_ice = {_fmt(ice['ice_pressure'] / 1e6,3)} MPa"
        f"（作用面积 {_fmt(ice['load_area'],3)} m²，压力-面积系数由调用方给定）\n"
        f"- 板格最大弯曲应力 = {_fmt(ice['maximum_bending_stress'] / 1e6,3)} MPa，"
        f"许用 = {_fmt(ice['allowable_stress'] / 1e6,3)} MPa，"
        f"许用利用率 = **{_fmt(ice['allowable_utilisation'],3)}**\n"
        f"- 屈服线极限压力 = {_fmt(ice['plastic_collapse_pressure'] / 1e3,3)} kPa"
        f"（机制 {ice['governing_mechanism']}，a/b = {_fmt(ice['yield_line_aspect'],3)}），"
        f"ULS 利用率 = **{_fmt(ice['uls_utilisation'],3)}**\n"
        f"- 结构侧反推的许用冰压：弹性许用应力法（σ_allow = 0.8σ_y）"
        f"{_fmt(ice['allowable_pressure_elastic'] / 1e3,3)} kPa，"
        f"刚性塑性上限法 {_fmt(ice['allowable_pressure_plastic'] / 1e3,3)} kPa；"
        f"规范（ISO 19906）给出的许用冰压本环境不可得，故\"与规范值偏差<5%\"判据 "
        f"**WITHHELD**，只给出结构侧可验证的误差项。\n"
        f"- 板格弯曲系数 M/(p b²) = {_fmt(ice['plate_field_coefficient'],4)}；"
        f"教科书方板值 {ice['textbook_square_coefficient']}，级数解相对偏差 "
        f"**{_fmt(ice['series_error'],3)}**\n")

    lines.append("\n### 3.3 CSR 总纵强度（Smith 渐进崩溃）\n")
    csr = hull["csr_check"]
    lines.append(
        f"- 中和轴 z = {_fmt(hull['collapse']['neutral_axis'],4)} m，剖面惯性矩 = "
        f"{_fmt(hull['collapse']['section_inertia'],4)} m⁴\n"
        f"- M_u = {_fmt(csr['ultimate_moment'] / 1e6,4)} MN·m，"
        f"完全塑性上限 M_p = {_fmt(hull['collapse']['upper_bound_fully_plastic_moment'] / 1e6,4)} MN·m\n"
        f"- 设计弯矩 M = {_fmt(csr['applied_moment'] / 1e6,4)} MN·m，"
        f"阻力侧分项系数 1.05，利用率 = **{_fmt(csr['utilisation'],3)}**，"
        f"通过 = {_fmt(csr['passed'])}\n"
        f"- 弯矩-曲率曲线在峰后进入平台（理想塑性极限应力模型无软化），"
        f"plateau_reached = {_fmt(hull['collapse']['plateau_reached'])}\n")

    combined = ice["combined_check"]
    lines.append("\n### 3.4 冰-波组合\n")
    lines.append(
        f"- 面内利用率 {_fmt(combined['in_plane_utilisation'],3)}，"
        f"弯曲利用率 {_fmt(combined['bending_utilisation'],3)}，"
        f"组合利用率 **{_fmt(combined['combined_utilisation'],3)}**"
        f"（{combined['interaction']}），通过 = {_fmt(combined['passed'])}\n")

    lines.append("\n## 4. 结果图\n")
    lines.append("图 1：左为 Shell4 弧长载荷-端缩曲线（含闭式板格极限水平线），"
                 "右为 Faulkner 板极限、面积加权板格极限与有限元峰值对板柔度 β 的分布。\n")
    lines.append("图 2：Smith 渐进崩溃的弯矩-曲率曲线、完全塑性上限与设计弯矩。\n")
    lines.append("图 3：冰压+面内压缩组合下板格横向挠度场（半跨模型，纵骨位于 y=0）。\n")

    lines.append("\n## 5. 注意点与局限\n")
    lines.append(
        "1. **没有实验对标**：Smith(1977) 加筋板试验数据本环境不可得，"
        "因此“与实验误差<10%”判据标记 **WITHHELD**，未用记忆中的数值冒充；"
        "替代证据是有限元与独立闭式链的比对 + 步长收敛。\n"
        "2. **ABAQUS Riks 对标同样 WITHHELD**：本机无 ABAQUS；"
        "替代证据是（a）弧长驱动器的弹性极限与 tensorfem 已验证的 "
        "solve_arc_length 逐点一致（测试相对差 1e-5），（b）Δs 与 Δs/2 的路径收敛。\n"
        "3. **横向网格必须能解析局部屈曲模态**：早期用 ny=2 的半跨网格（0.4 m 宽仅 2 单元）"
        "无法解析板件屈曲半波，结果退化为“压溃型”强度（1.17–1.21 σ_y）且没有极限点；"
        "该粗网格结果只作为网格敏感性证据保留，不作为强度预报。\n"
        "4. **本构为理想塑性**（硬化=0）、无残余应力、无腐蚀、无开孔；"
        "极限应力曲线无峰后软化，故总纵强度曲线是平台而非下降段，M_u 偏保守地定义为首个平台点。\n"
        "5. **规则系数全部由调用方给出**：ISO 19906 压力-面积系数（1.5 MPa @ 0.5 m²，指数 −0.3）"
        "与 CSR 阻力分项系数（1.05）都是**假定代表值**，不是从规范条文读取；"
        "任何设计用途必须用适用的规范版本替换。\n"
        "6. **冰-波组合面是筛选假设**（二次利用面），不是船级社相互作用公式。\n")

    lines.append("\n## 6. 误差判定（逐项）\n")
    lines.append("| 关键量 | 基准 | 相对偏差 | 是否 < 3% |\n|---|---|---|---|\n")
    series_error = ice["series_error"]
    lines.append(f"| 方板弯曲系数 M/(p b²) | Timoshenko 表值 0.0479 | {_fmt(series_error,3)} | "
                 f"{'是' if series_error < 0.03 else '否'} |\n")
    lines.append(f"| 方板屈服线极限压力 | 24 M_p/a²（同机制闭式） | "
                 f"{_fmt(abs(ice['square_yield_line_ratio'] - 1.0),3)} | "
                 f"{'是' if abs(ice['square_yield_line_ratio'] - 1.0) < 0.03 else '否'} |\n")
    for name, error in errors.items():
        lines.append(f"| FE 板格极限 σ_u/σ_y（{name}） | 闭式链 σ_u/σ_y | "
                     f"{_fmt(error['relative_error'],3)} | "
                     f"{'是' if error['relative_error'] < 0.03 else '否'} |\n")
    if convergence:
        lines.append(f"| 弧长步长 Δs 与 Δs/2 峰值 | 自比较 | "
                     f"{_fmt(convergence['relative_change'],3)} | "
                     f"{'是' if convergence['relative_change'] < 0.03 else '否'} |\n")
    lines.append("| Smith(1977) 实验值 | 不可得 | — | **无法判定（WITHHELD）** |\n")
    lines.append("| ABAQUS Riks 路径 | 不可得 | — | **无法判定（WITHHELD）** |\n")

    lines.append("\n### 判据状态\n")
    lines.append("```json\n" + json.dumps(criteria, indent=1, ensure_ascii=False) + "\n```\n")
    lines.append("\n产物：`artifacts/stiffened_panel_ultimate/evidence.json`、"
                 "`artifacts/stiffened_panel_ultimate/fe_traces.json`、"
                 "`docs/stiffened-panel-ultimate-strength.md`、"
                 "`docs/assets/stiffened_panel_ultimate/*.png`。\n")
    return "".join(lines)


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text())
    markdown = build_markdown(evidence)
    figures = []
    for entry in evidence.get("figures", []):
        path = pathlib.Path(entry["path"])
        if not path.exists():
            continue
        figures.append({"name": entry["name"], "caption": entry["caption"],
                        "png_base64": _png(path)})
    payload = {
        "title": "加筋板极限强度：弹塑性壳弧长路径 + ISO 19906/CSR 校核链 (m5097)",
        "summary": ("Stiffened-panel ultimate strength from an elasto-plastic Shell4 unit cell "
                    "with a stateful Riks path, plus an independent closed-form chain "
                    "(Faulkner/JO/Smith/Navier/yield-line) and ISO 19906 / CSR screening. "
                    "Smith 1977 and ABAQUS Riks comparisons are WITHHELD: the sources are not "
                    "available here."),
        "scenario_key": "ship.structure.stiffened_panel_ultimate",
        "mission_id": MISSION_ID,
        "tags": ["TensorFEM", "structure", "ultimate-strength", "arc-length", "ISO19906", "CSR"],
        "markdown": markdown,
        "figures": figures,
    }
    body = json.dumps(payload).encode()
    token = _login()
    request = urllib.request.Request(f"{_api_base()}/tensor/reports", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request) as response:
        print(response.status, response.read()[:400].decode(errors="replace"))
    print("payload bytes", len(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())