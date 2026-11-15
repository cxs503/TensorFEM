#!/usr/bin/env python3
"""Publish paper-quality progress report for GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE."""
from __future__ import annotations

import base64
import json
import math
import os
import urllib.request
from pathlib import Path

WORK = Path(__file__).resolve().parent
DOCS = WORK / "docs" / "grillage_ice_ultimate"
API_BASE = "http://127.0.0.1:8010/api/v1"
USER = "eco_admin"
PASS = "EcoTest!2026pass"


def get_token() -> str:
    url = f"{API_BASE}/auth/login"
    data = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())["access_token"]


def b64_png(name: str) -> str:
    path = DOCS / name
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_json(name: str) -> dict | list:
    path = DOCS / name
    with open(path) as f:
        return json.load(f)


def build_markdown() -> str:
    """Build the six-element paper-quality report."""
    combined = load_json("combined_cases.json")
    vmap = load_json("ultimate_map.json")
    cases = combined["cases"]

    # Build case table
    case_rows = ""
    for c in cases:
        case_rows += (
            f"| {c['scenario']} | {c['sigma_x_MPa']:.1f} | {c['ice_patch_MPa']:.1f} | "
            f"{c['patch_area_m2']:.3f} | {c['hydrostatic_MPa']:.2f} | "
            f"{c['utilisation_plate']:.2f} | {c['utilisation_stiffener']:.2f} | "
            f"{c['utilisation_frame']:.2f} | {c['ice_pressure_ultimate_MPa']:.2f} | "
            f"{c['governing_member']} |\n"
        )

    # Build ultimate map table (first 24 entries)
    map_rows = ""
    for entry in vmap.get("rows", vmap)[:24]:
        map_rows += (
            f"| {entry['sigma_x_over_yield']:.2f} | {entry['patch_area_ratio']:.2f} | "
            f"{entry['hydrostatic_pressure_MPa']:.2f} | {entry['ice_pressure_ultimate_MPa']:.2f} | "
            f"{entry['governing_member']} | {entry['utilisation_plate']:.2f} | "
            f"{entry['utilisation_frame']:.2f} |\n"
        )

    # Verification table from verification.json
    vdata = load_json("verification.json")
    checks = vdata["checks"]
    ver_rows = ""
    for ch in checks[:25]:
        err = ch.get("rel_error", 0)
        tol = ch.get("tolerance", 0)
        passed = ch.get("passed", False)
        mark = "✅" if passed else "❌"
        ver_rows += f"| {ch['id']} | {ch['value']:.6g} | {ch['reference']:.6g} | {err:.2e} | {tol:.2e} | {mark} |\n"

    md = f"""# 加筋板架在冰压斑 + 侧压组合下的极限强度 — 论文级进展报告

**锚点**: `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`
**仓**: TensorFEM
**模块**: `src/tensorfem/grillage_ice_ultimate.py` (1015 行)
**验证面**: T1–T13, 40 gates 全绿
**独立审计**: `docs/AUDIT-GAP-PP-5-GRILLAGE-ULTIMATE-independent-review.md` (approve with notes)

---

## ① 案例介绍（物理背景/几何/参数表）

冰区（Polar Class）舷侧冰带是"板 + 纵骨 + 横框架"的正交加筋板架：冰载荷为**局部压斑**，
静水/舱内压为**满布侧压**，船体梁总纵弯曲提供**面内压缩**。三者组合下必须先答
"该板架还能承受多大冰压斑"，才能给出安全裕度。这是 JOUR-ICE-STRUCT §5 局限 6 所指的
"板格应力 → 结构极限"的最后一跳。

| 参数 | 值 |
|---|---|
| 板厚 t | 20 mm |
| 纵骨间距 b / 横框架间距 a | 0.35 m / 2.10 m |
| 板格数 / 框架间距数 | 8 / 4 |
| 横框架跨距 | 2.80 m（附着板宽取 min(a, 0.4L) = 1.12 m，剪力滞限制）|
| 纵骨 / 横框架 | T250×10+80×12 / T550×10+120×14 mm |
| 材料 | E=206 GPa, ν=0.3, σ_y=355 MPa（理想弹塑性）|
| 板格柔度 β | 0.73（Faulkner 有效宽度比 1.00）|
| 板架轴压极限 σ_u | 329.9 MPa |

以上为**文档算例取值**，不是规范摘录。

## ② 方法（方法/流程/关键设置/工具链）

1. **侧向响应（精确解析）**：四边简支正交各向异性板 Navier 双三角级数（矩形压斑 + 满布压），
   给出板格局部弯矩与模态刚度分配 f_x/f_y/f_t（和恒为 1）。Dx 来自纵骨剖面、Dy 来自横框架剖面
   （含 `min(a,0.4L)` 附着板剪力滞上限），D3 = ν√(DxDy)+2Dxy。
2. **载荷路径（两体系）**：分配归一化后严格守恒（T9：残差 0），压斑力落到实际受载构件
   （受载纵骨、压斑所辖两根横框架），比"涂抹"分布保守。
3. **构件极限判据（三支并联，取先到者）**：
   - 板格局部弯曲：m_p = σ_y·t²/4，随膜压退化 m_p·√(1−(σ_x/σ_y)²)
   - 纵骨梁柱：σ_x + M·β_m/Z ≤ σ_L（β_m 为精确 secant 放大 2(sec u−1)/u²，
     σ_L 由有效宽度剖面的 Perry-Robertson / Johnson-Ostenfeld 取小）
   - 横框架刚塑性倒塌：F_u = 8M_p/(L(1−c/2L))
4. **极限求解**：对冰压斑压力二分（利用率单调），输出控制构件、极限值、安全裕度 p_ult/p_design，
   并机读入库 JSON（板格应力 → 极限载荷映射表）。
5. **独立交叉校核**：
   - T1/T2: Timoshenko Table 30（3% 门）
   - T7: Paik-Thayamballi 经验公式（15% 门）
   - T11: 独立 13 点 FD 双调和求解器（0.5% 门）
   - T12: 手算代数纵骨控制工况（0.5% 门）
   - T13: 刚塑性板架倒塌 ≥ 弹性成员极限

## ③ 计算结果（数字表，程序化注入零手抄）

### 3.1 三档组合工况（≥3 档，验收要求）

| 工况 | σ_x [MPa] | 冰压 [MPa] | 压斑面积 [m²] | 侧压 [MPa] | U_plate | U_stiff | U_frame | p_ice,ult [MPa] | 控制构件 |
|---|---|---|---|---|---|---|---|---|---|
{case_rows}
### 3.2 板格应力 → 极限载荷映射表（节选 24 行；全表 `docs/grillage_ice_ultimate/ultimate_map.json`，共 {len(vmap.get("rows", vmap))} 行）

| σ_x/σ_y | A_ice/A | p_h [MPa] | p_ice,ult [MPa] | 控制构件 | U_plate | U_frame |
|---|---|---|---|---|---|---|
{map_rows}
### 3.3 验证门限汇总（40 gates，前 25 项）

| 锚点 | 计算值 | 基准值 | 相对误差 | 门限 | 判定 |
|---|---|---|---|---|---|
{ver_rows}
## ④ 结果图

### 图 1: 三档组合工况极限冰压

![fig1](grillage_ice_ultimate/fig1_ultimate_pressure.png)

### 图 2: 极限冰压热力图（轴压比 × 斑面积比）

![fig2](grillage_ice_ultimate/fig2_ultimate_map.png)

### 图 3: 文献基准相对误差（Paik-Thayamballi）

![fig3](grillage_ice_ultimate/fig3_literature_errors.png)

## ⑤ 注意点（踩过的坑/局限/适用范围）

- **Navier 弯矩漏乘幅值**：初版弯矩表达式未乘 A_mn，位移对而弯矩随项数发散；T1b 立刻暴露。
- **重复折减**：初版同时用有效宽度（进入 A_e）与 σ_y·(be/b) 作柱限值上限，对板折减两次，
  β=1.73 时与 Paik 差 17%；去掉上限后三档降到 ≤ 6.3%。
- **局限**：侧向用小挠度板理论、轴压用有效宽度梁柱律，无完整载荷-缩短路径（峰值后后备承载力未计）；
  弹性模态分配、不追踪屈服后重分配（对延性板架偏保守）；未建模腹板局部屈曲/侧倾、残余应力。
- **规范值未内置**：IACS 压力-面积律的 p0 由调用方注入；本文 2.0 MPa 为示例设计压力。
- **成员控制分支**：原案例从未进入"板架/构件控制"分支；新增 `member_governed_geometry()`
  （t=26 mm / bay 0.55 m / T160×6+50×7），三档组合工况全部由**纵骨**控制。

## ⑥ 误差判定：关键量 vs 基准误差逐项列出

| 关键量类别 | 参照 | 最大误差 | 门限 | 是否 <3% |
|---|---|---|---|---|
| 解析解（Timoshenko 表 30） | α=0.00406, β=0.0479 | 0.058% | 3% | **是 ✅** |
| 模态分配对称性 | f_x = f_y | 1.3e-16 | 1e-9 | **是 ✅** |
| 载荷路径守恒 | f_x + f_y + f_t = 1 | 0.0 | 0 | **是 ✅** |
| 刚塑性倒塌闭式 | 16M_p/L, 8M_p/L | <1e-12 | 1e-12 | **是 ✅** |
| Paik-Thayamballi 经验式 | 三档柔度 | 6.3% | 15% | **是 ✅**（15% 门）|
| 独立 FD 交叉校核（T11） | 13 点双调和 | 0.018% | 0.5% | **是 ✅** |
| 手算代数交叉校核（T12） | 显式闭式 | 4.0e-5% | 0.5% | **是 ✅** |

**总体判定**：所有文献/解析基准对照误差均 <15%；求解器级（Timoshenko、FD、手算代数）全部 <3%。
honest-FAIL 无。

---
*Generated by TensorFEM grillage_ice_ultimate module*
*Anchor: GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE*
*Mission: m22070*
"""
    return md


def main():
    token = get_token()
    md = build_markdown()

    figures = []
    for name, caption in [
        ("fig1_ultimate_pressure.png", "三档组合工况极限冰压与安全裕度"),
        ("fig2_ultimate_map.png", "极限冰压热力图（轴压比 × 斑面积比）"),
        ("fig3_literature_errors.png", "文献基准相对误差（Paik-Thayamballi）"),
    ]:
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": b64_png(name),
        })

    report = {
        "title": "GAP-PP-5 加筋板架冰压斑+侧压组合极限强度：40 gates 全绿 + 独立审计 approve",
        "summary": (
            "TensorFEM grillage_ice_ultimate 模块完成验收：三档组合工况（板控）+ 三档纵骨控制工况，"
            "板格应力→极限载荷映射表 80 行机读入库；40 验证门全绿（T1-T13）；"
            "独立 FD 交叉校核误差 ≤0.018%；Paik-Thayamballi 经验式对照最大 6.3%（<15% 门）；"
            "独立审计 approve with notes（F1 成员控制分支已补、F2 文档主张已更正）。"
        ),
        "scenario_key": "GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE",
        "mission_id": 22070,
        "tags": ["TensorFEM", "grillage", "ice-structure", "ultimate-strength",
                 "verification", "Paik-Thayamballi", "JOUR-ICE-STRUCT", "m22070"],
        "markdown": md,
        "figures": figures,
    }

    url = f"{API_BASE}/tensor/reports"
    data = json.dumps(report).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    print(f"Report published: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
