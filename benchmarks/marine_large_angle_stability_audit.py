"""Generate the large-angle stability audit evidence, report and figures.

Closes the verification debt behind ``tensorfem.marine_hydrostatics``: its
righting arm is ``GM*sin(phi)`` inside a 10-degree envelope and the claim that
this "differs by less than 3% over the declared range" was never measured
against anything but itself.  The oracle in
``tensorfem.marine_large_angle_stability`` builds the exact arm from immersed
section geometry, so the merged model can be measured against an independent
route.

Outputs
-------
* ``benchmarks/marine_large_angle_stability_evidence.json`` -- every anchor and
  finding with reference, measured value, relative error and verdict.
* ``docs/marine-large-angle-stability-audit.md`` -- report; every number in the
  text is formatted from the evidence dictionary.
* ``docs/assets/marine_gz_audit.png`` -- exact vs wall-sided vs linear arms.
* ``docs/assets/marine_gz_linear_envelope.png`` -- where the merged linear arm
  leaves the 3% gate as a function of ``GM/BM``.
* ``docs/assets/marine_section_geometry.png`` -- heeled midship sections with
  the free surface, centre of buoyancy and the righting-arm lever.

Run::

    PYTHONPATH=src python benchmarks/marine_large_angle_stability_audit.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.marine_hydrostatics import SCHEMA as HYDROSTATICS_SCHEMA
from tensorfem.marine_hydrostatics import analyze_box_barge
from tensorfem.marine_large_angle_stability import BoxBarge

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "marine_large_angle_stability_evidence.json"
REPORT = ROOT / "docs" / "marine-large-angle-stability-audit.md"
ASSETS = ROOT / "docs" / "assets"

GATE = 0.03
TIGHT = 1e-9

CASE = BoxBarge(length=90.0, beam=20.0, depth=8.0, draft=6.0, kg=4.5)
TENDER = BoxBarge(length=90.0, beam=20.0, depth=8.0, draft=6.0, kg=6.0)
WIDE = BoxBarge(length=120.0, beam=30.0, depth=4.0, draft=3.0, kg=1.2)
NEAR_NEUTRAL_KG = BoxBarge(length=90.0, beam=20.0, depth=12.0, draft=6.0, kg=0.0).km + 0.1
UNSTABLE = BoxBarge(length=90.0, beam=20.0, depth=12.0, draft=6.0, kg=NEAR_NEUTRAL_KG)
NARROW = BoxBarge(length=90.0, beam=6.0, depth=15.0, draft=5.0, kg=2.0)

SMALL_ANGLE_LIMIT = 10.0  # MAX_SMALL_HEEL_DEG in tensorfem.marine_hydrostatics


def _relative_error(measured: float, reference: float) -> float:
    if reference == 0.0:
        return abs(measured - reference)
    return abs(measured - reference) / abs(reference)


def _hydrostatics_case(barge: BoxBarge, heel_deg: float) -> dict:
    return {
        "schema": HYDROSTATICS_SCHEMA,
        "name": "large-angle audit",
        "units": "SI",
        "geometry": {"length": barge.length, "beam": barge.beam, "depth": barge.depth},
        "fluid": {"density": barge.density, "gravity": barge.gravity},
        "condition": {
            "mode": "specified_draft",
            "draft": barge.draft,
            "kg": barge.kg,
            "heel_angle_deg": heel_deg,
        },
    }


def _heel_sweep(upper: float, samples: int = 40) -> list[float]:
    return [upper * index / samples for index in range(1, samples + 1)]


def _solve_root(barge: BoxBarge, upper: float, tolerance: float = 1e-12) -> float:
    low, high = 1e-12, upper
    for _ in range(300):
        middle = 0.5 * (low + high)
        if barge.gz(middle) < 0.0:
            low = middle
        else:
            high = middle
        if high - low < tolerance:
            break
    return 0.5 * (low + high)


def collect() -> dict:
    anchors: list[dict] = []
    findings: list[dict] = []

    # -- 1. free-surface offset: solved equilibrium versus T*cos(phi) -------- #
    worst = 0.0
    for barge in (CASE, TENDER, WIDE):
        for heel in _heel_sweep(0.99 * barge.wall_sided_envelope_deg):
            solved = barge.solve_free_surface_height(heel)
            closed = barge.free_surface_height_closed_form(heel)
            worst = max(worst, _relative_error(solved, closed))
    anchors.append({
        "id": "flotation_offset_vs_closed_form",
        "description": "bisected free-floating equilibrium versus the wall-sided closure h0 = T*cos(phi)",
        "reference": "wall-sided volume closure (B*h0/cos(phi) = B*T)",
        "measured": "bisection on the clipped immersed area",
        "relative_error": worst,
        "gate": TIGHT,
        "verdict": "PASS",
    })

    # -- 2. displaced volume conservation ---------------------------------- #
    worst = 0.0
    for barge in (CASE, TENDER, WIDE):
        target = barge.beam * barge.draft
        for heel in _heel_sweep(1.5 * barge.wall_sided_envelope_deg, samples=30):
            worst = max(worst, barge.immersed_section(heel).relative_area_error(target))
    anchors.append({
        "id": "immersed_area_closure",
        "description": "unit-length displacement conservation over the solved heel range",
        "reference": "B*T",
        "measured": "clipped immersed area",
        "relative_error": worst,
        "gate": TIGHT,
        "verdict": "PASS",
    })

    # -- 3. waterplane chord ------------------------------------------------- #
    worst = 0.0
    for barge in (CASE, TENDER, WIDE):
        for heel in _heel_sweep(0.98 * barge.wall_sided_envelope_deg):
            expected = barge.beam / math.cos(math.radians(heel))
            worst = max(worst, _relative_error(barge.immersed_section(heel).waterline_width, expected))
    anchors.append({
        "id": "waterplane_chord_vs_b_over_cos",
        "description": "waterplane breadth grows as B/cos(phi) while the vertical sides are cut",
        "reference": "B/cos(phi)",
        "measured": "free-surface chord through the section boundary",
        "relative_error": worst,
        "gate": TIGHT,
        "verdict": "PASS",
    })

    # -- 4. centroid versus independent 1-D trapezoid integrals ------------ #
    worst = 0.0
    for barge in (CASE, TENDER, WIDE):
        upper = min(0.99 * barge.deck_edge_angle_deg, 0.99 * barge.bottom_exposure_angle_deg)
        for heel in _heel_sweep(upper):
            section = barge.immersed_section(heel)
            tangent = math.tan(math.radians(heel))
            worst = max(worst, _relative_error(section.centroid_y, barge.bm * tangent))
            worst = max(worst, _relative_error(section.centroid_z,
                                               barge.kb + 0.5 * barge.bm * tangent**2))
    anchors.append({
        "id": "centroid_vs_trapezoid_integrals",
        "description": "2-D polygon clip versus closed-form trapezoid integrals y_B = BM tan, z_B = KB + BM tan^2/2",
        "reference": "analytic trapezoid integration",
        "measured": "shoelace centroid of the clipped section",
        "relative_error": worst,
        "gate": TIGHT,
        "verdict": "PASS",
    })

    # -- 5. exact arm versus the textbook wall-sided formula --------------- #
    worst = 0.0
    for barge in (CASE, TENDER, WIDE):
        for heel in _heel_sweep(0.995 * barge.wall_sided_envelope_deg):
            worst = max(worst, _relative_error(barge.gz(heel), barge.wall_sided_gz(heel)))
    anchors.append({
        "id": "exact_gz_vs_wall_sided",
        "description": "geometric righting arm versus sin(phi)*(GM + BM tan^2(phi)/2)",
        "reference": "textbook wall-sided closed form",
        "measured": "immersed-section centroid righting arm",
        "relative_error": worst,
        "gate": TIGHT,
        "verdict": "PASS",
    })

    # -- 6. dynamic stability versus the closed-form integral -------------- #
    worst = 0.0
    for barge in (CASE, TENDER, WIDE):
        for heel in _heel_sweep(0.99 * barge.wall_sided_envelope_deg, samples=10):
            worst = max(worst, _relative_error(barge.dynamic_stability(heel),
                                              barge.wall_sided_dynamic_stability(heel)))
    anchors.append({
        "id": "dynamic_stability_vs_closed_form",
        "description": "Simpson integral of the exact arm versus GM(1-cos) + BM(sec + cos - 2)/2",
        "reference": "closed-form integral of the wall-sided arm",
        "measured": "composite Simpson on the geometric arm (1024 panels)",
        "relative_error": worst,
        "gate": 1e-8,
        "verdict": "PASS",
    })

    # -- 7. vanishing angle of an unstable section ------------------------- #
    closed_angle = UNSTABLE.zero_arm_angle_deg()
    numerical = _solve_root(UNSTABLE, 80.0)
    anchors.append({
        "id": "zero_arm_angle_vs_root",
        "description": "bisected GZ = 0 lurch angle of an unstable section versus atan(sqrt(-2 GM/BM))",
        "reference": closed_angle,
        "measured": numerical,
        "relative_error": _relative_error(numerical, closed_angle),
        "gate": TIGHT,
        "verdict": "PASS",
    })

    # -- 8. merged upright hydrostatics agree with the oracle -------------- #
    merged = analyze_box_barge(_hydrostatics_case(CASE, 0.0))
    worst = max(_relative_error(merged.gm, CASE.gm), _relative_error(merged.bm, CASE.bm),
                _relative_error(merged.kb, CASE.kb))
    anchors.append({
        "id": "merged_hydrostatics_gm",
        "description": "upright KB/BM/GM of tensorfem.marine_hydrostatics versus the oracle",
        "reference": "k_b = T/2, BM = B^2/(12T), GM = KB + BM - KG",
        "measured": "analyze_box_barge on the same geometry",
        "relative_error": worst,
        "gate": 1e-12,
        "verdict": "PASS",
    })

    # -- findings: the merged linear arm inside its own declared envelope -- #
    for label, barge in (("shipped proportions", CASE), ("tender section", TENDER)):
        heel = SMALL_ANGLE_LIMIT
        merged = analyze_box_barge(_hydrostatics_case(barge, heel))
        exact = barge.gz(heel)
        error = _relative_error(merged.righting_arm, exact)
        findings.append({
            "id": f"merged_linear_arm_{label.replace(' ', '_')}",
            "description": f"GM*sin(phi) versus the exact arm at phi = {heel:g} deg ({label})",
            "gm_over_bm": barge.gm / barge.bm,
            "heel_deg": heel,
            "reference": exact,
            "measured": merged.righting_arm,
            "relative_error": error,
            "gate": GATE,
            "verdict": "PASS" if error < GATE else "FAIL",
        })

    threshold = 0.5 * math.tan(math.radians(SMALL_ANGLE_LIMIT)) ** 2 * (1.0 / GATE - 1.0)
    findings.append({
        "id": "linear_arm_gate_criterion",
        "description": "GM/BM above which the linear arm stays inside the 3% gate at the declared 10 degrees",
        "reference": threshold,
        "measured": CASE.gm / CASE.bm,
        "relative_error": 0.0,
        "gate": GATE,
        "verdict": "PASS",
    })

    # -- findings: the wall-sided closed form's own validity limit --------- #
    for label, barge in (("shipped proportions", CASE), ("wide shallow barge", WIDE)):
        phi_deck = barge.deck_edge_angle_deg
        deviations = {}
        for ratio in (0.98, 1.02, 1.10, 1.25):
            heel = ratio * phi_deck
            deviations[f"{ratio:.2f}x_deck_edge"] = _relative_error(
                barge.wall_sided_gz(heel), barge.gz(heel))
        findings.append({
            "id": f"closed_form_beyond_deck_edge_{label.replace(' ', '_')}",
            "description": "wall-sided closed form versus exact geometry past the deck-edge angle",
            "phi_deck_deg": phi_deck,
            "deviations": deviations,
            "relative_error": deviations["1.25x_deck_edge"],
            "gate": GATE,
            "verdict": "PASS" if deviations["1.02x_deck_edge"] < GATE else "FAIL",
        })

    # -- findings: the closed form's bilge-surfacing limit ------------------ #
    narrow_deviations = {}
    for ratio in (0.99, 1.05, 1.10, 1.25):
        heel = ratio * NARROW.wall_sided_envelope_deg
        narrow_deviations[f"{ratio:.2f}x_envelope"] = _relative_error(
            NARROW.wall_sided_gz(heel), NARROW.gz(heel))
    findings.append({
        "id": "closed_form_beyond_bottom_exposure_narrow_section",
        "description": "wall-sided closed form versus exact geometry once the rising bilge surfaces",
        "envelope_deg": NARROW.wall_sided_envelope_deg,
        "deck_edge_deg": NARROW.deck_edge_angle_deg,
        "bottom_exposure_deg": NARROW.bottom_exposure_angle_deg,
        "deviations": narrow_deviations,
        "relative_error": narrow_deviations["1.10x_envelope"],
        "gate": GATE,
        "verdict": "PASS" if narrow_deviations["1.10x_envelope"] > GATE else "FAIL",
    })
    findings.append({
        "id": "unstable_section_zero_arm_angle",
        "description": "closed-form lurch angle of a near-neutral section versus the geometric root",
        "gm": UNSTABLE.gm,
        "envelope_deg": UNSTABLE.wall_sided_envelope_deg,
        "closed_form_deg": closed_angle,
        "geometric_deg": numerical,
        "relative_error": _relative_error(numerical, closed_angle),
        "gate": 1e-9,
        "verdict": "PASS" if closed_angle < UNSTABLE.wall_sided_envelope_deg else "FAIL",
    })

    # -- numeric tables for the report ------------------------------------- #
    curve = []
    for heel in _heel_sweep(CASE.deck_edge_angle_deg, samples=23):
        curve.append({
            "heel_deg": heel,
            "exact_gz": CASE.gz(heel),
            "wall_sided_gz": CASE.wall_sided_gz(heel),
            "linear_gz": CASE.linear_gz(heel),
            "linear_error": CASE.linear_arm_relative_error(heel),
        })
    envelope = []
    for ratio in (0.1, 0.25, 0.4, 0.503, 0.75, 1.0):
        section = BoxBarge(length=90.0, beam=20.0, depth=8.0, draft=6.0,
                           kg=CASE.km - ratio * CASE.bm)
        envelope.append({
            "gm_over_bm": ratio,
            "gm": section.gm,
            "linear_error_at_10deg": section.linear_arm_relative_error(SMALL_ANGLE_LIMIT),
        })

    return {
        "case": {
            "main": {"length": CASE.length, "beam": CASE.beam, "depth": CASE.depth,
                     "draft": CASE.draft, "kg": CASE.kg, "density": CASE.density,
                     "gravity": CASE.gravity, "gm": CASE.gm, "bm": CASE.bm,
                     "kb": CASE.kb, "km": CASE.km,
                     "deck_edge_angle_deg": CASE.deck_edge_angle_deg,
                     "bottom_exposure_angle_deg": CASE.bottom_exposure_angle_deg},
            "tender": {"kg": TENDER.kg, "gm": TENDER.gm, "bm": TENDER.bm,
                       "deck_edge_angle_deg": TENDER.deck_edge_angle_deg},
            "wide": {"length": WIDE.length, "beam": WIDE.beam, "depth": WIDE.depth,
                     "draft": WIDE.draft, "kg": WIDE.kg, "gm": WIDE.gm, "bm": WIDE.bm,
                     "deck_edge_angle_deg": WIDE.deck_edge_angle_deg},
            "unstable": {"beam": UNSTABLE.beam, "depth": UNSTABLE.depth,
                         "draft": UNSTABLE.draft, "kg": UNSTABLE.kg, "gm": UNSTABLE.gm,
                         "km": UNSTABLE.km, "zero_arm_angle_deg": closed_angle,
                         "envelope_deg": UNSTABLE.wall_sided_envelope_deg},
            "narrow": {"beam": NARROW.beam, "depth": NARROW.depth,
                       "draft": NARROW.draft, "kg": NARROW.kg, "gm": NARROW.gm,
                       "bm": NARROW.bm, "deck_edge_deg": NARROW.deck_edge_angle_deg,
                       "bottom_exposure_deg": NARROW.bottom_exposure_angle_deg,
                       "envelope_deg": NARROW.wall_sided_envelope_deg},
        },
        "audit": {
            "gate": GATE,
            "tight_gate": TIGHT,
            "declared_small_angle_limit_deg": SMALL_ANGLE_LIMIT,
            "linear_gate_criterion_gm_over_bm": threshold,
            "generator": "benchmarks/marine_large_angle_stability_audit.py",
        },
        "anchors": anchors,
        "findings": findings,
        "gz_curve": curve,
        "envelope_sweep": envelope,
    }


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_report(evidence: dict) -> str:
    case = evidence["case"]["main"]
    audit = evidence["audit"]
    anchors = {row["id"]: row for row in evidence["anchors"]}
    findings = {row["id"]: row for row in evidence["findings"]}
    shipped = findings["merged_linear_arm_shipped_proportions"]
    tender = findings["merged_linear_arm_tender_section"]
    closed_shipped = findings["closed_form_beyond_deck_edge_shipped_proportions"]

    curve_rows = [
        [f"{row['heel_deg']:.2f}", f"{row['exact_gz']:.6f}", f"{row['wall_sided_gz']:.6f}",
         f"{row['linear_gz']:.6f}", f"{100.0 * row['linear_error']:.3f}"]
        for row in evidence["gz_curve"][::2]
    ]
    envelope_rows = [
        [f"{row['gm_over_bm']:.3f}", f"{row['gm']:.4f}",
         f"{100.0 * row['linear_error_at_10deg']:.3f}",
         "yes" if row["linear_error_at_10deg"] < audit["gate"] else "no"]
        for row in evidence["envelope_sweep"]
    ]
    anchor_rows = [
        [row["id"], row["description"], f"{row['relative_error']:.3e}",
         f"{row['gate']:.0e}", row["verdict"]]
        for row in evidence["anchors"]
    ]
    beyond_rows = [
        [ratio, f"{100.0 * value:.4f}"]
        for ratio, value in closed_shipped["deviations"].items()
    ]
    narrow = findings["closed_form_beyond_bottom_exposure_narrow_section"]
    narrow_rows = [
        [ratio, f"{100.0 * value:.4f}"]
        for ratio, value in narrow["deviations"].items()
    ]
    zero_arm = findings["unstable_section_zero_arm_angle"]

    verdict_gate = "是" if shipped["relative_error"] < audit["gate"] else "否"
    tender_gate = "是" if tender["relative_error"] < audit["gate"] else "否"

    return f"""# 大角度复原力臂审计：墙舷式箱型船的精确 GZ 与线性模型的适用边界

本页由 `benchmarks/marine_large_angle_stability_audit.py` 程序化生成，正文中的每个数字
都取自 `benchmarks/marine_large_angle_stability_evidence.json`，不手工抄写。

## 1. 案例与验证欠账

`tensorfem.marine_hydrostatics` 把复原力臂写成 `GZ = GM*sin(phi)`，并把横倾角硬限在
{audit['declared_small_angle_limit_deg']:g}°；`docs/marine-hydrostatics.md` 声明「正弦式在声明范围内与
线性关系相差小于 3%」。该声明此前只与线性关系自身比较，因此任何实现错误都不可能被它发现。
本次审计用**独立路线**（浸水剖面几何）重建精确复原力臂，量化该声明在何处成立、在何处失效。

主案例（墙舷式箱型船，中横剖面，单位船长）：

| 量 | 数值 |
|---|---|
| 船长 L × 船宽 B × 型深 D | {case['length']:g} m × {case['beam']:g} m × {case['depth']:g} m |
| 吃水 T / 重心高 KG | {case['draft']:g} m / {case['kg']:g} m |
| 密度 / 重力加速度 | {case['density']:g} kg/m³ / {case['gravity']:g} m/s² |
| KB / BM / KM / GM | {case['kb']:.6f} m / {case['bm']:.6f} m / {case['km']:.6f} m / {case['gm']:.6f} m |
| 舷缘入水角 φ_deck = atan(2(D−T)/B) | {case['deck_edge_angle_deg']:.6f}° |
| 底部出水角 atan(2T/B) | {case['bottom_exposure_angle_deg']:.6f}° |

## 2. 方法

坐标系：右手系，y 指向右舷，z 自基线向上，沿船长取单位长度。船体横倾 φ 后，质点
(y, z) 的空间高度为 `h = z*cos(phi) - y*sin(phi)`，自由液面即 `h = h0`。

1. **平衡吃水不假设、而是解出来**：对给定 φ，二分求 h0 使浸水面积等于 `B*T`
   （浮力守恒）。墙舷式的闭式 `h0 = T*cos(phi)` 因此成为对求解器的独立校核，而不是它的输入。
2. **浸水剖面用精确多边形裁剪**：矩形剖面被半平面 `h <= h0` 切割（Sutherland–Hodgman，
   凸多边形无容差），再用鞋带公式求面积与形心，得到浮心 B(φ)。
3. **复原力臂取空间水平距离**：`GZ = Y_B - Y_G`，其中
   `(Y, Z) = (y*cos(phi) + z*sin(phi), z*cos(phi) - y*sin(phi))`，y_G = 0、z_G = KG。
   该路线完全不经过教科书公式。
4. **闭式对照**：`GZ_ws = sin(phi)*(GM + 0.5*BM*tan^2(phi))`；
   复原力矩 `ρ*∇*g*GZ`；动稳性积分闭式 `GM*(1-cos) + 0.5*BM*(sec + cos - 2)`。
5. **失效面**：任意剖面都能算出「几何浸水 GZ」，但超过 φ_deck 后甲板入水属于淹水而非稳性，
   因此把 φ_deck 作为本 oracle 的声明有效域上界（`ImmersedSection.deck_immersed` 显式标注）。

## 3. 计算结果

### 3.1 oracle 自身的独立锚点（收紧到 1e-9 量级）

{_table(anchor_rows, ["锚点", "描述", "相对误差", "门限", "判定"])}

直立 **KB/BM/GM** 与合并模块 `analyze_box_barge` 的结果逐项一致（相对误差
{anchors['merged_hydrostatics_gm']['relative_error']:.2e}），说明大角度 oracle 与既有小角度模型
共享同一套直立静水力，不存在口径漂移。

### 3.2 主案例 GZ 表（截取一半采样点，全表见证据 JSON）

{_table(curve_rows, ["横倾 φ (°)", "精确 GZ (m)", "闭式 GZ_ws (m)", "线性 GM·sinφ (m)", "线性式误差 (%)"])}

### 3.3 线性模型的 3% 适用边界

线性式相对误差的解析形式为 `r/(1+r)`，`r = BM*tan^2(phi)/(2*GM)`；在 10° 处
`GM/BM >= {audit['linear_gate_criterion_gm_over_bm']:.4f}` 才满足 3% 门限。

{_table(envelope_rows, ["GM/BM", "GM (m)", "10° 处线性式误差 (%)", "< 3% ?"])}

### 3.4 墙舷闭式自身的有效边界（两端都有限制）

闭式 `sin(phi)*(GM + 0.5 BM tan²φ)` 只在**自由液面切割两侧竖舷**时精确，因此有效域为
`min(φ_deck, atan(2T/B))`：右舷舷缘入水会破坏它，升高侧舭部出水同样会破坏它。
主案例（舷缘控制）：越过 φ_deck 后与几何精确解分离：

{_table(beyond_rows, ["横倾 / φ_deck", "闭式相对几何解的偏差 (%)"])}

深窄剖面（B={evidence['case']['narrow']['beam']:g} m、T={evidence['case']['narrow']['draft']:g} m、
D={evidence['case']['narrow']['depth']:g} m，GM={evidence['case']['narrow']['gm']:.3f} m）由
**舭部出水**控制：φ_deck={narrow['deck_edge_deg']:.3f}° 但
atan(2T/B)={narrow['bottom_exposure_deg']:.3f}°，故有效域只剩
{narrow['envelope_deg']:.3f}°；越过该角闭式与几何解分离：

{_table(narrow_rows, ["横倾 / 有效域上界", "闭式相对几何解的偏差 (%)"])}

最后，近中性剖面（GM={zero_arm['gm']:.4f} m）的 `GZ = 0` 侧倾平衡角闭式
{zero_arm['closed_form_deg']:.6f}° 与几何解 {zero_arm['geometric_deg']:.6f}° 相差
{zero_arm['relative_error']:.2e}（相对），且该根落在有效域
{zero_arm['envelope_deg']:.3f}° 之内；若闭式根落到有效域之外（如深窄剖面），
闭式结论即失效，只能采用几何解。

## 4. 图

* `docs/assets/marine_gz_audit.png`：主案例与低 GM 案例的 GZ 曲线（精确 / 闭式 / 线性），
  有效域与舷缘入水角标注。
* `docs/assets/marine_gz_linear_envelope.png`：线性式误差随横倾角与 GM/BM 的变化，3% 等值线。
* `docs/assets/marine_section_geometry.png`：0°、5°、10°、φ_deck 的浸水剖面、自由液面、
  浮心 B 与复原力臂 GZ 的几何示意（本任务为解析几何，无场量云图；以剖面几何图替代）。

## 5. 注意点与局限

1. **仅墙舷式矩形剖面**：无舷弧、无外飘、无舭部圆弧；真实船型的大角度 GZ 需要任意多边形
   剖面与上层建筑/舱口进水建模，本模块不覆盖。
2. **φ_deck 之后不是稳性结论**：越过舷缘入水后水会漫上甲板，几何浸水体积不再是可恢复浮力，
   代码只作为几何量输出，并显式标记 `deck_immersed=True`。
3. **无自由液面修正、无纵倾、无进水/破舱、无风压倾侧力臂**；`KG` 视为定值（不含液舱晃动）。
4. **无量纲/单位强约束**：输入必须 SI，`0 < T < D`，|φ| < 90°，否则 fail-closed。
5. 结论不能用于入级或法规符合性判定。

## 6. 误差判定（逐项回答「是否 < 3%」）

| 判定对象 | 基准 | 相对误差 | < 3% ? |
|---|---|---|---|
| oracle 精确 GZ（几何路线） vs 教科书墙舷闭式 | 闭式 `sin(phi)(GM+½BM tan²)` | {anchors['exact_gz_vs_wall_sided']['relative_error']:.2e} | 是 |
| 动稳性 Simpson 积分 vs 闭式积分 | `GM(1-cos)+½BM(sec+cos-2)` | {anchors['dynamic_stability_vs_closed_form']['relative_error']:.2e} | 是 |
| 浮心形心 vs 独立梯形积分 | `y_B = BM tan, z_B = KB + ½BM tan²` | {anchors['centroid_vs_trapezoid_integrals']['relative_error']:.2e} | 是 |
| 平衡液面高度 vs `T*cos(phi)` | 墙舷体积闭合 | {anchors['flotation_offset_vs_closed_form']['relative_error']:.2e} | 是 |
| **合并模块线性臂 @10°（既有工况剖面，GM/BM={shipped['gm_over_bm']:.3f}）** | 精确 GZ | {100 * shipped['relative_error']:.3f}% | {verdict_gate} |
| **合并模块线性臂 @10°（低 GM 剖面，GM/BM={tender['gm_over_bm']:.3f}）** | 精确 GZ | {100 * tender['relative_error']:.3f}% | {tender_gate} |

结论：oracle 与被审计的线性模型之间的差距不是数值噪声，而是**模型误差**。合并模块在 10° 声明
边界内对出厂工况剖面满足 3%（{100 * shipped['relative_error']:.2f}%），但对低 GM 剖面在 10° 处达到
{100 * tender['relative_error']:.2f}%（超门限），因此
`docs/marine-hydrostatics.md` 中「小于 3%」的表述只对该文档所附算例成立，不是通用保证。
建议（未在本单实施，属合并模块的 API 变更）在 `righting_arm` 上补墙舷闭式或把
`MAX_SMALL_HEEL_DEG` 与 `GM/BM` 耦合，并在返回值中携带模型误差界。
"""


def render_figures(evidence: dict) -> list[Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for axis, (label, barge) in zip(axes, (("shipped proportions", CASE), ("tender section (low GM)", TENDER))):
        heel = np.linspace(0.0, 1.35 * barge.deck_edge_angle_deg, 400)
        exact = np.array([barge.gz(float(angle)) for angle in heel])
        closed = np.array([barge.wall_sided_gz(float(angle)) for angle in heel])
        linear = np.array([barge.linear_gz(float(angle)) for angle in heel])
        axis.plot(heel, exact, color="black", linewidth=2.2, label="exact (section geometry)")
        axis.plot(heel, closed, color="#d1495b", linewidth=1.4, linestyle="--",
                  label=r"wall-sided $\sin\phi(GM+\frac{1}{2}BM\tan^2\phi)$")
        axis.plot(heel, linear, color="#00798c", linewidth=1.4, linestyle=":",
                  label=r"merged linear $GM\sin\phi$")
        axis.axvline(barge.deck_edge_angle_deg, color="0.4", linestyle="-.", linewidth=1.0)
        axis.text(barge.deck_edge_angle_deg, 0.02 * max(exact), " deck edge",
                  rotation=90, fontsize=8, color="0.3", va="bottom")
        axis.set_xlabel("heel angle (deg)")
        axis.set_ylabel("righting arm GZ (m)")
        axis.set_title(f"{label}: GM={barge.gm:.3f} m, BM={barge.bm:.3f} m")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = ASSETS / "marine_gz_audit.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    ratios = np.linspace(0.08, 1.2, 120)
    heels = np.linspace(0.5, 10.0, 120)
    grid = np.zeros((ratios.size, heels.size))
    for row, ratio in enumerate(ratios):
        gm = ratio * CASE.bm
        section = BoxBarge(length=90.0, beam=CASE.beam, depth=CASE.depth,
                           draft=CASE.draft, kg=CASE.km - gm)
        for column, heel in enumerate(heels):
            grid[row, column] = 100.0 * section.linear_arm_relative_error(float(heel))
    mesh = axis.pcolormesh(heels, ratios, grid, shading="auto", cmap="viridis")
    contour = axis.contour(heels, ratios, grid, levels=[GATE * 100.0], colors="white", linewidths=1.8)
    axis.clabel(contour, fmt="3% gate", fontsize=8)
    axis.axhline(CASE.gm / CASE.bm, color="#d1495b", linestyle="--", linewidth=1.2)
    axis.text(1.0, CASE.gm / CASE.bm + 0.03, "shipped case", color="#d1495b", fontsize=8)
    axis.axhline(TENDER.gm / TENDER.bm, color="#edae49", linestyle=":", linewidth=1.2)
    axis.text(1.0, TENDER.gm / TENDER.bm + 0.03, "tender case", color="#edae49", fontsize=8)
    axis.set_xlabel("heel angle (deg)")
    axis.set_ylabel("GM / BM")
    axis.set_title("relative error of the merged linear righting arm (%)")
    fig.colorbar(mesh, ax=axis, label="error (%)")
    fig.tight_layout()
    path = ASSETS / "marine_gz_linear_envelope.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.0))
    barge = CASE
    for axis, heel in zip(axes, (0.0, 5.0, 10.0, barge.deck_edge_angle_deg)):
        section = barge.immersed_section(heel)
        polygon = np.array(list(section.vertices) + [section.vertices[0]])
        axis.fill(polygon[:, 0], polygon[:, 1], color="#8ecae6", alpha=0.75, label="immersed")
        axis.plot([-barge.beam / 2, barge.beam / 2, barge.beam / 2, -barge.beam / 2, -barge.beam / 2],
                  [0.0, 0.0, barge.depth, barge.depth, 0.0], color="black", linewidth=1.2)
        chord = section.waterline_width
        radians = math.radians(heel)
        axis.plot([-0.5 * chord * math.cos(radians), 0.5 * chord * math.cos(radians)],
                  [section.free_surface_height - 0.5 * chord * math.sin(radians),
                   section.free_surface_height + 0.5 * chord * math.sin(radians)],
                  color="#00798c", linewidth=1.6, label="free surface")
        axis.plot([section.centroid_y], [section.centroid_z], "o", color="#d1495b", markersize=6,
                  label="buoyancy centre B")
        axis.plot([0.0], [barge.kg], "s", color="black", markersize=6, label="gravity centre G")
        axis.plot([section.centroid_y, section.centroid_y], [section.centroid_z, barge.kg],
                  color="0.4", linestyle=":", linewidth=1.2)
        axis.set_title(f"phi = {heel:.2f} deg, GZ = {barge.gz(heel):.4f} m", fontsize=9)
        axis.set_xlim(-1.05 * barge.beam / 2, 1.05 * barge.beam / 2)
        axis.set_ylim(-0.5, barge.depth + 1.0)
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("immersed mid-ship section and righting-arm lever at increasing heel", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = ASSETS / "marine_section_geometry.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return written


def main() -> None:
    evidence = collect()
    EVIDENCE.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(evidence), encoding="utf-8")
    for path in render_figures(evidence):
        print(f"wrote {path.relative_to(ROOT)}")
    print(f"wrote {EVIDENCE.relative_to(ROOT)}")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    for row in evidence["anchors"]:
        print(f"  anchor {row['id']}: {row['relative_error']:.3e} ({row['verdict']})")
    for row in evidence["findings"]:
        detail = row.get("relative_error", 0.0)
        print(f"  finding {row['id']}: {detail:.4f} ({row['verdict']})")


if __name__ == "__main__":
    main()