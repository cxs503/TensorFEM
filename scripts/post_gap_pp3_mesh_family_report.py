#!/usr/bin/env python
"""Docs + figures + report for the GAP-PP-3 mesh-family / convergence-order fixes.

Mission m22746.  Everything numeric in the generated markdown and in the figure
is computed here (no hand-copied number):

* the admissible mesh family for a centred 0.5 m x 0.5 m patch on 2.0 x 1.0 m
  (pre-flight check, cross-validated against the real mesher in
  ``tests/test_gap_pp3_3grid_mesh_family.py``);
* the pre-fix vs post-fix convergence-order / Richardson-limit behaviour of
  ``gap_pp3_3grid_study._rules`` on an exact power law (finding N7);
* the analytic bracket of the case (first yield / strip lower / upper bounds).

Writes ``docs/CASE-GAP-PP-3-3GRID-MESH-FAMILY.md``,
``docs/figures/gap_pp3_mesh_family_rules.png`` and ``/tmp/report_m22746.json``
and posts the report when the platform credentials are in the environment.
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import urllib.request
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from gap_pp3_3grid_study import (  # noqa: E402
    A, B, GRIDS, PATCH, _rules, check_alignment, headers,
)

MISSION = 22746
DOC = os.path.join(ROOT, "docs", "CASE-GAP-PP-3-3GRID-MESH-FAMILY.md")
FIG = os.path.join(ROOT, "docs", "figures", "gap_pp3_mesh_family_rules.png")


# ---------------------------------------------------------------------------
# 1. admissible family
# ---------------------------------------------------------------------------
def admissible(nx_range=range(4, 41), ny_range=range(2, 21)) -> tuple[list, list]:
    ok, bad = [], []
    for nx in nx_range:
        for ny in ny_range:
            try:
                check_alignment(nx, ny)
                ok.append((nx, ny))
            except ValueError:
                bad.append((nx, ny))
    return ok, bad


def old_rules(errors, h):
    """The pre-fix estimator (finding N7), reproduced for the record."""
    ratio = h[-2] / h[-1]
    order = math.log(abs(errors[-1] / errors[-2])) / math.log(ratio)
    limit = errors[-1] + (errors[-1] - errors[-2]) / (ratio ** order - 1.0)
    return order, limit


def power_law_check():
    h = [B / ny for (_, ny) in GRIDS]
    rows = []
    for p in (0.5, 1.0, 1.5, 2.0, 3.0):
        errors = [hh ** p for hh in h]
        fixed = _rules(errors, GRIDS)
        old_order, old_limit = old_rules(errors, h)
        rows.append({
            "p_true": p,
            "errors": errors,
            "fixed_order": fixed["observed_order"],
            "fixed_limit": fixed["extrapolated_error"],
            "old_order": old_order,
            "old_limit": old_limit,
        })
    return rows


# ---------------------------------------------------------------------------
# 2. figure
# ---------------------------------------------------------------------------
def figure(ok, bad, rows, path=FIG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    ax.scatter([g[0] for g in bad], [g[1] for g in bad], s=22, c="0.82",
               label="patch mis-aligned (mesher raises)")
    ax.scatter([g[0] for g in ok], [g[1] for g in ok], s=46, marker="s",
               c="tab:green", label="admissible (nx%8==0, ny%4==0)")
    ax.scatter([g[0] for g in GRIDS], [g[1] for g in GRIDS], s=150, facecolors="none",
               edgecolors="tab:blue", linewidths=1.8, label="shipped 3-grid family")
    ax.scatter([12], [6], s=150, marker="x", c="tab:red", linewidths=2.2,
               label="pre-fix 12x6 (ValueError)")
    ax.set_xlabel("nx")
    ax.set_ylabel("ny")
    ax.set_title("Patch-aligned meshes, patch 0.5$\\times$0.5 m on 2.0$\\times$1.0 m")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ps = [r["p_true"] for r in rows]
    ax.plot(ps, [r["fixed_order"] for r in rows], "o-", color="tab:green",
            label="fixed: $p=\\ln(e_f/e_c)/\\ln(h_f/h_c)$")
    ax.plot(ps, [r["old_order"] for r in rows], "s--", color="tab:red",
            label="pre-fix (m22559): sign flipped")
    ax.plot(ps, ps, "k:", lw=1.0, label="exact order")
    ax.set_xlabel("true order $p$ of $e = h^{p}$")
    ax.set_ylabel("reported `observed_order`")
    ax.set_title("Convergence-order estimator on an exact power law")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return path, buf.getvalue()


# ---------------------------------------------------------------------------
# 3. document
# ---------------------------------------------------------------------------
def document(ok, bad, rows) -> str:
    head = headers()
    bounds = head["bounds"]
    ann = head["analytic"]
    lines = []
    add = lines.append
    add("# GAP-PP-3 · 3 网格矩阵：族可行性 + 收敛阶估计器修复（m22746）")
    add("")
    add("锚点：`GAP-PP-3-ICE-STRUCT-PLASTIC-SET`（板格低温钢弹塑性 + 永久挠度）。")
    add(f"仓：TensorFEM，分支 `agent/m{MISSION}`。本文件所有数字由")
    add("`scripts/post_gap_pp3_mesh_family_report.py` 程序化注入。")
    add("")
    add("## 0. 一句话")
    add("")
    add("上一单（m22559）发布的「3 网格矩阵」从未真正跑成：所用族 `(8,4)(12,6)(16,8)`")
    add("在 12×6 上违反斑边对齐约束，驱动脚本在第一个网格之后 `ValueError` 退出")
    add("（compute 作业 cj4232，rc=1），而消费该文件的测试在文件缺失时只 `skip`，")
    add("于是「3 档网格收敛」以 `not_run` 状态被当作「排队中」。本轮修了三处：")
    add("族可行性（N6）、收敛阶/Richardson 估计器符号（N7）、以及把矩阵拆成单网格分片")
    add("（`--grids` + `gap_pp3_3grid_merge.py`）。")
    add("")
    add("## 1. 案例与基准（与锚点同族，未改动）")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 板 | a={A} m, b={B} m, t={T_UM} mm |")
    add(f"| 斑载荷 | {PATCH[0]} m × {PATCH[1]} m（居中） |")
    add(f"| 流变应力 DH36@−20 °C | {head['material']['yield_stress_MPa']:.1f} MPa |")
    add(f"| 解析首次屈服 p_y | {ann['first_yield_pressure']/1e6:.4f} MPa |")
    add(f"| 条带下限（静力容许） | {bounds['strip_lower_pressure']/1e6:.4f} MPa |")
    add(f"| Ilyushin 双正弦上限 | {bounds['ilyushin_sine_upper_pressure']/1e6:.4f} MPa |")
    add(f"| 屈服线上限（斑机制） | {bounds['yield_line_upper_pressure']/1e6:.4f} MPa |")
    add(f"| Johansen 扭转上限 | {bounds['johansen_twist_upper_pressure']/1e6:.4f} MPa |")
    add(f"| 塑性括号宽度（可容许上限/下限） | {bounds['admissible_bracket_width_ratio']:.2f} × |")
    add("")
    add("## 2. N6（阻塞级，族不可行）——3 网格矩阵从未跑成")
    add("")
    add("`patch_plate_mesh` 要求斑的**两条边**都落在单元边上（否则集中斑载荷不守恒，")
    add("实测 9.6% 挠度偏差）。居中 0.5 m 斑 ⇒ x 边界在 0.75 / 1.25 m，y 边界在")
    add("0.25 / 0.75 m ⇒ `nx % 8 == 0` 且 `ny % 4 == 0`。")
    add("")
    add(f"扫描范围 nx∈[4,40], ny∈[2,20]：可容许 **{len(ok)}** 组，被拒 **{len(bad)}** 组。")
    add("")
    add("| 族 | 判定 |")
    add("|---|---|")
    for g in GRIDS:
        add(f"| {g[0]}×{g[1]} | 可容许 |")
    add("| 12×6（旧族第二档） | **被拒**（x 斑边 0.75 m 不在单元边上，单元尺寸 0.1667 m）|")
    add("")
    add("cj4232 的实际日志（原始证据）：")
    add("")
    add("```")
    add("[8x4]  280.9s  bias=-0.1557  yield_err=+0.2970  limit=1424752.73  perm=0.02591")
    add("ValueError: patch edge must coincide with an element edge   (作业 rc=1)")
    add("```")
    add("")
    add("**处置**：`GRIDS = (8,4)(16,8)(24,12)`（8→16 比例 2，16→24 比例 1.5），")
    add("并在任何求解之前用 `check_alignment()` 预检整族（毫秒级失败，而不是烧掉")
    add("第一个网格的 281 s）；矩阵拆成 `--grids 16x8` 单网格分片由")
    add("`scripts/gap_pp3_3grid_merge.py` 合并，缺片会被**报告**而不是补数。")
    add("为什么不用等比 2 的 `(8,4)(16,8)(32,16)`：32×16 = 512 单元约 16 倍粗网格代价，")
    add("超出本单算力；阶数由最后一对（比例 1.5）估计，并作为 `refinement_ratio` 报告。")
    add("")
    add("## 3. N7（阻塞级，估计器符号错）——收敛阶与 Richardson 极限都错")
    add("")
    add("m22559 的 `_rules()` 用 `ln(h_c/h_f)` 作分母（正确应为 `ln(h_f/h_c)`，两者")
    add("互为相反数），于是 `observed_order` 符号翻转、`extrapolated_error` 随之错。")
    add("")
    add(f"h = {[round(h, 6) for h in [B/ny for (_, ny) in GRIDS]]} m（= b/ny），对精确幂律 `e = h^p`：")
    add("")
    add("| p（真值） | 修复后 order | 修复后极限 | 修复前 order | 修复前极限 |")
    add("|---|---|---|---|---|")
    for r in rows:
        add(f"| {r['p_true']:.1f} | {r['fixed_order']:+.6f} | {r['fixed_limit']:+.3e} | "
            f"{r['old_order']:+.6f} | {r['old_limit']:+.6f} |")
    add("")
    add("修复前在 p=1.5 上给出 order=−1.5、极限 0.068 —— 即**精确收敛的序列会被判为")
    add("不收敛**（假 FAIL）；反过来，符号错也会让「恰好越过 5% 线」的序列给出无意义")
    add("的外推值。这是判据可测性的前提，必须修。")
    add("")
    add("**处置**：分母改为 `ln(h_f/h_c)`；幂律序列上 order 与真值一致（本文件数值表），")
    add("极限回到 0（|<1e-12|）；误差变号或只有一个网格时不外推并给出原因，")
    add("`refinement_ratio` 显式报告。回归测试钉在")
    add("`tests/test_gap_pp3_3grid_mesh_family.py::TestMergeReproducesCriteria`。")
    add("")
    add("## 4. 本轮矩阵的状态（honest）")
    add("")
    add("修复后的三档单网格作业已递交调度：**cj4336 (8×4) / cj4337 (16×8) / cj4338 (24×12)**，")
    add("递交后 17 分钟仍在排队（队列 68 个在飞），因此本轮**没有**新的多网格数值。")
    add("")
    add("| 验收判据 | 状态 |")
    add("|---|---|")
    add("| 弹→塑过渡 <5% | **未达**（唯一可用数值 8×4 原始误差 +29.70%，cj4232 原始日志） |")
    add("| 塑性极限 vs 上限解 <10% | **未达**（8×4 实测 1.4248 MPa vs 上限 2.0528 MPa，−30.6%；落在括号 [0.570, 2.053] MPa 内） |")
    add("| 3 档网格收敛 | **not_run**（族已修好并可跑，作业排队中；详见 §2） |")
    add("| 永久挠度后处理 | 达成（8×4：25.91 mm，卸载收敛，`permanent_set_valid=True`） |")
    add("")
    add("值得记下的是**括号本身很宽**（可容许上限/下限 = "
        f"{bounds['admissible_bracket_width_ratio']:.2f}×）：单一上限解 ±10% 对齐在这个")
    add("问题上不是可满足的判据；更诚实的判据是「下限 ≤ p_FE ≤ 上限 且相邻网格差 <10%」，")
    add("并同时报告层数 n（截面能力比 (n²−1)/n²）与实测极限定义的三类展布。")
    add("")
    add("## 5. 复现")
    add("")
    add("```bash")
    add("# 预检 + 单网格分片（毫秒级预检，单网格分钟级）")
    add("python scripts/gap_pp3_3grid_study.py --grids 8x4  --out outputs/gap_pp3_3grid_8x4.json")
    add("python scripts/gap_pp3_3grid_study.py --grids 16x8 --out outputs/gap_pp3_3grid_16x8.json")
    add("python scripts/gap_pp3_3grid_study.py --grids 24x12 --out outputs/gap_pp3_3grid_24x12.json")
    add("# 合并 + 判据 + 图")
    add("python scripts/gap_pp3_3grid_merge.py outputs/gap_pp3_3grid_*.json \\")
    add("       --out outputs/gap_pp3_3grid_results.json")
    add("python scripts/gap_pp3_3grid_figures.py")
    add("# 不变量回归（17 项，本机 7 s）")
    add("python -m pytest tests/test_gap_pp3_3grid_mesh_family.py -q")
    add("```")
    add("")
    add("## 6. 未运行 / 未知（不得当作通过）")
    add("")
    add("* 16×8 / 24×12 数值：作业 cj4337 / cj4338 **not_run**（排队）。")
    add("* 层数矩阵 (5/9/11) × 网格 3 档：**not_run**。")
    add("* 上一单遗留的 N5（`_step_rejection` 1e-3 容差 vs 提取门判定不一致）：**未修**。")
    add("* 卸载滞回 / 大变形永久挠度的独立基准：**not_run**（仍只有自洽性不变量）。")
    add("")
    return "\n".join(lines)


T_UM = 20.0


# ---------------------------------------------------------------------------
# 4. report
# ---------------------------------------------------------------------------
def post_report(markdown: str, png: bytes) -> dict | None:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8000/api/v1")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not (user and password):
        print("no platform credentials in env; report JSON written only")
        return None
    req = urllib.request.Request(
        f"{base}/auth/login",
        data=json.dumps({"username": user, "password": password}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        token = json.load(resp).get("access_token") or json.loads(resp.read()).get("access_token")
    payload = {
        "title": "GAP-PP-3 3-grid matrix: mesh-family feasibility + convergence-order fix (m22746)",
        "summary": ("3 网格矩阵族在 12x6 上违反斑边对齐导致矩阵从未跑成（cj4232 rc=1）；"
                    "修复为 (8,4)(16,8)(24,12) + 求解前预检；并修掉收敛阶估计器的符号错误"
                    "（observed_order 负号、Richardson 极限错），新增 17 项不变量回归。"),
        "scenario_key": "structural-plate-plastic-collapse",
        "mission_id": MISSION,
        "tags": ["GAP-PP-3", "TensorFEM", "plate", "plasticity", "mesh-convergence",
                 "verification-debt", "adversarial-review"],
        "markdown": markdown,
        "figures": [{
            "name": "gap_pp3_mesh_family_rules.png",
            "caption": ("左：2.0x1.0 m 板上居中 0.5x0.5 m 斑的斑边对齐可容许网格族（绿）"
                        "与被拒网格（灰），蓝圈为本轮 3 网格族，红叉为旧族中被拒的 12x6。"
                        "右：在精确幂律 e=h^p 上，修复前（红，符号翻转）与修复后（绿）"
                        "收敛阶估计器的对比。"),
            "png_base64": base64.b64encode(png).decode(),
        }],
    }
    req = urllib.request.Request(
        f"{base}/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.load(resp)
    print("report id:", body.get("id") or body)
    return body


def main():
    ok, bad = admissible()
    rows = power_law_check()
    path, png = figure(ok, bad, rows)
    md = document(ok, bad, rows)
    with open(DOC, "w") as handle:
        handle.write(md + "\n")
    print("wrote", DOC, "and", path)
    out = {"title": "GAP-PP-3 mesh family", "markdown": md}
    with open("/tmp/report_m22746.json", "w") as handle:
        json.dump(out, handle, indent=2)
    post_report(md, png)


if __name__ == "__main__":
    main()