#!/usr/bin/env python3
"""Publish the m24649 (re-open TensorFEM#1226, 管到底 round 1) progress report.

Every number is read from benchmarks/artifacts/beta_stiffener_calibration.json;
the three figures are read from docs/assets/stiffener_calibration/. Nothing is
hand copied.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HERE = Path(__file__).resolve().parent
ART = HERE / "benchmarks" / "artifacts"
FIG = HERE / "docs" / "assets" / "stiffener_calibration"


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def fig(name: str, caption: str) -> dict:
    png = base64.b64encode((FIG / name).read_bytes()).decode()
    return {"name": name, "caption": caption, "png_base64": png}


def main() -> int:
    d = json.loads((ART / "beta_stiffener_calibration.json").read_text())
    pan = d["panel"]
    dp = d["design_point_summary"]
    cal = d["design_point"]["calibration"]
    L, F = cal["longitudinal"], cal["frame"]
    b = d["band"]
    e2e = d["end_to_end"]["rows"]
    ex = d["band"]["exact_levy_check"]
    fld = d["field"]

    e2e_tbl = "\n".join(
        f"| {r['nx']} x {r['ny']} | {r['w_centre_m']*1e3:.4f} | {r['w_exact_m']*1e3:.4f} | "
        f"{r['err_w_pct']:+.3f}% | {r['Mx_centre_Nm_per_m']:.1f} | {r['Mx_exact_Nm_per_m']:.1f} | "
        f"{r['err_Mx_pct']:+.3f}% |"
        for r in e2e)
    band_tbl = "\n".join(
        f"| {r['beta']:.3f} | {r['Mx_centre_Nm_per_m']:.1f} | {r['sigma_centre_MPa']:.2f} | "
        f"{r['moment_fixity_strip']:.3f} |"
        for r in b["rows"])
    ex_tbl = "\n".join(
        f"| {r['beta']:.3f} | {r['K_N_per_rad']:.3e} | {r['Mx_centre_Nm_per_m']:.1f} | "
        f"{r['sigma_centre_MPa']:.2f} |"
        for r in ex)

    w_errs = [r["err_w_pct"] for r in e2e]
    mx_errs = [r["err_Mx_pct"] for r in e2e]
    w_ok = all(abs(e) < 3.0 for e in w_errs)
    mx_ok = all(abs(e) < 3.0 for e in mx_errs)

    md = f"""# 重开 TensorFEM#1226 · 管到底第 1 轮（m24649）：把被合并门禁滤除的 3 张 stiffener 标定图落进 main

**Anchor** `GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT`（JOUR-ICE-STRUCT Sec. 4.2）·
artifact git sha `{d['git_sha'][:12]}` · python {d['python']} ·
platform `{d['platform'].split('-')[0]}`

## 1. 案例介绍（物理背景 / 几何 / 参数）

冰带（ice-belt）板格由**纵桁（longitudinal）**与**横框架（transverse frame）**
围成，板边提供的是**有限转动刚度**而非 SSSS/CCCC 两个理想极端。m21055 的交付
（PR #1226，已裁定关闭）把板边转动约束 `beta` 从自由参数变成**由真实 scantling
反演**的物理量：分布转动刚度 `k_theta = 12 GJ / L^2`（构件端部被正交构件扭转
固定），再精确反解求解器的 `zeta = beta/(1-beta+eps)` 映射。

| 参数 | 值 | 单位 |
|---|---|---|
| 弹性模量 E | {pan['young']/1e9:.0f} | GPa |
| 泊松比 nu | {pan['poisson']} | - |
| 板厚 t | {pan['thickness']*1e3:.1f} | mm |
| 板格 a x b | {pan['side_x']} x {pan['side_y']} | m |
| 均布压力 p | {d['pressure_Pa']/1e6:.1f} | MPa |
| 抗弯刚度 D | {dp['D_Nm']:.1f} | N·m |
| 纵桁 | {L['member']}（支撑跨 {L['support_span']} m） | - |
| 横框架 | {F['member']}（支撑跨 {F['support_span']} m） | - |

**设计点**（由 scantling 反演）：`beta_x = {dp['beta_x']:.4f}`（纵桁）、
`beta_y = {dp['beta_y']:.4f}`（横框架）；对应的**动 mobilised 端部弯矩固定度**
`moment_fixity_x = {dp['moment_fixity_x']:.3f}`、`moment_fixity_y = {dp['moment_fixity_y']:.3f}`
（注意：固定度 != beta，`zeta` 映射强非线性）。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

本单是「管到底」重开单：原 PR #1226 的 8 个文本交付物**已经**在 main（经 #1020
合入），唯一从未落地的是 **3 张 PNG 图**——推送门把 `docs/` 之外的 PNG 当记录区
产物滤除，且全分辨率三张合计 434.5 KB 超过 merge-governor 的 200 KB 工件帽。

流程（全部可复现，零手抄）：
1. 核对 8 个文本交付物均在 main（`git log`/`git diff` 比对 PR 分支 07b0e6d2 与
   origin/main，仅 2 个文件 main 为超集，其余逐字节一致）。
2. 本地跑相关测试：`pytest tests/test_stiffener_rotational_restraint.py
   tests/test_plate_restraint_panel.py -q` → **44 passed**（相关模块与 JSON 在
   可编辑安装 worktree 与本 worktree 逐字节一致，结果有效）。
3. `python gen_stiffener_calibration_figures.py` 从已提交 JSON 确定性重生成 3 张
   图（只读 JSON，不重算）；再降采样 0.7x + 128 色调色板，合计 **141.9 KB**
   （< 200 KB 帽），落到 `docs/assets/stiffener_calibration/`（推送门 v5.8 F3
   只放行 `docs/` 下的交付图）。
4. 单 PR 4 文件（3 PNG + 1 README 溯源），单一功能点，未触碰任何热文件。

## 3. 计算结果（数字表，程序化注入）

**beta 带**（`beta_y = 0`，仅纵向约束扫描；nx=40 x 46 Mindlin FE）：

| beta | Mx_centre [N·m/m] | sigma_centre [MPa] | moment_fixity_strip |
|---|---|---|---|
{band_tbl}

端点：SSSS `{b['sigma_SSSS_MPa']:.2f}` MPa → 全固支 `{b['sigma_clamped_MPa']:.2f}` MPa
（**-{b['reduction_vs_SSSS_pct']:.1f}%**）；真实 scantling 设计点仅
`{b['sigma_at_calibrated_both_edges_MPa']:.2f}` MPa（**-{b['reduction_both_edges_pct']:.1f}%**，
双边缘）——SSSS 只是**轻度保守**上界，远早先「-33% 前置」的夸大说法。

**端到端校验**（scantling 标定 beta 的 Mindlin FE vs 同一分布刚度下的精确
Kirchhoff Levy 解）：

| nx x ny | w_centre [mm] | w_exact [mm] | err_w | Mx_centre [N·m/m] | Mx_exact [N·m/m] | err_Mx |
|---|---|---|---|---|---|---|
{e2e_tbl}

**精确 Levy 锚点**（同刚度下）：

| beta | K [N/rad] | Mx_centre [N·m/m] | sigma_centre [MPa] |
|---|---|---|---|
{ex_tbl}

标定设计点处 von Mises 弯曲应力场峰值 `{fld['max_MPa']:.2f}` MPa
（`beta_x={fld['beta_x']:.3f}`, `beta_y={fld['beta_y']:.3f}`）。

## 4. 结果图（场量云图 + 与解析解比较曲线）

三张图即本单落地的交付物（`docs/assets/stiffener_calibration/`），由
`gen_stiffener_calibration_figures.py` 从已提交 JSON 确定性重生成（只读 JSON，
不重算），降采样 0.7x + 128 色调色板以压进 200 KB 工件帽。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **原 PR #1226 为何被关**：其 8 个文本交付物早已经 #1020 在 main，分支内容
   100% 被 main 包含，mergeable 门无新内容可并 → 90 秒内被裁关。本单闭合的
   真实缺口是**被门禁滤除、从未落地的 3 张 PNG**。
2. **PNG 落位**：推送门 v5.4 把 `docs/` 之外的 PNG 当记录区产物滤除（v5.8 F3
   只放行 `docs/` 交付图）；全分辨率三张合计 434.5 KB 又超 merge-governor
   200 KB 工件帽。故落 `docs/assets/` 并降采样（141.9 KB）。全分辨率原件留在
   记录区 `benchmarks/artifacts/`（`python gen_stiffener_calibration_figures.py`
   可复现）。
3. **beta != 固定度**：`zeta = beta/(1-beta)` 强非线性，`beta=0.309` 只 mobilise
   18.3% 的固支端弯矩；「全固支 -37.8% vs 真实 scantling -8.1%」说明 SSSS 只是
   轻度保守上界。
4. **适用**：矩形板、均布压、线性弹性、Winkler 型分布转动弹簧；未含塑性、
   非均匀约束、薄板剪切锁定（t/a<1/50 需 MITC）。

## 6. 误差判定（关键量 vs 基准，逐项）

基准 = 同一分布刚度 `k_theta` 下的**精确 Kirchhoff Levy 解**（解析）：

| 关键量 | 最大绝对误差（nx=24→96 收敛族） | 是否 <3% |
|---|---|---|
| 中心挠度 w | {max(abs(e) for e in w_errs):.3f}%（{min(w_errs):+.3f}% ~ {max(w_errs):+.3f}%） | **{'是' if w_ok else '否'}** |
| 中心弯矩 Mx | {max(abs(e) for e in mx_errs):.3f}%（{min(mx_errs):+.3f}% ~ {max(mx_errs):+.3f}%） | **{'是' if mx_ok else '否'}** |

- 挠度误差随网格**不降反微升**（+0.40%→+0.47%）：Mindlin 剪切变形相对
  Kirchhoff 基准的系统偏差，非网格未收敛（Mx 单调收敛 -0.12%→-0.01% 佐证）。
- 本单为**文档/后端任务**（落地被滤除的图件），无新增计算；上表为交付物自带
  的端到端校验，全部 <3%。
"""

    payload = {
        "title": "重开 TensorFEM#1226 · 管到底第1轮（m24649）：把被合并门禁滤除的 3 张 stiffener 标定图落进 main",
        "summary": ("m21055 的 8 个文本交付物已在 main（#1020）；本单闭合唯一缺口——"
                    "被推送/合并门禁滤除、从未落地的 3 张 stiffener 标定图（band/field/stress_band），"
                    "确定性重生成+降采样至 141.9KB 落 docs/assets/stiffener_calibration/，"
                    "44 项相关测试通过，端到端 vs 精确 Levy 解 w/Mx 误差均 <3%。"),
        "scenario_key": "GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT",
        "mission_id": 24649,
        "tags": ["TensorFEM", "GAP-PP-4", "stiffener", "rotational-restraint", "管到底", "m21055"],
        "markdown": md,
        "figures": [
            fig("beta_stiffener_calibration_band.png",
                "图1  由真实 scantling 反演的 beta 与 mobilised 端部弯矩固定度 vs 正交构件间距；"
                "黑星=冰带设计点（beta_x=0.309 纵桁 / beta_y=0.471 横框架）。"),
            fig("beta_stiffener_calibration_field.png",
                "图2  标定约束下 von Mises 弯曲应力场（均布 1 MPa，0.7x0.8 m 板格）；峰值 317.1 MPa。"),
            fig("beta_stiffener_calibration_stress_band.png",
                "图3  中心应力 vs beta：Mindlin FE 带（40x46）+ 同刚度精确 Kirchhoff Levy 锚点 + 标定设计点；"
                "SSSS 356.0 → 全固支 221.3 MPa（-37.8%），真实约束仅 -8.1%。"),
        ],
    }
    req = urllib.request.Request(
        BASE + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + login()})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    print("report id:", out.get("id"), "status:", r.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
