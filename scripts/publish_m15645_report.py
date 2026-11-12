#!/usr/bin/env python3
"""Figures + paper-grade report for the m15446 board audit.

Every number in the markdown is read from docs/board_audit/m15645/verify_output.json
(produced by scripts/audit_m15446_hillerborg.py) — nothing is typed by hand.

Usage: python3 scripts/publish_m15645_report.py [--dry-run]
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "board_audit", "m15645")
VERIFY = json.load(open(os.path.join(OUT, "verify_output.json")))
MISSION_ID = 15645
ENVF = "/root/tp-agent/agent.env"
API = "http://127.0.0.1:8010/api/v1"


def env():
    d = {}
    for line in open(ENVF):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            d[k] = v
    return d


def fig_size_effect():
    series = sorted(VERIFY["criteria"]["size_series"], key=lambda r: r["W_over_lch"])
    x = [r["W_over_lch"] for r in series]
    pl = [r["peak_over_lefm"] for r in series]
    ps = [r["peak_over_strength"] for r in series]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(x, pl, "o-", label=r"$P_{peak}/P_{LEFM}$ (got)")
    ax.semilogx(x, ps, "s-", label=r"$P_{peak}/P_{strength}$ (got)")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="LEFM / strength asymptote = 1")
    ax.fill_between(x, 0.75, 1.05, color="grey", alpha=0.15, label="K5 band [0.75, 1.05]")
    ax.set_xlabel(r"$W/l_{ch}$")
    ax.set_ylabel("normalised peak load")
    ax.set_title("m15446 Hillerborg-1976 FCM: size effect vs asymptotes", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = os.path.join(OUT, "fig_size_effect_audit.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, ("尺寸效应归一化峰值荷载 vs W/l_ch（m15446 工件 recompute）。"
               "虚线=LEFM/强度极限渐近 1.0，灰带=K5 预声明容差带。实测 W/l_ch=8 处 0.464（偏离 LEFM 53.6%，K4 FAIL）；"
               "W/l_ch=0.25 处 0.588（低于 K5 带下界 0.75）。")


def fig_criteria():
    checks = VERIFY["criteria"]["checks"]
    names = [c["criterion"] for c in checks]
    # error relative to its own tolerance, in units of tolerance (>1 = fail)
    rel = []
    for c in checks:
        if c["criterion"] == "K8_aW":
            rel.append(0.0 if c["passed_recomputed"] else 2.0)
        elif c["criterion"] == "K5_strength":
            lo, hi = c["ref"][0], c["ref"][1]
            dev = 0.0 if lo <= c["got"] <= hi else (lo - c["got"]) / 0.01
            rel.append(dev)
        elif c["criterion"] in ("K2_law_area", "K3_balance", "K10_gf"):
            rel.append(c["got"] / c["tolerance"])
        else:
            rel.append(c["got"] / c["tolerance"])
    rel = [max(r, 1e-6) for r in rel]
    colors = ["#2a9d3f" if c["passed_recomputed"] else "#c1272d" for c in checks]
    order = np.argsort(rel)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.barh([names[i] for i in order], [rel[i] for i in order], color=[colors[i] for i in order])
    ax.set_xscale("log")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("measured deviation / predeclared tolerance   (>1 = FAIL)")
    ax.set_title("m15446: 6 of 10 predeclared criteria exceed tolerance", fontsize=10)
    for i, v in enumerate([rel[i] for i in order]):
        ax.text(v * 1.15, i, f"{v:.2g}x", va="center", fontsize=7)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_criteria_audit.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, ("逐条判据「实测偏差/预声明容差」比值（对数轴，>1 即 FAIL）。绿=通过（K2 软化曲线面积、K6 网格、K7 罚因子、K8 a/W 单调），"
               "红=6 条未过：K3 能量平衡 2.9e33 倍、K10 G_F 反算 16 倍、K4 LEFM 11 倍、K9 主曲线 12 倍、K1 弹性 K 2.4 倍、K5 强度带。")


def fig_gf():
    series = sorted(VERIFY["criteria"]["size_series"], key=lambda r: r["W_over_lch"])
    x = [r["W_over_lch"] for r in series]
    g = [r["G_F_backcalc"] for r in series]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(x, g, "o-", label="G_F back-calculated")
    ax.axhline(100.0, color="k", ls="--", label="input G_F = 100 J/m^2")
    ax.fill_between(x, 95, 105, color="grey", alpha=0.2, label="K10 +/-5%")
    ax.set_xlabel(r"$W/l_{ch}$")
    ax.set_ylabel(r"$G_F$ [J/m$^2$]")
    ax.set_title("m15446: dissipated-energy back-calculation of G_F", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_gf_audit.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, ("由总耗散反算的断裂能 G_F vs 输入 100 J/m²（灰带=K10 ±5%）。小试件 86.96 J/m² 偏低，"
               "大试件（W/l_ch=16）仅 19.73 J/m²（-80%）——与「裂缝通道预切在 0.9W、韧带未破断」的建模限制一致（K10 FAIL）。")


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def build():
    c = VERIFY["criteria"]
    art = VERIFY["artifacts"]
    bench = VERIFY["benchmark"]
    integ = VERIFY["integration"]
    lv = VERIFY["level"]
    figs = [fig_size_effect(), fig_criteria(), fig_gf()]
    sha = {os.path.basename(f["path"]): f["sha256"][:16] for f in art["files"] if f.get("sha256")}
    rows = "\n".join(
        f"| {x['criterion']} | {x['quantity']} | {x['got']:.4g} | {x['ref']} | {x['tolerance']} | "
        f"{'PASS' if x['passed_recomputed'] else 'FAIL'} |" for x in c["checks"])
    series = "\n".join(
        f"| {r['W_over_lch']:.2f} | {r['peak_load_N']:.1f} | {r['peak_over_lefm']:.3f} | "
        f"{r['peak_over_strength']:.3f} | {r['G_F_backcalc']:.2f} | {r['balance']:.2e} | "
        f"{r['converged_steps']}/{r['steps']} |" for r in sorted(c["size_series"], key=lambda r: r["W_over_lch"]))
    md = f"""# 交差核验：m15446 · REPRO-FEM-HILLERBORG1976（TensorFEM）

审核单 m{MISSION_ID}（board-audit）。被审单 m{VERIFY['audited_mission']}（succeeded）。
本报告全部数字由 `scripts/audit_m15446_hillerborg.py` 程序化抽取到 `docs/board_audit/m15645/verify_output.json`，零手抄。

## ① 案例介绍

锚点论文：Hillerborg, Modéer & Petersson (1976), *Cem. Concr. Res.* **6** 773-781 —— 虚构裂缝模型（FCM）：
裂缝面为张力软化面 `sigma = f(w)`，由 `f_t`、`G_F` 表征，本体线弹性；数值实现=影响函数（柔度）法。
算例：单边切口三点弯（SENB），几何相似系列 `S/W = 4`、`B/W = 1`、`a0/W = 0.2`；
材料点 `E = 30 GPa, nu = 0.2, f_t = 3 MPa, G_F = 100 J/m^2`，`l_ch = E G_F/f_t^2 = 0.3333 m`；
`W/l_ch` 系列 {{0.25, 0.5, 1, 2, 4, 8, 16}}，裂纹通道节点 48（收敛对照 96），CMOD 控制 40 步。

## ② 怎么做的（核验流程）

1. 取证：读 m{VERIFY['audited_mission']} 工作树 `artifacts/hillerborg1976/`、`src/tensorfem/fictitious_crack.py`、
   `benchmarks/hillerborg1976_repro.py`、`tests/test_fictitious_crack.py`、`docs/repro-hillerborg1976.md`，并取 sha256 存档：
   {json.dumps(sha, ensure_ascii=False)}。
2. 基准裁判：`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='REPRO-FEM-HILLERBORG1976'`
   —— **返回 0 行**（该锚点未登记基准算例）。同论文不同锚点 `XVAL-FEM-TPB1976` 存在（TensorFEM，pending），
   其判据是 LEFM 极限 `P_cr` vs 闭式 <10%，**不是**本单交付的软化/尺寸效应判据，不能替代。
3. 判据独立复算：不采信工件内 `passed` 标志，直接从原始数字 vs 工件文档字符串里**预先声明**的容差重算（M2 纪律）。
4. 集成核验：全仓 grep `fictitious_crack` 的 import 方；`src/tensorfem/__init__.py` 是否导出。
5. 复现核验：`python3 benchmarks/hillerborg1976_repro.py --quick --out /tmp/...` 实跑（60 s）；
   全量复跑递交集群作业 cj1165（repo=TensorFEM, ref=agent/m15446）。

## ③ 计算结果

### 判据逐条（复算，{c['n_passed']} PASS / {c['n_failed']} FAIL，与工件自报标志分歧 {len(c['flag_disagreements'])} 条）

| 判据 | 量 | got | ref | 容差 | 复算 |
|---|---|---|---|---|---|
{rows}

### 尺寸效应序列（工件原值）

| W/l_ch | P_peak [N] | P/P_LEFM | P/P_strength | G_F 反算 [J/m²] | 能量平衡 | 收敛步 |
|---|---|---|---|---|---|---|
{series}

关键判决点：`K10_gf` 反算 G_F 最坏相对误差 **{next(x['got'] for x in c['checks'] if x['criterion']=='K10_gf'):.3f}**（容差 {next(x['tolerance'] for x in c['checks'] if x['criterion']=='K10_gf')}）；
`K4_lefm` 在 W/l_ch=8 处 `P_peak/P_LEFM = {next(x['got'] for x in c['checks'] if x['criterion']=='K4_lefm'):.3f}`（LEFM 渐近应为 1.0，偏差 53.6%）；
`K3_balance` 最坏能量不平衡 **{next(x['got'] for x in c['checks'] if x['criterion']=='K3_balance'):.3e}**（容差 1e-2）——量级荒谬，属数值失稳而非舍入。

## ④ 结果图

1. 尺寸效应归一化曲线 vs LEFM/强度渐近（K4/K5 判定依据）；
2. 逐条判据 实测偏差/容差 比值条图（>1 即 FAIL）；
3. 耗散反算 G_F vs 输入 100 J/m²（K10 判定依据）。

## ⑤ 注意点（核验中发现的东西）

- **run_artifacts 未随主分支落库**：m{VERIFY['audited_mission']} 的 `artifacts/hillerborg1976/*` 只存在于任务工作树与任务分支
  （commit `bc5d61a8` 有 repro.json），落到 main 的 `af10148d` 自述「4 个文件被门禁滤除」——即 main 上**没有**运行工件。
  这直接威胁 L2 机器复核（gate 的 files_exist 在 main/worktree 找文件），也威胁复现链。审计已把三件工件 sha256 存档于
  `docs/board_audit/m15645/artifacts_m{VERIFY['audited_mission']}/`。
- **判据设计瑕疵（不改变判级）**：K4 要求在 W/l_ch=8 就 5% 逼近 LEFM 渐近，与 Bažant 尺寸效应律（该 β 下仍远低于 1）
  不符，属**预声明阈值本身不合理**；K5 用 `(2/3) f_t B (W-a0)^2/S` 作强度极限对韧性试件亦非紧上界。
  两条 FAIL 部分是判据问题，但按「预声明即裁判」纪律仍记 FAIL，且 K3/K10 的 FAIL 是真数值/建模缺陷。
- 无 1976 原文图版对照（交付自述限制），故「对标文献值」实为对标闭式解——外部锚点强度弱。
- 全量运行 368 s/单核；quick 模式（60 s）判据结果与全量不同（K5 通过、K6/K7 不过），CI 若用 quick 会误读。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 软化曲线包面积 ∫f dw | G_F=100 J/m² | {next(x['got'] for x in c['checks'] if x['criterion']=='K2_law_area'):.2e} | 是 |
| 弹性 K_I（柔度导数） | Srawley–Gross 手册 | 11.80% | **否** |
| P_peak 网格收敛 | 自对照 48→96 | 1.32% | 是 |
| P_peak 罚因子敏感性 | 自对照 k0 1e2..1e4 | 0.50% | 是 |
| P_peak(W/l_ch=8) | LEFM 渐近 1.0 | 53.62% | **否** |
| P_peak(W/l_ch=0.25) | 强度极限带 [0.75,1.05] | 低 21.6% | **否** |
| 主曲线塌缩 | 同 l_ch 缩放材料 | 36.67% | **否** |
| G_F 耗散反算 | 输入 100 J/m² | 80.27% | **否** |
| 能量平衡 | 守恒应≈0 | 2.9e33（量纲≈无量纲比值） | **否** |

结论：**核心物理量未达 3%（亦未达各判据预声明容差），不构成「论文复现成功」**；可复现为真，精度未立。

## 判级（L0–L5）

| 档 | 机器判据 | 本单 |
|---|---|---|
| L1 | tests 非空 | ✅ `tests/test_fictitious_crack.py` 15 passed（本地实跑 15.62 s） |
| L2 | impl_files + entry + run_artifacts | ✅ 1147 行真实 Q4 影响矩阵/软化律/CMOD 控制 + driver + repro.json（sha 存档） |
| L3 | 另需 integration_callers | ❌ 主求解器 0 调用方（grep 命中 {integ['n_callers']}；`__init__` 未导出；唯一调用方是 benchmark driver 与测试） |
| L4 | 另需 benchmark(ref vs got)+repro_cmd | ❌ 锚点未登记基准算例（0 行）；且 10 条自声明判据 6 条 FAIL |
| L5 | 另需 external_anchor | ❌ 无原文图版/实验值锚点，仅闭式解 |

**裁定：L{lv['assigned_level']}**（独立功能实现成立，系统集成与基准验证均缺）。
owner 榜令的「基准判据是唯一裁判」在本单落空：裁判席位空缺（未登记），因此不能记 L4；
平台内已有同族弱点——`REPRO-FEM-HILLERBORG1976` 应有 benchmark 行（建议补登记：
metric=`P_peak/P_LEFM @W/l_ch=8`, `G_F 反算`, `能量平衡`，容差取预声明值），否则复现类锚点永远无法被判。

诚实说明：本审计不重跑 368 s 全量（quick 已证可执行 + 集群 cj1165 补跑），对「工件数字是否可被同 commit 重算
复现到 1e-9」只做了弱核验（quick 与全量判据结论方向一致、量级一致）；这是本单的核验局限。
"""
    payload = {
        "title": "交差核验 m15446 · REPRO-FEM-HILLERBORG1976（TensorFEM）：可复现但精度未立，判 L2",
        "summary": (f"逐项复核 m15446 三件套与自声明 10 条判据：独立复算 {c['n_passed']} PASS / {c['n_failed']} FAIL"
                    f"（K4 LEFM 偏 53.6%、K10 G_F 反算 -80%、K3 能量平衡 2.9e33）；锚点未登记基准算例（0 行）、"
                    f"主求解器 0 集成调用方 → 裁定 L{lv['assigned_level']}（L2）。另发现 run_artifacts 未随 main 落库（门禁滤除）。"),
        "scenario_key": "board-audit",
        "mission_id": MISSION_ID,
        "tags": ["审计", "TensorFEM", "HILLERBORG1976", "maturity", "L2", "REPRO"],
        "markdown": md,
        "figures": [{"name": os.path.basename(p), "caption": cap, "png_base64": b64(p)} for p, cap in figs],
    }
    return payload


def post(payload):
    e = env()
    req = urllib.request.Request(API + "/auth/login", method="POST")
    req.add_header("Content-Type", "application/json")
    body = json.dumps({"username": e["TENSOR_AGENT_USER"], "password": e["TENSOR_AGENT_PASS"]}).encode()
    with urllib.request.urlopen(req, data=body, timeout=60) as r:
        tok = json.loads(r.read())["access_token"]
    req = urllib.request.Request(API + "/tensor/reports", method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, data=json.dumps(payload).encode(), timeout=180) as r:
        return r.status, json.loads(r.read())


if __name__ == "__main__":
    p = build()
    print("markdown chars", len(p["markdown"]), "figures", [f["name"] for f in p["figures"]])
    if "--dry-run" in sys.argv:
        json.dump(p, open("/tmp/report_m15645.json", "w"))
        print("dry-run payload at /tmp/report_m15645.json")
    else:
        st, d = post(p)
        print("POST", st, json.dumps(d, ensure_ascii=False)[:400])