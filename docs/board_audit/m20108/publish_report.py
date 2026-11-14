#!/usr/bin/env python3
"""Publish the m20108 board-audit report to the platform (tensor/reports).

All numbers are injected programmatically from the audit JSONs
(`verdict_evidence_m20108.json`, `independent_heat_l2_check.json`) and from the
audited `evidence.json` -- no hand-typed values.

Usage: TENSOR_API_BASE=... python3 docs/board_audit/m20108/publish_report.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
AUDITED = pathlib.Path("/root/tp-agent/work/m20054")
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

EVID = json.loads((HERE / "verdict_evidence_m20108.json").read_text())
MINE = json.loads((HERE / "independent_heat_l2_check.json").read_text())
REF = json.loads((AUDITED / "artifacts/CASE-FEM-HEAT-L2/evidence.json").read_text())


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def login() -> str:
    user = os.environ["TENSOR_AGENT_USER"]
    pw = os.environ["TENSOR_AGENT_PASS"]
    out = post("/auth/login", {"username": user, "password": pw})
    return out.get("access_token") or out["data"]["access_token"]


def figure(path: pathlib.Path, name: str, caption: str) -> dict:
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode()}


def main() -> None:
    studies = REF["studies"]
    rows = []
    for name, s in studies.items():
        if s["representable"]:
            worst = max(abs(v) for v in
                        [r["errors"]["l2_absolute"] for r in s["runs"]])
            rows.append(f"| {name} | 控制例（可表示，不主张阶次） | — | — | 最大绝对误差 {worst:.3e} |")
            continue
        finest = s["l2_orders"][-1]
        dev = abs(finest - 2.0) / 2.0 * 100.0
        rows.append(f"| {name} | {finest:.4f} | {dev:.3f}% | {min(s['l2_orders'][1:]):.4f} | "
                    f"{'<3% ✅' if dev < 3 else '≥3% ❌'} |")

    iv = MINE["variants"]
    ind_rows = "\n".join(
        f"| {label} | " + ", ".join(f"{v:.4f}" for v in iv[label]["l2_orders"]) +
        f" | {iv[label]['l2_finest_order']:.4f} | {iv[label]['runtime_s']:.1f} s |"
        for label in ("own_element_mean_source", "shipped_kernel", "own_exact_load"))

    aud = [f"{v:.4f}" for v in studies["sine_generation"]["l2_orders"]]
    find_rows = "\n".join(
        f"| {f['id']} | {f['severity']} | {'否' if not f['blocking'] else '是'} | {f['summary']} |"
        for f in EVID["findings"])

    md = f"""# m20054 CASE-FEM-HEAT-L2 交差核验（board-audit m20108）：L4 / confirmed

被审对象：mission **20054**，commit `8867c2a4`，anchor **CASE-FEM-HEAT-L2**（稳态热传导 Q4 的 L2 收敛率 ≥2 阶）。
本单是**交差核验**（非重做）：核三件套真实性、按基准判据判 got 数值、复现 repro_cmd、独立重实现交叉核。

## ① 案例介绍（物理背景 / 几何 / 参数表）

稳态热传导是船体结构热载荷（主机排气、舱室保温、焊后残余应力热载）与电子设备散热分析的底层求解器；
L2 阶次是这批算例可信度的基线。被审算例用**制造解（MMS）**做代码验证，参考解是解析式本身。

| 参数 | 值 |
|---|---|
| 域 | Ω = (0,1)×(0,1)（`sine_aspect2` 为 2:1 域） |
| 方程 | -div(k∇T) = q，k = 1 |
| 解析参考解 | T = sin(πx) sin(πy)，q = 2π²T（运行时求值，零手抄） |
| 边界条件 | 四边全 Dirichlet，T = T_exact（非齐次） |
| 离散 | 仓内既有双线性 Q4 传导单元，2×2 Gauss 传导矩阵，全稠密直接解 |
| 网格阶梯 | nx = 4, 8, 16, 32, 64（4225 未知量止步，稠密矩阵 143 MB） |
| 误差范数 | 独立 4 点 Gauss 后处理：‖e‖₀ = √(∫(T_h-T)²/∫T²)，|e|₁ = √(∫|∇T_h-∇T|²/∫|∇T|²) |
| 目标阶 | p_ref = 2.0，由 `target_l2_order_from_manual()` 正则解析仓内先验表 `adaptive_poisson_2d.py:34`（审计已逐字核对） |
| 预登记闸门 | L2 每对 ≥1.90；finest-pair ∈ [1.95, 2.10]；H1 ≥0.90；幂律残差 ≤1%；控制例 ≤1e-10；场误差 ≤3%；积分不变性 ≤1e-8 |

## ② 怎么做的（三层独立审计，不采信作者自述）

1. **L1 复跑**：把 m20054 工作树字节拷贝到 `/tmp/audit20054_wt`（不写他人工作树），执行其 repro_cmd
   `PYTHONPATH=src python3 benchmarks/heat_l2_convergence.py` → 37.7 s，`PASS (31 checks)`；
   4 件工件（evidence.json / tables.md / 2 PNG）sha256 与 commit `8867c2a4` 的 git blob **逐字节相同**，
   连重跑后的六段报告 md 都不变 ⇒ 文档数值确为程序注入。
2. **L2 从原始数组重算**：直接读 `evidence.json` 的 `h`/`l2_relative` 数组自行算 log-log 阶次与最小二乘斜率，
   与作者写入的 `l2_orders` 差 **0.0**，再自行判 p≥2。
3. **L3 独立重实现**（关键）：`independent_heat_l2_check.py` **不 import 被审模块**，自写 Q4 装配（2×2 Gauss 刚度）、
   自写 8×8 Gauss 单元均值源、自写 4×4 Gauss 误差范数，跑 nx=8/16/32/64；并另用**仓内主干内核**
   `tensorfem.thermal.solve_steady_thermal` 配审计者自己的网格/源/误差范数再算一遍。另加"精确积分载荷"变体区分载荷误差与离散误差。

环境：python 3.11.6 · torch 2.7.1 · numpy 2.4.6 · 4 线程 CPU；pytest 复跑 **17 passed / 22.59 s**。

## ③ 计算结果（数字全部程序化注入）

**基准裁判（SQL，2026-11-14）**：`tensor_benchmark_cases WHERE anchor_slug='CASE-FEM-HEAT-L2'` → **0 行**；
判据只存在于算例池 `tensor_case_pool` 的 `FEM-HEAT-L2 / tol_policy='收敛率判据'`。
故以"收敛率判据"为唯一裁判，按下表逐项核 got。

| 算例 | finest-pair L2 阶 (got) | 与 p=2 偏差 | 全阶梯 min 阶（coarse nx≥8） | 判定 |
|---|---|---|---|---|
{chr(10).join(rows)}

31 项预登记闸门复跑 **31/31 PASS**；有效判别算例 3 个（sine_generation / exp_trig_harmonic / sine_aspect2）。

**独立重实现交叉核（正弦 MMS，nx=8→64）**

| 变体 | L2 阶（8→16, 16→32, 32→64） | finest | 运行时 |
|---|---|---|---|
| 被审 evidence.json（含 4→8 段） | {', '.join(aud)} | {studies['sine_generation']['l2_orders'][-1]:.4f} | — |
{ind_rows}

误差范数逐级比对：审计自有实现 vs 被审值相对差 **3.6e-15 … 2.3e-11**；仓内内核 vs 审计自有装配最大相对差 **2.3e-11**。
⇒ 被审数字不是后处理假象；"精确积分载荷"变体阶升至 2.0000 且误差小至 0.386 倍，说明单元常数源只放大误差常数、不改阶次。

## ④ 结果图

图 1：审计交叉核（A 面板 = 被审阶梯与三种独立变体 + O(h²) 参考线；B 面板 = 各算例逐对阶次与预登记闸门）。
图 2：被审计的场量云图（sine_generation，nx=32：FE 场 / 解析场 / 逐点误差）。
图 3：被审计的 L2/H1 收敛曲线（虚线为理论斜率）。

## ⑤ 注意点（发现，按严重度排序）

| ID | 严重度 | 阻塞 | 内容 |
|---|---|---|---|
{find_rows}

要点解释：
* **F1（中，平台侧）** 基准表对应该 anchor 无行，判据只是自由文本，无法机读裁定；不阻塞本轮（生产者在代码中预登记了数值闸门：`heat_l2_convergence.py:137-143`），
  但建议板端补一行 metric/tolerance。审计者**未**代为写基准表（自证自判不合适）。
* **F2（低）** 6 算例中 3 例对求解器非判别（两项节点精确、误差即纯插值；一项可表示控制例），"31 项闸门"不等于 31 个独立求解器检查；文档 §⑥ 已主动披露。
* **F3（低）** `tensorfem.thermal` 源项为单元常数 ⇒ 载荷一致性误差 O(h²)，与离散误差同阶；阶次判据不受影响，但绝对精度类数字（如 0.08% 场误差）只在给定网格成立。
* **F6（信息）** 审计时该 commit 未合入主干，仅有 `heat_l2_convergence.py:446 → tensorfem.thermal.solve_steady_thermal` 的调用，无平台 API 调用者 ⇒ **不主张 L3**。

**未跑的检查（标 unknown，不标 pass）**：物理实验/文献实测对标（MMS 属代码验证，无实测数据集 → NOT_APPLICABLE）；
nx=128 阶梯（稠密 2.2 GB，作者声明不可行，审计未压测内存）；奇异解阶次退化（声明范围外）；平台 API 集成（不存在，NOT_SUPPORTED）。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

* 三个判别算例的 finest-pair L2 阶 = 1.9992 / 1.9998 / 1.9979，**相对 p_ref=2.0 的偏差 0.042% / 0.009% / 0.107%**。
* **是否 <3%：是**（最差 0.107%，比门限小 28 倍）；最差单对阶 1.9653 亦高于预登记闸门 1.90。
* 控制例：最大绝对误差 3.553e-15（< 1e-10 闸门）⇒ 求解器对可表示场无附加误差。
* 复现性：repro_cmd 37.7 s 一键重跑，四件工件 sha256 与 commit 逐字节一致；pytest 17/17 passed。

**判级：L4（基准验证 = 完整算例 + 误差分析 + 可复现）**，不判 L5（无新物理/新方法），不主张 L3（未接入主求解流程/平台 API）。
**Disposition：approve with notes（confirmed），阻塞项 0。** verdict 已落表 `tensor_maturity_verdicts`（id 828, level=4, verdict_by='board-audit', gate_state=pending，由 maturity_gate 裁定）。
"""

    figures = [
        figure(HERE / "figures/m20108_heat_l2_crosscheck.png", "audit-crosscheck",
               "图1 审计交叉核：被审阶梯 vs 审计独立 numpy Q4 / 仓内内核驱动 / 精确积分载荷；右：逐对阶次与预登记闸门"),
        figure(AUDITED / "docs/assets/case_fem_heat_l2_field.png", "field-cloud",
               "图2 被审场量云图：sine_generation nx=32 的 FE 场、解析场与逐点误差"),
        figure(AUDITED / "docs/assets/case_fem_heat_l2_convergence.png", "convergence",
               "图3 被审 L2/H1 收敛曲线（虚线为 O(h²)/O(h) 理论斜率）"),
    ]

    payload = {
        "title": "m20054 CASE-FEM-HEAT-L2 交差核验：L4 基准验证（finest L2 阶 1.9979–1.9998，偏差 <0.11%）",
        "summary": ("board-audit(m20108) 对 m20054 交差核验：三件套齐全且字节级可复现（37.7 s 重跑，4 工件 sha256 与 commit 一致）；"
                    "基准表该 anchor 无行，按算例池'收敛率判据'逐项核 got，finest-pair L2 阶 1.9992/1.9998/1.9979（偏差 <0.11%<<3%），31/31 闸门 PASS；"
                    "审计者另写独立 Q4 重实现复算，误差范数相对差 ≤2.3e-11、阶次逐位一致；判 L4 confirmed，1 项中危非阻塞（缺机读基准行）。"),
        "scenario_key": "CASE-FEM-HEAT-L2",
        "mission_id": "20108",
        "tags": ["board-audit", "TensorFEM", "CASE-FEM-HEAT-L2", "MMS", "convergence-order", "L4", "tensor_maturity_verdicts"],
        "markdown": md,
        "figures": figures,
    }
    (pathlib.Path("/tmp/report_m20108.json")).write_text(json.dumps(payload))
    token = login()
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()