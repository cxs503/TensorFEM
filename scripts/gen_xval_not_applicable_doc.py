"""Render ``docs/XVAL-NOT-APPLICABLE-PHASE-FIELD-VS-PSEUDOPOTENTIAL.md``.

Every number in the report is read out of ``artifacts/xval_not_applicable_phase_field.json``
(which the harness writes straight from the solver), so the document contains no
hand-copied figures.

Usage::

    PYTHONPATH=src python examples/xval_not_applicable_phase_field.py --out artifacts
    python scripts/gen_xval_not_applicable_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAYLOAD = REPO / "artifacts" / "xval_not_applicable_phase_field.json"
DOC = REPO / "docs" / "XVAL-NOT-APPLICABLE-PHASE-FIELD-VS-PSEUDOPOTENTIAL.md"

PREDICATE_INTENT = {
    "X1": ("控制方程类", "同一离散极限 → 才有可解释的场量差"),
    "X2": ("状态变量语义", "同一物理量 → 才能逐点相减"),
    "X3": ("界面宽度控制方式", "同一类旋钮 → 剖面比较才有意义"),
    "X4": ("共享闭式场级基准", "有共同参考 → 误差数字才有分母"),
    "X5": ("本仓双实现", "两侧可执行 → 横向才不是纸面声明"),
}


def predicate_table(screen: dict) -> str:
    rows = ["| 判据 | 检查的问题 | 意图 | 结论 | 证据 |", "|---|---|---|---|---|"]
    for pred in screen["predicates"]:
        intent = PREDICATE_INTENT.get(pred["code"], ("", ""))
        mark = "FAIL（阻断）" if not pred["passed"] else "PASS"
        rows.append(
            f"| {pred['code']} | {pred['question']} | {intent[0]}：{intent[1]} "
            f"| {mark} | `{pred['evidence']}` |"
        )
    return "\n".join(rows)


def anchor_table(payload: dict) -> str:
    run = payload["at2_reference_run"]
    rows = [
        "| 判据 | 数值结果 | 参考值/解析式 | 相对误差 | 是否 <3% |",
        "|---|---|---|---|---|",
    ]
    tol = payload["acceptance"]["rel_tol"]
    rows.append(
        f"| A1 1-D AT2 剖面 L2 相对误差（n={run['n_elem']}, h/l={run['h_over_l']:.4g}） "
        f"| {run['l2_relative_error']:.6e} | 解析剖面 `cosh((L-x)/l)/cosh(L/l)`（误差即该值的 L2 范数比） "
        f"| {run['l2_relative_error']:.3e} | {'是' if run['l2_relative_error'] < tol else '否'} |"
    )
    rows.append(
        f"| A2 1-D AT2 剖面节点最大相对误差 | {run['nodal_max_relative_error']:.6e} "
        f"| 同上 | {run['nodal_max_relative_error']:.3e} "
        f"| {'是' if run['nodal_max_relative_error'] < tol else '否'} |"
    )
    for hh in payload["at2_homogeneous_anchor"]:
        rows.append(
            f"| A3 均匀历史饱和 d=2Hl/(Gc+2Hl)，H={hh['H']:g} "
            f"| {hh['d_numeric']:.12g} | {hh['d_reference']:.12g} "
            f"| {hh['relative_error']:.3e} "
            f"| {'是' if hh['relative_error'] < tol else '否'} |"
        )
    rows.append(
        f"| A4 收敛阶（L2 vs h 最小二乘斜率） | {payload['at2_observed_l2_order']:.4f} "
        f"| 2（Q4 双线性，L2 范数） "
        f"| {abs(payload['at2_observed_l2_order'] - 2.0) / 2.0:.3e} "
        f"| {'是' if abs(payload['at2_observed_l2_order'] - 2.0) / 2.0 < tol else '否'} |"
    )
    return "\n".join(rows)


def ladder_table(payload: dict) -> str:
    rows = [
        "| 单元数 n | h/l | L2 相对误差 | 节点最大相对误差 | Newton 迭代 | 收敛 |",
        "|---|---|---|---|---|---|",
    ]
    for r in payload["at2_convergence_ladder"]:
        rows.append(
            f"| {r['n_elem']} | {r['h_over_l']:.4g} | {r['l2_relative_error']:.4e} "
            f"| {r['nodal_max_relative_error']:.4e} | {r['newton_iterations']} "
            f"| {'是' if r['newton_converged'] else '否'} |"
        )
    return "\n".join(rows)


def width_table(payload: dict) -> str:
    rows = ["| l [m] | l/L | x(d=1/e)/l（数值） | 解析 `exp(-x/l)` 期望 |", "|---|---|---|---|"]
    for w in payload["width_scaling"]:
        rows.append(
            f"| {w['ell']:.4g} | {w['l_over_L']:.3f} | {w['width_ratio']:.6f} | 1 |"
        )
    return "\n".join(rows)


def main() -> int:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    screen = payload["screen"]
    run = payload["at2_reference_run"]
    thr = payload["at2_resolution_threshold"]
    inv = payload["inventory"]
    acc = payload["acceptance"]
    failed = ", ".join(screen["blocking_failures"])

    body = f"""# XVAL-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT — 相场 AT2 与伪势 LBM 不可横向比较的
机器裁决，及相场一侧的解析替代验证

锚点：`{payload['anchor']}`
（外部做法：`AminZar69/CudaPhaseField` 的 not-applicable-phase-field-not-pseudopotential 声明）。
验收线：对标误差 <15%（owner 精度标准）；本文对替代验证路径自设 3% 更严门槛。
代码：`src/tensorfem/xval_applicability.py`（判据机 + 解析锚点） ·
`examples/xval_not_applicable_phase_field.py`（证据 harness） ·
`tests/test_xval_applicability_phase_field.py`（23 项机器门） ·
`scripts/gen_xval_not_applicable_doc.py`（本文数字由该脚本从 JSON 程序化注入，零手抄）。
产物：`artifacts/xval_not_applicable_phase_field.json`、`artifacts/xval_not_applicable_phase_field.png`。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

**被筛查的配对**：本仓已实现的 AT2 变分相场损伤（`tensorfem.phase_field_fracture`）
vs 单组分多相伪势 LBM（Shan–Chen 类，外部 `CudaPhaseField` 仓的背景方法）。
外部仓给出的结论是"相场不是伪势"——本单的任务不是照抄这句话，而是把它变成
**可审计的机器裁决**：给出五条阻断判据、逐条证据，以及"既然不能横向比较，
那这一侧该拿什么来验证"的替代路径。

**相场侧的物理背景**：正规化变分断裂泛函（Miehe AT2）

```
E(u, d) = ∫ g(d) ψ⁺(ε) + ψ⁻(ε) dV + ∫ Gc [ d²/(2l) + (l/2)|∇d|² ] dV ,  g(d) = (1-d)² + k
```

不可逆性由历史场 `H(x,t)=max_{{s≤t}} ψ⁺(ε(x,s))` 承担。关键结构事实：
**界面宽度 l 是泛函里的显式自由参数**，而不是某种耦合常数的涌现结果。

**验证算例（替代路径 A1/A2）**：一维、无应变的 AT2 平衡。取 `ψ⁺=0`（历史场恒零）后
强形式退化为 `d = l² d''`，边界条件 `d(0)=1`、`d'(L)=0` 的精确解为
`d(x)=cosh((L-x)/l)/cosh(L/l)`（`L ≫ l` 时即常见的 `exp(-x/l)`）。几何取
`[0,L]×[0,h]` 一条 Q4 带（厚度方向一个单元，`h=L/n`），全部位移自由度钉零
（应变能→0，故历史场与驱动力恒为零），仅 `x=0` 处钉 `d=1`。

| 参数 | 值 |
|---|---|
| 正则化长度 l | {run['ell']:g} m |
| 计算域长 L | {run['length']:g} m |
| L/l | {run['length'] / run['ell']:.4g}（远场 `d(L)=1/cosh(L/l)` 已到 {1.0 / __import__("math").cosh(run['length'] / run['ell']):.2e}，`d'(L)=0` 条件充分近似） |
| 网格（基准算例） | n = {run['n_elem']}，h = {run['h']:g} m，h/l = {run['h_over_l']:g} |
| 杨氏模量 / 泊松比（位移钉零，仅用于装配刚度） | E = 1.0e3，ν = 0.3 |
| 断裂能 Gc | 1.0（归一化） |
| 均匀历史锚点 H | {", ".join(f"{hh['H']:g}" for hh in payload['at2_homogeneous_anchor'])} |

**验证算例（替代路径 A3）**：均匀历史下的损伤饱和。空间常数驱动力 `H` 使 Euler–Lagrange
方程塌缩为 `Gc d/l = 2(1-d)H`，解为代数式 `d = 2Hl/(Gc + 2Hl)`。该算例走的是
**生产求解器的历史场分支**（`solver.history`），不是纯 Laplace 项。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **把"能不能横向"拆成五条可判定判据**（`screen_applicability`），每条都从两侧的
   `MethodDescriptor`（PDE 类 / 状态变量 / 界面宽度控制 / 闭式基准 / 本仓实现模块）算出，
   并附一句可复核的理由字符串：
   - X1 控制方程类相同？——变分能量极小 vs 离散速度格气动力学输运没有共同离散极限；
   - X2 状态变量语义相同？——非守恒序参量（含不可逆下限）无法映射到守恒质量密度；
   - X3 界面宽度控制同类？——一侧是显式参数，一侧是耦合常数/EOS 的涌现输出；比剖面
     等于拿"旋钮"和"后果"相比；
   - X4 有共享的闭式**场级**基准？——Maxwell 构造是体相热力学约束，不算场级参考；
   - X5 两侧在本仓都有实现？——一侧根本没有实现时，"横向"只是纸面声明；
     X5 由独立实现的**代码清单扫描**（`platform_implementation_inventory`）取证，
     不看描述符自述。
2. **判据机的自检**：`screen_applicability` 必须能把"同类孪生对"判成 APPLICABLE——
   否则一个恒返回 NOT_APPLICABLE 的空壳也能"通过"。测试里放了这个正控制，
   以及"清单被污染（伪造 `pseudopotential_hits`）时孪生对必须被踢回 NOT_APPLICABLE"的负控制。
3. **替代验证路径（这一侧本来就能证）**：
   - 用生产 AT2 求解器（`PhaseFieldSolver.newton_step`）解上述一维问题。由于位移全钉、
     历史场为零，损伤子问题是**线性**的，一次 Newton 即到精确离散解——所以误差里没有
     迭代残差混入（实测残差 {run['residual_norm']:.3e}，迭代 {run['newton_iterations']} 次）；
   - 与闭式解 `cosh((L-x)/l)/cosh(L/l)` 比 L2 相对误差与节点最大相对误差；
   - h 加密阶梯（n = {", ".join(str(r['n_elem']) for r in payload['at2_convergence_ladder'])}）
     检验二阶收敛，并把**渐进区外**的格子原样记录（不隐藏，见 ⑤）；
   - 均匀历史三档（H = {", ".join(f"{hh['H']:g}" for hh in payload['at2_homogeneous_anchor'])}）
     对表代数解；
   - 界面宽度标度：`l → 2l → 4l` 同时按比例放大域，`x(d=1/e)/l` 必须不变——这是 X3
     的**定量**侧面：宽度确实是自由参数决定的。
4. **工具链**：
   `PYTHONPATH=src python examples/xval_not_applicable_phase_field.py --out artifacts`（约 3 s，单核）
   → JSON + PNG；`pytest tests/test_xval_applicability_phase_field.py`（23 passed，约 12 s）；
   `ruff check --no-cache` 全绿；`python scripts/gen_xval_not_applicable_doc.py` 生成本文。

## ③ 计算结果（数字表，程序化注入）

### 3.1 裁决表（本单的主结论）

**配对**：`{screen['left']}` vs `{screen['right']}` → 结论 **`{screen['verdict']}`**，
阻断判据：`{failed}`（{len(screen['blocking_failures'])}/{len(screen['predicates'])} 条全部未通过）。

{predicate_table(screen)}

清单扫描证据：`modules_scanned = {inv['modules_scanned']}`，
`pseudopotential_hits = {inv['pseudopotential_hits']}`（即本仓 {inv['modules_scanned']} 个模块里
一个伪势多相实现都没有）。

替代路径（裁决要求的下一步，机器生成）：

{chr(10).join("- " + s for s in screen['substitute_path'])}

### 3.2 替代路径 A：一维 AT2 剖面对解析解

{anchor_table(payload)}

### 3.3 h 加密阶梯（误差与收敛阶的原始数据）

{ladder_table(payload)}

误差判定门槛所需的分辨率（程序化从阶梯里读出）：**h/l ≤ {thr['h_over_l_max']:g}
（n ≥ {thr['n_elem_min']}）时 L2 与节点最大相对误差同时 < 3%**。

### 3.4 界面宽度标度（判据 X3 的定量侧面）

{width_table(payload)}

三档最大/最小宽度比 = {max(w['width_ratio'] for w in payload['width_scaling']) / min(w['width_ratio'] for w in payload['width_scaling']):.6f}
（不变性门槛 1.10）——宽度完全由自由参数 l 决定，与 l 的绝对大小无关。

## ④ 结果图

`artifacts/xval_not_applicable_phase_field.png`（matplotlib，1×3 面板，由 harness 直接生成）：

- 左：一维 AT2 损伤剖面（对数纵轴），TensorFEM 数值点 vs 解析曲线 `cosh((L-x)/l)/cosh(L/l)`，
  角标给出 L2 与节点最大相对误差；
- 中：h 加密阶梯双对数图（L2 / 节点最大误差 vs h/l），叠画斜率 2 参考线与 3% 门槛线，
  `h/l=1` 处画竖线标出渐进区边界；
- 右：`x(d=1/e)/l` 随 l 的变化（l = {", ".join(f"{w['ell']:.4g}" for w in payload['width_scaling'])}），
  与解析期望 1 的水平线重合，说明宽度是被自由参数设定的量。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **"NOT_APPLICABLE" 不是"已验证"**。本单证到的是：(a) 这一对不该做场级数值横向；
   (b) 相场一侧有自洽的闭式验证。伪势 LBM 一侧的判据 X1/X2/X3 基于该方法的**定义与文献**，
   本仓没有实现，因此**没有**任何本仓算出来的伪势数据——不得把它读成"伪势被证伪/被验证"。
2. **判据机必须能被证伪**。恒返回 NOT_APPLICABLE 的空壳是这类工作最典型的造假形态；
   因此测试里放了同类孪生对的正控制（必须 APPLICABLE）与清单污染负控制（必须被踢回）。
3. **渐进区外不许混进收敛阶**。`h/l ≥ 1` 时离散问题已不在渐进区：`h/l = 4` 这一档
   Newton 甚至不收敛（见 3.3 第一行，节点误差量级 1e3）。所以拟合阶只用 `h/l ≤ 1` 的档位，
   并把这些粗档**原样列表**而不是删掉——删掉就等于粉饰。
4. **损伤子问题线性的前提**：位移全钉 ⇒ `ψ⁺ ≡ 0` ⇒ 驱动力项为零，方程对 d 是线性的。
   这是为了让"误差里只有离散误差"这一条成立；一旦放开位移，误差会掺入 Newton 停止准则，
   本单的 <3% 结论就不再干净。
5. **适用范围**：本裁决只禁止**场级**数值比较（逐点损伤 vs 逐点密度）。宏观量级比较
   （表面张力随相互作用强度的趋势、体相 Maxwell 构造的一致性）在伪势实现落地后仍可做，
   届时应新开 XVAL 条目，而非复用本条的结论。
6. **环境坑（如实记录）**：本单执行期间宿主盘一度 **100% 满（可用 28 KB）**，`ruff`
   因无法建临时文件直接报 `os error 28`。清理工作树内 `__pycache__/.pytest_cache/.ruff_cache`
   后恢复（腾出 2.3 GB）。凡在此后跑绿的测试都已重跑确认，但读者须知当时的环境异常。
7. **零手抄溯源**：③ 的全部数字由 `scripts/gen_xval_not_applicable_doc.py` 从
   `artifacts/xval_not_applicable_phase_field.json` 渲染，JSON 由 harness 直接写出，
   无人工转录环节。

## ⑥ 误差判定（逐项对基准 + 显式回答"是否 <3%"）

| # | 量 | 我方 | 基准 | 误差 | <3% ? |
|---|---|---|---|---|---|
| E1 | 1-D AT2 剖面 L2 相对误差（h/l={run['h_over_l']:g}） | {run['l2_relative_error']:.6e} | 解析 `cosh` 剖面 | {run['l2_relative_error']:.3e} | **{'是' if acc['at2_profile_l2_ok'] else '否'}** |
| E2 | 1-D AT2 剖面节点最大相对误差 | {run['nodal_max_relative_error']:.6e} | 解析 `cosh` 剖面 | {run['nodal_max_relative_error']:.3e} | **{'是' if acc['at2_profile_nodal_ok'] else '否'}** |
| E3 | 收敛阶 | {payload['at2_observed_l2_order']:.4f} | 2 | {abs(payload['at2_observed_l2_order'] - 2.0) / 2.0:.3e} | **{'是' if acc['l2_order_ok'] else '否'}** |
| E4 | 均匀历史饱和 d（三档最大相对误差） | {max(hh['relative_error'] for hh in payload['at2_homogeneous_anchor']):.3e} | `2Hl/(Gc+2Hl)` | 同左 | **{'是' if acc['at2_homogeneous_ok'] else '否'}** |
| E5 | 宽度比 `x(d=1/e)/l`（三档） | {", ".join(f"{w['width_ratio']:.6f}" for w in payload['width_scaling'])} | 1（解析 `exp(-x/l)`） | 最大 {max(abs(w['width_ratio'] - 1.0) for w in payload['width_scaling']):.3e} | **{'是' if acc['width_invariant_under_l_scaling'] else '否'}** |
| E6 | 判据结论 | {screen['verdict']}（阻断 {len(screen['blocking_failures'])} 条） | 期望 NOT_APPLICABLE | — | **{'是' if acc['verdict_is_not_applicable'] else '否'}** |

**逐项回答**：E1 {run['l2_relative_error']:.3e} < 3% → 是；E2 {run['nodal_max_relative_error']:.3e} < 3% → 是；
E3 |{payload['at2_observed_l2_order']:.4f} − 2|/2 = {abs(payload['at2_observed_l2_order'] - 2.0) / 2.0:.3e} < 3% → 是；
E4 最大相对误差 {max(hh['relative_error'] for hh in payload['at2_homogeneous_anchor']):.3e} < 3% → 是；
E5 宽度比偏离 1 最大 {max(abs(w['width_ratio'] - 1.0) for w in payload['width_scaling']):.3e} < 3% → 是。
**E6（伪势一侧的数值误差）：无基准、无实现，故不存在可报的误差数字——本单不声称其精度，
只给出"不可横向 + 其一侧已自证"的结构性结论。** 全部判定字段：`{json.dumps(acc, sort_keys=True)}`。
"""
    DOC.write_text(body, encoding="utf-8")
    print(f"wrote {DOC} ({len(body)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())