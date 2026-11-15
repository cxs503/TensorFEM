# CASE-FEM-HEAT-L2 — 验证欠账闭环：机读基准行注册（m21445）

**锚点** CASE-FEM-HEAT-L2 ｜ **性质** 验证欠账闭环（verification-debt closure）｜
**判据** 收敛率 ≥ 2 阶（ref: manufactured solution）

## 一句话

锚点 `CASE-FEM-HEAT-L2`（稳态热传导 Q4 制造解 L2 收敛率 ≥2 阶）的**实质能力已由
m20054 交付并获 L4 板审**（commit `8867c2a4`，verdict id=828，31 闸门全 PASS，
finest-pair L2 阶 1.999165 ∈ [1.95, 2.10]）。本单不重做该能力，而是**闭环板审 m20108
标记的 F1 元数据缺口**：`tensor_benchmark_cases` 对该锚此前 **0 行**，判据仅为
`tensor_case_pool.tol_policy` 自由文本，导致"got 是否在容差内"无法机读裁定。

本单把**机读基准行**注册进 `tensor_benchmark_cases`（id=148，status=passed），
并把 `feature_ledger` 状态从 `open` 更新为 `verified`——锚点从此 L4 机读可裁定，
不再依赖人工按文本判读。

## 1. 为什么这是"验证欠账"而非"重做"

| 判据 | 现状（本单核对时） | 结论 |
|---|---|---|
| 实质能力 | m20054 已交付（`src/tensorfem/heat_l2_convergence.py`，6 制造解 + 真积分误差范数 + 31 预登记闸门），L4 板审 confirmed | **不重做**（任务规则：done/verified 不得重做） |
| 机读基准 | `tensor_benchmark_cases` 对 `CASE-FEM-HEAT-L2` **0 行**（板审 m20108 F1 明确标记） | **欠账**：补一行机读基准 |
| 账本状态 | `feature_ledger.status='open'`（与已完成的实质能力不一致） | **欠账**：更新为 `verified` |

## 2. 做了什么（最小、可验证）

1. **注册机读基准行**（`benchmarks/register_heat_l2_benchmark_case.py`，数值全部从
   commit `8867c2a4` 的 `artifacts/CASE-FEM-HEAT-L2/evidence.json` 程序化读取，零手抄）：

   | 列 | 值 |
   |---|---|
   | anchor_slug | `CASE-FEM-HEAT-L2` |
   | case_name | 稳态热传导 Q4 制造解（sine_generation MMS）L2 收敛率 |
   | ref_source | MMS（Salari & Knupp SAND2000-1009J）；目标阶 p_ref=2.0 由仓内版本化先验表 `src/tensorfem/adaptive_poisson_2d.py:34` 程序化解析 |
   | metric | finest-pair L2 order of ‖e‖₀ on sine_generation MMS (32→64) |
   | tolerance | [1.95, 2.10]（板审 m20108 建议原文） |
   | repo | TensorFEM |
   | status | passed |

   **got**（从 evidence 读取）：`sine_generation` 最细对 L2 阶 = **1.999165**，
   落在 [1.95, 2.10] 内 ⇒ 机读裁定 **PASS**。

2. **账本状态更新**：`feature_ledger` 该锚 `open → verified`（附证据指针：
   L4 板审 m20108 verdict id=828 + 基准行 id=148 + 31 闸门 PASS）。

## 3. 验证

- 注册脚本 `--dry-run` 输出 SQL 与预期一致；实跑 `INSERT 0 1`，随后
  `SELECT ... WHERE anchor_slug='CASE-FEM-HEAT-L2'` 返回 **id=148 / passed**。
- 账本 `UPDATE ... RETURNING` 返回 `verified`。
- 数值来源：`git show 8867c2a4:.../evidence.json` 的 `sine_generation.l2_orders[-1]`
  = 1.9991651663236134（与板审 m20108 独立复算一致）。

## 4. 未做的检查（如实声明）

- 未重跑 65×65 稠密装配（本机 load15≈62，且该能力已由 m20108 字节级复跑
  37.7s 确认；本单只做轻量元数据闭环）。
- 未覆盖非均匀/扭曲网格、张量热导 k(x)、三维单元、瞬态项——这些是
  m20054 已披露的锚点强度局限，不属本单范围。

## 5. 给下次自找活的建议新条目

**建议新条目**（slug+量化验收+所属域）：
- `CASE-FEM-HEAT-L2-TRANSIENT`（TensorFEM/热传导）：瞬态热传导（Crank-Nicolson /
  隐式欧拉）时间步收敛率 ≥2 阶，ref=制造解 `T=e^{-t}sin(πx)sin(πy)`，容差
  [1.95, 2.10]——把本锚的稳态 L2 能力扩展到瞬态，复用同一套真积分误差范数
  与机读基准行模式。
- `CASE-FEM-HEAT-L2-ANISO`（TensorFEM/热传导）：张量热导 k(x) 各向异性稳态
  热传导 L2 收敛率 ≥2 阶——覆盖 m20054 明确未做的 k(x) 项。
