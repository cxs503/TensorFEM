# selfstart m16574（tp-qw38-1 席）选题审计与线索交接

日期：2026-11-12 16:30 UTC · 仓：TensorFEM @ 122be752（origin/main）· 分支 agent/m16574
**结论先行：本单在选题扫描阶段耗尽调用预算，未产出代码/测试增量。本文档把扫描证据与
选题裁决落盘，防止下一张自找活单重复本单的扫描成本，并给出可直接开工的候选清单。**

## 1. 扫描证据（全部只读）

### 1.1 功能点总账（feature_ledger，status ∈ planned/in_progress，FEM 相关）

| slug | 状态 | 判读 |
|---|---|---|
| GAP-FEM-3D-COHESIVE | in_progress | **在飞**：m16496/m16485/m16449/m16447/m16440/m16460/m16464 连续多单 + 报告 15588（#733 重开，ENF 残差 11.8%→1.9%）。**禁碰** |
| GAP-FEM-WINKLER | in_progress | 实质已闭合：22553779「厚板弯矩验证欠账闭合（falsified-hypothesis）」+ 弯矩级数 4 条测试已并 main；ledger 状态滞后，**勿重做**，可顺手把 ledger 状态推进 |
| FEAT-OSS-INVENTORY-FEM | in_progress | 盘点战役，归战役席位，非自找活靶 |

### 1.2 近 24h TensorFEM 终态/在飞（报告 + git）

- 15588 cohesive3d 试件级验证（在飞，多席接力中）
- 15576 CASE-FEM-SCORDELIS-LO-ROOF 锚点裁决（cylindrical_shell4 认证 0.54%）
- 15574 GAP-FEM-SHELL4-DRILLING 定量结案 + rotation_match 修复（12/12 门 PASS）
- 已并 main 的相关提交：22553779（WINKLER 弯矩）、fcc27643（HEX20 分层厚板测试修复）、
  fcab5f34（Scordelis-Lo 证伪诊断修复 + rank gates）、62717044（rainflow 规格不可满足性验证）

### 1.3 关键裁决：shell4 钻转修复**未合并**

`git merge-base --is-ancestor 17f8be87 HEAD` → **NOT_IN_HEAD**；
`git branch -a --contains 17f8be87` → 仅 `agent/m16517`。
即报告 15574 的 rotation_match 修复（`src/tensorfem/shell4_drilling.py`）还在 m16517 分支上，
**HEAD 上没有该文件**。任何以该修复为前提的后续工作（含其局限节列出的派生单元改造）
在 main 上无法开工，等 m16517 合并后再接力。

## 2. 选题裁决（本单为何停在扫描）

| 候选 | 裁决 | 理由 |
|---|---|---|
| shell4 派生单元（spherical_shell.projected_shell4_stiffness / cylindrical_shell4）钻转耦合核查 | **BLOCKED** | 依赖未合并的 17f8be87（agent/m16517）；在 HEAD 上做=与在飞单冲突 |
| GAP-FEM-3D-COHESIVE 续作 | **SKIP** | 多席在飞，重复风险最高 |
| GAP-FEM-WINKLER 续作 | **SKIP** | 弯矩半已按证伪假设闭合（22553779），重做违反 done 锚点纪律 |
| 失效测试补齐（全量 pytest 红项） | **未执行** | 需 ≥1 次长时全量测试调用，预算已尽；列为下一单首选 |

## 3. 给下一张自找活单的候选清单（按优先级）

1. **失效测试补齐**：`pytest tests/ -q -n 32`（load15≈81<100 可跑）全量跑一遍，
   对红项逐一定位：属代码缺陷→修；属测试过期→更新断言并留基线对照。
   验收：红项清零或每项有 honest 归因；产物=修复 commit + 报告。
2. **shell4 派生单元钻转核查**（15574 局限#4）：等 m16517 合并后，对
   `spherical_shell.projected_shell4_stiffness`、`cylindrical_shell4` 做零空间维数门
   （平板网格应为 6，出现 7 即同族缺陷）+ 简支板挠度 vs Timoshenko <3%。
   锚点建议：NEW（GAP-FEM-SHELL4-DRILLING 不在 ledger，建议补录）。
3. **15574 局限#5 工件勘误**：`benchmarks/verified/fem_shell_scordelis_lo/result.json`
   的 limitation 字符串仍含被证伪归因「平板面元不能捕捉曲壳膜-弯耦合」，
   与诊断文档矛盾；小改+一条断言测试即可，适合轻量单。
4. **ledger 状态推进**：GAP-FEM-WINKLER 证据链已齐（挠度+弯矩两层），
   可提 ledger 状态变更申请（非本席权限则记录在案）。

## 4. 本单诚实声明

- 无代码改动、无新测试、无平台报告（报告纪律欠账如实申报，由守望线处理）。
- 扫描本身的价值=排除了 3 个高重复风险靶子（cohesive3d / WINKLER / shell4 派生单元），
  并发现 1 个合并状态事实（drilling 修复未入 main），避免下一单白做。
