# m17064 交差核验审计报告：`tensorfem.inverse`（GAP-PP-4-opt-loop）

- 审计单：m17084（board-audit）｜被审单：m17064（TensorFEM，status=succeeded）
- 被审交付：分支 `agent/m17064`，commit `378b368d57bfb2fa6df2e20d7f2c6b97f6be7aca`（base `442d393a`）
- 结构化核对档：`artifacts/audit_m17064/verify.json`（本报告所有数字由脚本从工件读出，零手抄）
- 落表：`tensor_maturity_verdicts` id=481，`TensorFEM / m17064 GAP-PP-4-opt-loop / level=2 / board-audit`

## ① 案例介绍（审计对象与判据）

被审交付是一套反问题优化基础设施 `tensorfem.inverse`：Levenberg-Marquardt 循环（有限差分
Jacobian、严格下降接受、回溯最速下降保底）、可组合损失（学生化数据残差 + Tikhonov 正则块）、
多起点种子控制、可辨识性诊断（加权 Jacobian 的 rank / effective_rank / 零空间弱方向 / 线性化
标准误 / 相关系数）、机读收敛日志（`convergence.jsonl` + `summary.json`，带 sha256 锁定）。

三个算例（真值均为脚本自设）：

| 算例 | 前向模型 | 观测量 | 未知数 | 自设真值 | 噪声 |
|---|---|---|---|---|---|
| 1 exponential-decay | 闭式 `Tinf + A·exp(-t/tau)` | T(t), 41 点 | A, tau, Tinf | tau=4.0 s | 1 % |
| 2 axial-bar | 真实 `TrussModel` + `solve_linear_static`（4 单元） | 4 节点位移 | E | 210 GPa | 0.5 % |
| 3 axial-bar-product | 同上，但 E 与 A 同时未知 | 同上 | E, A | E=210 GPa, A=1e-3 m² | 同上 |

判据（任务书规定「基准判据是唯一裁判」）：

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='GAP-PP-4-opt-loop';      -- 实测：0 行
SELECT count(*) FROM tensorbench_results WHERE mission_id=17064;  -- 实测：0
GET /api/v1/tensor/bench/l4-verdict/17064   -- CLAIMED_NOT_PROVEN, rows=0
```

## ② 怎么做的（审计方法）

1. **取证**：读 m17064 工作树与 `tensor_reports` 报告；逐项核对 impl/entry/run_artifacts/tests/repro_cmd。
2. **哈希自洽**：独立重算 5 份 `convergence.jsonl` 的 sha256，与各自 `summary.json` 的
   `convergence_log_sha256` 比对 → **5/5 匹配**。
3. **独立复现**（关键）：在 `/tmp/aud` 建**干净树**＝parent HEAD `442d393a` 的 `src/tensorfem`
   ＋**只加交付的 4 个新文件**（不借 m17064 的任何其它改动），`PYTHONPATH=/tmp/aud/src`：
   - `pytest tests/test_inverse_parameter_identification.py -q` → **10 passed in 8.09 s**（`pytest_independent.log`）
   - `python examples/inverse_parameter_identification.py` → exit 0，重生成 5 算例工件
   - 逐算例比对重跑与交付工件的参数 → **5/5 最大相对差 0.0（逐位一致）**
4. **对抗性探针**（adversarial-review）：
   - 交付自称「2 条红灯为 HEAD 既有漂移」→ 在 base `442d393a`（无任何新文件）跑
     `tests/test_public_api_contract.py`：`test_public_api_snapshot_is_exact_and_packaged_copy_matches`
     确实 FAIL（1 failed / 3 passed / 35.68 s）→ **说法成立**；
   - 「hash-pinned」口径：跨树重跑日志 sha256 不同，差异**仅**在 `header.provenance.git_commit`（同树内字节稳定）；
   - 「3 种子」口径：`decay_problem()` 数据种子固定=7，`SeedControl(seed=1/2/3)` 只变多起点初值 → 三 τ 差异 <3e-8；
   - 集成面：全仓 grep 反向调用方 0 处；`src/tensorfem/__init__.py` `__all__` 未导出；`api/public-api-v1.json` 无 `inverse` 条目。

## ③ 计算结果（数字表，均由 `verify.json` 程序化注入）

| 检查项 | 交付声明 | 实测 | 结论 |
|---|---|---|---|
| run_artifacts 存在 | 12 文件 | 12 文件齐（5×convergence.jsonl+summary.json+metrics.json+png） | ✅ |
| 日志哈希自洽 | 5/5 | 5/5（重算 sha256 全等） | ✅ |
| τ 反演误差 | 0.848 %（3 种子均 <3 %） | 0.8482 / 0.8482 / 0.8482 % | ✅ |
| E 反演误差 | 0.311 % | 0.31115 %（2.1065e11 vs 2.10e11 Pa） | ✅ |
| E·A 不可辨识如实报 | effective_rank=1 | rank=2, effective_rank=1, cond=4.453e11, 弱方向 (0.70710, −0.70711) | ✅ |
| 独立复现（干净 parent-HEAD 树） | — | 10/10 测试；5/5 参数逐位一致 | ✅ |
| 平台基准裁判行 | — | `tensor_benchmark_cases` **0 行**、`tensorbench_results` **0 行** | ❌ 无裁判 |
| 公共 API / 主求解器调用方 | — | `__all__` 未导出、API 快照无、调用方 0 处、分支未进 main | ❌ 未集成 |

## ④ 结果图

![audit](../artifacts/audit_m17064/audit_figure.png)

- 左：三个起点的 τ 相对误差（0.848 %）与 3 % 验收门、审计独立重跑的复现值（与交付重合）。
- 右：E 与 E·A 的 ref vs got（0.311 % 偏差，条形高度差在图上不可见——这正是结论）。
- 交付自带图：`artifacts/audit_m17064/delivered_runs/inverse-identification.png`（重跑件 `rerun_figure.png` 与之一致）。

## ⑤ 注意点（发现与局限）

| # | 级别 | 发现 | 处置建议 |
|---|---|---|---|
| F1 | 阻断 L4 | anchor `GAP-PP-4-opt-loop` 在 `tensor_benchmark_cases` **0 行**、`tensorbench_results` 0 行：无机器裁判行，`l4-verdict` API 返回 CLAIMED_NOT_PROVEN | 注册案例行（metric=tau/E 相对误差，tolerance=3 %）+ 集群重算落库 |
| F2 | 阻断 L3 | `tensorfem.inverse` 未进 `__all__`/API 快照，全仓 0 调用方，分支 `agent/m17064` 未进 `origin/main` | 导出 + API 快照 + 至少一个生产调用方（如 CLI/benchmarks 入口） |
| F3 | 提示 | 「3 种子」仅变多起点初值（数据种子固定=7），三 τ 一致到 1e-8 → 属优化器鲁棒性，非噪声实现鲁棒性；报告 §3 已披露，未夸大但易被误读 | 补噪声种子扫描（不同数据实现） |
| F4 | 提示 | `convergence_log_sha256` 含 `header.provenance.git_commit` → 跨树/跨 commit 不稳定 | 哈希口径写清「同树/同 commit」 |
| F5 | 提示 | 报告称 `inverse.py`「约 700 行」，实测 1006 行（低报） | 更正行数 |
| F6 | 已证清白 | 交付登记的 2 条红灯确为 HEAD 既有漂移（base commit 无新文件亦 FAIL） | 另单修 API 快照漂移 |

未跑：loong64 集群重算、GitHub Actions x86 CI、外部解析解复核（本案参照值自合成，无外部源）。

## ⑥ 误差判定（vs 判据逐项，显式回答）

| 关键量 | 基准（ref 来源） | got | 相对误差 | <3 %？ |
|---|---|---|---|---|
| tau | 自合成真值 4.0 s | 4.033929 s | **0.8482 %** | **是**（但与外部基准无关，属自洽） |
| E | 自合成真值 210 GPa（真实 TrussModel 前向） | 2.1065e11 Pa | **0.3112 %** | **是**（同上） |
| E·A | 自合成真值 2.10e8 | 2.1065e8 | **0.3112 %** | **是**（同上） |
| 「平台基准裁判」误差 | **不存在**（0 行注册） | — | 不可计算 | **否——无基准，L4 不成立** |

对照实验/参考值误差 >15 % 的不可接受线：本案自洽误差全部 <1 %，**不触发**；但按「无外部锚点即
不得称基准验证」的纪律，只能记 **L4 未达**，不得以「误差 0.3 %」反推 L4。

## 裁定

**L2**。逐级：L0 文档✅；L1 数学/单元验证✅（fd Jacobian 对解析 Jacobian 1.2e-6<5e-6、
central<1e-9；弱方向 ±0.7071 与理论一致；loss 单调；chi2/边界语义）；L2 独立真实现✅（1006 行
模块，干净 parent-HEAD 树 10/10 绿、5 算例端到端、参数逐位可复现）；L3❌（未集成：无 __all__/
API 快照/调用方，未进 main）；L4❌（`tensor_benchmark_cases` 0 行、`tensorbench_results` 0 行、
l4-verdict=CLAIMED_NOT_PROVEN，参照值自合成无外部锚点）。

交付质量本身是**诚实**的：10/10 测试断言了不变量而非仅执行代码，不可辨识性被如实上报而非
假装收敛，报告 §5 主动登记了 6 条坑与既有红灯。扣级原因不是造假，而是**交付形态**停在
库内模块：无平台基准行、无集成面。补 F1/F2 即可升 L4。