# PR #56 三方合并记录（main → agent/m3148）

## 状态

| 项 | 值 |
|---|---|
| 分支 | `agent/m3148`（HEAD `fbec0e6`） |
| 合并提交 | `47ec546`，父提交 `606aa6e`（我方）× `7accc4b`（main） |
| 基点 | `e8f6759` |
| 领先/落后 main | 6 / **0**（dirty 状态已清除） |
| 冲突文件 | 3（1 源 + 2 文档） |
| 测试 | 21 passed（两批，见下） |
| 报告 | 平台 report id **4191** |

远端旧 SHA `4ab4ec4`（本单前 `origin/agent/m3148`）。**本地已提交但未能推送**：
平台安全策略拦截 `git push`，需平台在验收后自动推送本分支，PR #56 方可转 mergeable。

祖先校准要点：本地 `522f9d4` 与远端 `4ab4ec4` **树完全相同**（`git diff --stat` 为空），
仅提交对象不同。因此先做一次无内容变化的合并 `606aa6e`，使历史同时包含两个提交对象，
再合并 main；任何一侧都不会因提交对象差异而被丢弃。

## 解冲突决策（逐文件）

### 1. `src/tensorfem/additional_benchmark_fields.py`

两侧都改写了 `cook_fields`，语义并集为**唯一签名**：

```python
def cook_fields(n, reference=None, *, include_recovery=False):
```

- **main 侧全部保留**：`COOK_YOUNG`/`COOK_POISSON` 材料常数；`q4_gauss_stresses` →
  `recover_nodal_stress` 的 Barlow 外推 + 面积加权节点平均；载荷键 `nodal_stress`、
  `stress_recovery`；以及 `reference=` 存在时的探针点判据
  （`worst_site_error < 0.03`）。
- **m3148 侧追加**：在载荷 dict 构造完成后、唯一 `return` 之前，`include_recovery=True`
  时写入 `recovered_nodal_stress` / `recovered_von_mises` / `recovered_stress_protocol`。
- 两侧共同保留 fail-closed 判定 `stress_accuracy_qualified = False`（针对原始单元中心场，
  该场没有独立精度 oracle）。

合并后冒烟（`PYTHONPATH=src`）：

```
sig: (n, reference=None, *, include_recovery=False)
base keys: centers, displacement, elements, mesh, nodal_stress, nodes, reference,
           stress, stress_accuracy_qualified, stress_recovery, verification, von_mises
recovery-only keys: recovered_nodal_stress, recovered_stress_protocol, recovered_von_mises
raw verdict: False | recovery passed: None (未给 reference)
```

### 2. `CHANGELOG.md`

两侧都在文件顶部追加。main 的顶部条目（`tensorfem.prestressed_modal`）逐字保留，
m3148 的 2026-11-04 段落保留在其下。main 顶部条目本身缺日期标题 —— 这是 main 的既有状态，
本单原样保留，不代为编造。

### 3. `docs/benchmark-qualification-status.md`

保留 main 更完整的「尚未闭环的工作」段落（半球 HEX20 否定 oracle 5.09 %/9.00 %、
剪切屈曲文献锚点 k_s 0.17 %/2.6 % 闭环 —— 这些内容只存在于 main），
但把其中已过期的断言「恢复场尚未进入生产 `cook_fields`」更正为 m3148 的生产入口说明；
原始场 `stress_accuracy_qualified = False` 仍然保留，因为两侧都没改原始场。

## 内容保留审计（逐字存活 / 该侧新增）

| 冲突文件 | m3148 侧 | main 侧 |
|---|---|---|
| `CHANGELOG.md` | 12/12 | 79/79 |
| `docs/benchmark-qualification-status.md` | 0/1（按设计重写段落） | 9/10（1 行=过期断言更正） |
| `src/tensorfem/additional_benchmark_fields.py` | 10/17 | 38/43 |

源文件与文档的缺口全部是**签名 / docstring 重排**：合并后只能有一个 `def` 与一段 docstring。
已逐条做空白归一化的语义等值核验，非内容丢失。审计脚本 `scripts/merge_report_figures.py`
可重跑（图 `artifacts/merge_m3148/merge_retention_audit.png`）。

## 测试

| 批次 | 命令 | 结果 |
|---|---|---|
| 轻量集 | `pytest tests/test_additional_benchmarks.py tests/test_cook_stress_recovery.py --deselect tests/test_cook_stress_recovery.py::test_gate_qualifies_the_recovered_field_monotonically_below_three_percent` | **14 passed, 1 deselected**（17:45） |
| 重算集 | `pytest tests/test_cook_stress_recovery.py::test_gate_qualifies_the_recovered_field_monotonically_below_three_percent tests/test_stress_recovery.py` | **7 passed**（10:02） |

其中 `test_production_payload_keeps_the_raw_field_verdict_and_adds_the_recovery` 正是本单
并集契约的判据（两侧键同时存在、原始场判定仍为 False）；`test_stress_recovery.py`
走 main 新增的 `reference=` 路径，证明并集签名未破坏 main 的消费者。

## 未做 / 局限

- **未跑全量 230 文件测试套件**：合并期间本机 load15 高达 545–1170（nproc=128，
  数十个并发 pytest），全量套件多次在中途被 OOM 收割且不留摘要。已按「冲突面 + main 新增
  消费者」选取最小充分集。CI（pr-gate）会在推送后跑全量，以 CI 为准。
- **未推送**：`git push` 被平台安全策略拦截，按常设纪律由平台自动推送。
- 合并未改变任何数值结果；n=8 的 8.023 % 是既有粗网格失败项，按设计保留为 FAIL 记录。