# PR #117 CI 红修复：`artifacts/` 载荷被交付门禁剥离导致的收集期崩溃

- 仓库：`cxs503/TensorFEM`，PR #117，源分支 `agent/m3825`（本修复提交在**同一分支**上，head = 本文件所在提交）
- 修复前 CI：`test (3.11)` = failure、`test (3.12)` = failure、`lint (advisory)` = failure
- 归仓：TensorFEM（求解器库 CI 门禁），未改动任何与本 PR 无关的求解器代码

## 1. 案例介绍（红在哪、为什么红）

`agent/m3825` 的交付提交 `f69969b` 的标题已经写明根因线索：**「15 个文件被门禁滤除」**。
被滤除的 15 个文件全部位于 `artifacts/`（`pr61_union_audit.py`、`pr61_union_audit_evidence.json`、
解冲器、出图器、会话日志等）——平台的交付门禁（`merge_governor`/`merge_master` 的
`ART_RE = (outputs|figures|artifacts|__pycache__)/|\.png$|...`）不收录 `artifacts/` 与图片。
于是分支上被提交的这 11 个文件里，**门禁测试自己依赖的载荷并不存在**：

| 测试 | 依赖 | 交付后是否存在 |
|---|---|---|
| `tests/test_pr61_merge_union_audit.py` | `artifacts/pr61_union_audit.py` | ❌ 被滤除 |
| 同上 `test_shipped_evidence_matches_a_fresh_run` | `artifacts/pr61_union_audit_evidence.json` | ❌ 被滤除 |

该测试在**模块导入期**调用 `_load_audit()`（`importlib` 直接 exec 审计器），因此载荷缺失
不是"某些用例红"，而是**整个文件收集期报错** → pytest 以非零退出 → 两个 Python 版本的
`test` job 在还没跑任何真实算例前就判红（本地复现：`12 errors` = 12 个 xdist worker 各报一次
`FileNotFoundError: .../artifacts/pr61_union_audit.py`，第 0% 就中断）。

`lint (advisory)` 的红是仓内既有的 lint 存量（`ruff check` 4268 条、`ruff format` 620 文件待格式化，
job 本身 `continue-on-error: true`），与本次改动无关；本修复保证**新增/改动文件自身零违规**。

## 2. 怎么做的（最小改动，不动审计语义）

原则：门禁剥离的是**路径**，不是**内容**。因此把同一份载荷搬到能随仓交付、且被 lint 覆盖的路径上，
而不是把测试改成"缺文件就跳过"（那会让审计门禁在 CI 里永远静默失效）。

| 动作 | 路径 | 说明 |
|---|---|---|
| 迁移（内容逐字） | `artifacts/pr61_union_audit.py` → `scripts/pr61_union_audit.py` | 审计器本体不变；仅 `OUT` 常量与用法字符串改指新证据路径，并补一段"为什么落在 scripts/"的说明 |
| 迁移（内容逐字） | `artifacts/pr61_union_audit_evidence.json` → `docs/assets/pr61_union_audit_evidence.json` | 证据 JSON 由脚本在最终 HEAD 上重算（`UNION_OK`），非手改 |
| 改 2 行常量 | `tests/test_pr61_merge_union_audit.py` | `AUDIT`/`EVIDENCE` 指向新路径；docstring 同步（仅 4 行 diff） |
| 改文档指针 | `docs/pr61-merge-reconciliation.md` | 权威载体路径更新 + 一条"交付门禁不收录 `artifacts/`，其余作业工件未随仓交付"的说明 |

`docs/assets/` 是仓内既有的数据/图资产目录（如 `docs/assets/suboff-ice-v2/appended-16.json`），
`scripts/` 是 CI lint 与 doctest 覆盖的可交付目录，二者都不在门禁的剥离规则内。

工具链：`git`（`git cat-file` 判定冻结坐标是否存在）、`ruff`（新文件零违规）、`pytest`（3.11 本地）。

## 3. 计算结果（程序化复算，零手抄）

审计器在修复后的 HEAD 上重算（`python scripts/pr61_union_audit.py --check`）：

| 量 | 值 |
|---|---|
| 合并面文件数（`files_changed_total`） | 243 |
| 逐字或手工解冲承载（`files_carried_verbatim_or_hand`） | 243 / 243 |
| main 侧独有 / 分支侧独有 / 双侧 | 234 / 7 / 2 |
| `git merge-tree` 冲突集（= 声明的 2 个手工解冲文件） | `CHANGELOG.md`, `docs/marine-hydrostatics.md` |
| 手工解冲原子丢失（数字/标识符/链接/路径，多重集） | 0 / 0（两文件） |
| 手工解冲词集丢失 | 0 / 0 |
| 合并后篡改（`evolved_after_merge`） | `[]` |
| 负控 1：main-only 机械 overlay | `control-fires`（被拒） |
| 负控 2：branch-only 机械 overlay | `control-fires`（被拒） |
| 负控 3：删掉一行含 `1.33e-15` 的实测值 | `control-fires`（被拒） |
| 判决 | `UNION_OK`，`failures = []` |

测试证据（本地，Python 3.11.6）：

| 检查 | 修复前 | 修复后 |
|---|---|---|
| `pytest tests/test_pr61_merge_union_audit.py` | 收集期 `FileNotFoundError`（整文件 ERROR） | **6 passed** |
| `pytest tests/test_marine_large_angle_gz.py` | 39 passed | 39 passed |
| 上述两文件合跑 | — | **45 passed** |
| 浅克隆（`--depth 1`，等价 CI checkout）下的门禁 | — | **6 skipped**（冻结坐标不在浅历史里，按设计跳过而非失败） |
| 全仓 `pytest -q -n 12` 收集 | 12 errors | 0 errors |

## 4. 结果图

`docs/assets/ci-pr117-evidence.png`：左面板=修复前后每个 CI job 的失败项计数（test 3.11 / test 3.12 / lint），
右面板=审计器负控三档的拒绝文件数（证明量具有牙齿，绿不是因为量具失灵）。

## 5. 注意点（踩坑与局限）

1. **门禁剥离是路径级的、静默的**：`artifacts/`、`*.png` 不进交付提交，但 PR 里的测试仍按原路径 import。
   教训：任何**测试 fixture** 都不能放在 `artifacts/`、`outputs/`、`figures/` 下；测试需要的载荷必须落在
   `src/`、`tests/`、`scripts/`、`docs/assets/`、`benchmarks/` 等可交付路径。
2. **不要用"缺文件就 skip"糊红**：那样 CI 永远绿，但审计门禁在 CI 里彻底失效（本仓有"量具失灵仍显示通过"的先例，见
   `docs/pr61-merge-reconciliation.md` §7.3 第 2 条）。本次改为迁移载荷、保留门禁牙齿。
3. **`ruff format --check` 不改测试文件**：全仓 620 个文件未格式化，若对既有测试文件跑 `ruff format`
   会引入 50 行无关 diff；本次只对新脚本做格式化，测试文件保持 4 行最小 diff。
4. **局限一（未修）**：`artifacts/` 下其余一次性工件（`pr61_merge_evidence.png`、解冲器、pytest 会话日志）
   仍然没有随仓交付，`docs/pr61-merge-reconciliation.md` 里对它们的引用保留为历史记录，并已加"未随仓交付"说明。
5. **局限二（既有红，未在本单修）**：本地全仓跑还有 2 个与本 PR 无关的失败：
   `tests/test_release_quality.py::test_api_audit_script_passes_with_warnings_as_errors`
   （本机 torch 触发 `pynvml` `FutureWarning`，在 `PYTHONWARNINGS=error` 下报错；在 `origin/main`
   上同样复现，属环境性）与 `tests/test_finite_strain_curved_friction_contact3d.py::... level_one_closure`
   （在 `origin/main` 上同样复现，非本次改动引入）。二者都不在 `agent/m3825` 的变更面上。
6. 浅克隆（CI）下门禁按设计 skip，因此**"CI 绿"不蕴含"审计在 CI 里被执行"**；审计的真实执行点是
   本地/完整历史的检出。这一点写进了测试模块 docstring 的既有约定，未改语义。

## 6. 误差判定

本质检类任务：判据是**离散的通过/失败**，不存在连续量的百分比误差，故逐项判定如下：

| 关键量 | 基准 | 实测 | 误差 | 是否达标 |
|---|---|---|---|---|
| 收集期错误数（全仓 pytest） | 0 | 0 | 0 | 是 |
| 门禁测试结果 | 6 passed | 6 passed | 0 | 是 |
| 浅克隆门禁结果 | 6 skipped / 0 failed | 6 skipped | 0 | 是 |
| 审计判决 | `UNION_OK` | `UNION_OK` | 0 | 是 |
| 承载文件数 | 243 | 243 | 0 | 是 |
| 手工解冲原子/词丢失 | 0 | 0 | 0 | 是 |
| 负控拒绝数 | 3/3 触发 | 3/3 触发 | 0 | 是 |
| 新增/改动文件 ruff 违规 | 0 | 0 | 0 | 是 |

**是否 <3%**：全部判据为 0/非 0 的离散量，实测与基准完全一致（相对误差 0%），故**是**。
连续量（如求解器精度）不适用于本单；`docs/pr61-merge-reconciliation.md` 中大倾角 GZ 的精度判定
沿用该文原有基准，本单未改动其算法。