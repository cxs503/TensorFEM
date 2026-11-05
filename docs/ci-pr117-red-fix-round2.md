# PR #117 CI 红修复（第 2 轮）：修复内容已在分支上，红只因网关交付面缺载荷

- 仓库：`cxs503/TensorFEM`，PR #117，源分支 `agent/m3825`
- 第 1 轮（m4099）判定的根因见 `docs/ci-pr117-red-fix.md`：**测试 fixture 依赖的载荷被交付网关按路径滤除**
  （`artifacts/` 属纯记录区），导致 pytest 在**收集期**就 `FileNotFoundError`，两个 Python 版本的
  `test` job 在跑任何算例前即判红。
- 第 2 轮的任务是"把红修掉并合进 main"。本轮工作 = **在 PR 头所在的提交上复现红、确认修复内容完整、
  用合并树（PR 头 ∪ main）重跑全量、并把可交付性逐文件核对**。

## 1. 案例介绍（红在哪、为什么第 1 轮之后仍然是红）

第 1 轮把载荷从 `artifacts/` 迁到可交付路径（`scripts/`、`docs/assets/`），但平台推送器把该成果推成了
`refs/heads/agent/m4099`（推送状态可见 `tp_pusher_state.json`：`agent/m4099 → 829392a80d62（15 文件）`），
而 **PR #117 的头仍是 `agent/m3825` 上缺载荷的那一版**（`tp_pusher_state.json`：
`agent/m3825 → f69969b216a0（11 文件）`，即 15 个改动文件里有 4 个被网关滤除）。于是 CI 复跑仍然是红。

本轮在 PR 头内容上复现（`git archive f69969b` 取树 + 只做收集，不跑算例）：

```text
$ python3 -m pytest -q --collect-only -p no:cacheprovider
ERROR tests/test_pr61_merge_union_audit.py - FileNotFoundError: [Errno 2] No ...
!!!!!! Interrupted: 1 error during collection !!!!!!
1731 tests collected, 1 error in 36.55s
```

即：**1 个文件在收集期报错 → pytest 退出码非 0 → `test (3.11)`/`test (3.12)` 双红**，与 CI 观测一致；
`lint (advisory)` 的红是仓内既有 ruff 存量（见 §5），与本 PR 无关。

## 2. 怎么做的（不改审计语义，只让载荷可交付）

本轮不新增算法改动，交付的是**第 1 轮已成熟的载荷迁移**加上**可交付性核对与合并树全量复验**：

| 项 | 内容 |
|---|---|
| 审计器载荷 | `scripts/pr61_union_audit.py`（原 `artifacts/pr61_union_audit.py`），内容逐字 |
| 证据载荷 | `docs/assets/pr61_union_audit_evidence.json`（原 `artifacts/…json`），由审计器在最终 HEAD 上重算 |
| 门禁测试 | `tests/test_pr61_merge_union_audit.py` 指向新路径（4 行 diff） |
| 交付面核对 | 按推送器的过滤规则（`^artifacts?/|^runs/|results[-/]`、`\.(db|sqlite3?|png|log)$`，`docs/` 下 png/svg 放行）逐条核过 15 个改动文件 |

15 个改动文件的交付判定：**15/15 全部落在可交付面**（`docs/assets/*.png` 走 `docs/` 放行分支），
无一个落在 `artifacts/`、`runs/`、根级 `*.png`、`*.log` 等记录区。这正是第 1 轮的迁移目标，也是本轮复核项。

另有一条硬约束（本轮实测）：`scripts/pr61_union_audit.py` 的 **post-merge hygiene** 判据规定
「合并提交之后只允许新增文件、不许改动已合并内容」。因此本轮**不对已合并文件做纯格式化**：
`ruff format` 会改写 `src/tensorfem/marine_large_angle_gz.py`、`tests/test_marine_large_angle_gz.py`、
`scripts/plot_large_angle_gz.py` 三个已合并文件，实测使审计判据直接给出

```text
FAIL commit(s) after the merge modified/deleted merged content:
     ['scripts/plot_large_angle_gz.py', 'src/tensorfem/marine_large_angle_gz.py', 'tests/test_marine_large_angle_gz.py']
UNION_FAIL
```

故本轮**回退了这 3 个文件的格式化**（工作树与 `HEAD` 一致），保留审计门禁的牙齿，不为了 lint（advisory、且
FEM 合并网关 `ignore_checks` 已显式忽略）去破坏一条会永久生效的完整性判据。

## 3. 计算结果（程序化，零手抄）

| 量 | 修复前（PR 头 f69969b） | 修复后（本提交） |
|---|---|---|
| 全仓 pytest 收集错误数 | 1（整文件收集期 ERROR） | **0**（1737 tests collected） |
| `tests/test_pr61_merge_union_audit.py` | 收集期 FileNotFoundError | **6 passed**（完整历史） |
| 同上（浅克隆 `--depth 1`，等价 CI checkout） | 收集期 FileNotFoundError | **6 skipped / 0 failed**（按设计跳过） |
| 合并树（本分支 ∪ `origin/main` c5168e4）全量 | 收集期中断 | 网格化跑到 99%，唯一失败 = `test_release_quality::test_api_audit_script_passes_with_warnings_as_errors`（`origin/main` 上同样失败，环境性；见 §5） |
| 分支自有测试 `tests/test_marine_large_angle_gz.py` | 39 passed | 39 passed |
| `scripts/pr61_union_audit.py --check` | 无法导入载荷 | **UNION_OK，failures = []** |
| 改动面 ruff（`ruff check` 15 个改动文件） | — | All checks passed |
| 合并冲突（`git merge-tree` vs `origin/main`） | — | 0 冲突，mergeable |

审计器在最终 HEAD 上的复算值：`files_changed_total = 243`、`files_carried_verbatim_or_hand = 243/243`、
`files_main_only = 234`、`files_branch_only = 7`、`files_both_sides = 2`、手工解冲（`CHANGELOG.md`、
`docs/marine-hydrostatics.md`）原子/词丢失 0/0、三条负控全部 `control-fires`。

## 4. 结果图

本单是 CI/交付面修复，**没有场量、没有求解器输出可供画云图**；因此配图为"红→绿"状态与交付面的程序化条形图，
不含任何示意性伪场量：`docs/assets/ci-pr117-round2-evidence.png`（左：三个 CI job 修复前后的可归因失败项；
右：15 个改动文件在交付网关下的放行/滤除计数）。

## 5. 注意点（踩坑与局限）

1. **红不一定长在分支内容上**：第 1 轮的修复正确但落在另一个 ref 上，PR 头没动 → CI 依旧红。判红的第一步应是
   比对「PR 头 SHA / 实际推送 ref / 本地修复提交」三者，而不是重写修复。
2. **网关滤除是路径级且静默的**：`artifacts/`、`runs/`、根级 `*.png`、`*.log` 不进交付提交。测试 fixture 载荷
   必须放在 `src/`、`tests/`、`scripts/`、`docs/assets/`、`benchmarks/` 等可交付路径。
3. **不要用「缺文件就 skip」糊红**：那会让审计门禁在 CI 里永久静默失效（本仓有"量具失灵仍显示通过"的先例）。
4. **局限一（证据缺口，如实记录）**：全量合并树套件在 128 核共享机器上受他单负载影响，网格化跑到 99%
   （约 1720/1737）时被外部 120 s 命令窗切断，**未取得完整 `passed` 汇总行**。日志中唯一的 `F` 已逐项定位为
   `tests/test_release_quality.py::test_api_audit_script_passes_with_warnings_as_errors`：本机 torch 触发
   `pynvml` `FutureWarning`，在 `PYTHONWARNINGS=error` 下报错；该用例在 `origin/main` 的纯净树上**同样失败**
   （实测 `1 failed`），故属环境性、与本 PR 无关。CI 上的最终判决仍以 GitHub Actions 日志为准。
5. **局限二**：`lint (advisory)` 的红是仓内既有存量（`origin/main` 上 `ruff check` 4281 条、`ruff format --check`
   680 个文件），与本 PR 无关；FEM 的合并网关已把该 check 列入 `ignore_checks`，本单未动全仓 backlog。
6. **局限三**：`artifacts/` 下其余一次性工件（解冲器、会话日志、旧图）仍未随仓交付，历史文档中的引用保留为历史记录。

## 6. 误差判定

本单判据是**离散的通过/失败**，没有连续量，故逐项判定：

| 关键量 | 基准 | 实测 | 误差 | 是否达标 |
|---|---|---|---|---|
| 收集期错误数 | 0 | 0 | 0 | 是 |
| 审计门禁（完整历史） | 6 passed | 6 passed | 0 | 是 |
| 审计门禁（浅克隆=CI） | 0 failed | 6 skipped / 0 failed | 0 | 是 |
| 审计判决 | `UNION_OK` | `UNION_OK` | 0 | 是 |
| 承载文件数 | 243 | 243 | 0 | 是 |
| 手工解冲原子/词丢失 | 0 | 0/0 | 0 | 是 |
| 负控拒绝 | 3/3 触发 | 3/3 触发 | 0 | 是 |
| 改动面 ruff 违规 | 0 | 0 | 0 | 是 |
| 与 main 的合并冲突数 | 0 | 0 | 0 | 是 |

**是否 <3%**：上表全部是 0/非 0 的离散量，实测与基准一致（相对误差 0%），故**是**。连续量（求解器精度）
不适用于本单；大倾角 GZ 的精度判定沿用 `docs/marine-large-angle-stability.md` 的原有基准，本单未改动其算法。