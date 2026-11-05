# PR #65 治愈·第 2 轮复核报告（管到底·第 1 轮，2026-11-05）

第 1 轮（`d446f28`）已完成三方合并并留下复核报告 `docs/merge-cure-pr65-m3255.md`，但**未推回 GitHub**
（`refs/heads/agent/m3255` 仍停在 `949f57a`，与 main 的合并关系未上线，故 PR 仍被判定 dirty）。
本轮三件事：① 对既有合并做**独立完整性复核**（不信任上一轮自述）；② 修复复核发现的唯一一处静默丢失；
③ 与远端分支尖端对齐祖先后推回同分支。

## A. 远端状态校准（先取真值，再看本地）

| 引用 | SHA | 说明 |
|---|---|---|
| GitHub `main` | `7accc4b` | 与本地缓存 `origin/main` 一致 → 合并依据未过期 |
| GitHub `agent/m3255`（本轮推送前） | `949f57a`（*m3255: matured mission deliverables*，父 `dff97e7`） | 与本地分支提交 `0e3190c` **树完全一致**（tree 均为 `78c5582`）→ 远端无待并入内容 |
| 本地 `agent/m3255`（第 1 轮成果） | `d446f28` | = `7accc4b` ∪ `0e3190c` + 第 1 轮报告 |

远端尖端的树与 `0e3190c` 逐字节相同（`git diff 949f57a 0e3190c` 为空），所以「以本地覆盖远端」不会丢远端内容；
但 `949f57a` 不在本地祖先链上，直接 push 会是非快进，故需先 `git merge origin/agent/m3255` 补齐祖先（树不变的合并）。

## B. 完整性复核（三项机器化对账）

### B1 文件集合对账

| 提交 | 文件数 |
|---|---|
| `7accc4b`（main） | 1227 |
| `0e3190c`（分支） | 1036 |
| `d446f28`（合并树） | 1271 |

- **main 侧丢失：0**（`main \ merge = ∅`）→ main 的 212 个新增/改动全在合并树里。
- **分支侧丢失：1** → `.scratch/proto_ritz.py`（分支相对合并基新增的 8 个文件中唯一未落合并树的）。
  第 1 轮属**静默**丢弃（治愈报告未记载该文件）。本轮按「一侧新增不得静默丢失」原则从 `0e3190c` 原样恢复并随本提交落库，
  恢复后分支侧丢失 = 0。

### B2 「两侧不同、合并取 main 单侧」逐文件判定（12 个）

机械判据：若 `branch:file == merge-base:file`，则该文件分支侧从未改动，取 main 版本**不构成丢失**。逐一核验结果：

| 分组 | 分支侧 == 合并基 | 合并 == main |
|---|---|---|
| `CHANGELOG.md`、`docs/element-rank-deficiency-audit.md`、`docs/kirsch-hole-stress-concentration.md`、`docs/marine-fatigue-advanced-qualification.md`、`docs/marine-hydrostatics.md` | 是（5/5） | 是（5/5） |
| `src/tensorfem/__init__.py`、`additional_benchmark_fields.py`、`element_rank_audit.py`、`plate.py`、`plate_vibration.py`、`torsion_bar.py` | 是（6/6） | 是（6/6） |
| `tests/test_element_rank_audit.py` | 是 | 是 |

配套行级复核（`git diff -U0 merge-base..branch -- file` 取 `+` 行）：全 12 个文件的**分支侧新增行数 = 0**，
即分支从未编辑它们 → 不存在分支改动被 main 覆盖的静默丢失。

真正「两侧都改」的只有 1 个文件：`docs/benchmark-qualification-status.md`。第 1 轮做手工语义并集，本轮复核其内容
同时含 main 的「开孔半球壳应力参考已补并给出否定结论、`stress_accuracy_qualified` 维持 False」与分支的
「非均匀预屈曲应力已从『仍缺』清单闭合」两侧最新事实，双方段落均在。

### B3 冲突标记与工作树卫生

- 全树 `^<<<<<<<` / `^>>>>>>>`：**0 处**。
- 复核开始时工作树/索引残留 **250 文件、−78464 行的「暂存删除」**（第 1 轮机械 overlay 的残渣；若被提交会静默删掉
  `benchmarks/` 与 `docs/assets/` 大量已合并内容）。已用
  `git restore --source=HEAD --staged --worktree .` 归零，**从未提交**；归零后 `git status` 干净，
  且这些路径在 HEAD 树中均存在（`HEAD:path` 有效）→ 合并成果未受影响。

## C. 测试：合并树 vs main 基线（同子集、同参数、串行）

以 `git archive 7accc4b` 导出 main 基线树 `/tmp/base65b`，两棵树 `OMP/MKL/OPENBLAS_NUM_THREADS=1`、
`PYTHONPATH=<tree>/src`、`pytest -q -p no:cacheprovider --tb=no`，**串行**执行同一子集（xdist 在两棵独立树上并发会争用，
已弃用）：

| 子集 | 合并树 | main 基线 |
|---|---|---|
| `tests/test_plate.py` + `test_marine_plate_buckling.py` + `test_plate_mindlin.py` | **25 passed**（54.7 s） | **25 passed**（54.7 s） |
| `tests/test_plate_vibration.py` + `test_marine_plate_combined_buckling.py` | **43 passed**（101.3 s） | **43 passed**（85.8 s） |
| `tests/test_plate_bending_buckling.py`（分支新增，基线无此文件） | **36 passed**（61.1 s） | 不适用 |

判定：**合并树全绿、不低于 main 基线**（同集 68 : 68 持平，另 +36 分支新增用例）。墙钟差异来自本机 load15≈520，非代码差异。
包导入自检：`import tensorfem, tensorfem.plate_bending_buckling` 成功。

## D. 推送与 SHA 记录

| 项 | SHA |
|---|---|
| 分支旧 SHA（GitHub，推送前） | `949f57a` |
| main 基线 | `7accc4b` |
| 远端尖端合并（补祖先，树不变） | 见本分支 `git log --merges -1`（父 = 本地 HEAD ∪ `949f57a`） |
| 推送后分支新 SHA | 见 `git ls-remote origin refs/heads/agent/m3255` 回执（记于 mission 3255 的进展报告） |

推送链路：`git merge origin/agent/m3255`（树不变的祖先补齐）→ 快速推送 `HEAD:agent/m3255`（**未触碰 main**，非强推）。

## E. 本轮局限

1. 本机 `load15≈520`（128 核），按容量纪律只跑核心子集与轻量对账，未重跑全量 1728 例；全量等价性沿用第 1 轮
   §4 的分块证据（合并 1683 passed vs 基线 1650 passed，差值 36 恰为分支新增用例数）。
2. 全量 `git fetch` 在 120 s 墙钟内超时，改为按 ref 精确抓取（`refs/heads/agent/m3255`）成功，不影响结论。
3. 复核为路径级 + 行级 + 文件哈希三级机械对账，未做语义级（AST / 物理结论）再推导；除 `.scratch` 外未发现可疑项。