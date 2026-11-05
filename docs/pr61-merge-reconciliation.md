# PR #61 治愈：把 `agent/m3202` 重新对齐到 `main`（冲突清单 · 逐文件解法 · 测试证据）

- 仓库：`cxs503/TensorFEM`（有限元求解器库，`src/tensorfem/`）
- 被治愈的 PR：#61，源分支 `agent/m3202`，被判定 `mergeable=false`（对 main 变脏）
- 本次作业分支：`agent/m3783`（工作树 `/root/tp-agent/work/m3783`）
- 合并形态：把 `main` 三方并入 `agent/m3202`（保留 PR 的 9 个文件全部内容）
- 证据文件：`artifacts/pr61_merge_union_evidence.json`（程序化生成）、
  审计器 `artifacts/pr61_merge_union_check.py`、解冲器 `artifacts/pr61_resolve_conflicts.py`、
  出图器 `artifacts/pr61_merge_figure.py`、图 `artifacts/pr61_merge_evidence.png`

## 1. 案例介绍：这次"脏"到底是什么脏

PR #61 的业务内容（分支侧 2 个提交 `3c667a8`、`2ff75a8`，共 9 个文件、+3828 行）是船舶
**大倾角稳性**：为墙侧（wall-sided）棱柱形驳船给出精确截面几何下的 `GZ` 曲线与 `KN`
横交曲线族，补齐 `marine_hydrostatics` 文档里声明的"不支持大倾角 GZ / 横交曲线"欠账。
它用的基准算例就是仓内自带的小几何：

| 量 | 值 | 出处 |
|---|---|---|
| 船长 L | 40.0 m | `examples/marine/large_angle_box_barge.json` |
| 船宽 B | 12.0 m | 同上 |
| 型深 D | 6.0 m | 同上 |
| 吃水 T | 2.0 m（`mode = specified_draft`） | 同上 |
| 重心高 KG | 2.1 m | 同上 |
| 水密度 ρ / 重力 g | 1000 kg/m³ / 9.81 m/s² | 同上 |

"变脏"的成因不在业务内容，而在**目标分支的移动**：从 merge-base `e8f6759` 起，`main`
被持续 drain，作业结束时 `main` 侧已带 236 个文件变更；两侧**同时改写**的文件只有 2 个，
而它们恰好都是"补充型文档"（CHANGELOG 与 `docs/marine-hydrostatics.md`），于是三方合并
在文本层报 2 处 content 冲突 —— 这就是 GitHub 上 `mergeable=false` 的本地可复现成因。

> 作业期间观察到一个必须记录的事实：`origin/main` 在本次作业中途从 `14ca2fd` 前进到
> `7accc4b`（`Merge PR #73`, owner 手动 drain）。因此"按某个 main 快照算出的冲突集合"
> 会过期，本报告的基线一律用**实测量到的 main SHA**（见下表），merge_governor 下一周期
> 会以当时的 main 复验。

| 角色 | SHA |
|---|---|
| merge-base | `e8f6759aab2d203db641b2e94502ee5865848e3f` |
| `main`（合并目标，实测） | `7accc4b9cfb45369958c432a8aa2f0ebbb846a5b` |
| `agent/m3202`（旧，PR 头） | `2ff75a8158492b1d2232348d1a1d8e002ce1c009` |
| 合并提交（可快进 PR 分支） | `bafc0fa`（父 `7accc4b` + `2ff75a8`） |

## 2. 怎么做的：并集义务先于动作

1. **建工作树、拉两支**：`git worktree add` 出 `m3783`，`git fetch origin main`；
   不依赖仓内 `main` 本地引用 —— 实测它比 `origin/main` 落后 84 个提交（陈旧引用陷阱）。
2. **算清义务面**：`git merge-base` + `git diff --name-status` 两侧变更集求交，
   得到"两侧同时改写"的文件集合 = {`CHANGELOG.md`, `docs/marine-hydrostatics.md`}；
   `main` 侧另有 1 个删除（`.scratch/proto_ritz.py`），分支侧无删除。
3. **三方合并**：`git merge --no-ff --no-commit agent/m3202`（ort 策略），
   得到 2 处 content 冲突，其余 234+7 个单侧文件自动带入。
4. **逐文件语义并集解冲**（脚本化 `artifacts/pr61_resolve_conflicts.py`，可复算）：
   - `CHANGELOG.md`：两侧都在文件顶部插入条目。解法 = **两条目并存**，按本文件
     "新在前"的既有次序拼接：main 的 `tensorfem.prestressed_modal` 条目在上，
     分支的 `## 2026-11-04 大倾角稳性…` 小节紧随其后，其后接原有
     `## 2026-11-03 Cook 膜应力审计证据收尾`。**不删除任何一行**，仅插入一个空行分隔。
   - `docs/marine-hydrostatics.md` → `## Deliberate limitations`：
     main 侧说"本模块还没有大倾角 GZ，精确墙侧 oracle 在
     `tensorfem.marine_large_angle_stability`"，分支侧说"大倾角 GZ / KN 由
     `tensorfem.marine_large_angle_gz` 单独提供，机精度对标墙侧闭式"。
     两侧都对且互补（合并后两个模块都存在），故解法 = **两个服务并列陈述**：
     保留两边的模块名、审计报告链接、`docs/marine-large-angle-stability.md` 指针、
     "clipped-section geometry / prismatic barges / machine precision" 等断言；
     仅重排换行、把重复的"It is not a …"连接句合并成一句
     "Beyond those two services this module offers no large-angle `GZ`, …"，
     句尾的免责列表（free-surface / damage stability / compartment flooding / trim /
     arbitrary-hull / wave-load / seakeeping / mooring / classification-rule）逐项保留。
5. **并集审计（防机械 overlay 静默丢内容）**：`artifacts/pr61_merge_union_check.py`
   直接读 git 对象库，对两侧同时改写的文件施加**并集义务**：
   - 自动合并块（tier 1）：合并后的行多重集必须包含"任一侧相对 base 独有"的每一行；
   - 手工解冲块（tier 2）：合并后文本必须保留两侧的全部**原子**
     （反引号标识符、markdown 链接目标、文件路径、数字）与全部内容词 ——
     允许重排换行，不允许丢断言；
   - 删除项单列审计：单侧删除且另一侧未触碰 → 必须真的不在合并结果里；
     删/改并存 → 必须报出来而不是悄悄解决。
   审计器自带**两个负控**：把"合并结果"换成 `main` 单侧（机械 overlay = 取 main）
   与换成 `agent/m3202` 单侧（overlay = 取分支），必须判 FAIL。
6. **测试**：同一命令 `python3 -m pytest tests -q -p no:cacheprovider -n 8`
   分别在合并树与 `main` 基线工作树（`git worktree add` 于 `origin/main`，`PYTHONPATH=src`）跑全量。

## 3. 计算结果（程序化产出，零手抄）

### 3.1 规模与并集义务

| 指标 | 数值 |
|---|---|
| main 侧变更文件数 | 236 |
| 分支侧变更文件数 | 9 |
| 两侧同时改写（冲突面） | 2 |
| 仅 main 侧 / 仅分支侧 | 234 / 7 |
| main 侧删除（`.scratch/proto_ritz.py`） | 1（已按语义传播） |
| 合并后残留冲突标记 | 0 |

### 3.2 并集审计（`artifacts/pr61_merge_union_evidence.json`）

| 文件 | 层级 | main 独有原子 | 分支独有原子 | **丢失** | 判定 |
|---|---|---|---|---|---|
| `CHANGELOG.md` | tier 2（原子+词） | 136 | 27 | **0 / 0** | union-preserved |
| `docs/marine-hydrostatics.md` | tier 2（原子+词） | 13 | 3 | **0 / 0** | union-preserved |
| 负控：overlay = main 单侧 | tier 2 | — | — | 分支侧丢 **30** 个原子 | FAIL（应fail，已fail） |
| 负控：overlay = 分支单侧 | tier 2 | — | — | main 侧丢 **149** 个原子 | FAIL（应fail，已fail） |

负控是这套审计的"量具校准"：它证明"零丢失"不是审计器太松，而是本次合并真的没丢东西。

### 3.3 测试证据（同一命令、同一时刻、同一台机，`-n 8`）

| 树 | 汇总行（pytest 原样） | passed | failed | skipped |
|---|---|---|---|---|
| 合并树（main `7accc4b` + PR #61） | `3 failed, 1427 passed, 6 skipped, 1 xpassed, 31 warnings in 805.22s` | **1427** | 3 | 6 |
| 基线 `main` `7accc4b` | `2 failed, 1389 passed, 6 skipped, 1 xpassed, 30 warnings in 862.98s` | **1389** | 2 | 6 |

- **通过数：1389 → 1427（+38，全部来自 PR #61 的 `tests/test_marine_large_angle_gz.py`），无一项回归。**
- 两边共有的失败是环境性/既有欠账：
  `test_finite_strain_curved_friction_contact3d.py::test_real_curved_nonmatching_tet4_friction_level_one_closure`、
  `test_release_quality.py::test_api_audit_script_passes_with_warnings_as_errors`。
- 多出来的那 1 个失败是**墙钟断言**，不是合并回归：
  `tests/test_rigid_contact_graph.py::test_thousand_constraint_sparse_islands_benchmark`
  断言 `elapsed < 10 s`，合并跑出 `13.31 s`。归因证据三条：
  ① 该测试与其被测源码在合并前后**字节相同**（`git diff origin/main HEAD -- tests/test_rigid_contact_graph.py
  src/tensorfem/rigid_contact*.py` 为空）；
  ② 在基线树上单跑同一条测试**同样失败**（46.4 s）；
  ③ 判据是纯墙钟，当时机器 load average ≈ 2000（128 核，多智能体共用）。

## 4. 结果图

`artifacts/pr61_merge_evidence.png`（`artifacts/pr61_merge_figure.py` 生成，双面板；
输入=审计 JSON + 归档的 pytest 会话汇总 `artifacts/pytest_session_summaries.json`）：

- 左：并集审计的"内容丢失量"柱图 —— 三方合并 (0, 0) vs 两个机械 overlay 负控
  （取 main 丢分支 30 个原子、取分支丢 main 149 个原子）；
- 右：测试套件合并树 vs 基线的 passed/failed/skipped 对比。

**为什么没有场量云图**：本任务是版本控制/集成任务，不产生任何物理场；PR #61 自身的
物理证据（GZ 曲线对墙侧闭式、动稳性积分）已由 PR 带上的
`docs/assets/marine_large_angle_gz.png` 与 `benchmarks/large_angle_gz_reference.json` 提供，
本次合并**原样保留、未重算**，故不重复作图。

## 5. 注意点（踩过的坑）

1. **共享 `.git` 上 main 会边干边走**：作业中途 `origin/main` 由 `14ca2fd` 前进到
   `7accc4b`。第一轮合并（针对 `14ca2fd`）提交后立即过期；随后 `git diff origin/main HEAD`
   会把 main 新加的 173 个文件显示成"被删除"，极易误判为"合并丢内容"。正确姿势：
   每次判定冲突/基线前重新 `git fetch` 并记录实测 SHA。
2. **仓内 `main` 本地引用陈旧**（落后 `origin/main` 84 个提交）。用 `origin/main` 做基线。
3. **`/tmp` 日志名会撞车**：多智能体共用 `/tmp`，`/tmp/base_pytest.log` 被别的进程写过，
   导致一次基线统计污染。最终用带任务号的唯一路径复跑。
4. **`pkill -f <模式>` 会误杀同伴**：模式同时匹配自己的 shell 与他人命令行，曾误杀两个
   邻居任务的 shell。杀进程只用精确 PID。
5. **墙钟断言在共用机上不可用**：`elapsed < 10 s` 这类判据在 load ≈ 2000 时必然翻车，
   归因时必须给出"源码零差异 + 基线上同样失败"两条证据，不能只写"不稳"。
6. **GitHub `mergeable=false` 与本地冲突面一一对应**：本地 ort 合并报的 2 处 content 冲突
   就是 dirty 的成因；若改用 `-X ours` 类机械 overlay 可以"消冲突"，但会静默丢内容
   （负控实测丢 30 / 149 个原子）——所以本次坚持逐文件并集 + 审计。

## 6. 误差判定（关键量 vs 基准，逐项）

本任务无物理基准量（无实验/解析解对比对象），故以"内容丢失率"和"测试通过数差"为判据：

| 关键量 | 基准 | 实测 | 相对偏差 | 是否 <3% |
|---|---|---|---|---|
| 并集原子丢失（main 侧） | 0 个 | 0 个 | 0 % | **是** |
| 并集原子丢失（分支侧） | 0 个 | 0 个 | 0 % | **是** |
| 合并后残留冲突标记 | 0 处 | 0 处 | 0 % | **是** |
| 测试通过数 | 基线 1389 | 1427 | +2.7 %（≥ 基线） | **是**（判据为"不得低于基线"） |
| 测试失败数 | 基线 2 | 3（墙钟型 1 项，基线同测亦失败） | 见 §3.3 归因 | **是**（源码零差异，非回归） |

结论：合并结果相对基线**不低于**，文档冲突按并集解决、两侧断言零删除，
本 PR 可常规合并；`merge_governor` 下一周期以当时 main 复验即可。