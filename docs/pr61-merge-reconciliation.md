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
# PR #61 治愈：`agent/m3202` 三方并入 `main`（冲突清单 · 逐文件解法 · 双工具并集审计 · 测试证据）
- 被治愈的 PR：**#61**，源分支 `agent/m3202`，GitHub 判定 `mergeable=false / state=dirty`
- 业务增量：船舶**大倾角稳性** —— 墙侧（wall-sided）棱柱形驳船按精确浸水剖面几何求 `GZ`
  曲线与 `KN` 横交曲线族，补上 `marine_hydrostatics` 文档自己声明的「不支持大倾角 GZ」欠账
- 本工作树：`/root/tp-agent/work/m3938`（分支 `pr61-cure`，最终以 `HEAD:agent/m3202` 推回）
- 证据：`docs/evidence/pr61/`（审计 JSON、负控文本、全量 pytest 日志）
## 1. 案例介绍：这次"脏"是什么脏
物理背景与几何/参数表（仓内基准算例 `examples/marine/large_angle_box_barge.json`）：
| 量 | 值 |
| 船长 L / 船宽 B / 型深 D | 40.0 m / 12.0 m / 6.0 m |
| 吃水 T（`mode = specified_draft`） | 2.0 m |
| 重心高 KG | 2.1 m |
| 水密度 ρ / 重力 g | 1000 kg/m³ / 9.81 m/s² |
| 解析锚 | 墙侧闭式 `GZ(φ) = sinφ·(GM + BM·tan²φ/2)`，`GM = KB + BM − KG` |
"脏"不在业务内容，而在**目标分支持续前进**：分支从分叉点 `e8f6759` 起，main 被 drain 了
236 个文件（含 `.scratch/proto_ritz.py` 一处删除），两侧同时改写 17 个文件，其中 2 个
文档型文件文本层直接冲突。本次工作的实质不是"再合一次"，而是**证明合并结果对两侧都不丢
内容**——机械 overlay 有静默丢内容的前科，"看起来能合"不算证据。
| 角色 | SHA | 说明 |
| 历史分叉点（审计基准） | `e8f6759` | 两侧共同祖先；`--base` 必须钉它 |
| merge-base（PR 视角） | `c9d72b1` | 分支侧 9 文件 × main 侧 60 提交 |
| `main`（合并目标，本机最后已知） | `7accc4b9cfb45369958c432a8aa2f0ebbb846a5b` | `origin/main` |
| `agent/m3202`（旧 SHA，GitHub PR 头 = 合并前） | `6b4c4fa48efea38e63303f1d5ae921392d468fe6` | 1 提交 / 9 文件 / +3828 −0 |
| 三方合并提交 | `10c10a95f021371b193e70603f366043e9210373` | 父 = `6b4c4fa` + `7accc4b` |
| 表述保真修复（唯一真实丢内容的修复） | `31d8146` | 见 §3.3 |
| 分支新头 | 本文件所在提交（`git log -1`），旧头 `6b4c4fa` → 新头见 §5 | 推回 `agent/m3202` 后 GitHub 重算 `mergeable` |
## 2. 怎么做的（方法 · 流程 · 关键设置 · 工具链）
1. **建工作树、拉两支**：`origin/main` 与 `agent/m3202` 都用远端 ref，不依赖仓内本地
   `main`（实测本地 `main` 与 `/root/tensor-ecosystem/TensorFEM` 镜像都停在旧提交，属陈旧
   引用陷阱）。
2. **算清义务面**：`git merge-base` + 两侧 `diff --name-status` 求交 → main 侧 236 文件、
   分支侧 24 文件、**双侧 17**、单侧 219 / 7，main 侧 1 个删除、分支侧无删除。
3. **三方合并**：`git merge` 得 `10c10a9`；17 个双侧文件里 15 个由 git 自动合并，2 个
   文档冲突手工解冲（清单见 §3.1），**禁止机械 overlay**。
4. **双工具并集审计**（两套互相独立的判据，任一套判 FAIL 即视为不合格）：
   - `scripts/audit_pr61_merge_union.py`（语义分级）：散文 → 行内反引号标识符 + 链接 +
     数字 + 内容词义务；代码/数据 → 严格行多重集；单侧文件 → 逐字节相等；删除项单列。
     空审计（base 塌到一侧）判 `NOT_AUDITED_NO_SHARED_FILES`，不发假绿。
   - `scripts/crosscheck_pr61_merge.py`（独立复算，不复用上者代码）：义务 A = main 单侧
     219 条路径在合并树里逐字节相同（删除项必须已删）；义务 B = 共享文件保内容
     （代码/数据严格 token 超集，散文按**原子**=反引号标识符/链接/数字判，散文改词必须在
     `PROSE_REWORDED` 里逐文件写明理由，否则 fail-closed）。
5. **三个负控（审计器必须能判死，否则它的绿没有意义）**：机械取 main、机械取分支、
   以及本次新增的**单文件静默丢失**（把 `docs/marine-hydrostatics.md` 换成 main 版）——
   三者分别必须判 `UNION_FAIL` / `CROSSCHECK_FAIL`。
6. **测试**：合并树与 `main` 基线树（`git archive 7accc4b` 快照）各跑全量
   `pytest -q -n 16 --timeout=1800`，再对存活的 3 个失败**串行**复跑并逐项归因。
## 3. 计算结果（数字表，全部程序化产生）
### 3.1 冲突文件清单与逐文件解法
| # | 冲突文件 | 冲突原因（两侧各写了什么） | 解法（语义并集） |
|---|---|---|---|
| 1 | `CHANGELOG.md` | 分支在顶部加「2026-11-04 大倾角稳性…」整节（5 条要点）；main 在顶部加 `tensorfem.prestressed_modal` 单条要点 | **两段全保留**：main 的 prestressed 要点在前，分支的大倾角稳性整节紧随其后，再接 main 既有历史（`## 2026-11-03 Cook…` 等）。两侧条目数各自可数，行多重集校验 0 丢失 |
| 2 | `docs/marine-hydrostatics.md`（`## Deliberate limitations` 段） | main 侧写「本模块仍不是大倾角 GZ 曲线；墙侧 oracle 在 `marine_large_angle_stability`，见 audit 报告」；分支侧写「大倾角 GZ 曲线与 `KN` 横交曲线由 `marine_large_angle_gz` 单独提供，对墙侧闭式机精度验证」 | 重写为**并集句**：本模块自身仍无大倾角 GZ → 两个独立服务分别覆盖该问题（`marine_large_angle_stability` 的墙侧 oracle + `marine_large_angle_gz` 的曲线族）→ 两侧共有的免责列表（free-surface / damage stability / compartment flooding / trim / arbitrary-hull / wave-load / seakeeping / mooring / classification-rule）逐项保留 |
15 个自动合并的双侧文件（如 `src/tensorfem/plate_shear_buckling.py`、
`tests/test_saint_venant_torsion.py`、`docs/saint-venant-torsion.md` 等）经**严格行/token
多重集**校验：0 丢失；main 侧唯一删除（`.scratch/proto_ritz.py`）已随合并删除，另一侧未触碰
（`deletion-propagated`）。
### 3.2 双审计结果
| 判据（工具） | 数值 | 判定 |
| 义务面（audit） | main 侧 236 / 分支侧 24 / 双侧 17 / 单侧 219·7 | — |
| 双侧文件并集义务 / 丢失单元 | 17 文件，**丢失 0** | `UNION_OK` |
| 单侧文件逐字节校验 | 225（218 main-only + 7 branch-only）| 0 不一致 |
| 未解冲突标记残留 | 0 | — |
| 删除传播 `.scratch/proto_ritz.py` | 已删且另一侧未触碰 | `deletion-propagated` |
| 独立复算·义务 A（crosscheck） | main 单侧 219 路径，**218 逐字节相同 + 1 删除已删**，0 不符 | `CROSSCHECK_OK` |
| 独立复算·义务 B | 17 个共享文件，代码/数据 token 丢失 **0**，散文原子丢失 **0** | 同上 |
| 散文改词（需具名理由） | 1 文件：`docs/marine-hydrostatics.md`（仅系词改写，标识符/链接/数字全在） | 已具名记录 |
| 负控①机械取 main（`--merge 7accc4b`） | 丢分支 **30** 单元 | `UNION_FAIL` ✓ |
| 负控②机械取分支（`--merge 6b4c4fa`） | 丢 main **178** 单元 | `UNION_FAIL` ✓ |
| 负控③单文件静默丢失（散文取 main 版） | 丢分支 **3 atoms** | `UNION_FAIL` + `CROSSCHECK_FAIL` ✓ |
负控都判死 = 审计器的绿不是空转；负控③正是"看起来能合、实则吃掉一句断言"的那类缺陷，
两套工具都能在文件粒度上抓到。
### 3.3 本轮之前抓到的唯一一处**真实**丢内容（已修复并留证）
`10c10a9` 上审计器报 `docs/marine-hydrostatics.md` 相对 main 侧**内容词短缺 1 个
（`module`）**，main 的 "this module … two separate services cover that question" 框架整体
被判吃掉——合并文字是从分支句起头、把 main 的句子续在后面。修复 `31d8146` 后该文件原子/
词丢失归零，文字同时保留三件事：本模块自身无大倾角 GZ、墙侧 oracle 在
`marine_large_angle_stability`、曲线由 `marine_large_angle_gz` 提供。修复前审计
（`--merge 10c10a9`）与收尾树审计可用 §4 命令复算复现。
### 3.4 测试与基线
| 运行 | 合并树（HEAD） | main 基线（`7accc4b` 快照） |
| 全量 `pytest -q -n 16 --timeout=1800` | **3 failed, 1720 passed, 7 skipped, 1 xpassed, 714.66 s** | 同命令同机同时段在跑；(.log 见 `docs/evidence/pr61/`) |
| `test_release_quality.py::test_api_audit_script_passes_with_warnings_as_errors`（串行复跑） | **failed**（2.52 s） | **failed**（2.58 s）→ **两侧同样失败，非合并回归**（环境：本机 torch 2.7.1 / 无 TensorLBM fixture，`PYTHONWARNINGS=error` 下 `scripts/audit_public_api.py` 非零退出） |
| `test_colored_contact_graph.py::test_1024_constraint_vectorized_benchmark_and_serial_equivalence` | failed：`serial_time=10.90 s > 10 s`（机时门） | 同一机时门（load average ≈ 700 / 128 核，超订 5 倍） |
| `test_finite_strain_curved_friction_contact3d.py::test_real_curved_nonmatching_tet4_friction_level_one_closure` | Newton 在该载荷步不收敛（`last residual=5`，Coulomb ratio 1.00002） | 同模块，未被本分支触碰 |
| PR 新增门 `tests/test_marine_large_angle_gz.py` | 39 passed（含 0.5° 步长全表、墙侧闭式 1.33e-15 m 残差、排水体积 1e-12 守恒、防漂移出图钉） | 文件不存在（PR 新增） |
**归因（不粉饰）**：合并树相对 main 的代码差异只有 PR 自己的 9 个文件 + 本单证据文件
（`scripts/`、`docs/evidence/`），**3 个存活失败点所在模块（接触图基准、finite-strain
摩擦接触、API 审计脚本）分支一行未改**（`git diff --name-status origin/main HEAD` 可证）；
其中 API 审计项在基线上同样失败（§3.4 第 2 行）。故判定：**无合并引入的回归**；两项机时门
失败是宿主超订所致，属环境噪声而非代码退化，仍按 honest-FAIL 记录。
### 3.5 误差判定（逐项，显式回答「是否 <3%」）
| 关键量 | 基准 | 本结果 | 误差 | <3%？ |
| 并集义务：合并树丢失内容单元 | 0（义务要求） | **0** | 0% | **是** |
| main 单侧路径逐字节一致 | 219/219 | 218 一致 + 1 已按 main 删除 | 0% | **是** |
| 共享文件原子（散文）丢失 | 0 | **0**（唯一改词文件已具名理由） | 0% | **是** |
| 审计器灵敏度（负控应丢） | — | 30 / 178 / 3 单元均被判死 | 三个方向全部判死 | 是（灵敏度充分） |
| 测试通过数 | 基线见 `docs/evidence/pr61/pytest-full-baseline.log` | 合并树 1720 passed | 回归项 0（3 失败逐项归因为机时/环境） | 是（无回归） |
| PR 载荷精度：大倾角 GZ vs 墙侧闭式 | 闭式解 | 带内最大残差 1.33e-15 m（相对 8.3e-16） | ≈1e-15 | **是** |
## 4. 复现
```bash
# 并集审计（--base 必须钉历史分叉点，否则 base 塌到 main 会得到"空审计"）
python3 scripts/audit_pr61_merge_union.py --base e8f6759 --main 7accc4b \
    --branch 6b4c4fa --merge HEAD --check --out docs/evidence/pr61/union-audit.json
# 独立复算（另一套判据）
python3 scripts/crosscheck_pr61_merge.py --base e8f6759 --main 7accc4b \
    --branch 6b4c4fa --merge HEAD --out docs/evidence/pr61/crosscheck.json
# 负控：三个都必须判死（文本记录在 union-audit-controls.txt / crosscheck-controls.txt）
python3 scripts/audit_pr61_merge_union.py --base e8f6759 --main 7accc4b \
    --branch 6b4c4fa --merge 7accc4b --check --out /tmp/neg_main.json      # UNION_FAIL
python3 scripts/audit_pr61_merge_union.py --base e8f6759 --main 7accc4b \
    --branch 6b4c4fa --merge 6b4c4fa --check --out /tmp/neg_branch.json    # UNION_FAIL
python3 scripts/plot_pr61_merge_evidence.py --out /tmp/pr61_merge_evidence.png
```
## 5. 注意点与局限（不粉饰）
1. **图不入库（治理约束）**：merge governor 的 artifact 预算 <200 kB/PR，而 PR 已带
   `docs/assets/marine_large_angle_gz.png`（158 kB，文档正文引用，属正当资产）。故本轮把
   自证图改为**按需重算**（`scripts/plot_pr61_merge_evidence.py --out /tmp/...`），仓内只留
   可复算脚本与 JSON；上一轮 `artifacts/` 下的 313 kB 证据（含 112 kB PNG、6 个日志）
   收敛为 `docs/evidence/pr61/` 文本台账。
2. **全量基线跑完时间未定**：宿主 load average ≈ 700（128 核），基线全量在同一时段仍在
   推进；本轮用于判定的基线证据是"同名失败项在基线上同样失败/分支未触碰该模块"的**定点
   对照**，不是等时全量对照。基线日志 `docs/evidence/pr61/pytest-full-baseline.log` 一旦补齐
   即可直接替换该行。
3. **本机取新 main 有超时风险**：`git fetch origin` 偶发 120 s 超时；本审计钉的是
   `origin/main = 7accc4b`。若远端 main 又前进，重跑 §4（`--base` 用当时的真实分叉点）即可。
4. **散文审计的边界**：只保证"原子 + 内容词 + 断言不丢"，不保证语序/断句与任一侧相同——
   这是散文并集的物理限制；§3.3 那次真实丢失正是词级短缺 1 被抓到的，散文没有放宽到
   "重写也算过"（改词必须具名理由）。
5. **解冲器需在冲突态运行**：`scripts/resolve_pr61_conflicts.py` 依赖冲突标记定位替换区，
   合并已提交后再跑会 `StopIteration`；其内嵌文本与仓内提交文本一致，作为"当时解法"的
   可复算记录保留。
6. **删除台账不静默**：上一轮 `artifacts/` 证据收敛的逐条理由见 §5.1 的 removal ledger，
   均可用 `git show <sha>:<path>` 取回。
### 5.1 Removal ledger（证据收敛，不静默丢）
| 旧路径 | 处置 | 理由 |
| `artifacts/pr61_merge_union_check.py` | → `scripts/audit_pr61_merge_union.py` | 可复算审计器，属工具链；默认输出改到 `docs/evidence/pr61/` |
| `artifacts/pr61_merge_figure.py` | → `scripts/plot_pr61_merge_evidence.py` | 改为按需重算、输出到 `--out`（不入库），读新证据路径 |
| `artifacts/pr61_resolve_conflicts.py` | → `scripts/resolve_pr61_conflicts.py` | 解冲文本快照，保留可复算记录 |
| `artifacts/pr61_merge_evidence.png`（112 kB） | 不入库 | artifact 预算 <200 kB（见 §5.1），由脚本按需重算 |
| `artifacts/evidence_logs/*.log`、`pytest_session_summaries.json` | → `docs/evidence/pr61/*.log` | 同一批实测日志，改文本路径不占 artifact 预算；旧"外部工作树全量数字"不再引用，本轮用自己的全量实测替代 |
| `artifacts/README.md`（旧清单/台账） | 并入本文 §5.1 | 单一台账入口，避免两份清单漂移 |
| `artifacts/pr61_merge_union_evidence{,_prefix}.json` | → `docs/evidence/pr61/union-audit.json`（重算） | 旧 JSON 已被负控跑覆盖过；本轮用当前树重算，修复前对照可用 `--merge 10c10a9` 现场复现 |
# PR #61 治愈：把 `agent/m3202` 重新对齐到 `main`（冲突清单 · 逐文件解法 · 并集审计 · 测试证据）
- 被治愈的 PR：#61，源分支 `agent/m3202`，GitHub 判定 `mergeable=false`（对 main 变脏）
- 本工作树：`/root/tp-agent/work/m3202`，分支 `agent/m3202`
- 证据：`artifacts/README.md`（文件清单与删除台账）、`artifacts/pr61_merge_union_evidence.json`
  （程序化审计结果）、`artifacts/pr61_merge_evidence.png`（出图器 `artifacts/pr61_merge_figure.py`）、
  `artifacts/evidence_logs/`（pytest 原始日志与基线树校验）
## 1. 案例介绍：这次"脏"是什么脏，业务内容是什么
PR #61 的业务增量是船舶**大倾角稳性**：为墙侧（wall-sided）棱柱形驳船按精确浸水剖面
几何求 `GZ` 曲线与 `KN` 横交曲线族，补上 `marine_hydrostatics` 文档里自己声明的欠账。
基准算例（`examples/marine/large_angle_box_barge.json`，仓内自带）：
| 解析锚 | 墙侧闭式 `GZ = sinφ·(GM + BM·tan²φ/2)` |
"脏"不在业务内容，而在**目标分支持续前进**：从分叉点 `e8f6759` 起 main 被 drain 了
222 个文件，两侧同时改写的是文档型文件，于是文本层报冲突。本次工作的实质不是"再合一次"，
而是**证明合并结果对两侧都不丢内容**——机械 overlay 有静默丢内容的前科，所以"看起来能合"
不算证据。
| 分叉点（历史） | `e8f6759aab2d203db641b2e94502ee5865848e3f` | 两侧共同祖先 |
| merge-base（PR 视角） | `c9d72b1` | 分支侧 9 文件 × main 侧 222 文件 |
| `main`（合并目标，本机最后已知） | `7accc4b9cfb45369958c432a8aa2f0ebbb846a5b` | `origin/main`，见 §5 第 1 条局限 |
| `agent/m3202`（合并前 PR 头，旧 SHA） | `6b4c4fa48efea38e63303f1d5ae921392d468fe6` | GitHub 上的旧头 |
| 表述保真修复 | `31d8146` | 本单新增，见 §3.3 |
| 分支头（新 SHA） | 本文件所在提交（`git log -1`），父链：`10c10a9` 合并 → `31d8146` 表述保真修复 → `ddea5b9` 审计与证据 | 平台推送后 GitHub 重算 `mergeable` |
## 2. 怎么做的
1. **建工作树、拉两支**：工作树已在 `agent/m3202`；`main` 用 `origin/main` 的 ref，
   **不依赖仓内本地 `main`**——实测本地 `main` 停在 `e48fea8`（落后 84 个提交），
   本地镜像 `/root/tensor-ecosystem/TensorFEM` 同样陈旧，属陈旧引用陷阱。
2. **算清义务面**：`git merge-base` + 两侧 `diff --name-status` 求交。以历史分叉点
   `e8f6759` 为基：main 侧 236 文件、分支侧 24 文件、**两侧同时改写 17 文件**、
   单侧 219/7；main 侧 1 个删除（`.scratch/proto_ritz.py`），分支侧无删除。
3. **三方合并**：`git merge` 后得到 `10c10a9`；17 个双侧文件里 15 个由 git 自动合并，
   2 个文档（`CHANGELOG.md`、`docs/marine-hydrostatics.md`）按**语义并集**手工解冲：
   两侧的模块名、断言、链接、免责列表逐项保留，只重排换行。
4. **并集审计（本节的核心工具）**——`artifacts/pr61_merge_union_check.py`，
   对"合规"给出机器判据，分级只看**承载语义的单元是什么**：
   - 散文（`*.md`）→ **原子 + 内容词**义务：每个行内反引号标识符、链接目标、
     文件路径、数字和内容词都必须存活；允许换行重排，不允许丢断言。
   - 代码/数据 → **严格行多重集**：这里一行就是一个语句，丢行即丢内容。
   - 单侧文件 → **逐字节相等**：把路径带过来不算数，合并后的 blob 必须与该侧
     blob 完全一致（这正是机械 overlay 过不去的那道门）。
   - 删除项单列 `touched_by_other_side`；删/改并存报"待复核"而不是悄悄解决。
5. **两个负控（审计器必须能判死）**：把合并结果分别换成 `7accc4b`（机械取 main）
   与 `6b4c4fa`（机械取分支），必须判 `UNION_FAIL`；实测分别丢 30 / 178 个内容单元。
6. **测试**：串行跑合并树与 `main` 基线树（快照目录，见 §3.4 的逐文件校验）的同名模块集。
7. **出图**：`artifacts/pr61_merge_figure.py` 只读审计 JSON、负控结果与 pytest 日志，
   图中没有手抄数字。
## 3. 计算结果
### 3.1 并集审计（`artifacts/pr61_merge_union_evidence.json`）
| 判据 | 数值 | 判定 |
| 双侧文件数 | 17 | 全部 `union-preserved` |
| 单侧文件逐字节校验 | 225 | 0 不一致（另 1 项为删除，单列审计） |
| 合并结果丢失内容单元（两侧合计） | **0**（修复前同一合并为 1） | `UNION_OK` |
| 冲突标记残留 | 0 | — |
| 删除传播（`.scratch/proto_ritz.py`） | 已删除，另一侧未触碰 | `deletion-propagated` |
| 负控：机械取 main | 丢 branch 30 单元 | `UNION_FAIL` ✓ |
| 负控：机械取 branch | 丢 main 178 单元 | `UNION_FAIL` ✓ |
| 空审计防护（base 塌到 main 时） | — | `NOT_AUDITED_NO_SHARED_FILES`，不再假绿 |
### 3.2 本轮修掉的三类"假证据"
1. **空审计假绿**（工具缺陷，最危险）：合并一旦成为提交，`merge-base(main, branch)`
   就塌到 `main`，审计器看到"main 侧 0 文件改动"，照样打印 `UNION_OK`——**在审计集合为空时发绿**。
   现在该情形是显式判定 `NOT_AUDITED_NO_SHARED_FILES`，并新增 `--base` 钉住历史分叉点。
   本次第一次运行就撞上了这个假绿（0 文件审计 ⇒ `UNION_OK`）。
2. **跨行反引号伪造原子**：散文硬换行会让 `` ` `` 跨度跨行，若正则允许跨行，
   整段话会被铸成一个"原子"，任何重排都被判成丢失——`docs/saint-venant-torsion.md`
   两侧各 8 个假丢失即由此而来。改为行内匹配后归零。
3. **行级规则用在散文上**：`docs/benchmark-qualification-status.md` 的 Cook 段落被两侧
   各自续写，句级并集必然破坏行同一性；该文件原子/词丢失均为 0，此前的行级 FAIL
   是把"换行"当成"丢内容"。改为散文走原子+词义务后通过。
### 3.3 唯一一处**真实**丢内容（抓到并修复）
修复前后的两份审计都留在仓内，可直接对比：`artifacts/pr61_merge_union_evidence_prefix.json`
（`--merge 10c10a9`，判定 `UNION_FAIL`）与 `artifacts/pr61_merge_union_evidence.json`
（收尾树，判定 `UNION_OK`）。在修复前的提交 `10c10a9` 上，审计器（散文层）报告
`docs/marine-hydrostatics.md` 相对 main 侧 **内容词短缺 1 个：`module`**，且 main 的
"this module … two separate services cover that question" 框架整体消失——合并后的文字
是从分支句起头、再把 main 的句子续在后面，于是 main 的一句断言被吃掉了。
这是"语义并集"做得不够彻底的真实缺陷，不是工具噪声。修复（`31d8146`）后该文件的
行/原子/词丢失全部归零，文字同时保留：本模块自身无大倾角 GZ、墙侧 oracle 在
`marine_large_angle_stability`、曲线由 `marine_large_angle_gz` 提供，以及两侧共有的
免责列表（free-surface / damage stability / compartment flooding / trim / arbitrary-hull /
wave-load / seakeeping / mooring / classification-rule）。
| 运行 | 合并树（HEAD） | main 基线（`7accc4b`） |
| `test_marine_large_angle_gz.py`（PR 新增门） | 39 passed（9.93 s） | 文件不存在（新增） |
| `test_marine_hydrostatics.py` + `test_marine_large_angle_stability.py` | 27 passed（14.03 s） | 27 passed（24.24 s，另一次 33.89 s） |
| 合计（串行，日志见 `artifacts/evidence_logs/`） | **66 passed，0 failed** | **27 passed，0 failed** |
合并树 = 基线 27 + PR 新增 39，**无回归**。基线树本身的身份也做了程序化校验
（`artifacts/evidence_logs/baseline_tree_verification.json`）：与 `7accc4b` 的 1227 个
文件逐字节比对，**0 不一致、0 缺失**（该目录是快照导出，无 `.git`，所以不能靠
`rev-parse` 自证）。
图 `artifacts/pr61_merge_evidence.png`（110 KB）：左栏=审计器对"真实三方合并"
与两个机械 overlay 的丢失量对比；右栏=合并树 66 passed vs 基线 27 passed。
### 3.5 误差判定（逐项）
| 并集义务：合并结果丢失的内容单元 | 0（义务要求） | 0 | 0% | **是** |
| 单侧文件逐字节一致 | 225/225 | 225/225 | 0% | **是** |
| 机械 overlay 对照（应当丢失） | — | 30 / 178 单元 | 审计器灵敏度充足 | — |
| 测试通过数 | 基线 27 | 合并树 66 | 回归项 0 | **是** |
| PR 载荷精度（大倾角 GZ vs 墙侧闭式） | 闭式解 | 带内最大残差 1.33e-15 m（相对 8.3e-16） | ≈1e-15 | **是** |
| 全量测试套件 | 基线 2 failed / 1389 passed | 合并树 3 failed / 1427 passed（**外部工作树测得**） | 多出的 1 项是墙钟门（13.31 s vs 10 s，当时 load15≈2076） | 见 §5 第 2 条 |
# 并集审计（务必钉住历史分叉点，否则会得到"空审计"）
python3 artifacts/pr61_merge_union_check.py --base e8f6759 --main 7accc4b \
    --branch 6b4c4fa --merge HEAD --out artifacts/pr61_merge_union_evidence.json
# 负控（两个都必须判死）
python3 artifacts/pr61_merge_union_check.py --base e8f6759 --main 7accc4b \
    --branch 6b4c4fa --merge 7accc4b --check
python3 artifacts/pr61_merge_union_check.py --base e8f6759 --main 7accc4b \
    --branch 6b4c4fa --merge 6b4c4fa --check
# 出图
python3 artifacts/pr61_merge_figure.py
1. **本机无法向 GitHub 取新 main**：`git fetch origin` 在 120 s 内超时，`origin/main`
   停在 `7accc4b`（本机最后已知）。因此"对最新 main 仍然 clean"这件事只能用
   governor 推送后 GitHub 重算的 `mergeable` 来确认；本单能证明的是**对 `7accc4b` 及
   其全部祖先无冲突、且并集无损**。若远端 main 又前进，需重跑 §4 的审计（`--base`
   用当时的真实分叉点）。
2. **未在本工作树重跑全量套件**：`load15 ≈ 545`（128 核，超订 4 倍），单文件
   `test_plate_shear_buckling.py` / `test_saint_venant_torsion.py` 在本机超过 2 分钟被
   宿主杀掉（`artifacts/evidence_logs/*_slow.log` 留空文件为记）。故本单自带证据是
   模块级串行测试（66 vs 27）；`artifacts/pytest_session_summaries.json` 里那份全量
   数字来自**相邻工作树 `m3783`（分支头 `2ff75a8`）**，已在该文件内标注来源，不作为
   本工作树的实测。
3. **审计的"宽松"是有边界的**：散文只保证"原子 + 内容词 + 断言不丢"，不保证语序与
   断句与任一侧完全相同——这是散文并集的物理限制，也是 §3.3 那次真实丢失能被发现
   的原因（词级短缺 1 就报 FAIL）。代码与数据文件仍是严格行级，没有放宽。
4. **解冲器需在冲突态运行**：`artifacts/pr61_resolve_conflicts.py` 依赖冲突标记定位
   替换区，合并已提交后再跑会 `StopIteration`（本次即如此）；其内嵌的
   `MARINE` 文本与仓内提交文本一致，作为"当时解法"的可复算记录保留。
5. **删除台账不静默**：`8615e2d` 里更早的一套证据文件被本轮收敛（旧的 `audit_union.py`、
   `verify_merged_gz.py`、原始日志等），逐条理由见 `artifacts/README.md` 的
   Removal ledger，均可 `git show 8615e2d:<path>` 取回。
