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
- 复核/加固作业分支：`agent/m3825`（工作树 `/root/tp-agent/work/m3825`，追加 §7）
- 权威审计证据：`artifacts/pr61_union_audit_evidence.json`（冻结坐标、可复算）、审计器 `artifacts/pr61_union_audit.py`、常驻门禁 `tests/test_pr61_merge_union_audit.py`
- 第一版审计证据：`artifacts/pr61_merge_union_evidence.json`（程序化生成）、
## 7. 复核与加固（`agent/m3825`，2026-11-05 02:15 UTC 起）
本节由后续作业 `agent/m3825` 追加。复核的第一步就发现一个**与内容无关、但与"为什么还是
dirty"有关**的事实：合并提交早已存在（`bafc0fa`，父 `7accc4b`+`2ff75a8`），但 `git fetch
origin` 之后远端**没有** `origin/agent/m3202` —— 也就是说这一次治愈的产物从未推上 GitHub，
GitHub 侧看到的分支头仍是最初的 `2ff75a8`，所以 PR #61 的 `mergeable=false` 并非"上次没修好"，
而是"上次没送到"。远端在本单期间不可达（fetch 静默失败），按纪律不自行 push，交由平台在验收后推送；
本单对本地 `agent/m3202` / `agent/m3825` 两支做了对齐，并补齐可复算证据。
### 7.1 复核结论：合并内容本身成立
复核不复用第一版的结论，而是重新独立判定三件事：
1. **冲突面**用 `git merge-tree`（不与第一版共享任何代码路径）复算，仍只有
   `CHANGELOG.md` 与 `docs/marine-hydrostatics.md` 两处 content 冲突，与"手解集"一致 ——
   这条同时封死了"把 git 本可自动合并的文件也声称手解、从而掩盖丢内容"的口子；
2. **243 个变更文件**逐一核对：合并后的 blob 必须**等于**该侧 blob（单侧文件）或属声明的手解集，
   否则判失败。结果是 243/243 精确命中，main 侧与分支侧内容**字节级未变**；
3. **手工解冲的两个文件**保留两侧全部原子（多重集）与全部内容词（集合），丢失量 0/0。
### 7.2 复核中修掉的三个"证据级"缺陷
| # | 缺陷 | 后果 | 修法 |
| 1 | 第一版审计器把 ref 名写死（`origin/main`、`agent/m3202`） | 分支指针一旦指向合并提交，`merge-base(main, agent/m3202)` 退化成 main 自身，审计变成"0 个冲突文件"的**假通过**，且证据 JSON 不可复现 | 新版 `artifacts/pr61_union_audit.py` 把 PR 坐标冻结为常量（可 `--base/--main/--branch/--merge` 覆盖），证据里记全 5 个 SHA |
| 2 | 第一版图脚本的"取分支"负控用 `--merge agent/m3202` | 分支指针一动，负控变成"合并结果 vs 合并结果"，**永远通过**——量具失效但图上仍显示"负控已fail" | 新版 `artifacts/pr61_union_figure.py` 直接 import 审计模块、对冻结 SHA 求值，并把"删掉一行"负控一并画出 |
| 3 | 第一版对内容词用**多重集**判定 | 合法改写被判成丢词：main 侧 `It is not a large-angle GZ curve` 与分支侧句子在并集时合并为 `Beyond those two services this module offers no large-angle GZ, ...`，词元 `curve` 出现次数 2→1，被误报 | 改为**原子用多重集**（数字/标识符/路径/链接一次都不许少）+ **内容词用集合**（允许重排改写，不许整词消失），并在本节写明该粒度取舍 |
负控是这套审计的"量具校准"，实测三档（见 7.4 表与 `artifacts/pr61_union_audit.png`）：
取 main 单侧丢分支 156 个内容单元、取分支单侧丢 main 479 个、删掉一行丢 3 个；
而交付的三方合并丢 0 —— 说明"零丢失"不是因为审计太松。
### 7.3 一处既有缺陷的澄清（不是本次合并造成的）
合并后 `CHANGELOG.md` 第 21/23 行出现"`## 2026-11-03 Cook 膜应力审计证据收尾` 空节、
紧跟 `## 2026-11-04 — plane-strain B-bar Q4`"的结构。核对 `git show 7accc4b:CHANGELOG.md`
第 3/5 行，**main 侧本身**就是这个结构 —— 属既有文档欠账，本次合并未新增结构性破坏。
按最小改动原则不动它，仅在此记录，留给后续文档整理单。
### 7.4 生成表格（`artifacts/pr61_report_tables.py` 程序化注入，零手抄）
<!-- BEGIN GENERATED: pr61-audit-tables -->
#### 并集审计（权威记录：`artifacts/pr61_union_audit_evidence.json`）
| `main`（合并目标） | `7accc4b9cfb45369958c432a8aa2f0ebbb846a5b` |
| `agent/m3202`（合并前 PR 头） | `2ff75a8158492b1d2232348d1a1d8e002ce1c009` |
| 合并提交（三方合并结果） | `bafc0faf2b438de53fb2da52e8478997687c69ad` |
| 分支当前 tip（含本次证据提交） | `00813c78ce476d38ce3017d5e62130756c1f9108` |
| main 侧变更文件 | 236 |
| 分支侧变更文件 | 9 |
| 两侧同时改写（= 冲突面） | 2 |
| 仅 main 侧 | 234 |
| 仅分支侧 | 7 |
| 合并相对 merge-base 的变更文件总数 | 243 |
| 按“单侧 blob 原样/手工解冲”核对通过 | 243 |
| 合并提交之后新增（仅追加，无改动/删除） | 8 |
| `git merge-tree` 独立判定的冲突文件 | `CHANGELOG.md`, `docs/marine-hydrostatics.md` |
| 越界文件（两侧都没有的新路径） | 0 |
| main 是否为合并提交祖先（PR 可快进） | True |
| 手工解冲文件 | main 独有原子 | 分支独有原子 | 丢失原子(main/分支) | 丢失内容词(main/分支) | 判定 |
| `CHANGELOG.md` | 200 | 34 | **0 / 0** | **0 / 0** | union-preserved |
| `docs/marine-hydrostatics.md` | 18 | 4 | **0 / 0** | **0 / 0** | union-preserved |
| 负控（量具校准） | 预期 | 实测 |
| main-only overlay | 必须被拒 | control-fires |
| branch-only overlay | 必须被拒 | control-fires |
| single-line deletion: 1.33e-15 m，相对 8.3e-16）；90° 精确几何 | 必须被拒 | control-fires |
| 树 | 来源 | tests | passed | failed | skipped |
| 合并树（main `7accc4b` + PR #61） | 归档会话汇总行（本次会话未落 XML） | 1430 | **1427** | 3 | 6 |
| 基线 `main` `7accc4b` | 归档会话汇总行 | 1391 | **1389** | 2 | 6 |
<!-- END GENERATED: pr61-audit-tables -->
### 7.5 测试证据（本次会话，2026-11-05）
- **PR 载荷自身**：`tests/test_marine_large_angle_gz.py` + `tests/test_marine_large_angle_stability.py`
  → `58 passed`（6.98 s）—— 合并没有破坏 PR 要交付的物理能力。
- **合并内容常驻门禁（本次新增）**：`tests/test_pr61_merge_union_audit.py` → `6 passed`。
  它把并集审计变成仓库里的测试：合并满足并集义务、审计器必须拒绝两个单侧 overlay 与单行删除、
  以"改一个测量数字"验证篡改可被察觉、仓内证据 JSON 必须与现场复算一致（防手改）。历史里
  若不含 PR #61 的提交（例如无分支历史的导出），这些测试 skip 而不是 fail。
- **全量**：见 7.4 表（同一命令 `-n 8`，同一台机）。本次会话在 load average 6000+ 的共享机上
  重跑全量到 91% 时预算耗尽、未落 JUnit XML，故统计仍取同日归档会话（诚实标注，不混用两次数据）。
### 7.6 未完成与局限（如实）
1. **没有执行真实 push**：远端不可达（`git fetch origin` 无输出、无 `origin/agent/m3202`），
   按纪律由平台在验收后推送。本单完成的是"本地两支对齐 + 可复算证据 + 常驻门禁"。
   记录：`agent/m3202` 旧头 `2ff75a8`（PR 原始头）→ 合并提交 `bafc0fa` → 本次 tip 见 7.4 表。
2. **基线跑分沿用同日归档**（main `7accc4b` 的 `1389 passed / 2 failed` 会话汇总行；原始
   `/tmp` 日志已被邻居任务覆盖）。本次会话只重跑了合并树。公平性用结构论证补强：合并树
   = main 树 + 分支新增，243 个变更文件逐一核对到精确 blob（main 侧字节不变），因此 main 的
   测试结果不可能被合并不利影响；此外分支侧 9 个文件里 `src/tensorfem/` 只有**新增**
   `marine_large_angle_gz.py`，没有改动任何既有模块。
3. **墙钟型断言在共享机上不可用于归因**（load average 6000+）：`elapsed < 10 s` 一类判据的
   结论只能靠"源码零差异 + 基线上同样失败"两条证据，不能写进回归统计，见 §5.5。
