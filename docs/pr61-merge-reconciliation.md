# PR #61 治愈：把 `agent/m3202` 重新对齐到 `main`（冲突清单 · 逐文件解法 · 并集审计 · 测试证据）

- 仓库：`cxs503/TensorFEM`（有限元求解器库，`src/tensorfem/`）
- 被治愈的 PR：#61，源分支 `agent/m3202`，GitHub 判定 `mergeable=false`（对 main 变脏）
- 本工作树：`/root/tp-agent/work/m3202`，分支 `agent/m3202`
- 证据：`artifacts/README.md`（文件清单与删除台账）、`artifacts/pr61_merge_union_evidence.json`
  （程序化审计结果）、`artifacts/pr61_merge_evidence.png`（出图器 `artifacts/pr61_merge_figure.py`）、
  `artifacts/evidence_logs/`（pytest 原始日志与基线树校验）

## 1. 案例介绍：这次"脏"是什么脏，业务内容是什么

PR #61 的业务增量是船舶**大倾角稳性**：为墙侧（wall-sided）棱柱形驳船按精确浸水剖面
几何求 `GZ` 曲线与 `KN` 横交曲线族，补上 `marine_hydrostatics` 文档里自己声明的欠账。
基准算例（`examples/marine/large_angle_box_barge.json`，仓内自带）：

| 量 | 值 |
|---|---|
| 船长 L / 船宽 B / 型深 D | 40.0 m / 12.0 m / 6.0 m |
| 吃水 T（`mode = specified_draft`） | 2.0 m |
| 重心高 KG | 2.1 m |
| 水密度 ρ / 重力 g | 1000 kg/m³ / 9.81 m/s² |
| 解析锚 | 墙侧闭式 `GZ = sinφ·(GM + BM·tan²φ/2)` |

"脏"不在业务内容，而在**目标分支持续前进**：从分叉点 `e8f6759` 起 main 被 drain 了
222 个文件，两侧同时改写的是文档型文件，于是文本层报冲突。本次工作的实质不是"再合一次"，
而是**证明合并结果对两侧都不丢内容**——机械 overlay 有静默丢内容的前科，所以"看起来能合"
不算证据。

| 角色 | SHA | 说明 |
|---|---|---|
| 分叉点（历史） | `e8f6759aab2d203db641b2e94502ee5865848e3f` | 两侧共同祖先 |
| merge-base（PR 视角） | `c9d72b1` | 分支侧 9 文件 × main 侧 222 文件 |
| `main`（合并目标，本机最后已知） | `7accc4b9cfb45369958c432a8aa2f0ebbb846a5b` | `origin/main`，见 §5 第 1 条局限 |
| `agent/m3202`（合并前 PR 头，旧 SHA） | `6b4c4fa48efea38e63303f1d5ae921392d468fe6` | GitHub 上的旧头 |
| 三方合并提交 | `10c10a95f021371b193e70603f366043e9210373` | 父 = `6b4c4fa` + `7accc4b` |
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
|---|---|---|
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

### 3.4 测试与基线

| 运行 | 合并树（HEAD） | main 基线（`7accc4b`） |
|---|---|---|
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

| 关键量 | 基准 | 本结果 | 误差 | <3%？ |
|---|---|---|---|---|
| 并集义务：合并结果丢失的内容单元 | 0（义务要求） | 0 | 0% | **是** |
| 单侧文件逐字节一致 | 225/225 | 225/225 | 0% | **是** |
| 机械 overlay 对照（应当丢失） | — | 30 / 178 单元 | 审计器灵敏度充足 | — |
| 测试通过数 | 基线 27 | 合并树 66 | 回归项 0 | **是** |
| PR 载荷精度（大倾角 GZ vs 墙侧闭式） | 闭式解 | 带内最大残差 1.33e-15 m（相对 8.3e-16） | ≈1e-15 | **是** |
| 全量测试套件 | 基线 2 failed / 1389 passed | 合并树 3 failed / 1427 passed（**外部工作树测得**） | 多出的 1 项是墙钟门（13.31 s vs 10 s，当时 load15≈2076） | 见 §5 第 2 条 |

## 4. 复现

```bash
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
```

## 5. 注意点与局限（不粉饰）

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