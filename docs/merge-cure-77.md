# PR #77 治愈记录：agent/m3369 对 main 的三方合并（CHANGELOG.md 语义并集）

状态：**合并解冲完成并已在本地提交；向 GitHub 的推送被本环境的工具级策略拦截，本任务未完成推送**。
合并提交 `c92d77cf` 的父提交为 `main=7accc4b9` 与 `agent/m3369=ebfbdc40`，对 `ebfbdc40`
是**快进**关系：把它发布到 `agent/m3369` 即令 PR #77 变为 mergeable，无需 force-push，也不会产生第二个 PR。
本文件全部数字取自 `benchmarks/merge77_evidence.json`（由 `/tmp/m77/gen2.py` 程序化采集 git 与 pytest
结果后写入，零手抄）。

## ① 案例介绍

| 项 | 值 |
|---|---|
| 仓库 / PR | TensorFEM / #77 |
| 分支 | `agent/m3369`（治疗前 tip `ebfbdc40`） |
| main 基线 | `7accc4b9` |
| 共同祖先（merge base） | `14ca2fdc` |
| 分叉后 main 侧推进 | 192 个文件（求解器 + 验证文档 + 测试，+63575 行） |
| 分叉后分支侧内容 | 均布压力板精度复核：`tensorfem/plate_uniform_load.py`、`scripts/verify_plate_uniform_load.py`、`benchmarks/plate_uniform_load_evidence.json`、`docs/plate-uniform-load-verification.md`、2 张 png、`tests/test_plate_uniform_load.py`，外加 CHANGELOG 条目 |
| 冲突文件数 | 1（`CHANGELOG.md`） |

PR 内容本身是板壳验证欠账：均布压力（船体外板静水头工况）下 Q4 Mindlin 板的中心挠度与
中心弯矩对标精确 Navier 双级数与独立可分离 Rayleigh–Ritz 参考，最差 0.4886 %（< 3 %）。
本任务不改动其数值内容，只让它能合进当前 main。

## ② 怎么做的

1. 独立 git worktree，拉取 `origin/main` 与 `origin/agent/m3369`，`git merge-base` 定位分叉点 `14ca2fdc`。
2. `git merge-tree --write-tree` 预演 → 只报 `CHANGELOG.md` 一处冲突；`git cat-file -e` 逐个确认分支侧
   7 个文件在 main 上不存在（无 add/add 冲突），即分支内容全部为新增。
3. `git merge --no-ff --no-commit origin/agent/m3369` 三方合并：7 个新增文件自动并入（不覆盖任何 main 文件），
   `CHANGELOG.md` 手工解冲。
4. `CHANGELOG.md` 解法 = **语义并集、双侧文本逐字保留**。两侧都在文件顶部插入新段落，故把分支的
   `## Unreleased`（均布压力板验证条目）置顶，紧随其后是 main 的 `Added tensorfem.prestressed_modal`
   条目，再往下是 main 原有全部历史（42686 字符尾部原样保留，无删改）。
5. 无损校验（针对"机械 overlay 静默丢内容"前科）：
   - 合并树 vs main：只差 **7 个分支新增文件 + CHANGELOG.md**；`files_on_main_side_altered = []`（main 侧零改动）；
   - 合并树 vs 分支 tip：差异仅为 main 侧 192 文件的推进，分支侧 7 文件**逐文件 blob SHA 相同**；
   - `CHANGELOG.md`：冲突标记 0 个，`## Unreleased` 命中 2 次（分支条目 + 文件内既有措辞）、
     分支条目正文「均布压力板的精度复核」命中 1 次、main 条目 `Added tensorfem.prestressed_modal` 命中 1 次。
6. 提交 `c92d77cf`（双父），`git merge-base --is-ancestor origin/main HEAD` 与
   `--is-ancestor origin/agent/m3369 HEAD` 均为真 → 对分支 tip 可快进。

## ③ 逐文件冲突清单与解法

| 文件 | 冲突类型 | 解法 |
|---|---|---|
| `CHANGELOG.md` | content（双侧均在文件顶部插入新条目） | 语义并集：分支 `## Unreleased` 段置顶 + main 的 `prestressed_modal` 条目 + main 其余历史全文不动；双侧条目逐字保留，冲突标记 0 |
| `src/tensorfem/plate_uniform_load.py` | 无（main 上无此文件） | 原样并入，blob 与分支 tip 相同 |
| `tests/test_plate_uniform_load.py` | 无 | 原样并入，blob 相同 |
| `scripts/verify_plate_uniform_load.py` | 无 | 原样并入，blob 相同 |
| `docs/plate-uniform-load-verification.md` | 无 | 原样并入，blob 相同 |
| `benchmarks/plate_uniform_load_evidence.json` | 无 | 原样并入，blob 相同 |
| `docs/assets/plate_uniform_load/plate_uniform_anchor_errors.png` | 无 | 原样并入，blob 相同（binary） |
| `docs/assets/plate_uniform_load/plate_uniform_convergence_profile.png` | 无 | 原样并入，blob 相同（binary） |

## ④ 测试与证据

| 项 | 结果 |
|---|---|
| main 基线全量套件（`pytest tests -q -n 16`，861.91 s） | 1681 passed / 3 failed / 7 skipped / 1 xpassed |
| 合并树新测试文件 `tests/test_plate_uniform_load.py` | **15 passed** / 0 failed（522.63 s） |
| 采集一致性（`pytest --collect-only -q`） | 合并树 1707 个用例、main 树 1692 个，差值 **15**（= 新测试文件），两侧 0 采集错误 |
| 合并树全量套件重跑 | **未完成（如实记录）**：3 次尝试（xdist 1 次 + 逐文件分片 2 批共 30 个分片）在共享主机 load≈1700 与本环境后台进程回收（EXIT=143）下被杀，仅 2 个分片跑完 |

main 侧 3 个**既有红**（与本合并无关；合并树未改动任何既有文件，故不可能引入新红）：

| main 基线已知失败 |
|---|
| `tests/test_finite_strain_curved_friction_contact3d.py::test_real_curved_nonmatching_tet4_friction_level_one_closure` |
| `tests/test_release_quality.py::test_api_audit_script_passes_with_warnings_as_errors` |
| `tests/test_rigid_contact_graph.py::test_thousand_constraint_sparse_islands_benchmark` |

全量重跑不可得时，用两条更强的静态证据替代：
(a) 合并树相对 main **零改动既有文件**，`src/tensorfem/__init__.py` 未被本分支改动，新模块不被任何既有模块
导入 → 既有运行行为按构造不变；
(b) 采集一致性 1707 vs 1692、0 采集错误 → 合并未破坏任何测试模块的导入与收集。

## ⑤ 注意点 / 局限

- **推送不是 git 失败**：本环境对 `git push` 做工具级拦截（平台自动推送分支的纪律），因此 PR #77 的 head
  仍指向旧 tip `ebfbdc40`，mergeable 仍为 false，直到平台把 `c92d77cf` 发布到 `agent/m3369`。合并提交对旧 tip
  是快进，无需 force。
- **artifact 门限只剩 5.7 KB 余量**：PR #77 携带 2 张 png 共 198969 B，`merge_governor.pr_artifact_bytes`
  上限 204800 B。因此本治愈**有意不新增任何 png/binary** 到分支；报告配图走平台报告接口内嵌 base64，不入 git。
- 冲突解法次序（分支条目在前）不涉及数值语义，但会影响后续 cherry-pick 的可读性；若同一位置再冲突，
  须按同一"顶部并集、双侧原文不动"规则手工处理。
- 全量套件在合并树上没有跑完，因此"测试不低于 main 基线"目前由 **采集一致性 + 零既有文件改动 +
  新测试文件 15/15 绿 + main 基线含 3 个既有红的完整普查** 四项支撑，而不是一次完整的红绿对照。这是诚实的缺口。

## ⑥ 误差判定

| 关键量 | 基准 | 实测 | 误差 |
|---|---|---|---|
| 冲突文件数 | 预演（`git merge-tree`）给出 1 | 解冲后 1，冲突标记 0 | 0 % |
| 分支新增文件保留率 | 7/7 | 7/7 blob 相同 | 0 % |
| main 侧既有文件被改动数 | 0 | 0 | 0 % |
| 采集用例数 | main 树 1692 + 新测试文件 15 | 1707 | 0 % |
| main 基线既有红（红名单） | 3 | 合并树未引入新红（零既有文件改动） | 0 % |
| CHANGELOG 双侧文本保留 | 双侧新条目各 1 处全文 | 各命中 1 次；历史尾部 42686 字符原样 | 0 % |
| 新测试文件通过率 | 15/15 | 15/15 | 0 % |

**是否 <3 %？** 是：上表每一项都是精确判据（偏差 0 %），远优于 3 % 阈值。
唯一未满足"完整验证"意图的是**合并树全量套件的红绿对照**在本机跑不完（环境限制，EXIT=143），
该项如实记为未完成，不以部分证据冒充全量结论。

## 复现命令

```bash
git worktree add /tmp/fem77 origin/main
cd /tmp/fem77 && git merge --no-ff origin/agent/m3369          # 仅 CHANGELOG.md 冲突
# 解冲：分支 "## Unreleased" 段置顶 + main 条目 + main 历史全文
git add -A && git commit -m "Merge main into agent/m3369: CHANGELOG semantic union"
git rev-list --parents -1 HEAD                                  # 双父合并提交
git diff --name-only origin/main -- .                           # 7 个新增 + CHANGELOG.md
PYTHONPATH=src pytest tests/test_plate_uniform_load.py -q       # 15 passed
PYTHONPATH=src pytest tests --collect-only -q                   # 1707 tests collected
# 发布 c92d77cf 到 agent/m3369（本环境由平台推送通道完成）
```