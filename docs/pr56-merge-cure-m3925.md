# PR #56（agent/m3148）治愈记录：三方合并语义并集 + 双侧保留审计 + 测试基线对表

任务：TensorFEM 仓库 PR #56 对 `main` 变脏（`mergeable=false`）。本记录给出**冲突清单、逐文件解法、双侧内容保留审计、测试证据与门禁结论**，所有数字由 `scripts/merge_cure_pr56_evidence.py` 从 git 历史、保留审计 JSON 与 pytest 日志程序化生成，未手抄。

## 1. 案例介绍（分支几何）

| 项 | 值 |
|---|---|
| 仓库 | TensorFEM（本地远端 `tensorfem_local` = `/root/tensor-ecosystem/TensorFEM`） |
| PR | #56，源分支 `agent/m3148` |
| merge base | `7accc4b9cfb45369958c432a8aa2f0ebbb846a5b` |
| `main` 基线 | `7a8f4afeb6dd79c2ad3be480a6117ed2550c2347`（base 之后 6 个提交） |
| 分支旧尖端 | `de6195a4349bf31c615cbbbccccdefe5fc458433`（base 之后 7 个提交） |
| 冲突面 | 主定三方合并（`git merge-tree` 预演）只有 2 个内容冲突，其余 32 个路径自动合并 |

分支侧交付内容（本 PR 主体）：Cook 膜应力恢复算子进入生产入口 `cook_stress_recovery.recovered_nodal_stress` + fail-closed 门禁 `qualify`、`additional_benchmark_fields.cook_fields(include_recovery=True)`、审计过的收敛证据与 7 项新测试。
`main` 侧在 base 之后新增：板面内线性变化压缩屈曲（`plate_bending_buckling.py`、独立差分条路线、经典系数 k=4.0/7.81/23.9/25.6 校验、图与资格文档）、预压模态（`prestressed_modal`）与 PR #72 的治愈档。

## 2. 怎么做的（方法与关键设置）

1. **工作树隔离**：仓库内 `git worktree`（合并态）与独立基线工作树分离，基线工作树只用于跑 `main` 的原始测试，不动合并工作树。
2. **合并方向**：`main` → `agent/m3148`（因为验收要求成果回写原 PR 分支，`main` 侧可用快进收编）。
3. **冲突解法不是机械 overlay**：两个冲突文件分别走两种策略，并带**机器守卫**（`artifacts/resolve_pr56_conflicts.py`）：
   - `CHANGELOG.md`：两侧各自在**同一锚点**前置了一个新章节。解法=两章节**全部保留**（分支 11-04 Cook 生产化在前、main 11-03 板屈曲在后），仅删三条冲突标记 → 两侧新增行 100% 进入结果。
   - `docs/benchmark-qualification-status.md`：两侧**重写了同一段**「尚未闭环的工作」。纯拼接会出现互相矛盾的两段，故做**语义并集**：以 main 的新段为骨架（含线性变化预屈曲压缩已闭环、欠账收窄到「面内沿 x 变化/非线性预屈曲分布」），把分支「恢复场已进入生产、带 fail-closed 门禁」的分句替换进去。守卫脚本在写入前断言必须存在的关键事实（`cook_stress_recovery.qualify`、`include_recovery=True`、`marine-plate-linear-stress-buckling-validation.md`、`面内应力沿 x 变化或非线性的预屈曲分布`、`5.09%`），缺任一即拒写。
4. **双侧保留审计**（`artifacts/audit_merge_retention.py`）：对 base→两侧每一路径逐行做**多重集差分**，统计「该侧新增行」在合并结果中是否仍存在；二进制文件按存在性核对；两个被重写的文件降到**句级**粒度（按 `；。` 切分）逐句核对，只有被对方**显式取代**的陈旧论断才允许不出现，且取代理由必须写进配置表。结果落盘 `artifacts/pr56-merge-audit.json`。
5. **测试对表**：同一命令在合并态与 `main` 基线工作树各跑一次全量 `tests/`（`PYTHONPATH=src`, `OMP_NUM_THREADS=1`, `MPLBACKEND=Agg`, `pytest -v --tb=line -n 32 --timeout=900`），比较通过/失败/跳过计数与**失败用例集合**。

## 3. 计算结果

### 3.1 冲突文件与逐文件解法

| 冲突文件 | 两侧改动 | 解法 | 保留校验 |
|---|---|---|---|
| `CHANGELOG.md` | 双方各加 1 个日期章节（13 行 / 12 行） | 双章节全保留，仅删标记 | 两侧新增行 100% 在位 |
| `docs/benchmark-qualification-status.md` | 双方各重写同段（937 字符 / 935 字符） | 语义并集（main 骨架 + 分支生产化分句） | 5 条关键事实断言通过；2 条陈旧论断显式取代 |

被显式取代的 2 条论断（本轮唯一的内容裁决，均记录在审计 JSON）：

1. main 侧「…恢复场尚未进入生产 `cook_fields`，故 `stress_accuracy_qualified` 保持 False」——被分支的生产化取代（恢复场已进 `cook_fields` 并有门禁；`stress_accuracy_qualified` 仍保持 False，因为**原始单元中心场未变**，语义未失真）。
2. 分支侧欠账项「非均匀预屈曲应力」——被 main 的线性变化预屈曲压缩闭环取代，欠账收窄为「面内应力沿 x 变化或非线性的预屈曲分布」。

### 3.2 双侧保留审计（34 个路径 = 27 文本 + 7 二进制）

| 指标 | 值 |
|---|---|
| 审计路径数 | 34（main 独有新增 19、分支独有新增 12、双方同改 2、分支独改 1） |
| main 侧新增行 | 3750 |
| 分支侧新增行 | 1285 |
| **静默丢失行** | **0** |
| 句级审计（重写文件） | 缺失句 0；显式取代 2 句（理由见 3.1） |
| 二进制文件 | 7 个全部在位（哈希清单由 git 保证） |

### 3.3 测试证据（全量 `tests/`，1729 用例）

| 指标 | `main` 基线（7a8f4af） | 合并态（PR #56 cure） | 判定 |
|---|---|---|---|
| passed | 1712 | **1719** | +7（分支新增 `tests/test_cook_stress_recovery.py` 7 项），不低于基线 |
| failed | 2 | 2 | 失败集合**完全相同**，非回归 |
| skipped | 7 | 7 | 一致 |
| xpassed | 1 | 1 | 一致 |
| 墙钟 | 735.86 s | 431.94 s | 机器负载不同，时长不作比较 |
| 摘要行 vs 逐条记录交叉核对 | 一致 | 一致 | `summary_matches_verbose=true` |

两侧共同的 2 个既存失败（在 `main` 上就失败，与本合并无关）：

- `tests/test_release_quality.py::test_api_audit_script_passes_with_warnings_as_errors`
- `tests/test_finite_strain_curved_friction_contact3d.py::test_real_curved_nonmatching_tet4_friction_level_one_closure`

### 3.4 门禁

| 门禁 | 结果 |
|---|---|
| 无静默内容丢失（`dropped_total == 0`） | ✅ |
| 不低于 main 测试基线（passed ≥、failed ≤ 且失败集合相同） | ✅ |
| 执行用例数不低于基线（1721 vs 1714） | ✅ |
| 摘要与逐条记录一致 | ✅ |

## 4. 结果图

- `docs/assets/merge_cure_pr56/retention_audit.png`：按路径的**双侧新增行数**（蓝=main，橙=分支）与**合并后仍在位的行数**（黑点）。所有黑点与柱顶重合，即 34 个路径无一行丢失。
- `docs/assets/merge_cure_pr56/test_parity.png`：合并态 vs main 基线的 passed/skipped/xfailed/xpassed/failed 逐类比，标注比值；passed 比值 1.0041，failed 比值 1.0000。

两图均由 `scripts/merge_cure_pr56_evidence.py` 从审计 JSON 与 pytest 日志直接绘制（matplotlib 3.10.8，Agg 后端），无手工输入。

## 5. 注意点（踩坑与局限）

- **xdist 收尾崩溃**：首轮 `pytest -q -n 16` 在两侧都于 `pytest_sessionfinish` 抛出 `OSError: cannot send (already closed?)`（Pluggy 旧式 hookwrapper 告警），进程结束但**没有摘要行**，日志仅留在 `artifacts/pytest_{merge,main_base}.log`。改用 `-v` 重跑：逐条 `PASSED/FAILED/SKIPPED` 记录 + 完整摘要都拿到，并用逐条计数交叉校验摘要（`summary_matches_verbose=true`），避免「无摘要=无结论」。该现象是测试框架收尾噪声，两侧对称出现，与本合并改动无关。
- **参数化用例名含空格**：逐条记录的测试 ID 正则不能按空格切分，否则 `[override0-two deformable]` 一类 ID 会被误判为「未执行」。
- **`.scratch/` 未被 `.gitignore` 覆盖**：本仓内建基线工作树若放在 `.scratch/` 下，`git add -A` 会误纳整棵工作树；本轮基线工作树用后即删，提交按显式路径逐个添加，未做目录级 add。
- **内容裁决只有两处**，都是「一方的声明被另一方的工作取代」，且两侧都留痕于审计 JSON；没有任何一侧的新增/删除被静默丢弃。
- **未做**：不重跑被跳过（7 个 skip）与既存失败的用例去修因——它们与 main 一致，属既有欠账，不属本单范围；不重新认证 PR #56 涉及的 Cook 恢复精度（该精度证据由分支自带，本单只验证「合并不丢不坏」）。

## 6. 误差判定（显式回答）

本单是**合并治愈**，不产生新的物理量，因此没有「解析解/实验值」可比。等价的可机器裁决基准是**集合与计数恒等式**，其允许误差为 0，比 3% 严格得多：

| 关键量 | 基准 | 实测 | 误差 | <3% ? |
|---|---|---|---|---|
| 静默丢失行数 | 0（硬门） | 0（共审 5035 行新增） | 0 | ✅ 精确成立 |
| 缺失句数（重写文件） | 0 | 0（另 2 句显式取代并留痕） | 0 | ✅ 精确成立 |
| passed 数 vs main 基线 | ≥ 1712 | 1719 | +0.41%（正向） | ✅ |
| failed 集合 | 与 main 相同 | 相同（2=2，同名） | 0 | ✅ 精确成立 |
| 执行用例数 | ≥ 1714 | 1721 | +0.41%（正向） | ✅ |

## 7. 复现

```bash
# 合并与冲突解法（在 agent/m3148 检出上）
python3 artifacts/resolve_pr56_conflicts.py
# 双侧保留审计
python3 artifacts/audit_merge_retention.py
# 测试（两侧各一次，-v 保证逐条记录不依赖 sessionfinish）
OMP_NUM_THREADS=1 PYTHONPATH=src pytest tests -v --tb=line -n 32 --timeout=900
# 证据与图
MPLBACKEND=Agg python3 scripts/merge_cure_pr56_evidence.py
```