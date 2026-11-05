# PR #70 治愈记录：`agent/m3297` → `main` 三方合并解冲

本文件是一次**合并治愈**的可审计记录：冲突清单、逐文件解法、语义并集依据、无丢失校验与测试证据。
不含任何求解结果的重算，只处理「分支对 main 变脏」这一工程问题。

## 1. 三方状态（合并前）

| 角色 | 引用 | SHA | 说明 |
|---|---|---|---|
| 合并基 | `14ca2fd` | `14ca2fdc1fd06467f3edb33db1d9e549b6a901a2` | Merge #51 (m3098) via merge-governor |
| 目标 | `origin/main` | `7accc4b` | Merge PR #73（owner 手动 drain） |
| 来源 | `origin/agent/m3297` | `092f326` | PR #70，1 个提交，10 个路径 |
| 合并提交 | 本分支 HEAD | 见 §6 | `main` 之上无 `main` 侧回退 |

`main` 自合并基起领先 52 个提交（192 个文件、+63575/−65），分支领先 1 个提交。GitHub 报 `mergeable=false`，
`git merge-tree` 复现：**只有 1 个文件真冲突**，其余 9 个路径可自动合并。

## 2. 冲突清单与逐文件解法

| 文件 | 合并类型 | 解法 |
|---|---|---|
| `docs/benchmark-qualification-status.md` | **CONFLICT (content)，1 个 hunk（第 94–98 行）** | 手工**语义并集**，见 §3 |
| `src/tensorfem/plate_shear_buckling.py` | 自动（仅分支侧改动，main 未触碰） | 接受分支版本；见 §4 |
| `src/tensorfem/plate_bending_buckling.py` | 自动新增（697 行） | 接受分支版本，逐字节相同 |
| `tests/test_plate_bending_buckling.py` | 自动新增（242 行） | 接受分支版本，逐字节相同 |
| `benchmarks/plate_bending_buckling.py` | 自动新增 | 接受分支版本 |
| `benchmarks/plate_bending_buckling_evidence.json` | 自动新增 | 接受分支版本 |
| `docs/plate-bending-buckling-validation.md` | 自动新增（143 行） | 接受分支版本 |
| `docs/assets/plate_bending_buckling_{mode,reference,sweep}.png` | 自动新增 3 个二进制 | 接受分支版本（47068/55446/41566 B） |

## 3. 冲突段落：逐句合成表

冲突段落是 `## 尚未闭环的工作` 下的单行长段落，两侧本是**同一句话的不同演进**，因此冲突粒度极细。
合成规则：**每一句都必须在合成结果中出现；状态性句子取「证据更新」的一侧，并在下文标注取舍理由。**

| 子句 | base `14ca2fd` | main `7accc4b` | 分支 `092f326` | 合成结果 |
|---|---|---|---|---|
| Cook 膜 5.02% 残差归因 + `stress_accuracy_qualified=False` | 有 | 有 | 有 | 保留（三方一致） |
| 开孔半球壳应力参照状态 | 「仍缺独立应力参考」 | **HEX20 oracle 已补并否定**（5.09%/9.00%）+ oracle 自标定（Lamé 8.6e-7、薄板 0.26%） | 句与 base 同 | 取 **main**：base 句被 main 明确取代，分支未提供新信息 |
| 板屈曲：单轴压缩 | 已闭环 | 已闭环 | 已闭环 | 保留 |
| 板屈曲：面内剪切/压-剪（k_s=9.34、5.34，0.17%/2.6%） | 有 | 有 | 有 | 保留 |
| 板屈曲：**非均匀预屈曲（纯面内弯曲）** | 列为「仍缺」 | 列为「仍缺」 | **已闭环** k_b=23.8818 vs 23.9（0.076%），另有 Hermite C1 交叉印证 4.97e-5 | 取**分支**：新增闭环，并**从「仍缺」清单中删除该项**（唯一一处对 main 措辞的删改，理由见 §5） |
| 仍缺：壳单元几何刚度、Hertz 接触、其他经典壳体案例 | 有 | 有 | 有 | 保留（除上一行已闭环项） |
| 两个 1.0 阻塞项 / 未提升版本 | 有 | 有 | 有 | 保留（三方一致，未冲突区） |

分支侧**没有**提出对抗 main 的断言：两处差异（半球 oracle 句、弯曲屈曲闭环句）方向一致，
因此并集是唯一无信息损失的解，不存在需要裁决的矛盾。

## 4. 分支对共享求解器的修正（自动合并，但需说明）

`src/tensorfem/plate_shear_buckling.py::_smallest_positive_from_symmetric_indefinite` 原先把
`L Lᵀ = K_b` 的铅笔变换写成 `L⁻ᵀ G L⁻ᵀ`（`solve(factor.T, ...)`），只有 `L` 为对角时（等/正交各向同性）
才与正确的合同变换 `L⁻¹ G L⁻ᵀ` 相同；特征向量回代 `φ = L⁻¹ y` 也应为 `L⁻ᵀ y`。
分支把两者改为合同变换，目的是让新增的 Hermite C1 参考（刚度非对角）不得出错误特征值。

风险面已核：函数只有 3 个调用点——`plate_shear_buckling.py:266`（等/正交各向同性，行为不变）、
`plate_bending_buckling.py:314/568`（本分支新增）。`main` 侧无新增调用者，也无断言旧行为的测试。
原有剪切锚点（k_s=9.34、5.34）由 `tests/test_plate_shear_buckling.py` 与
`test_plate_bending_buckling.py::test_shear_module_anchors_are_unchanged` 双向锁死，实测全通过（§5）。

## 5. 无丢失校验（机器判据，非自述）

```bash
git diff --diff-filter=D --name-only origin/main  HEAD | wc -l   # 0  ← 未删 main 任何文件
git diff --diff-filter=D --name-only origin/agent/m3297 HEAD | wc -l   # 0  ← 未删分支任何文件
git ls-tree -r --name-only origin/main | wc -l    # 1227
git ls-tree -r --name-only HEAD        | wc -l    # 1235 = 1227 + 分支新增 8 个新路径
git diff --stat origin/agent/m3297 HEAD           # 只有 main 的 52 提交（分支内容一字未失）
git diff --stat origin/main HEAD                  # 恰好是分支那 10 个路径，无夹带
# 分支新增/修改的 9 个非冲突路径，逐文件与 origin/agent/m3297 逐字节相同：
git diff --quiet origin/agent/m3297 HEAD -- <file>   # 全部 SAME（含 3 个 PNG）
# 唯一与分支不同处 = main 侧更新（表格半球行 + 半球 oracle 句），diff 仅 2 行
```

唯一一处「删除 main 措辞」：把「非均匀预屈曲应力」从「仍缺」清单移除。理由：分支已用
k_b=23.8818 vs 文献 23.9（0.076%）+ 独立 Hermite C1 参考闭环该子项，并新增了
`docs/plate-bending-buckling-validation.md`；若同时保留「仍缺」将产生自相矛盾的陈述。
该删改在 §3 表格中显式登记，不属静默丢失。

## 6. 测试证据（核心用例，main 基线对照）

同一套核心用例在两棵树上各跑一遍（`OMP_NUM_THREADS=1`，逐文件独立进程，日志见 `artifacts/m3297_merge/`）：

| 测试文件 | 合并树 | main 基线 |
|---|---|---|
| `test_plate_bending_buckling.py`（分支新增） | 20 passed | 文件不存在（新增） |
| `test_plate_shear_buckling.py`（共享求解器） | 15 passed | 15 passed |
| `test_marine_plate_buckling.py` | 9 passed | 9 passed |
| `test_marine_plate_combined_buckling.py` | 32 passed | 32 passed |
| `test_marine_plate_postbuckling.py` | 13 passed | 13 passed |
| `test_plate.py` | 6 passed | 6 passed |
| `test_plate_mindlin.py` | 10 passed | 10 passed |
| `test_shell4_buckling_qualification.py` | 12 passed | 12 passed |
| `test_marine_large_angle_stability.py` | 19 passed | 19 passed |
| **合计** | **136 passed / 0 failed / 0 error** | **116 passed / 0 failed / 0 error** |

结论：合并后核心用例 **136 ≥ 116**（差值＝分支新增的 20 个用例），无回归，无失败。

全量 `pytest tests`（1712 用例）的并行跑在同一台机器上被其它并行任务干扰、
日志落盘互相覆盖，未产出可判读的完整计数，**因此不作为证据**；本记录只对
逐文件核心集负责（见 §7 局限）。

## 7. 注意点与局限

1. 自动化合并（overlay/机械并集）在本仓有静默丢内容前科，本次**未使用**；冲突段落逐句手工合成并留表。
2. 文档冲突看起来「只是文字」，但它承载验收声明（`stress_accuracy_qualified`）。若把两侧整段二选一，
   会丢掉半球 oracle 的否定结论或弯曲屈曲闭环之一——两者都是 fail-closed 判据的依据，故必须并集。
3. 分支的求解器修正是**语义性**的（改变非对角刚度铅笔的特征值）。已核对全部调用点与锚点测试，
   但未做「同一物理算例修前/修后」的独立扫描；该扫描属于 Plate 屈曲族的验证账，不在本单范围。
4. 全量测试在共享机上的计数未取得，核心集以外的回归风险不能完全排除；`merge_governor`
   下一周期复验时应以其自身的 CI 结果为准。
5. 本文档不改变任何数值结论：屈曲系数、误差百分比等数字全部引自分支既有证据文件，
   未在本单重算。

## 8. 复现

```bash
git fetch origin main agent/m3297
git log --oneline 14ca2fd..origin/main | wc -l        # 52
git merge-tree --write-tree --name-only origin/main origin/agent/m3297   # 唯一冲突：本文件所在路径
git merge --no-ff origin/agent/m3297                  # 手工解冲（§3）
bash artifacts/run_core.sh <tree> <log>               # 核心用例逐文件跑
```