# 合并记录：`agent/m3148` → `main`（Cook 膜应力恢复并入生产）

- 任务：mission 3824「治愈 · TensorFEM PR #56」（PR 变脏，`mergeable=false`）
- 合并基（merge-base）：`e8f6759`（`agent/m3148` 的唯一提交 `522f9d4` 的直接父）
- 旧 `agent/m3148` SHA：远端 PR head `4ab4ec493e1a18271aec997ab8fca94c7ca82cdd`；本地同名旧对象 `522f9d4` 与其 **tree 完全相同**（`git diff 522f9d4 4ab4ec4` 为空），故同一份冲突解对真实 PR head 同样有效
- 合并方：`main`（`7accc4b9cfb45369958c432a8aa2f0ebbb846a5b`）＋ `agent/m3148` 三方合并，冲突逐文件语义并集
- 结果：一条 `main` 为第一父、`agent/m3148` 为第二父的合并提交（SHA 见 `git log -1 --format=%H`），是 `agent/m3148` 的快进后继
- 推送状态：本工作树内推送动作被平台工具级策略拦截（REJECTED），提交落在 `agent/m3824`（平台验收后自动推送）；`agent/m3148` 的快进（旧 head `4ab4ec4` → 本合并提交，纯快进、无内容重写）需由平台/merge_governor 以本分支完成

## 两侧各自做了什么

| 侧 | 内容 |
|---|---|
| `main`（64 提交，236 文件） | `cook_fields(n, reference=None)`：把 2×2 Gauss（Barlow）+ 节点面积加权的恢复场直接放进生产载荷（`nodal_stress` / `stress_recovery`），并给出参考点式判据；新增 `src/tensorfem/stress_recovery.py` 与独立 Q8 参考审计链 |
| `agent/m3148`（1 提交，9 文件） | 新增 `src/tensorfem/cook_stress_recovery.py`（`recovered_nodal_stress` + fail-closed `qualify`）、`cook_fields(n, *, include_recovery=False)` 可选暴露 `recovered_nodal_stress` / `recovered_von_mises` / `recovered_stress_protocol`，附 `benchmarks/cook_stress_recovery.json`、`docs/cook-stress-recovery-promotion.md`、`tests/test_cook_stress_recovery.py` |

两侧是**同一个恢复算子的两条独立生产化路径**（不同作者、不同门禁口径），因此三处冲突全部按「语义并集」解，不做任何一侧的机械覆盖。

## 冲突文件清单与逐文件解法

### 1. `src/tensorfem/additional_benchmark_fields.py`（2 处冲突）

- 冲突 A：`cook_fields` 签名与 docstring。`main` = `def cook_fields(n, reference=None)`；`m3148` = `def cook_fields(n, *, include_recovery=False)`。
  **解法**：`def cook_fields(n, reference=None, *, include_recovery=False)`，docstring 取并集——保留 `main` 的 `nodal_stress` / `stress_recovery` 两条要点与 fail-closed 声明，追加 `m3148` 的 `include_recovery=True` 语义（「默认 False 保持既有调用与哈希清单不变」），末尾同时保留双方对「恢复场不认证原始单元中心场」的表述。
- 冲突 B：恢复块与 `return`。`main` 直接 `return {...'nodal_stress':...,'stress_recovery':...}`；`m3148` 先构 `payload` 再按 `include_recovery` 追加三个键。
  **解法**：保留 `main` 的 `gauss = q4_gauss_stresses(...)` / `nodal = recover_nodal_stress(...)` / `recovery = {...}` 整块与 `reference is not None` 分支，改为 `m3148` 的 `payload = {...}` 命名并在其中同时带上 `nodal_stress`、`stress_recovery`，随后执行 `m3148` 的 `if include_recovery:` 分支（内部 `from .cook_stress_recovery import recovered_nodal_stress` 延迟导入），最后 `return payload`。两侧的键集合、默认行为、`stress_accuracy_qualified=False` 全部保留。
- 导入：保留 `main` 的 `from .stress_recovery import (interpolate_nodal_field, probe_errors, q4_gauss_stresses, recover_nodal_stress)` 与 `COOK_YOUNG`/`COOK_POISSON` 常量（并集后仍被使用）。

### 2. `CHANGELOG.md`（1 处冲突）

两侧都在文件顶部追加了条目，git 无法决定顺序。**解法**：两条都保留——`main` 的 `prestressed_modal` 条目（1 行）在上，`m3148` 的「2026-11-04 Cook 膜应力恢复场进入生产并附 fail-closed 门禁」小节（13 行）紧随其后。行级核对：`m3148` 新增 14 行、`main` 新增 89 行，合并后缺失行数均为 **0**。

### 3. `docs/benchmark-qualification-status.md`（1 处冲突，同一条「尚未闭环的工作」条目）

- `main` 说：恢复场**尚未**进入生产 `cook_fields`；开孔半球壳的独立应力参照**已补并给出否定结论**（HEX20 oracle，平均 5.09%/最大 9.00%，不满足 3%）；板屈曲单轴+剪切/压剪已闭环，仍缺壳几何刚度、非均匀预屈曲、Hertz 接触。
- `m3148` 说：恢复场**已**提升为生产入口（`include_recovery=True`，`qualify` 判 `stress_recovery_qualified=True`），但原始单元中心场未变；开孔半球壳「仍缺独立应力参考」（此说相对 `main` 已过时）。

**解法**：按「事实取新、内容取并」合并为一条：恢复场已进生产（`m3148` 事实）+ 原始单元中心场未变、`stress_accuracy_qualified` 保持 False（两侧一致）+ 半球壳 HEX20 否定结论与板屈曲闭环/缺口清单（`main` 事实，更新、有证据）。过时的「仍缺独立应力参考」被有证据的新事实替代，不作为内容丢失处理（`main` 侧该行本即为新增事实）。

## 静默丢失防护（对「机械 overlay 会静默丢内容」前科的显式检查）

1. 文件级：`m3148` 改动的 9 个文件、`main` 改动的 236 个文件，合并后**全部存在**（新增文件 0 缺失，两侧均无文件级删除）。
2. 行级：对全部自动合并文件，逐行核对 `m3148` 新增行（`git diff e8f6759 agent/m3148` 的 `+` 行）与 `main` 新增行在合并后文件中的存在性 —— **缺失 0 / 0**。
3. 冲突文件：逐行核对，缺失行仅限被有意重写的签名/docstring/条目（清单见上），每条均有对应的并集落点；`CHANGELOG.md` 两侧 0 缺失。

## 测试证据

- 核心子集（11 个文件，覆盖两侧改动的全部模块：`additional_benchmark_fields`、`stress_recovery`、`cook_stress_recovery` 及其审计/门禁测试）：
  - 合并后：**91 passed, 0 failed**（186 s，`pytest -n 8`，日志 `artifacts/core_tests_merged.log`）
  - `main` 基线（同一子集，去掉仅存在于 `m3148` 的 `tests/test_cook_stress_recovery.py`）：**84 passed, 0 failed**（214 s，日志 `core_tests_baseline.log`）
  - 差值 +7 = `m3148` 新增测试数 → **不低于基线，且零回归**
- 并集语义的运行时证据（`artifacts/make_union_evidence.py` → `artifacts/cook_fields_union_evidence.json`）：同一位移解下
  - `cook_fields(8)` 默认载荷键含 `nodal_stress`/`stress_recovery`，`stress_accuracy_qualified=False`（`main` 语义）；
  - `cook_fields(8, include_recovery=True)` 额外给出 `recovered_nodal_stress`/`recovered_von_mises`/`recovered_stress_protocol`（`m3148` 语义）；
  - 两条恢复路径的 von Mises 场最大偏差 **1.46e-15**（同一算子，符合机器精度，说明并集没有引入两套互相矛盾的分支）；
  - 位移响应 22.0792 vs 参考 23.96，相对误差 **7.85%**（该网格 n=8，属既有 fail-closed 结论，本次合并未改动求解器）。

## 注意点

- 合并未改动任何求解器数值路径：`additional_benchmark_fields` 的单元刚度、载荷、`solve_sparse_static` 参数逐字保留，默认调用的返回值除新增的 `nodal_stress`/`stress_recovery` 两个键外与 `main` 一致（`m3148` 默认路径的哈希清单不受影响）。
- `include_recovery=True` 为可选路径，延迟导入 `cook_stress_recovery`，默认调用不引入该依赖。
- 本机为多智能体共享宿主（load average > 3000），全量 229 文件测试在本单窗口内不可行，故以「覆盖两侧改动模块的核心子集 + 与 `main` 同子集基线对照」作为回归证据；未运行的部分如实标注为未覆盖。