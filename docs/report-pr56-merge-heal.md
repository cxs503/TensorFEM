# 报告：TensorFEM PR #56 治愈（mission 3824）

- 仓库：TensorFEM　分支：agent/m3824　合并提交：`04adfa0d`
- 旧 `agent/m3148` head：`4ab4ec4`（本地旧对象 `522f9d4`，tree 相同）
- 合并记录：`docs/merge-m3148-cook-recovery-union.md`　证据：`artifacts/`

---

# TensorFEM PR #56 治愈：`agent/m3148` → `main` 语义并集合并（Cook 膜应力恢复）

## ① 案例介绍

**物理背景**：Cook 锥形膜是线弹性平面应力（E=1, ν=1/3, t=1）的经典基准：一端固支、另一端受单位竖向均布剪，位移响应有公认参考值 23.96（尖端）。它的应力场长期是 TensorFEM 的**未闭环项**——原始单元中心应力是 O(h) 精度，任一点式判据都不收敛，因此生产载荷里一直挂 `stress_accuracy_qualified=False`。

**本单不是新物理，而是两条独立生产化路线的合并治理**（PR #56 `mergeable=false`）：

| 项 | `main`（64 提交，236 文件） | `agent/m3148`（1 提交，9 文件） |
|---|---|---|
| 生产入口 | `cook_fields(n, reference=None)` | `cook_fields(n, *, include_recovery=False)` |
| 恢复实现 | `stress_recovery.py`：2×2 Gauss（Barlow）外推 + 节点面积加权，茌参考点时给点式判据 | `cook_stress_recovery.py`：`recovered_nodal_stress` + fail-closed `qualify`（最细网 <3%、逐级严格单调、观测阶 >1、参考 sha256 入账） |
| 载荷字段 | `nodal_stress` / `stress_recovery` | `recovered_nodal_stress` / `recovered_von_mises` / `recovered_stress_protocol` |

几何/参数：锥形膜，8×8~64×64 Q4 网格；合并基 `e8f6759`；`main`=`7accc4b`；`agent/m3148` 远端 head `4ab4ec4`（与本地旧对象 `522f9d4` tree 完全相同）。

## ② 怎么做的

1. 建工作树，fetch `main` 与 `agent/m3148`，确认 `merge-base(e8f6759)`、`main` 领先 64 提交、分支侧 1 提交。
2. `git merge --no-ff agent/m3148` → 冲突 3 文件：`src/tensorfem/additional_benchmark_fields.py`（2 处）、`CHANGELOG.md`、`docs/benchmark-qualification-status.md`。
3. 逐文件**语义并集**（禁止机械 overlay）：
   - 代码：签名合为 `cook_fields(n, reference=None, *, include_recovery=False)`；保留 `main` 的 `q4_gauss_stresses`/`recover_nodal_stress`/`probe_errors` 恢复块与 `reference` 分支，载荷改名为 `payload` 同时带 `nodal_stress`/`stress_recovery`，再执行 `m3148` 的 `if include_recovery:` 分支；原始单元中心场与 `stress_accuracy_qualified=False` 逐字保留。
   - `CHANGELOG.md`：两侧条目全保留（`main` 1 行 + `m3148` 13 行）。
   - `docs/benchmark-qualification-status.md`：同一段「尚未闭环」按**事实取新、内容取并**重写——恢复场已进生产（`m3148` 事实）+ 原始场未变、`strain_accuracy_qualified` 保持 False + 半球壳 HEX20 oracle 否定结论、板屈曲闭环与缺口清单（`main` 事实）；`m3148` 侧「仍缺独立应力参考」已被有证据的新事实取代。
4. **静默丢失防护**（针对机械 overlay 前科）：文件级 9 + 236 个变更文件合并后全在；行级逐行核对两侧新增行在合并后文件中的存在性 → 缺失 **0 / 0**；冲突文件缺失行仅为被有意重写的签名/docstring/条目，逐条有并集落点。
5. 测试：核心子集 11 文件（覆盖两侧改动模块）在合并树与 `main` 基线树（`git archive` 导出，去棹 `m3148` 独有测试）上各跑一遍，`pytest -n 8`。
6. 提交为 `main` 为第一父、`agent/m3148` 为第二父的合并提交（`04adfa0d`）。

## ③ 计算结果

**测试证据（数字表，程序化取自 pytest 输出）**

| 树 | 子集 | 结果 | 耗时 |
|---|---|---|---|
| 合并后（`04adfa0d`） | 11 文件 | **91 passed, 0 failed** | 186 s |
| `main` 基线（`7accc4b`） | 同子集缺 `test_cook_stress_recovery.py` | **84 passed, 0 failed** | 214 s |

差值 +7 恰为 `m3148` 新增测试数 → **不低于基线，零回归**。

**并集语义运行时证据**（`cook_fields(8)` 同一次位移解，`artifacts/cook_fields_union_evidence.json`）

| 量 | 值 |
|---|---|
| 默认载荷键 | 含 `nodal_stress`、`stress_recovery`（`main` 语义） |
| `include_recovery=True` 新增键 | `recovered_nodal_stress`、`recovered_von_mises`、`recovered_stress_protocol`（`m3148` 语义） |
| 两条恢复路径 von Mises 最大偏差 | **1.46e-15**（同一算子，机器精度） |
| `stress_accuracy_qualified` | `False`（两侧一致，未被合并放松） |
| 位移响应 n=8 | 22.0792 vs 参考 23.96 |
| 恢复场收敛（`benchmarks/cook_stress_recovery.json`） | n=8/16/32/64 最大点式误差 8.023%/1.951%/0.484%/0.120%，观测阶 2.01 |

## ④ 结果图

图（合并后代码现场跑出，非手抄）：左 = 恢复场最大点式 von Mises 误差 vs h，叠 h² 参考斜率与 3% 门限；右 = n=8 节点 von Mises 场云图，标注两条恢复路径差 1.46e-15。

## ⑤ 注意点

- 合并**未动**任何求解器数值路径：刚度/载荷/`solve_sparse_static` 参数逐字保留；默认调用除新增两个键外与 `main` 一致（`m3148` 的默认 False 路径哈希清单不变）。
- `include_recovery=True` 为延迟导入的可选路径，默认调用不引入 `cook_stress_recovery` 依赖。
- 本工作树内推送动作被平台工具级策略拦截，提交落在 `agent/m3824`（平台验收后自动推送）；`agent/m3148` 的快进（`4ab4ec4` → `04adfa0d`，纯快进、无内容重写）需由平台/merge_governor 完成。
- 主机为多智能体共享（load average >3000），229 文件全量套件在本单窗口内不可行，故以「覆盖两侧改动模块的 11 文件子集 + 同子集 `main` 基线对照」作回归证据；子集外的文件未运行，属未覆盖。
- 残余语义风险：两份恢复实现算法同源但代码独立，长期看建议由 `m3148` 的 `qualify` 单一门禁收口，避免两套门禁口径漂移。

## ⑥ 误差判定

| 关键量 | 基准 | 合并后值 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 恢复场最大点式 von Mises（n=64） | 冻结独立 Q8 参考 | 0.120% | 0.12% | 是 |
| 两套恢复算子互差（n=8） | 互为基准 | 1.46e-15（绝对值量级） | ~1e-14 | 是 |
| 位移响应（n=8） | 23.96 | 22.0792 | 7.85% | **否**（n=8 粗网格的历史 fail-closed 结论，`stress_accuracy_qualified=False` 保留，本合并未涉及求解器） |
| 测试通过数 | `main` 同子集 84 | 91 | +7（=分支新增） | 是（零回归） |

原始单元中心场在有意义的判据下**仍未达 3%**（`main` 审计：n=64 点式 5.02%，协议落单元边界不收敛），合并**不声明**它被认证；恢复场通过不等于原始场通过。
