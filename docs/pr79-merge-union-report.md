# PR #79（agent/m3402 → main）三方合并与语义并集报告

日期：2026-11-05 · 任务：TensorFEM#79 治愈（分支 `agent/m3873` 出工，成果推回 `agent/m3402`）

## 1. 背景与对象

PR #79 的分支 `agent/m3402` 与 `main` 都从 `14ca2fd` 分叉，两侧各自新增了一整套
**Q4 应力恢复（Gauss/Barlow 恢复）** 实现，于是 PR 变成 `mergeable=false`：

| 侧 | 提交内容 | 实现族 |
|---|---|---|
| `agent/m3402` | `stress_recovery.py`（新增）、`cook_probe_protocol.py`（改为再导出）、`cook_recovery_qualification.py`、`scripts/build_cook_recovery_qualification.py`、证据 JSON、文档、`tests/test_cook_recovery_qualification.py` | **NumPy** 审计族 |
| `main`（同期） | `stress_recovery.py`（新增）、`additional_benchmark_fields.py` 扩展 `cook_fields` 载荷、`scripts/cook_recovery_production_evidence.py`、`tests/test_stress_recovery.py` | **Torch** 生产族 |

两侧数学相同（2×2 Gauss 点取值 → 线性外推到节点 → 面积加权节点平均），代码却互不共享。
机械 overlay（任取一侧）会**静默丢失**另一侧的整族实现与其消费者，因此本次采用语义并集。

案例物理背景（两族共用的验证病例）：经典 Cook 膜，平面应力，E=1、ν=1/3、单位厚度，
四角 `(0,0)/(48,44)/(48,60)/(0,44)` mm，左边界（x=0）固支，右边界均布单位合力剪切，
载荷边中点 (48,52) 的已发表挠度参考 23.96。应力探针 5 点取自独立 Q8 Richardson 参考
`benchmarks/cook_membrane_stress_reference.json`。

## 2. 合并方法

1. 工作树 `/root/tp-agent/work/m3873`（分支 `agent/m3873`，基线 `origin/main` = `7accc4b`），
   `git merge --no-commit --no-ff agent/m3402`。
2. 仅 2 处文本冲突 + 1 处 API 断言冲突（见 §3）；其余 192 个 main 侧文件与 13 个 m3402 侧文件
   由 git 自动合并（`CHANGELOG.md`、`docs/cook-probe-protocol-audit.md` 自动合并成功）。
3. 解冲不是手抄：`.scratch/resolve_sr.py` 以 **AST/正则按函数块抽取**两侧 blob 后拼装并集，
   每处替换都断言"锚点唯一命中"，避免手抄数值内核出错。
4. 冲突解决后：`ast.parse` 通过、无重复顶层定义（22 个函数）、`compileall` 通过、
   无残留冲突标记（脚本自检 marker 计数 = 0）。
5. 提交并推回 `agent/m3402`（记录旧/新 SHA，见 §6）。

## 3. 逐文件解法

### 3.1 `src/tensorfem/stress_recovery.py`（add/add，3 段冲突）

**解法 = 语义并集（两族全部保留）**，以 main 的 322 行 Torch 版为骨架，尾部追加 m3402 的
NumPy 审计族（逐字保留），三个同名符号用**显式分派规则**统一，并在模块 docstring 中写明：

| 符号 | main 侧语义 | m3402 侧语义 | 并集解法 |
|---|---|---|---|
| `EXTRAPOLATION_1D` | 2×2 嵌套 tuple，行索引后再取元素 | `np.ndarray`，`[i, j]` 索引 | 改为 2×2 `np.ndarray`，两种索引风格都成立，数值不变 |
| `q4_shape(xi, eta, …)` | 返回 `torch.Tensor`（`dtype`/`device` 关键字） | 返回 `np.ndarray` | `dtype=None and device=None` → NumPy；给定 `dtype`/`device` → Torch。两族调用点均不改 |
| `element_areas(nodes, elements)` | Torch：两三角形叉积和 | NumPy：节点环 shoelace | 按 `nodes` 类型分派；凸四边形上两者恒等（本报告 §4.2 实测吻合到 0 位差） |

其余同名函数按语义差异**分开保留**（互不覆盖）：
`q4_gauss_stresses`/`gauss_point_stresses`、`recover_nodal_stress`/`nodal_recovered_stress`、
`interpolate_nodal_field`/`interpolate_nodal_stress`、`locate_point`/`containing_elements`、
`normalized_error`/`recovery_error`，以及 `extrapolation_matrix`、`local_coordinates`、
`q4_strain_matrix`（Torch 公开名保留；NumPy 族所需的
`cook_stress_reference.plane_stress_matrix/q4_strain_matrix` 以私有别名
`_reference_*` 引入，供 m3402 的 `_element_stress` 使用，行为与原实现逐位一致）。
`__all__` 取两侧并集（24 个符号）。

未丢失性检查：main 侧 322 行除上述 2 个函数与 1 个常量外逐行未变（`diff` 可见）；
m3402 侧 8 个函数 + `_element_stress` 逐字进入并集（仅 2 处私有别名替换）。

### 3.2 `docs/benchmark-qualification-status.md`（content 冲突，1 段）

两侧都在改写"尚未闭环的工作"里 Cook 膜那段：

* m3402 新增：恢复场**已进入生产路径**（n=64 von Mises 0.120%、最差分量 1.068%；n=32
  0.484%/2.393%；n=8/16 失败项如实保留），并保留"开孔半球壳仍缺独立应力参考"。
* main 新增：开孔半球壳的**独立 HEX20 oracle 已补并给出否定结论**（平均 5.09%、最大 9.00%，
  孔边 6.87%、点载附近 7.56%），`stress_accuracy_qualified` 维持 False。

**语义并集**：两句都保留；仅把被 main 证据**取代**的半句"开孔半球壳仍缺独立应力参考"替换为
main 的 oracle 结论（这是唯一被删除的表述，且属"新证据取代旧陈述"，不是静默丢内容）。
其余段落（板屈曲 k_s=9.34/5.34、壳几何刚度等缺口清单）两侧一致，保持一致。

### 3.3 `tests/test_cook_recovery_qualification.py`（非文本冲突，测试断言与 main 新增 API 冲突）

合并后 `test_stored_production_field_still_carries_no_stress_claim` 失败：
`cook_fields(8)` 的键集多了 main 新增的 `nodal_stress`、`stress_recovery`（main 把恢复场写进
生产载荷），而 m3402 的测试用**等于号**断言历史键集。

**解法**：断言意图改为"历史键一个都不能少 + 新增两键必须存在且不翻转 fail-closed 旗标"
（`historical_keys <= set(case)` 且 `{"nodal_stress","stress_recovery"} <= set(case)`、
`stress_accuracy_qualified is False`、`len(nodal_stress) == len(nodes)`）。
这样既保住 m3402 测试的原始语义（常数单元场协议未被放松），也接受 main 的 API 扩展，
不靠删测试通过。

## 4. 计算结果（程序化生成，零手抄）

生成器 `scripts/verify_stress_recovery_union.py`，产物
`benchmarks/cook_recovery_union_verification.json`（键排序、含 sha 与逐点误差）。

### 4.1 两族实现互检（同一 Cook 位移场，n=8/16/32/64）

| n | 单元数 | max｜NumPy−Torch｜/尺度 | 探针跨单元 spread（Torch） | von Mises 最大误差 | 最差分量误差 |
|---|---|---|---|---|---|
| 8 | 64 | 5.26e-15 | 3.6e-15 | 8.02311% | 32.97504% |
| 16 | 256 | 1.10e-14 | 4.4e-15 | 1.95144% | 8.94680% |
| 32 | 1024 | 1.94e-14 | 5.2e-15 | 0.48379% | 2.39274% |
| 64 | 4096 | 4.69e-14 | 6.8e-15 | 0.12019% | 1.06786% |

* 两族差异在**机器精度**（≤5e-14），证明并集没有改变任何一侧的数值语义。
* 逐点误差序列与仓库已存证据 `benchmarks/cook_recovery_qualification.json` **逐位一致**
  （`max_abs_stored_delta = 0.0`）。
* 观测收敛阶 von Mises = **2.02**（二阶，符合 Barlow 恢复的理论阶），`monotone=True`，
  `recovery_qualified=True`（n=64：0.120% / 1.068%，均在 3% 门限内）。

### 4.2 分派统一性（`q4_shape` / `element_areas`）

| 检查 | 结果 |
|---|---|
| `q4_shape(0.37,-0.62)` 返回类型 / 给定 dtype 时 | `ndarray` / `Tensor` |
| 两种返回的最大差 / 形函数和 | 0.0 / 1.0 |
| 单位正方形面积（NumPy / Torch） | 2.0 / 2.0 |
| 锥形四边形（Cook 型）面积（NumPy / Torch） | 2.51 / 2.51 |

## 5. 测试证据

| 树 | 命令 | 结果 |
|---|---|---|
| 合并后（并集） | 11 个核心文件（恢复/审计/基准/连续体） | **106 passed**（80.8 s，EXIT=0），日志 `runs/pytest_merged_core.log` |
| main 基线 | `test_stress_recovery` + `test_cook_probe_protocol` + `test_additional_benchmarks` | **22 passed**（15.4 s + 41.3 s） |
| 合并后（并集） | 同三文件逐文件对表 | **22 passed**（99.5 s） |
| 冲突专项 | `tests/test_stress_recovery.py` + `tests/test_cook_recovery_qualification.py` | 20 passed |
| 证据再生 | `scripts/build_cook_recovery_qualification.py` 重跑并比对存档 JSON | 除内嵌生成路径外**逐位一致**（0 处数值差异） |

基线对表结论：在两侧都存在的三个测试文件上，main 基线 22 passed，并集树 22 passed，
**不低于基线**；并集树另外还多跑过 m3402 侧独有的 `test_cook_recovery_qualification.py`
（全部通过）。全仓库 228 个测试文件的整轮运行受机器负载（load average >1000）阻塞，
已如实记为未完成项（§7.3）。

核心文件清单：`test_stress_recovery`、`test_cook_recovery_qualification`、
`test_cook_probe_protocol`、`test_cook_stress_audit_closure`、
`test_cook_stress_recovery_verification`、`test_cook_stress_reference`、
`test_cook_oracle_operator`、`test_additional_benchmarks`、`test_benchmark_field_gates`、
`test_elasticity_analytic_verification`、`test_continuum`。

## 6. SHA 记录

| 项 | SHA |
|---|---|
| 合并前 `agent/m3402` | `528f042` |
| 基线 `main`（origin） | `7accc4b` |
| 合并基 | `14ca2fd` |
| 合并后 `agent/m3402`（本报告提交） | 见 §6 更新（推送后回填） |

## 7. 注意点与局限

1. **同名符号分派是"可读性换兼容"**：`q4_shape` 的默认返回类型从 Torch `Tensor` 变成
   NumPy `ndarray`。仓库内所有调用点都显式传 `dtype`/`device`（Torch 族）或不传（NumPy 族），
   故无实际影响；但**外部代码**若依赖裸 `q4_shape(...)` 返回 Tensor，需要补 `dtype`。
2. `element_areas` 的两套公式仅在**凸四边形**上恒等（Cook 网格与所有 Q4 用例都是凸的）；
   强凹四边形上两族会有差异 —— 这是分派而非统一公式的原因（保留各自历史数值）。
3. 未做（明确声明）：未在并集上重跑全仓库 228 个测试文件的完整套件（机器 load average 长期
   >1000，整轮与分批基线跑均被超时收割），只跑了与合并面相关的 11 个核心文件 + 冲突专项 +
   三文件基线对表；未改动
   `additional_benchmark_fields` 与 `cook_probe_protocol` 的逻辑（它们是两侧的非冲突自动合并结果）。
4. merge_governor 复验后如仍报脏，最可能是 `benchmarks/` 下证据 JSON 的 sha 漂移，而非本文件。

## 8. 误差判定（是否 <3%？）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| n=64 探针 von Mises 最大误差 | 独立 Q8 Richardson 参考 | 0.1202% | **是** |
| n=64 最差应力分量 | 同上（尺度归一化，5% 下限） | 1.0679% | **是** |
| n=32 探针 von Mises 最大误差 | 同上 | 0.4838% | **是** |
| n=32 最差分量 | 同上 | 2.3927% | **是** |
| n=16 / n=8 最差分量 | 同上 | 8.947% / 32.975% | 否（粗网格，如实保留为失败项） |
| 并集 vs 已存证据（逐点误差） | `cook_recovery_qualification.json` | 0.0（逐位一致） | **是** |
| NumPy 族 vs Torch 族恢复场 | 彼此（零解析基准） | ≤5e-14（机器精度） | **是** |

结论：并集在**数值上零改变**（两族互差 ≤5e-14、与已存证据逐位一致），生产判据在
n≥32 满足 3% 门限，n=8/16 的失败项按原样保留、未被并集"修好"或抹平。