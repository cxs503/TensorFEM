# 标量基准注册表的证据分级审计（69 项逐项归类）

**结论先行**：注册表当前登记 **69** 项标量检查（`docs/benchmark-qualification-status.md` 里
"55 个标量检查"的说法已过期）。按证据类别逐项归类后，**只有 43 项（62.3 %）是"离散解 vs 闭式/文献"
的真数值验证**，9 项（13.0 %）是"数值求积/Ritz/围道积分 vs 闭式"，16 项（23.2 %）是**闭式 vs 闭式**
（同一物理公式的两份独立编码，误差全部 ≤ 8.97e-16，**结构上不可能发现 FE 内核缺陷**），
1 项（1.4 %）是算法自满足的**一致性恒等式**。本审计未发现数值错误；发现的是**覆盖度口径问题**：
"69/69 通过"若被引用为 FE 覆盖度，会把真实的离散验证面高估约 1.6 倍（69 vs 43）。
该口径现已由 fail-closed 测试锁死（见 §5）。

## 1. 案例介绍（对象与量纲）

审计对象不是单一物理案例，而是 `tensorfem.benchmark_registry.run_registered_benchmarks()`
返回的全部标量证据条目。每条证据的结构是
`(id, capability, quantity, unit, source, computed, reference, tolerance ≤ 0.03, passed)`，
`passed` 的语义是 `|computed − reference| / |reference| < tolerance`。

注册表覆盖的域：杆/桁架/框架/连续体 Q4/TET4/TET10/HEX8 静力与模态、Mindlin 板（薄与厚）、
屈曲与后屈曲、Newmark/中心差分动力学、J2 塑性、patch test（Robinson A/B/C）、内聚律积分、
稀疏装配求解、稳态热（含对流与约束热应力）、轴对称壳（球/球面体/椭圆体）、
以及船海结构族（船体梁、加筋板、板格屈曲、极限强度、疲劳、断裂、缺陷条带）。

未变动的量：本审计**只读**注册表结果，不修改任何容差、参考值或 `passed` 语义。

## 2. 怎么做的（方法、流程、工具链）

1. **取证据**：`OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/registry_evidence_audit.py`
   运行注册表（单核约 90 s），落 `runs/registry_evidence_class/registry_report.json`
   （原始 `computed/reference/error/tolerance`，零手抄）。
2. **逐项定级**：在 `src/tensorfem/registry_evidence_audit.py::EVIDENCE_CLASSES` 里为 **69 个 id
   逐一**声明类别与一行"依据"（依据指向源码里 `computed` 的真实来源）。类别枚举：
   - `discretized`：网格/有限差分/特征值/时间步进求解 vs 闭式或文献；
   - `trial_space`：在解析试空间上做数值求积/极小化/围道积分（无网格）；
   - `closed_form`：闭式 vs 另一份闭式编码（无任何数值求解）；
   - `identity`：算法按构造自满足的一致性恒等式。
3. **fail-closed 语义**：**未登记的 id 直接抛错**，不默认成 `discretized`；
   注册表里消失的 id 记为 `stale_ids`（防止"删检查"式通过）。
4. **独立 oracle**：对最薄余量的 `shell.spheroid.displacement_equator`，用膜理论**重新推导**
   赤道法向位移闭式解（见 §4），与仓库内 Rayleigh–Ritz 参考比对，判断 2.31 % 误差到底是 FE
   几何误差还是参考本身有偏。
5. **门禁**：`tests/test_registry_evidence_audit.py`（8 项，全绿）含反空转对照（§5）。

## 3. 计算结果（程序化注入，零手抄）

| 类别 | 条数 | 占比 | `min error` | `max error` | 含义 |
|---|---:|---:|---:|---:|---|
| `discretized` | 43 | 62.32 % | 0.0（patch/节点精确） | **2.3073e−2** | 网格/差分/特征值/时步 vs 闭式或文献 |
| `trial_space` | 9 | 13.04 % | 0.0 | 7.5502e−4 | 数值求积/Ritz/围道 vs 闭式 |
| `closed_form` | 16 | 23.19 % | 0.0 | **8.9706e−16** | 闭式 vs 闭式（无离散化） |
| `identity` | 1 | 1.45 % | 1.8608e−16 | 1.8608e−16 | 一致性恒等式（按构造满足） |
| 合计 | 69 | 100 % | — | — | `unclassified_ids = []`，`stale_ids = []` |

**余量最薄的两项**（`error/tolerance ≥ 0.5`，即已吃掉一半以上容差）：

| 检查 | 类别 | error | error/tolerance |
|---|---|---:|---:|
| `shell.spheroid.displacement_equator` | discretized | 2.3073 % | 0.769 |
| `continuum.cook` | discretized | 2.2103 % | 0.737 |

**结构上不可能发现 FE 缺陷的 17 项**（`algebra_only_ids`，误差 ≤ 1e-9）：
`marine.hydrostatics.{displacement,gm,restoring}`、`marine.fatigue.{hot_spot.linear,hot_spot.quadratic,sn.single_block,miner.multi_block}`、
`marine.fatigue_advanced.{hot_spot.path_linear,rainflow.triangle_count,miner.variable_amplitude,paris.m2_closed_form}`、
`marine.fracture.{hot_spot,scl_membrane,mode_i_k,mode_i_j}`、`contact.spring`、`plasticity.j2`。
它们是**公式互证**（两份独立编码互裁）与**一致性恒等式**，有价值，但不能计入 FE 数值验证面。

图：`docs/assets/registry-evidence-classes/evidence_classes.png`（各类别条数，由脚本生成）。
逐项依据与完整表：`runs/registry_evidence_class/audit.json`、`docs/assets/registry-evidence-classes/audit.md`。

## 4. 独立 oracle：椭圆体赤道位移参考值是否可信

`shell.spheroid.displacement_equator` 的仓库内参考来自"膜势能 Rayleigh–Ritz 极小化"，
本身是近似方法，其截断误差此前**未被量化**——若参考偏 2 %，2.31 % 的门限通过就失去意义。
本审计用膜理论重新推导（`spheroid_equator_displacement_closed_form`）：

```
r = a sinθ, z = b cosθ；球面体赤道：R1 = R2 = a²/b, sinφ = 1
盖截轴向平衡  N_φ = p r /(2 sinφ) = p a / 2
法向平衡      N_φ/R1 + N_θ/R2 = p  ⇒  N_θ = p a (1 − a²/(2b²))
平面应力 Hooke w = a ε_θ = a (N_θ − ν N_φ)/(E t) = (p a²/(E t))(1 − a²/(2b²) − ν/2)
```

算例 `a=1, b=0.5, t=0.01, E=1, ν=0.3, p=1`（注册表 `facets=256`）：

| 量 | 值 |
|---|---:|
| 注册表参考（Ritz） | −114.99999999694278 |
| 独立闭式（本审计） | −114.99999999999999 |
| 参考 vs oracle 相对差 | **2.66e−11** |
| FE（256 面元） vs oracle 相对差 | **2.3073 %** |
| 注册表登记的 FE 误差 | 2.3073 % |

即：Ritz 参考在该算例上与精确膜解一致到 2.7e−11，**2.31 % 是真实的 FE 几何离散误差**
（2 节点环单元每单元常曲率，球面体被逐段圆拱近似），不是参考假象。同时也说明这项检查的
容差余量只剩 23 %，再细的判据会立即翻转。

球极限自检：`a=b=2, t=0.05, E=7e10, ν=0.33, p=1e5` 时闭式退化为经典
`w = pR²(1−ν)/(2Et)`，两者相对差 < 1e−12（见测试）。

## 5. 注意点（局限、适用范围、踩过的坑）

- **口径**：`discretized` 只表示"`computed` 来自离散求解"，**不**表示该检查已具备完整的
  三维场/应力精度认证。域级认证仍以 `docs/benchmark-qualification-status.md` 的逐案例报告为准。
- **`trial_space` 不是网格验证**：Navier 级数/Ritz/Morison 求积是对同一解析试空间的数值处理，
  能发现级数截断与求积误差，发现不了单元级缺陷。
- **抗空转对照**：把任一未登记 id 注入报告 → 审计报 `unclassified_ids`；删掉一个已登记 id →
  报 `stale_ids`；把 oracle 参考人为改动 1 % → 相对差立即越过 1e−9 阈值。
  三条反例均写成测试，证明门禁不是"永远绿"。
- **类别是人工声明**：新加检查必须同步在 `EVIDENCE_CLASSES` 里写类别与依据，否则测试红。
  这是刻意的摩擦——避免"先通过再补依据"。
- 文档漂移已修：`benchmark-qualification-status.md` 的 "55 个标量检查" 实为 69 项。

## 6. 误差判定（逐项 vs 基准，显式回答是否 < 3 %）

| 被判据 | 值 | 基准 | 判定 |
|---|---:|---|---|
| 参考值独立性（椭圆体赤道位移） | 2.66e−11 | 独立膜理论闭式解 | **< 3 %（实为 < 1e−9）** |
| 球极限退化 | < 1e−12 | `pR²(1−ν)/(2Et)` | **< 3 %** |
| 离散验证面占比 | 62.32 %（43/69） | 无外部基准（计数可逐项复核） | 不适用（非误差量） |
| `discretized` 类最大误差 | 2.3073 % | 膜理论精确解（经 oracle 确认） | **< 3 %，余量仅 23 %** |
| `closed_form` 类最大误差 | 8.9706e−16 | 两份独立编码互裁 | **< 3 %（位级一致）** |

**未做的事（honest 边界）**：未对 43 项 `discretized` 检查逐个建立外部独立 oracle（仅对最薄余量的
椭圆体赤道位移做了完整 oracle）；未认证三维壳体几何刚度、加筋板峰后收敛等已知阻塞项；
本审计不提升任何域级能力等级。

## 复现

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/registry_evidence_audit.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_registry_evidence_audit.py -q
```

（实测：脚本 gate PASS；pytest 8 passed，86.8 s，单核。）