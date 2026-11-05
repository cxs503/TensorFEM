# Cook 膜恢复应力：fail-closed 资格门禁（库级 API + 可复算记录）

本单把 Cook 膜**恢复场**的资格判定从「证据脚本里的一段内联布尔运算」提升为**可导入、可复算、
可被负控反驳**的库级门禁，并落盘一份机器生成的资格记录。基线不变：

* 原始逐单元中心场 `cook_fields['stress_accuracy_qualified']` **保持 `False`**——恢复场是同一位移解的
  二阶后处理，不能用来认证原始场（口径替换）。
* 参考解仍为冻结的独立 Q8 参考 `benchmarks/cook_membrane_stress_reference.json`，本轮**不重解、不改写**，
  只记录其 sha256。

## 1. 案例与参数

| 项 | 取值 |
|---|---|
| 几何 | Cook 膜，48×44 梯形板（`continuum_benchmarks.cook_membrane`），单位厚度 |
| 材料 | 平面应力，`E = 1`，`ν = 1/3` |
| 载荷/边界 | 右端均布剪切 6.25/单位长度；左端固支 |
| 离散 | 结构化 Q4，n = 8 / 16 / 32 / 64 |
| 恢复算子 | 2×2 Gauss（Barlow）一维两点外推到单元节点 + 单元面积加权节点平均 + 探针处双线性重插值 |
| 参考 | 独立 Q8 serendipity（3×3 积分）12/24/48 三级 Richardson 外推，逐点存于上述 JSON |
| 判据 | 逐级有限、逐级严格单调下降、最细网格最差站点 ≤ 3%、观测阶 > 1、参考摘要入账 |

## 2. 做法

`src/tensorfem/cook_stress_recovery.py`（新增，Torch）：

1. `load_reference(path=None)`：读冻结参考，返回 `path / sha256 / formulation / levels / points`，
   并把 Richardson 外推的 von Mises 作为逐点目标。摘要是每次判定都重算的，不是抄来的常量。
2. `recovered_nodal_stress(nodes, elements, displacement)`：**文档化入口**。它不新增任何算法，
   只是调用生产原语 `stress_recovery.q4_gauss_stresses` + `stress_recovery.recover_nodal_stress`，
   并给结果打上协议字符串——防止下游把恢复场当成 `cook_fields['stress']` 原始场导出。
3. `measure(levels, reference_path)`：**走生产载荷** `cook_fields(n, reference=<冻结参考>)`，
   逐级读回 `stress_recovery.worst_site_error` 与逐点块。被判定的数就是消费者拿到的数，
   没有第二条私有路径。
4. `qualify(levels, reference_path, tolerance=0.03, measured=None)`：**fail-closed** 判定。
   四条硬性条件（有限 / 严格单调 / 最细 ≤ 容差 / 观测阶 > 1）任一不满足即
   `stress_recovery_qualified=False`，并给出 `blocking_reasons`；同时输出 `verdict_sha256`。
   `tolerance` 只允许 `0 < t ≤ 3%`，超出即 `ValueError`——3% 是门限不是旋钮。
   `measured=` 参数允许对已有实测数据重新判定，这是负控（见 §4）唯一入口。
5. `scripts/qualify_cook_stress_recovery.py`：生成 `benchmarks/cook_stress_recovery.json` 与
   `docs/assets/cook-stress-recovery/convergence.png`。

观测阶用**误差差商** `p = ln(|e_L − e_{L−1}| / |e_{L−1} − e_{L−2}|) / ln(h_{L−1}/h_L)`，
而不是原始误差比：阶梯误差由一个平滑的 `h^p` 项主导，差商能消掉常数偏移，比原始比更保守。

## 3. 结果

逐级最差站点归一化 von Mises 误差（生产载荷，n 为网格分划数）：

| n | 最差站点误差 | 最差界面离散度 | 原始场 `stress_accuracy_qualified` |
|---|---|---|---|
| 8 | 8.0231% | ≤ 9.4e-17（单值） | False |
| 16 | 1.9514% | ≤ 9.4e-17 | False |
| 32 | 0.4838% | ≤ 9.4e-17 | False |
| 64 | 0.1202% | ≤ 9.4e-17 | False |

* 观测阶 **2.013**（差商，末三级）——二阶收敛，与 2×2 Gauss 恢复的理论阶一致。
* 门禁判定：`blocking_reasons = []`，`stress_recovery_qualified = True`，
  `stress_accuracy_qualified = False`（原始场未认证）。
* 交叉校验（n = 8，应力尺度 0.39059）：
  `recovered_nodal_stress` vs 生产载荷 `nodal_stress` 最大差 **0.0**（同一函数）；
  生产载荷 vs 独立 NumPy 审计实现 `cook_probe_protocol.nodal_recovered_stress` 最大相对差
  **5.26e-15**（机器精度）。
* 图：`docs/assets/cook-stress-recovery/convergence.png`（双对数误差-`h` 曲线 + 二阶导则 + 3% 门限线，
  每个点标注误差百分数，程序化生成）。

重算记录：`benchmarks/cook_stress_recovery.json`（含 `verdict_sha256`、逐点块、交叉校验与
`qualification_recorded=true`）。

## 4. 负控（为什么这个门禁不是摆设）

`tests/test_cook_stress_recovery.py`（19 项）里，除正向用例外含**注入式负控**，
每条都断言门禁给出具体 `blocking_reason`：

| 注入 | 期望 | 实际 |
|---|---|---|
| 误差序列 5%/6%/2%（非单调） | 拒绝 | `non_monotone_error` |
| 9%/5%/4%（最细超 3%） | 拒绝 | `finest_above_tolerance` |
| 9%/4.5%/2.01%（一阶） | 拒绝 | `observed_order_not_above_one` |
| NaN / inf（非有限） | 拒绝 | `non_finite_error` |
| 单级网格 | 抛 `ValueError` | "at least two mesh levels" |
| 级别非递增 | 抛 `ValueError` | "strictly increasing" |
| 实测级别与声明不一致 | 抛 `ValueError` | "do not match" |
| `tolerance ∈ {5%, 3.1%, 0, −1%, NaN}` | 抛 `ValueError` | 门限不可放宽 |

另有两条正向实现性测试：均匀应变态下恢复算子对精确常应力场误差 < 1e-13（含跨单元插值单值性），
以及文档化入口与生产载荷逐位一致（最大差 0.0，防止出现第二份实现漂移）。
负控还证明 `verdict_sha256` 随被判数据变化（改一位误差摘要即变），不是常量。

## 5. 注意点与适用范围

* **不是三维厚度应力证书**，也不是工业验收；只声明 5 个声明材料点上的平面应力 von Mises 精度。
* 固支角点与加载端部的奇异区**不在**覆盖范围，那里不做任何逐点声明（von Mises 在奇异区本身发散）。
* 参考本身是数值解（Q8 三级 Richardson），其 GCI 已在参考文件里记录（逐点 `gci_fine_relative`
  量级 1e-4）；因此 0.12% 应读作「与参考解之差」，不是与解析解之差。
* 夹具罚：`cook_fields(n, reference=...)` 三级阶梯 + 复算交叉校验约 24 s（四核、单线程 BLAS），
  测试里用 n=8/16/32 三级把模块测试压到 ~9 s。
* 与 `docs/cook-recovered-stress-production.md` 互补：那份报告证明**恢复算子进了生产路径**，
  本报告证明**资格判定本身可复算、可被负控反驳**；两者都不改动原始场的 `False`。

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 最差站点 von Mises（n=64，恢复场） | 独立 Q8 Richardson 外推 | **0.1202%** | 是 |
| 最差站点 von Mises（n=32） | 同上 | 0.4838% | 是 |
| 观测阶（末三级，差商） | 理论 2（二阶恢复） | 2.013 vs 2 → 0.65% | 是 |
| 恢复场实现一致性（n=8） | `cook_probe_protocol`（NumPy 独立实现） | 5.26e-15 相对 | 是 |
| 原始逐单元中心场（n=64） | 同上参考 | 3.5–4.0%（一阶） | **否**（故 `stress_accuracy_qualified=False`） |

结论：恢复场在 3% 门限内且二阶收敛，资格判定 fail-closed 且有负控；原始场**未**通过，
本单没有放宽任何既有判定。