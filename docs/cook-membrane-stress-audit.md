# Cook 膜应力场独立审计（验证欠账清单条目）

## 结论摘要

`docs/benchmark-qualification-status.md` 记录的欠账是「Cook 膜位移响应通过、缺独立应力参考」。
本单为该条目建立独立审计通道，结论如下（honest-FAIL，不做粉饰）：

| 项 | 结论 | 证据 |
|---|---|---|
| 已发表参考位移 23.96（加载边中点 48,52） | **确认**：两套独立离散均落在 3% 内；Q4 四级网格 Richardson 外推 23.967（0.03%） | `benchmarks/cook_membrane_stress_reference.json` → `platform.response`、`reference.probe` |
| 独立 Q8 参考解自收敛性 | **通过**：五个内部采样点 von Mises 的 GCI(最细) 0.049% | `reference.points[*].von_mises.gci_fine_relative` |
| 生产 Q4 应力场精度认证 | **仍未闭环（WITHHELD）**：审计自带的点式 Q4 应力恢复与 `cook_fields` 存盘单元中心应力不一致，点式比较不可认证 | `recovery_cross_check` |
| 加载边（y=44..60）位移型线 | **未解释分歧**：Q4 近水平（23.2→25.0），Q8 强倾斜（7.6→52.2），两法应收敛到同一解 | `loaded_edge_profiles` |

因此本次审计**没有**把 `cook_fields` 的 `stress_accuracy_qualified` 翻成 True，也没有改动
`docs/benchmarks/tutorials/cook_membrane_full.md` 及其哈希清单：报告正文的「无独立应力参考」表述
在同一次审计中依然成立。

## 方法与工具链

1. **独立参考实现**（`src/tensorfem/cook_stress_reference.py`，纯 NumPy，不复用 `tensorfem.continuum`
   的装配与求解代码）：
   - 8 节点 serendipity（Q8）平面应力单元，3×3 Gauss 积分，等参双线性映射；
   - 自建 COO 汇总→CSR 结构，Jacobi 预处理共轭梯度，容差 1e-13；
   - 加载边按二次边形函数施加均布竖向面力，合力恒为 1（核验 `load_resultant`）；
   - 固定左边界 x=0 全部自由度。
2. **收敛与不确定度**：网格 n=12/24/48，对五个固定材料点（父坐标 (0.5,0.5)、(0.3,0.5)、
   (0.7,0.5)、(0.5,0.25)、(0.5,0.75)）的应力分量做 Richardson 外推 + Roache GCI（安全系数 1.25）。
   采样点全部位于内部，远离夹持角点与加载边端点处的载荷间断奇异区。
3. **生产代码侧**：`tensorfem.additional_benchmark_fields.cook_fields(n)`，n=8/16/32/64，
   原样调用，不修改生产路径。
4. 复现：`OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_cook_membrane_stress.py`，
   产物即上表引用的 JSON。

## 计算结果（程序化注入，零手抄）

Q8 独立解（容差 1e-13）：

| n | 节点 | 单元 | CG 迭代 | 相对残差 | 合力 |
|---:|---:|---:|---:|---:|---:|
| 12 | 481 | 144 | — | <1e-13 | 1.000000 |
| 24 | 1825 | 576 | — | <1e-13 | 1.000000 |
| 48 | 7105 | 2304 | — | <1e-13 | 1.000000 |

具体迭代数、逐级应力、Q4 四级位移与五个采样点的参考/计算对照表全部写在
`benchmarks/cook_membrane_stress_reference.json`（含环境版本、问题定义、样点父坐标与单元定位），
本文件不重复抄写以免出现手抄漂移；表格由 `scripts/audit_cook_membrane_stress.py` 生成。

## 注意点与局限

- **加载边型线分歧未定论**：Q4 在 n=16 与 n=64 上的型线都接近水平，Q8 在 n=12/24/48 上持续强倾斜，
  二者是同一边值问题的两套协调位移法，理论上必须收敛到同一解。审计未能在本单预算内定位该分歧
  （候选原因：加载边端点载荷间断的边界层、Q4 完全积分在弯曲主导区的剪切锁定、以及审计自身 Q4
  点式恢复的未验证性）。该项作为开放问题保留，**不得**据此宣称任一方「更准」。
- **点式恢复交叉校验失败**：审计用自写 `q4_strain_matrix` 复算 `cook_fields` 存盘的单元中心应力，
  相对差达数百 %，说明审计自身的点式恢复尚未通过验证；因此五点的 Q4 vs Q8 对照（JSON 中
  `comparison`）只能作为「分歧存在」的证据，不能作为 Q4 误差量。任何把该对照当作误差百分比的引用
  都是误用。
- **适用范围**：本审计只讨论线弹性平面应力、E=1、ν=1/3、厚度 1、单位合力剪切载荷下的 Cook 梯形膜；
  不涉及几何非线性、塑性、板壳或接触。采样点结论不覆盖夹持角点与加载边端点的峰值应力。
- **奇异区**：夹持边与自由边的交界 (0,44)、以及加载边端点的载荷间断处应力奇异/不连续，点式值随网格
  发散，任何「峰值应力误差 x%」的说法在这些位置都无意义。

## 后续建议（给下次自找活）

1. 用有限差分对 `field_stress`/`q4_strain_matrix` 的 B 矩阵做独立的数值微分校验（`u` 插值后中心差分），
   一次性判定 Q4 点式恢复与 Q8 应力回代孰对孰错——这是解开型线分歧的最短路径。
2. 分歧定位后，再把 `cook_fields` 的 `stress_accuracy_qualified` 与
   `scripts/validate_additional_benchmark_reports.py` 的硬性禁止条款一并改为「有证据才放行」的门禁。

## 定谳（m2430，2026-11-03）：型线分歧已闭环，缺陷在审计自身

上一节两条开放问题由同一根因解释，且**生产代码无缺陷**。审计模块 `src/tensorfem/cook_stress_reference.py`
的两处单元算子把物理梯度写成了 `dN @ inv(J)`，正确形式是 `dN @ inv(J).T`（行存 `J[a,b]=dX_b/dxi_a`
时链式法则为 `dN/dx_b = sum_a (J^-1)_{b,a} dN/dxi_a`）；两者仅在 Jacobian 对称（矩形单元）时等价。
Cook 梯形映射的 Jacobian 非对称，因此差异被放大到 O(1)。此外 Q4 的父坐标导数表把节点 3/4 的
`1+eta`、节点 2/3 的 `1+xi` 误写为 `eta`、`xi`，且缺 1/4，使返回的 measure 偏大 16 倍。

### 判据一：与代码约定无关的三项独立性检验（skew 四边形 (0,0),(10,6),(10,9),(0,7)）

| 检验 | 修复前 | 修复后 |
|---|---|---|
| 常应变 patch（线性位移场在 4/9 个 Gauss 点的恢复应变 vs 解析值） | 偏差 1.1e-3 ~ 2.4e-3（量级等于应变本身） | 6.3e-18（Q4）/ 6.1e-18（Q8） |
| 常应变能量恒等式 `½uᵀKu = ½σ:ε·A·t`（不用被测代码的任何约定） | 单元能量/精确值 = **8.466**（+747%） | **1.000**（−6.4e-14） |
| 恢复梯度 vs 位移插值的有限差分（Newton 反演物理点） | 不一致 | 一致至 1e-12 |

### 判据二：两套离散的加载边型线

| 来源 | y=44 处 u_y | y=60 处 u_y | 加载边中点(48,52) |
|---|---:|---:|---:|
| 生产 Q4，n=64 | 23.160 | 25.043 | 23.925（对 23.96：0.148%） |
| 审计 Q8，d=24（修复后） | 23.186 | 25.110 | 23.949（0.046%） |
| 审计 Q8，d=12（修复后） | 23.138 | 25.016 | 23.919（0.172%） |
| 审计 Q8，d=12（修复前） | 7.569 | 51.605 | 24.210 |

修复后两套离散在加载边全长上相对差 < 0.3%（离散度水平），修复前相差 6.8 倍。d=48 时 Q8 中点
23.9605（0.002%），Richardson/GCI(finest)=0.068%。

### 判据三：生产 Q4 未被改动

`src/tensorfem/continuum.py::q4_stiffness` 的 `deriv` 使用 `jac = Jᵀ` 与 `dN @ inv(jac)`，等价于
`dN @ inv(J).T`，本身正确：同一常应变能量检验下其单元能量相对误差 **+5.5e-14**，且矩形单元上与
独立实现逐位一致（差值 0.0）。本单只改审计模块，未动生产路径。

### 残留欠账（honest，不粉饰）

1. **Q4 逐点应力仍未 3% 闭环**：n=8/16/32/64 的最大 von Mises 相对误差 30.52% / 28.83% / 8.02% / 5.02%，
   最细网格仍 > 3%，故 `cook_fields` 的 `stress_accuracy_qualified` 保持 False。该比较是「单元中心存盘
   应力 vs 内部材料点」，含一阶采样偏差（误差随 h 近似线性下降，30.5%→5.0% 对应 n 增大 8 倍），
   因此 5.02% 不能直接当作单元精度结论；要闭环需按材料点做单元内插值恢复。
2. **点式恢复交叉校验仍 fail-closed**：`recovery_cross_check` 仍报 inf（16 个样本中有分量为 0 的
   除数），说明采样集合包含近零应力分量，需要改为按尺度归一化后再比较；本单未改该项判据。

### 复现

```
OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_cook_membrane_stress.py   # → benchmarks/cook_membrane_stress_reference.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_cook_oracle_operator.py -q  # 10 passed
```

图：`docs/assets/cook-oracle-resolution.png`（左：加载边型线三套离散对照；中：响应收敛；
右：常应变能量恒等式修复前后）。回归测试 `tests/test_cook_oracle_operator.py` 用 patch/能量/FD 三项
约定无关判据锁死该缺陷，并用 fail-closed 断言保持第 1 条欠账为 False。
