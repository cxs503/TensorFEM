# GAP-PP-3-ICE-STRUCT-PLASTIC-SET 验证报告

## 1. 案例介绍（物理背景/几何/参数表）

**物理背景**: 冰-结构相互作用中，板格在大斑载荷下的失效模式由塑性储备与永久挠度控制，
而非弹性应力上限。本验证针对北极级船体结构钢（DH36/EH36/FH36）在低温下的弹塑性板单元能力，
解决 JOUR-ICE-STRUCT §5 局限2 中指出的"现有板能力仅线弹性，无法回答允许斑压力的塑性判据"问题。

**几何参数**:

| 参数 | 值 | 单位 |
|------|-----|------|
| 板长 a | 2.0 | m |
| 板宽 b | 1.0 | m |
| 板厚 t | 0.02 | m |
| 斑载荷尺寸 | 0.5×0.5 | m |
| 钢种 | DH36 | - |
| 温度 | -40.0 | °C |
| 屈服应力 | 400 | MPa |
| 弹性模量 | 206 | GPa |
| 泊松比 | 0.3 | - |
| 硬化模量 | 1000 | MPa |

**参考标准**: DNV-RP-C204, CSR Polar Ships §5, Ilyushin (1948) Plasticity.

## 2. 方法（怎么做的）

### 2.1 低温钢流变应力模型
采用分段线性插值，基于 DNV-RP-C204 和 CSR Polar Ships 发布的 Arctic-grade 钢材数据点。
温度范围 -60°C 至 +20°C，覆盖 DH36/EH36/FH36 三个等级。

### 2.2 弹性参考解（Navier 双正弦级数）
简支矩形板在均布/斑载荷下的弹性挠度和弯矩，采用 Navier 双正弦级数求解（41 项截断）。
独立于有限元网格，作为弹性-塑性过渡点的解析基准。

### 2.3 塑性极限载荷上下界
- **Strip 下界**: 将板视为独立条带，取各条带塑性极限的最小值
- **Yield-line 上界**: 基于虚功原理的屈服线机制，最小化两种机构族
- **Sine 上界**: Ilyushin 正弦模态上限解，考虑斑载荷的傅里叶幅值

### 2.4 弹塑性过渡验证
采用二分法精化弹性-塑性过渡点，载荷步 50 步 + 10 次二分精化，
分辨率达到 ~0.02% 的解析屈服载荷。

### 2.5 永久挠度计算
通过弹性回弹（eigenstrain springback）计算：将塑性应变作为初始应变，
求解线性弹性问题得到残余挠度。

## 3. 计算结果（数字表，程序化注入）

### 3.1 参考解汇总

| 量 | 值 | 单位 |
|----|-----|------|
| 首次屈服压力 (Navier) | 0.2623 | MPa |
| Yield-line 上界 (均布) | 0.5656 | MPa |
| Strip 下界 (斑载) | 0.6095 | MPa |
| Sine 上界 (斑载) | 1.8237 | MPa |
| Yield-line 上界 (斑载) | 2.1943 | MPa |
| 屈服处中心挠度 | 5.7257 | mm |
| 塑性弯矩 Mp | 40.00 | kN·m/m |

### 3.2 弹塑性过渡验证（4×2 网格）

| 量 | 值 | 单位 |
|----|-----|------|
| 解析屈服载荷 | 355.0000 | MPa |
| FE 屈服载荷 | 355.0004 | MPa |
| 相对误差 | 0.00 | % |
| 载荷分辨率 | 0.000650 | MPa |
| 是否通过 (<5%) | 是 | - |

### 3.3 误差判定

| 对比项 | 误差 | 判定 |
|--------|------|------|
| Navier 1×1 vs Timoshenko 0.0479 | 0.03% | ✓ <3% |
| Navier 2×1 vs Timoshenko 0.1017 | 0.02% | ✓ <3% |
| Navier strip vs 0.125 | 0.10% | ✓ <3% |
| 弹塑性过渡点 | 0.00% | ✓ <5% |

## 4. 结果图

### 图1: 弹性载荷-挠度曲线与塑性极限标记
![elastic_load_deflection_with_markers](figure:elastic_load_deflection_with_markers)

### 图2: 塑性极限载荷上下界对比
![plastic_limit_load_bounds](figure:plastic_limit_load_bounds)

### 图3: 低温钢流变应力-温度曲线
![flow_stress_temperature](figure:flow_stress_temperature)

### 图4: Navier 弯矩系数 vs 长宽比
![navier_moment_coefficients](figure:navier_moment_coefficients)

### 图5: 弹塑性过渡验证（4×2 网格）
![elastic_plastic_transition](figure:elastic_plastic_transition)

### 图6: 永久挠度机制示意图
![permanent_deflection_schematic](figure:permanent_deflection_schematic)

## 5. 注意点（踩过的坑/局限/适用范围）

### 5.1 已修复的审计缺陷
- **F1**: 弹塑性过渡采用二分法精化（原为步长量化，误差被步长网格人为限制）
- **F2**: 移除 FE 失败时的 false-pass 回退
- **F3**: 修正屈服线公式（原方形板结果偏低 50%）
- **F4**: 修正首次屈服压力（原使用无来源的 alpha 猜测值）
- **F5**: 修正网格收敛（原使用硬编码 target_load 覆盖计算结果）
- **F6**: 修正文档字符串（无弧长延续法）

### 5.2 局限性
- 流变应力模型为分段线性插值，未考虑应变率效应和韧脆转变
- Ilyushin 解假设理想塑性（无硬化），与 FE（含硬化）的比较为下界检查
- 斑载荷通过 tributary area 施加为节点力，未实现切割单元上的压力积分
- 永久挠度通过弹性回弹计算，忽略卸载过程中的几何非线性

### 5.3 适用范围
- 温度范围：-60°C 至 +20°C（数据点范围内）
- 板厚范围：薄板至中厚板（Mindlin 壳理论适用）
- 载荷类型：均布压力、矩形斑载荷
- 边界条件：简支（Navier 解）、固支（FE 网格）

## 6. 误差判定总结

| 验收判据 | 要求 | 实际 | 状态 |
|----------|------|------|------|
| 弹性→塑性过渡点误差 | <5% | 0.00% | ✓ 通过 |
| 塑性极限载荷 vs Ilyushin | <10% | 待 HPC 全网格验证 | ⏳ 待完成 |
| 3 档网格收敛 |  successive <5% | 待 HPC 全网格验证 | ⏳ 待完成 |
| pytest ≥10 绿 | ≥10 | 35+16=51 tests | ✓ 通过 |

**结论**: 解析参考解与经典文献（Timoshenko 表、Ilyushin 解、yield-line 理论）一致，
弹塑性过渡验证在粗网格上满足 <5% 误差要求。塑性极限载荷和网格收敛的完整验证
需要 HPC 资源（已在 cj3469 作业中提交）。

---
*报告生成时间: 2026-11-15T18:18:35.755697*
*验证脚本: verify_gap_pp3_plastic_set.py + gen_gap_pp3_plastic_set_report.py*
*测试文件: tests/test_ice_plate_plasticity.py (35 tests) + tests/test_ice_plate_plasticity_permanent.py (16 tests)*
