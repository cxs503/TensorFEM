# GAP-PP-3-ICE-STRUCT-PLASTIC-SET: 板格低温钢弹塑性 + 永久挠度

## 1. 案例介绍

### 物理背景
冰-结构相互作用中，大斑载荷（ice patch pressure）作用下的板格结构，其实际控制量是**塑性储备**与**永久挠度**，而非弹性应力上限。现有板能力仅线弹性 Kirchhoff/Mindlin，无法回答"允许斑压力"的塑性判据。

### 几何与参数
| 参数 | 值 | 单位 |
|------|-----|------|
| 板长 a | 1.0 | m |
| 板宽 b | 0.5 | m |
| 板厚 t | 0.01 | m |
| 弹性模量 E | 206 | GPa |
| 泊松比 ν | 0.3 | - |
| 屈服应力 σ_y (DH36, 20°C) | 355 | MPa |
| 硬化模量 H | 1000 | MPa |
| 低温钢等级 | DH36/EH36/FH36 | - |
| 温度范围 | 20°C ~ -60°C | °C |

### 斑载荷工况
- 斑载荷区域：板中心 50%×50% 区域
- 载荷类型：均匀压力（模拟冰压力）
- 边界条件：四边简支 / 四边固支

## 2. 方法

### 2.1 低温钢流变应力模型
采用分段线性插值的温度相关屈服应力模型，覆盖 DH36/EH36/FH36 三个 Arctic-grade 钢种，温度范围 20°C ~ -60°C。数据源自 DNV-RP-C204 和 CSR Polar Ships。

### 2.2 Ilyushin 板极限分析
Ilyushin 全塑性极限曲面（1948）作为塑性极限载荷的解析基准：
- 单向压缩：σ_limit = σ_y
- 横向压力：p_limit = 4·Mp/b² · (2b/a + a/b)，其中 Mp = σ_y·t²/4

### 2.3 斑载荷施加
通过 tributary area 法将均匀压力分布为节点力，支持矩形斑载荷区域。

### 2.4 永久挠度后处理
从塑性状态出发，通过弹性回弹（eigenstrain 方法）计算卸载后的残余挠度。

### 2.5 弹塑性过渡验证
通过增量加载检测首个积分点屈服，与解析屈服载荷对比（验收：<5%）。

### 2.6 塑性极限载荷验证
通过增量加载追踪极限点，与 Ilyushin 解对比（验收：<10%）。

### 2.7 网格收敛
三档网格（4×2, 8×4, 16×8）收敛性研究。

## 3. 计算结果

### 3.1 低温钢流变应力
| 钢种 | 温度 (°C) | 屈服应力 (MPa) |
|------|-----------|----------------|
| DH36 | 20 | 355 |
| DH36 | -20 | 380 |
| DH36 | -40 | 400 |
| DH36 | -60 | 420 |
| EH36 | -40 | 440 |
| FH36 | -60 | 495 |

### 3.2 Ilyushin 极限载荷
| 工况 | 解析极限 | 单位 |
|------|----------|------|
| 单向压缩 (σ_y=355MPa) | 355 | MPa |
| 横向压力 (a=1,b=0.5,t=0.01) | 计算值 | Pa |

### 3.3 弹塑性过渡
- 解析屈服载荷：σ_y = 355 MPa（单向压缩参考载荷 = 1 Pa）
- FE 屈服载荷：通过增量加载检测首个积分点屈服
- 误差：待集群验证（目标 <5%）

### 3.4 塑性极限载荷
- Ilyushin 极限：σ_y = 355 MPa
- FE 极限：通过增量加载追踪
- 误差：待集群验证（目标 <10%）

### 3.5 网格收敛
三档网格收敛性研究：
- 粗网格：4×2 = 8 单元
- 中网格：8×4 = 32 单元
- 细网格：16×8 = 128 单元

## 4. 结果图

### 图1：低温钢屈服应力-温度曲线
（见 artifacts/ice_plate_flow_stress.png）

### 图2：Ilyushin 极限曲面
（见 artifacts/ilyushin_yield_surface.png）

### 图3：弹塑性过渡验证
（见 artifacts/ep_transition.png）

### 图4：网格收敛曲线
（见 artifacts/mesh_convergence.png）

## 5. 注意点

### 5.1 局限性
1. 流变应力模型为分段线性插值，未考虑应变率效应和韧脆转变
2. Ilyushin 解假设理想塑性（无硬化），与 FE 结果（含硬化）的比较为下限检验
3. 斑载荷通过 tributary area 施加，未实现切割单元的压力积分
4. 永久挠度通过弹性回弹计算，忽略卸载过程中的几何非线性

### 5.2 适用范围
- 小应变 Mindlin 板壳
- 等向硬化 J2 塑性
- 浅翘曲单元（warp ratio < 0.25）
- 温度范围：20°C ~ -60°C

## 6. 误差判定

| 验证项 | 基准 | 误差阈值 | 状态 |
|--------|------|----------|------|
| 弹塑性过渡 | 解析屈服载荷 | <5% | 待集群验证 |
| 塑性极限载荷 | Ilyushin 解 | <10% | 待集群验证 |
| 网格收敛 | 细网格参考 | <5% | 待集群验证 |
| 流变应力插值 | 文献数据 | <0.1% | ✅ 通过 |
| Ilyushin 单向极限 | σ_y | 0% | ✅ 通过 |
| Ilyushin 横向极限 | t² 缩放 | <10% | ✅ 通过 |
| Ilyushin 组合极限 | Q=1 | <1e-10 | ✅ 通过 |

### 测试通过情况
- **24 项轻量测试全部通过**（流变应力、Ilyushin 极限、斑载荷网格、永久挠度提取、模块集成）
- **6 项重量测试**（弹塑性过渡、塑性极限、网格收敛）已递交集群 cj3123，待验证

## 7. 代码结构

```
src/tensorfem/ice_plate_plasticity.py  # 主模块（~900 行）
tests/test_ice_plate_plasticity.py     # 测试（30 项）
docs/ice_plate_plasticity_report.md    # 本报告
```

### 新增 API
- `flow_stress_at_temperature(grade, temperature)` — 低温钢屈服应力
- `LowTempSteelModel` — 温度相关板壳模型包装器
- `ilyushin_plate_limit_stress(a, b, t, ...)` — Ilyushin 极限载荷
- `ilyushin_combined_limit(Nx, Ny, ...)` — Ilyushin 组合屈服函数
- `patch_load_plate_mesh(...)` — 斑载荷板网格
- `compute_permanent_deflection(...)` — 永久挠度计算
- `extract_max_permanent_deflection(...)` — 最大永久挠度提取
- `verify_elastic_plastic_transition(...)` — 弹塑性过渡验证
- `verify_plastic_limit_load(...)` — 塑性极限载荷验证
- `verify_mesh_convergence(...)` — 网格收敛验证
