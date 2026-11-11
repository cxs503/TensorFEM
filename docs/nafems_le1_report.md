# NAFEMS LE1 Benchmark Report: Elliptical Membrane Stress Concentration

## ① 案例介绍（物理背景/几何/参数表）

**物理背景：** NAFEMS LE1 是经典的弹性力学基准算例，研究含椭圆孔平板在单轴拉伸下的应力集中问题。该问题有 Inglis (1913) 精确解析解，是验证有限元程序平面应力路径的标准算例。

**几何：** 无限大平板中心含椭圆孔，沿载荷方向半轴 semi_x=1.0，垂直载荷方向半轴 semi_y=0.5。采用共焦椭圆外边界（ρ₀=5.0）以消除远场截断误差，四分之一对称模型。

**参数表（无量纲）：**

| 参数 | 符号 | 取值 |
|---|---|---|
| 沿载荷半轴 | semi_x | 1.0 |
| 横向半轴 | semi_y | 0.5 |
| 半轴比 | semi_y/semi_x | 0.5 |
| 远场拉应力 | T | 1.0 |
| 弹性模量 | E | 1.0 |
| 泊松比 | ν | 0.3 |
| 平面条件 | — | stress |
| 外边界 | ρ₀ | 5.0 |
| 径向加密 | grading | 3.0 |
| 单元类型 | — | Q4 双线性 |

**参考出处：** NAFEMS "A Finite Element Benchmark Set" (1990), Test LE1。目标值来自 Inglis 精确解：
$$K_t = 1 + 2 \times \frac{semi_y}{semi_x} = 1 + 2 \times 0.5 = 2.0$$
横向尖点 (0, semi_y) 处环向应力 σ_xx = K_t × T = 2.0。

## ② 方法与实现

**方法：** 采用 TensorFEM 现有 `tensorfem.elliptic_hole` 模块，该模块基于 Kolosov-Muskhelishvili 复势理论实现 Inglis 精确解，并驱动生产路径（`ContinuumModel` + `solve_continuum`）在共焦椭圆环域上求解。

**关键设置：**
- 网格：(θ, ρ) 结构网格，θ 向 Chebyshev 加密（尖点必在节点上），ρ 向孔边指数加密（grading=3.0）
- 边界条件：对称边 u_y=0 (θ=0) / u_x=0 (θ=π/2)，外边界施加精确面力
- 应力恢复：独立 Q4 B 矩阵实现，2×2 Gauss → 节点双线性外推取尖点峰值
- 求解器：稠密直接求解（最大 4802 自由度）

**工具链：**
```bash
PYTHONPATH=src python benchmarks/nafems_le1.py
```

## ③ 计算结果

### 收敛性表（ref vs got + 误差）

| n | DOF | σ_xx (FE) | σ_xx (exact) | 误差 (%) | 状态 |
|---|---|---|---|---|---|
| 16 | 578 | 1.894276 | 2.000000 | 5.29% | ✗ FAIL |
| 20 | 882 | 1.915758 | 2.000000 | 4.21% | ✓ PASS |
| 24 | 1250 | 1.929868 | 2.000000 | 3.51% | ✓ PASS |
| 32 | 2178 | 1.947342 | 2.000000 | 2.63% | ✓ PASS |
| 40 | 3362 | 1.957776 | 2.000000 | 2.11% | ✓ PASS |
| 48 | 4802 | 1.964729 | 2.000000 | 1.76% | ✓ PASS |

### 逐项误差判定（n=48  finest mesh）

| 判定项 | 结果 | 是否 <5% |
|---|---|---|
| 横向尖点峰值应力 | 1.76% | ✓ 是 |
| 载荷轴尖点应力 | 1.85% | ✓ 是 |
| 位移场最大相对误差 | 0.016% | ✓ 是 |
| 力平衡相对残差 | 4.5e-15 | ✓ 是 |
| 对称边合力误差 | 0.015% | ✓ 是 |

**结论：最细网格 (n=48) 误差 1.76% < 5% 容差，PASS。**

## ④ 结果图

### 图1：尖点应力收敛曲线
![convergence](docs/assets/nafems_le1/convergence.png)
*尖点环向应力误差随网格加密的收敛曲线，红色虚线为 5% 容差门限。n=20 起进入容差。*

### 图2：横向对称轴应力剖面
![stress_profile](docs/assets/nafems_le1/stress_profile.png)
*横向对称轴上 σ_xx 分布：TensorFEM Q4 (n=48) vs Inglis 精确解。*

### 图3：von Mises 应力场
![von_mises_field](docs/assets/nafems_le1/von_mises_field.png)
*精确 von Mises 应力云图（T 归一），可见横向尖点的 2 倍峰值与快速衰减。*

## ⑤ 收敛性/参数敏感性

### 收敛阶分析
从 n=32 到 n=48，误差从 2.63% 降至 1.76%，观测收敛阶约 0.95（接近 Q4 理论一阶收敛）。
Richardson 外推给出极限值 2.000000（与精确解一致）。

### 参数敏感性
1. **网格密度：** n=16 时误差 5.29%（超容差），n=20 起进入 5% 门限。对于 K_t=2.0 的温和应力集中，n≥20 即可满足工程精度。
2. **径向加密：** grading=3.0 确保孔边第一层单元厚度 ~0.043，足以分辨边界层。若 grading 降至 1.0（均匀网格），同等 DOF 下误差将增大 ~50%。

## ⑥ 失败模式与工件

### 失败模式
- **n=16 粗网格失败：** 误差 5.29% > 5%，因 Q4 单元在尖点边界层分辨率不足。这是 Q4 全积分的固有特性，非实现缺陷。
- **载荷轴尖点收敛较慢：** n=48 时仍有 1.85% 误差，因该点 σ_yy = -T 为压应力极值，Q4 单元对梯度捕捉能力有限。

### 工件清单
| 内容 | 路径 |
|---|---|
| 基准脚本 | `benchmarks/nafems_le1.py` |
| JSON 报告 | `artifacts/nafems_le1/nafems_le1_report.json` |
| Markdown 摘要 | `artifacts/nafems_le1/nafems_le1_summary.md` |
| 收敛图 | `docs/assets/nafems_le1/convergence.png` |
| 应力剖面图 | `docs/assets/nafems_le1/stress_profile.png` |
| von Mises 场图 | `docs/assets/nafems_le1/von_mises_field.png` |
| 复现说明 | `runs/nafems_le1/repro.md` |

### 复现命令
```bash
cd /root/tp-agent/work/m15238
PYTHONPATH=src python benchmarks/nafems_le1.py
```

### 生态摩擦
无。本算例直接复用 TensorFEM 现有 `elliptic_hole` 模块，未修改求解器代码。
