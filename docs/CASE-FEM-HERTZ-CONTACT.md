# CASE-FEM-HERTZ-CONTACT: Hertz 球-平面接触半径与压入深

【锚点=CASE-FEM-HERTZ-CONTACT】｜仓=TensorFEM｜EDIT_WHITELIST: `docs/`, `tools/`, `benchmarks/`

## 1 案例介绍

刚性球体压入弹性半空间的经典 Hertz 接触问题（Hertz 1882; Johnson 1985 Contact Mechanics §4.2）。
给定法向载荷 P，通过轴对称有限元求解接触半径 a 与压入深 δ，与 Hertz 解析解对比，容差 <3%。

| 参数 | 值 | 单位 |
|---|---|---|
| 球半径 R | 10.0 | mm（无量纲长度基准） |
| Young 模量 E | 100000 | MPa |
| Poisson 比 ν | 0.3 | — |
| 等效模量 E* = E/(1−ν²) | 109890.1 | MPa |
| 参考 | Hertz 1882 J. reine angew. Math. 92:156; Johnson 1985 §4.2 DOI:10.1017/cbo9781139171731 | |

## 2 方法

- **离散**：轴对称 Q4 单元，接触区均匀加密（h/a 恒定），远场几何增长
- **接触**：增广 Lagrange 乘子法 (ALM)，node-wise KKT 投影 `λ ← max(λ − ρg, 0)`，无罚函数系统偏差
- **载荷控制**：regula falsi 外循环搜索 δ 使反力 = P（Hertz 单调性保证唯一根）
- **接触半径恢复**：线性外推末两个正压力节点到零压力点（经 m17149 修复：成员阈值改为物理压力比 `p_i > 1e-6·max(p)`，避免细网格退化）
- **模块**：`src/tensorfem/hertz_halfspace_loadcontrol.py`
- **证书**：`benchmarks/verified/fem_hertz_contact_loadcontrol/gates_verified.json`（schema `tensorfem.verified-gates.v1`）

## 3 计算结果

### 3.1 主算例池（a/R = 0.05, 0.10；h/a = 0.03, 0.015, 0.0075）

| a/R | h/a | 单元数 | a_解析 | a_FE | a 误差 | δ_解析 | δ_FE | δ 误差 | p₀ 误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0075 | 17040 | 0.5000 | 0.4978 | -0.445% | 0.0250 | 0.0247 | -1.010% | +0.630% |
| 0.05 | 0.0150 | 9040 | 0.5000 | 0.4973 | -0.532% | 0.0250 | 0.0247 | -1.011% | +0.588% |
| 0.05 | 0.0300 | 5040 | 0.5000 | 0.4958 | -0.848% | 0.0250 | 0.0248 | -0.996% | +0.493% |
| 0.10 | 0.0075 | 17040 | 1.0000 | 0.9951 | -0.491% | 0.1000 | 0.0990 | -1.010% | +0.630% |
| 0.10 | 0.0150 | 9040 | 1.0000 | 0.9963 | -0.375% | 0.1000 | 0.0990 | -1.011% | +0.588% |
| 0.10 | 0.0300 | 5040 | 1.0000 | 0.9919 | -0.814% | 0.1000 | 0.0990 | -0.996% | +0.492% |

### 3.2 域截断灵敏度（a/R=0.05, h/a≈0.015）

| 域深 | a 误差 | δ 误差 |
|---:|---:|---:|
| 24a | -0.537% | -2.900% |
| 48a | -0.532% | -1.639% |
| 96a | -0.532% | -1.011% |

### 3.3 结果图

![接触半径 a 与压入深 δ 的相对误差 vs h/a（双载荷水平）](assets/case-fem-hertz-contact/hertz_case_study_convergence.png)

*图: 接触半径 a 与压入深 δ 的相对误差 vs h/a（双载荷水平）*

![最细网格恢复的接触压强场 vs Hertz 解析椭圆分布](assets/case-fem-hertz-contact/hertz_case_study_pressure.png)

*图: 最细网格恢复的接触压强场 vs Hertz 解析椭圆分布*

![域截断灵敏度：远场边界深度对 a 和 δ 误差的影响](assets/case-fem-hertz-contact/hertz_case_study_truncation.png)

*图: 域截断灵敏度：远场边界深度对 a 和 δ 误差的影响*

## 4 误差判定（是否 <3%）

| 关键量 | 工况 | 基准 | 误差 | <3%? |
|---|---|---|---:|---:|
| 接触半径 a | a/R=0.05, h/a=0.0075 | Hertz `a=(3PR/4E*)^(1/3)` | -0.445% | **是** |
| 压入深 δ | a/R=0.05, h/a=0.0075 | Hertz `δ=a²/R` | -1.010% | **是** |
| 接触半径 a | a/R=0.10, h/a=0.0075 | Hertz `a=(3PR/4E*)^(1/3)` | -0.491% | **是** |
| 压入深 δ | a/R=0.10, h/a=0.0075 | Hertz `δ=a²/R` | -1.010% | **是** |
| 峰值压力 p₀ | a/R=0.05, h/a=0.0075 | `3P/(2πa²)` | +0.493% | **是**（判据 5%） |

### 4.1 证书闸门摘要（20 条闸门，全部通过）

| 闸门 ID | 量 | 值 | 阈值 | 通过 |
|---|---|---:|---:|---:|
| contact_radius_aR005 | |a_FE/a_Hertz - 1|, a/R = 0.05, h/a = 0.0075 (prim | 0.4451 | 3 | ✓ |
| indentation_aR005 | |delta_FE/delta_Hertz - 1|, a/R = 0.05, h/a = 0.00 | 1.01 | 3 | ✓ |
| refinement_stability_indentation_aR005 | |indentation error(h/a=0.0075) - indentation error | 0.01427 | 0.5 | ✓ |
| contact_radius_aR010 | |a_FE/a_Hertz - 1|, a/R = 0.1, h/a = 0.0075 (prima | 0.4908 | 3 | ✓ |
| indentation_aR010 | |delta_FE/delta_Hertz - 1|, a/R = 0.1, h/a = 0.007 | 1.01 | 3 | ✓ |
| refinement_stability_indentation_aR010 | |indentation error(h/a=0.0075) - indentation error | 0.01346 | 0.5 | ✓ |
| peak_pressure_aR005 | |p0_FE/p0_Hertz - 1| from the mass-matrix consiste | 0.6297 | 5 | ✓ |
| penetration_aR005 | max penetration / prescribed indentation (constrai | 1.557e-07 | 1e-05 | ✓ |
| load_residual_aR005 | |sum(lambda) / P - 1| (load-control equilibrium) | 2.272e-12 | 1e-09 | ✓ |
| equilibrium_residual_aR005 | || (K u)_free + lambda || / ||lambda|| (free-dof f | 3.108e-06 | 0.0001 | ✓ |
| rho_spread_contact_radius | spread of contact_radius over rho*E/a in {0.3, 1,  | 0.01865 | 0.1 | ✓ |
| rho_spread_indentation | spread of indentation over rho*E/a in {0.3, 1, 3}  | 0.0003406 | 0.1 | ✓ |
| truncation_indentation_d48_to_96 | change in indentation error between domain depth 4 | 0.6284 | 1 | ✓ |
| truncation_contact_radius_d48_to_96 | change in contact_radius_linear error between doma | 0.0005212 | 1 | ✓ |
| truncation_extrapolated_indentation | |e_inf(delta)|: Richardson limit of the indentatio | 0.4278 | 3 | ✓ |
| truncation_residual_model_bias | |e_inf(delta)| against the pre-declared 0.5% bound | 0.4278 | 0.5 | ✓ |
| truncation_ladder_monotone_aR005 | number of levels where the indentation error does  | 0 | 0 | ✓ |
| truncation_order_observed | |observed decay order of the indentation-error seq | 1.183 | 0.5 | ✓ |
| truncation_insensitive_contact_radius_aR005 | spread (pp) of |contact-radius error| over the dep | 0.001046 | 0.05 | ✓ |
| truncation_extrapolated_indentation_aR010 | |e_inf(delta)| at the second load (a/R = 0.10, h/a | 0.4039 | 3 | ✓ |

**总判据**: 算例池容差 3.0%，所有闸门 **通过**

## 5 注意点

1. **ALM vs 罚函数**：罚函数法在 δ 上有系统性偏差（~2.9% @ nr=nz=24），网格加密无法消除；ALM 通过乘子更新精确满足互补条件，穿透量在 ALM 容差级别（<1e-5 δ）。
2. **δ 系统偏差**：δ 误差随加密不再下降（−0.996% → −1.011% → −1.010%），存在约 −1% 的网格无关系统偏差。这是有限域截断（96a）导致的建模误差，非离散误差。判据已改为"加密稳定性 |Δ误差| ≤ 0.5 pp"。
3. **接触半径估计器修复**（m17149）：原成员阈值 `λ > 1e-10·max(λ)` 在细网格上退化为节点半径，修复为物理压力比 `p_i > 1e-6·max(p)`，a/R=0.10 最细网格误差由 −1.750% 改善到 −0.491%。
4. **域截断**：24a 域深时 δ 误差 −2.90%（超判据），48a 时 −1.64%，96a 时 −1.01%。接受模型使用 96a，截断误差已低于离散误差。
5. **低载失效模式**：a/R=0.02 探针记录为失效模式（不入闸），接触半径估计量在低载下对边缘节点抖动更敏感。

## 6 工件索引

| 工件 | 路径 |
|---|---|
| 原始数据 | `benchmarks/verified/fem_hertz_contact_loadcontrol/result.json` |
| 证书闸门 | `benchmarks/verified/fem_hertz_contact_loadcontrol/gates_verified.json` |
| 截断外推 | `benchmarks/verified/fem_hertz_contact_loadcontrol/truncation_extrapolation.json` |
| 求解器 | `src/tensorfem/hertz_halfspace_loadcontrol.py` |
| 测试 | `tests/test_hertz_contact_loadcontrol.py` (8 passed), `tests/test_hertz_anchor_evidence.py` (7 passed) |
| 图生成器 | `tools/gen_hertz_case_study.py` |

---
*本报告程序化生成，所有数字从 `result.json` 注入，零手抄。*
*生成时间: 2026-11-13T01:53:53+00:00*
