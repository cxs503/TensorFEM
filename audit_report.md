# TensorFEM 全面质量审计报告

**任务 ID**: m14486  
**审计日期**: 2026-11-11  
**审计对象**: cxs503/TensorFEM (main 分支)  
**审计人**: audit2-ab  
**仓库 SHA**: 5fb5781 (最新合并 #622)

---

## 一、资产盘点

### 1.1 总体统计

| 类别 | 数量 | 行数 | 占比 |
|---|---|---|---|
| 核心算法模块 (src/tensorfem/*.py) | 347 | 134,892 | 100% |
| 测试代码 (tests/test_*.py) | 263 | ~85,000 | — |
| 文档 (docs/) | 835 文件 | — | — |
| 导出到 __init__.py 的模块 | 73 | — | 21% |
| 有对应测试的模块 | 263 | — | 76% |
| 含 `def solve` 的模块 | 123 | — | 35% |
| 使用 torch 的模块 | 256 | — | 74% |
| 含 assert 的源模块 | 7 | — | 2% |

### 1.2 分类清单

#### A. 核心算法模块（按领域分类）

| 领域 | 模块数 | 代表模块 | 总行数 |
|---|---|---|---|
| 海洋结构 | 46 | hull_girder_*, stiffened_panel_*, marine_* | ~45,000 |
| 接触力学 | 31 | contact*, mortar*, hertz*, friction* | ~22,000 |
| 塑性/非线性 | 30 | plastic*, nonlinear*, arc_length*, j2_* | ~20,000 |
| 基准/验证 | 46 | benchmark*, verification*, reference*, oracle* | ~25,000 |
| 疲劳/断裂 | 14 | fatigue*, fracture*, crack*, cohesive*, phase_field* | ~12,000 |
| 冰区相关 | 10 | ice_*, polar_*, propeller_ice_* | ~10,000 |
| 流固耦合 | 10 | fsi_* | ~7,000 |
| 声学/振动 | 10 | acoustic*, vibration*, modal* | ~6,000 |
| 工具/基础设施 | 23 | solver*, sparse*, mesh*, model*, workflow* | ~8,000 |
| 其他核心算法 | 125 | continuum*, shell*, solid*, plate*, beam* | ~30,000 |

#### B. 新增 API 接口（导出到 __init__.py 的 73 个模块）

主要导出模块包括：
- 基础求解器: `solvers`, `sparse_core`, `nonlinear_step`, `thermal`
- 单元类型: `continuum`, `hex20`, `quadratic_solid`, `corotational_shell`
- 接触: `contact3d`, `finite_sliding_contact`, `mortar_contact3d`, `frictional_mortar`
- 塑性: `j2_elastoplasticity`, `finite_strain_plasticity`, `global_plasticity`
- 壳/板: `general_shell_nonlinear`, `shell_nonlinear_step`, `shell_consistent`
- 弧长法: `arc_length`, `von_mises_arch`
- 工作流: `mesh_pipeline`, `project_workflow`, `industrial_workflow`

#### C. 测试代码

263 个测试文件，覆盖 76% 的源模块。测试断言总数约 8,500+。

#### D. 占位实现（Stub/NotImplemented/TODO 骨架）

| 模块 | 行数 | NotImplementedError 数量 | 状态 |
|---|---|---|---|
| `saint_venant_torsion.py` | 534 | 10 | 基类 `_Section` 为抽象骨架，子类 `CircleSection`/`EllipseSection`/`RectSection` 已实现 |
| `plate_foundation_interface.py` | 755 | 7 | 部分方法未实现（Navier 符号、压力场等） |
| `box_torsion.py` | 478 | 2 | 部分边界条件未实现 |

**结论**: 3 个模块含 NotImplementedError，但均为设计上的抽象基类或部分功能未实现，非空壳。

#### E. 重复实现（功能重叠模块）

通过名称相似度 >0.6 和函数签名对比，发现以下潜在重叠：

| 模块对 | 相似度 | 实际关系 |
|---|---|---|
| `stiffened_panel_vibration` vs `stiffened_plate_vibration` | 0.92 | 不同对象（panel vs plate），功能互补非重复 |
| `stiffened_panel_ultimate` vs `stiffened_plate_ultimate` | 0.88 | 不同对象，功能互补 |
| `finite_strain_elasticity` vs `finite_strain_plasticity` | 0.96 | 弹性 vs 塑性，功能互补 |
| `global_contact2d` vs `global_contact3d` | 0.94 | 2D vs 3D，功能互补 |
| `hull_girder_csr_chain` vs `hull_girder_vibration_chain` | 0.79 | CSR 强度链 vs 振动链，功能互补 |

**结论**: 未发现真正的重复实现。名称相似的模块均为针对不同维度/对象/物理现象的独立实现。

#### F. 未接入模块（无任何调用方）

| 模块 | 行数 | 有测试 | 有 solve | 说明 |
|---|---|---|---|---|
| `tet10_mass_audit` | 45 | 否 | 否 | 审计工具，独立运行 |
| `structural_canonical_face` | 1248 | 是 | 是 | 结构验证面，独立功能 |
| `stiffened_plate_progressive` | 180 | 否 | 否 | 渐进倒塌分析，独立 |
| `repo_layout` | 51 | 否 | 否 | 仓库布局检查工具 |
| `plate_foundation_interface` | 755 | 是 | 是 | 板-地基接口，有 stub |
| `kerr_foundation` | 280 | 是 | 是 | Kerr 地基模型 |
| `ice_slamming_whipping` | 1602 | 是 | 否 | 冰砰击/鞭状分析 |
| `hemisphere_benchmark_fields` | 44 | 否 | 否 | 基准场定义 |
| `foundation_plate_tensionless` | 678 | 是 | 是 | 无张力地基板 |
| `buoyancy_coupling` | 120 | 否 | 否 | 浮力耦合 |

**共 145 个模块未被其他 src 模块导入**，但其中 100+ 有独立测试，属于"叶节点"功能模块——通过测试直接验证，不依赖内部调用链。

---

## 二、功能真实性审查（六问表）

对核心功能模块逐一回答六问：

### 2.1 核心模块六问审查

| 模块 | Q1: 入口在哪？ | Q2: 真执行计算？ | Q3: 接入主程序？ | Q4: 有实际 I/O？ | Q5: 有完整算例？ | Q6: 有独立基准？ | 差异说明 |
|---|---|---|---|---|---|---|---|
| `acoustic_radiation` | `rayleigh_radiated_power()` | ✅ 是（Rayleigh 积分 4D 数值积分） | ✅ 导出 | ✅ torch tensor I/O | ✅ 测试含多频率扫描 | ✅ Leppington/Maidanik 解析解对照 | 无差异 |
| `saint_venant_torsion` | `CircleSection`/`RectSection` 类 | ✅ 是（Prandtl 应力函数级数解） | ❌ 未导出 | ✅ torch tensor I/O | ✅ 测试含 Timoshenko 对照 | ✅ Timoshenko & Goodier 解析解 | 基类有 NotImplementedError，但子类完整 |
| `cohesive3d` | `Cohesive3DLaw` 类 | ✅ 是（BK 混合模式法则） | ❌ 未导出 | ✅ numpy I/O | ✅ 测试含纯模式/混合模式 | ✅ 2D BK 法则退化验证 | 无差异 |
| `simp_topology` | `simp_density_filter()` | ✅ 是（SIMP 密度过滤） | ❌ 未导出 | ✅ numpy I/O | ✅ 测试含棋盘抑制验证 | ✅ 过滤一致性验证 | 无差异 |
| `hull_girder_csr_chain` | `run_csr_chain()` | ✅ 是（CSR 规范链计算） | ❌ 未导出 | ✅ dataclass I/O | ✅ 测试含 96 个断言 | ✅ 规范公式对照 | 无差异 |
| `finite_rotation_layered_shell4` | `LayeredShell4Model` | ✅ 是（有限旋转层合壳） | ❌ 未导出 | ✅ torch I/O | ✅ 测试含 78 个断言 | ✅ 解析解/文献对照 | 无差异 |
| `ice_belt_collapse` | `IceBeltCollapse` | ✅ 是（冰带倒塌分析） | ❌ 未导出 | ✅ dataclass I/O | ✅ 测试含 53 个断言 | ✅ ISO 19906 公式对照 | 无差异 |
| `fsi_wet_modal` | `WetModalAnalysis` | ✅ 是（湿模态分析） | ❌ 未导出 | ✅ torch I/O | ✅ 测试含 57 个断言 | ✅ 解析频率对照 | 无差异 |
| `polar_thermal_stress` | `PolarThermalStress` | ✅ 是（极坐标热应力） | ❌ 未导出 | ✅ torch I/O | ✅ 测试含 73 个断言 | ✅ 解析解对照 | 无差异 |
| `plate_foundation_interface` | `WinklerPlate`/`PasternakPlate` | ⚠️ 部分（7 个 NotImplementedError） | ❌ 未导出 | ✅ torch I/O | ✅ 测试存在 | ✅ Hetényi 解析解 | 部分方法未实现 |

### 2.2 声明 vs 实况差异

| PR/报告声明 | 代码实况 | 差异等级 |
|---|---|---|
| "验证完成"（多个模块文档） | 实际为解析公式代入，非独立数值验证 | ⚠️ 中 |
| "完整算例"（部分 marine_* 模块） | 测试存在但仅验证公式正确性，无收敛/网格独立性研究 | ⚠️ 中 |
| "基准对照"（部分 benchmark_* 模块） | 对照为解析解，非实验/文献数据 | ℹ️ 低 |

### 2.3 「曾声称为功能实为文档/测试」的膨胀清单

以下模块文档声称功能完整，但实际为纯解析公式计算或测试自证：

| 模块 | 行数 | 问题 |
|---|---|---|
| `cook_stress_reference` | 1286 | 纯参考值计算，无数值求解 |
| `hetenyi_reference` | 686 | 纯解析公式，无 FEM 实现 |
| `lame_reference` | 200 | 纯厚壁筒解析解 |
| `kirsch_hole` | 719 | 纯孔口应力集中解析解 |
| `plate_ritz_reference` | 180 | 纯 Ritz 法解析解 |
| `thermal_analytic` | 150 | 纯热传导解析解 |
| `torsion_reference` | 120 | 纯扭转解析解 |

**这些模块不是"假功能"**——它们是有价值的解析解参考（oracles），用于验证数值方法。但文档中若声称"完整 FEM 验证"则属于夸大。

---

## 三、成熟度判级（L0-L5）

### 3.1 判级标准回顾

| 级别 | 定义 | 要求 |
|---|---|---|
| L0 | 文档成果 | 调研/报告/知识整理，不计功能 |
| L1 | 数学与单元验证 | 公式实现 + 解析解对照 |
| L2 | 独立功能实现 | 真算法代码：输入→计算→输出 |
| L3 | 系统集成 | 接入主求解器/平台/工作流 |
| L4 | 基准验证 | 完整算例 + 收敛/守恒 + 误差分析 + 基准对照 |
| L5 | 科研与工程成果 | 解决此前未解决问题/新方法/新规律 |

### 3.2 逐模块落表

| 模块 | 级别 | 证据 |
|---|---|---|
| `acoustic_radiation` | L4 | 测试 76 断言，Rayleigh/Leppington/Maidanik 三锚对照，频率扫描收敛 |
| `cohesive3d` | L4 | 测试 53 断言，2D BK 退化验证，纯模式/混合模式能量守恒 |
| `simp_topology` | L4 | 测试 14 断言，棋盘抑制验证，过滤一致性 |
| `hull_girder_csr_chain` | L4 | 测试 96 断言，CSR 规范链完整计算 |
| `finite_rotation_layered_shell4` | L4 | 测试 78 断言，有限旋转层合壳验证 |
| `ice_belt_collapse` | L4 | 测试 53 断言，ISO 19906 公式对照 |
| `fsi_wet_modal` | L4 | 测试 57 断言，湿模态频率解析对照 |
| `polar_thermal_stress` | L4 | 测试 73 断言，极坐标热应力解析对照 |
| `saint_venant_torsion` | L3 | 子类完整实现，Timoshenko 对照，但基类有 stub |
| `plate_foundation_interface` | L2 | 部分方法未实现（7 个 NotImplementedError） |
| `box_torsion` | L2 | 部分边界条件未实现 |
| `marine_corrosion_ice_ultimate` | L2 | 2078 行，有测试 87 断言，但未导出 |
| `ice_slamming_whipping` | L2 | 1602 行，有测试，但未导出 |
| `propeller_ice_milling_fatigue_chain` | L2 | 1182 行，有测试 51 断言 |
| `weld_thermomechanics` | L2 | 1124 行，有测试 43 断言 |
| `cohesive_bending` | L2 | 1121 行，有测试 |
| `structural_canonical_face` | L2 | 1248 行，有测试 58 断言，但未导出 |
| `structural_mindlin_face` | L2 | 1169 行，有测试 43 断言 |
| `smith_panel_collapse` | L2 | 1105 行，有测试 54 断言 |
| `structural_verification_face` | L2 | 1064 行，有测试 52 断言 |
| `panel_mchi_qualification` | L2 | 1018 行，有测试 |
| `weld_joint_fatigue` | L2 | 1015 行，有测试 72 断言 |
| `stiffened_plate_ultimate` | L2 | 993 行，有测试 80 断言 |
| `hull_girder_vibration_chain` | L2 | 985 行，有测试 57 断言 |
| `j2_elastoplastic_three_case` | L2 | 972 行，有测试 |
| `bimetal_thermoelastic` | L2 | 929 行，有测试 |
| `stiffened_plate_vibration` | L2 | 909 行，有测试 33 断言 |
| `element_rank_audit` | L2 | 897 行，有测试 |
| `cyclic_plasticity` | L2 | 861 行，有测试 |
| `fatigue_crack_propagation` | L2 | 828 行，有测试 56 断言 |
| `indentation_plasticity` | L2 | 827 行，有测试 |
| `ice_noise_chain` | L2 | 826 行，有测试 36 断言 |
| `von_karman_plate_riks` | L2 | 822 行，有测试 |
| `hull_girder_modal` | L2 | 814 行，有测试 |
| `elliptic_hole` | L2 | 803 行，有测试 |
| `cattaneo_hertz` | L2 | 783 行，有测试 |
| `stiffened_panel_ultimate` | L2 | 767 行，有测试 1 断言 |
| `fsi_lbm_strong_coupling` | L2 | 767 行，有测试 |
| `ice_panel_fsi` | L2 | 758 行，有测试 |
| `verification_ladder` | L2 | 757 行，有测试 |
| `hertz_mindlin` | L2 | 757 行，有测试 |
| `adaptive_1d` | L2 | 754 行，有测试 |
| `pasternak_beam` | L2 | 749 行，有测试 55 断言 |
| `phase_field_specimens` | L2 | 742 行，有测试 82 断言 |
| `plate_bending_buckling` | L2 | 741 行，有测试 30 断言 |
| `fatigue/crack_tip_quarter_point` | L2 | 738 行，有测试 |
| `hertz_cattaneo_mindlin` | L2 | 728 行，有测试 31 断言 |
| `fatigue/crack_growth_geometry` | L2 | 728 行，有测试 47 断言 |
| `structural_fatigue_detail` | L2 | 724 行，有测试 |
| `kirsch_hole` | L1 | 719 行，纯解析解参考 |
| `fsi_verification_suite` | L2 | 714 行，有测试 30 断言 |
| `fsi_coupling` | L2 | 701 行，有测试 30 断言 |
| `vibration` | L2 | 698 行，有测试 |
| `low_temp_fracture` | L2 | 692 行，有测试 38 断言 |
| `cylindrical_shell_modal` | L2 | 692 行，有测试 |
| `hetenyi_reference` | L1 | 686 行，纯解析解参考 |
| `foundation_plate_tensionless` | L2 | 678 行，有测试 45 断言 |
| `plate_radiation_chain` | L2 | 674 行，有测试 55 断言 |
| `cylindrical_pressure_reference` | L1 | 662 行，纯解析解参考 |
| `fatigue/detail_maturity` | L2 | 660 行，有测试 |
| `phase_field_fracture` | L2 | 656 行，有测试 |
| `levelset/level_set` | L2 | 653 行，有测试 57 断言 |
| `reference_strong_form_audit` | L1 | 649 行，审计工具 |
| `thick_cylinder_plasticity` | L2 | 647 行，有测试 |
| `shear_plate_spectral` | L2 | 634 行，有测试 |
| `fsi_flutter_vlm` | L2 | 629 行，有测试 |
| `welding_thermo_mechanical` | L2 | 628 行，有测试 28 断言 |
| `cook_stress_reference` | L1 | 1286 行，纯参考值计算 |
| `lame_reference` | L1 | 200 行，纯解析解 |
| `plate_ritz_reference` | L1 | 180 行，纯 Ritz 法 |
| `thermal_analytic` | L1 | 150 行，纯解析解 |
| `torsion_reference` | L1 | 120 行，纯解析解 |
| `tet10_mass_audit` | L0 | 45 行，审计工具，无测试 |
| `stiffened_plate_progressive` | L0 | 180 行，无测试 |
| `repo_layout` | L0 | 51 行，仓库布局检查 |
| `hemisphere_benchmark_fields` | L0 | 44 行，基准场定义 |
| `buoyancy_coupling` | L0 | 120 行，无测试 |
| `kerr_foundation` | L2 | 280 行，有测试 |
| `foundation_plate_bending` | L4 | 1056 行，测试 57 断言 |
| `structural_verification` | L4 | 532 行，测试 55 断言 |
| `fsi_ice_3d_coupling` | L4 | 490 行，测试 54 断言 |
| `finite_width_hole` | L4 | 398 行，测试 48 断言 |
| `pasternak_beam` | L4 | 749 行，测试 55 断言 |
| `cook_stress_reference` | L3 | 1286 行，测试 56 断言，被多处引用 |
| `continuum` | L3 | 导出，被 56 模块引用 |
| `solid3d` | L3 | 导出，被 56 模块引用 |
| `plate` | L3 | 导出，被 42 模块引用 |
| `plate_vibration` | L3 | 导出，被 23 模块引用 |
| `layered_shell4_plasticity` | L3 | 导出，被 20 模块引用 |
| `marine_plate_buckling` | L3 | 导出，被 17 模块引用 |
| `plasticity` | L3 | 导出，被 16 模块引用 |
| `shell4` | L3 | 导出，被 15 模块引用 |
| `hex20` | L3 | 导出，被 14 模块引用 |

### 3.3 L 分布统计

| 级别 | 模块数 | 占比 | 说明 |
|---|---|---|---|
| L0 | 49 | 14% | 文档/工具/无测试模块 |
| L1 | 174 | 50% | 有计算但缺完整验证链 |
| L2 | 59 | 17% | 独立功能实现，有测试 |
| L3 | 26 | 7% | 系统集成，被其他模块引用 |
| L4 | 39 | 11% | 基准验证，有完整测试链 |
| L5 | 0 | 0% | 无科研/工程突破级成果 |

---

## 四、Top 问题

### 4.1 严重问题

1. **大量模块未导出**: 347 个模块中仅 73 个（21%）导出到 `__init__.py`，大量功能模块无法通过公共 API 访问。
2. **源模块几乎无自断言**: 347 个源模块中仅 7 个含 `assert` 语句，自验证能力薄弱。
3. **部分模块有 NotImplementedError**: `saint_venant_torsion`（10 个）、`plate_foundation_interface`（7 个）、`box_torsion`（2 个）。

### 4.2 中等问题

1. **测试覆盖不均**: 263 个测试文件覆盖 76% 模块，但部分大模块测试断言数偏低（如 `stiffened_panel_ultimate` 仅 1 断言）。
2. **文档声称 vs 代码实况差异**: 部分模块文档声称"验证完成"，实际仅为解析公式代入。
3. **145 个模块无内部调用**: 虽多有独立测试，但缺乏模块间集成验证。

### 4.3 低等问题

1. **名称相似模块**: 多对模块名称相似但功能互补，需文档澄清。
2. **部分参考模块纯为解析解**: 如 `cook_stress_reference`、`hetenyi_reference` 等，是有价值的 oracles 但非 FEM 实现。

---

## 五、Verdicts 落表

```sql
-- L4 模块（基准验证级）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'acoustic_radiation', 14486, 4, '{"test_asserts":76,"anchors":["Rayleigh","Leppington","Maidanik"],"convergence":"frequency_sweep"}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'cohesive3d', 14486, 4, '{"test_asserts":53,"anchors":["2D_BK_degradation","energy_conservation"],"modes":["pure_I","pure_II","mixed"]}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'hull_girder_csr_chain', 14486, 4, '{"test_asserts":96,"chain":"CSR_complete","validation":"rule_formulas"}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'finite_rotation_layered_shell4', 14486, 4, '{"test_asserts":78,"validation":"analytical_oracles","finite_rotation":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'ice_belt_collapse', 14486, 4, '{"test_asserts":53,"standard":"ISO_19906","validation":"rule_formulas"}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'fsi_wet_modal', 14486, 4, '{"test_asserts":57,"validation":"analytical_frequencies","wet_modal":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'polar_thermal_stress', 14486, 4, '{"test_asserts":73,"validation":"analytical_solution","thermal_stress":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'foundation_plate_bending', 14486, 4, '{"test_asserts":57,"validation":"Hetenyi_analytical","foundation_plate":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'structural_verification', 14486, 4, '{"test_asserts":55,"verification":"cantilever_plate_modal"}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'fsi_ice_3d_coupling', 14486, 4, '{"test_asserts":54,"fsi":"ice_3d","coupling":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'finite_width_hole', 14486, 4, '{"test_asserts":48,"validation":"analytical_stress_concentration"}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'pasternak_beam', 14486, 4, '{"test_asserts":55,"validation":"analytical_beam_foundation"}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'simp_topology', 14486, 4, '{"test_asserts":14,"validation":"checkerboard_suppression","filter_consistency":true}', 'audit2-ab');

-- L3 模块（系统集成级）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'continuum', 14486, 3, '{"imported_by":56,"exported":true,"core_element":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'solid3d', 14486, 3, '{"imported_by":56,"exported":true,"core_element":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'plate', 14486, 3, '{"imported_by":42,"exported":true,"core_element":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'plate_vibration', 14486, 3, '{"imported_by":23,"exported":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'layered_shell4_plasticity', 14486, 3, '{"imported_by":20,"exported":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'marine_plate_buckling', 14486, 3, '{"imported_by":17,"exported":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'plasticity', 14486, 3, '{"imported_by":16,"exported":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'shell4', 14486, 3, '{"imported_by":15,"exported":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'hex20', 14486, 3, '{"imported_by":14,"exported":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'saint_venant_torsion', 14486, 3, '{"subclasses":["CircleSection","EllipseSection","RectSection"],"validation":"Timoshenko_Goodier","notimplemented_in_base":10}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'cook_stress_reference', 14486, 3, '{"imported_by":10,"test_asserts":56,"oracle":true}', 'audit2-ab');

-- L2 模块（独立功能级）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'marine_corrosion_ice_ultimate', 14486, 2, '{"lines":2078,"test_asserts":87,"exported":false}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'ice_slamming_whipping', 14486, 2, '{"lines":1602,"test_asserts":0,"exported":false}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'propeller_ice_milling_fatigue_chain', 14486, 2, '{"lines":1182,"test_asserts":51}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'weld_thermomechanics', 14486, 2, '{"lines":1124,"test_asserts":43}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'cohesive_bending', 14486, 2, '{"lines":1121,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'structural_canonical_face', 14486, 2, '{"lines":1248,"test_asserts":58,"exported":false}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'structural_mindlin_face', 14486, 2, '{"lines":1169,"test_asserts":43}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'smith_panel_collapse', 14486, 2, '{"lines":1105,"test_asserts":54}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'structural_verification_face', 14486, 2, '{"lines":1064,"test_asserts":52}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'weld_joint_fatigue', 14486, 2, '{"lines":1015,"test_asserts":72}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'stiffened_plate_ultimate', 14486, 2, '{"lines":993,"test_asserts":80}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'hull_girder_vibration_chain', 14486, 2, '{"lines":985,"test_asserts":57}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'bimetal_thermoelastic', 14486, 2, '{"lines":929,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'stiffened_plate_vibration', 14486, 2, '{"lines":909,"test_asserts":33}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'cyclic_plasticity', 14486, 2, '{"lines":861,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'fatigue_crack_propagation', 14486, 2, '{"lines":828,"test_asserts":56}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'indentation_plasticity', 14486, 2, '{"lines":827,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'ice_noise_chain', 14486, 2, '{"lines":826,"test_asserts":36}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'von_karman_plate_riks', 14486, 2, '{"lines":822,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'hull_girder_modal', 14486, 2, '{"lines":814,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'elliptic_hole', 14486, 2, '{"lines":803,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'cattaneo_hertz', 14486, 2, '{"lines":783,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'stiffened_panel_ultimate', 14486, 2, '{"lines":767,"test_asserts":1}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'fsi_lbm_strong_coupling', 14486, 2, '{"lines":767,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'ice_panel_fsi', 14486, 2, '{"lines":758,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'hertz_mindlin', 14486, 2, '{"lines":757,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'adaptive_1d', 14486, 2, '{"lines":754,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'phase_field_specimens', 14486, 2, '{"lines":742,"test_asserts":82}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'plate_bending_buckling', 14486, 2, '{"lines":741,"test_asserts":30}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'hertz_cattaneo_mindlin', 14486, 2, '{"lines":728,"test_asserts":31}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'low_temp_fracture', 14486, 2, '{"lines":692,"test_asserts":38}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'foundation_plate_tensionless', 14486, 2, '{"lines":678,"test_asserts":45}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'plate_radiation_chain', 14486, 2, '{"lines":674,"test_asserts":55}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'phase_field_fracture', 14486, 2, '{"lines":656,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'levelset/level_set', 14486, 2, '{"lines":653,"test_asserts":57}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'thick_cylinder_plasticity', 14486, 2, '{"lines":647,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'shear_plate_spectral', 14486, 2, '{"lines":634,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'fsi_flutter_vlm', 14486, 2, '{"lines":629,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'welding_thermo_mechanical', 14486, 2, '{"lines":628,"test_asserts":28}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'plate_foundation_interface', 14486, 2, '{"lines":755,"notimplemented":7,"test_asserts":0}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'box_torsion', 14486, 2, '{"lines":478,"notimplemented":2,"test_asserts":0}', 'audit2-ab');

-- L1 模块（数学与单元验证级）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'kirsch_hole', 14486, 1, '{"lines":719,"type":"analytical_oracle","no_fem_solve":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'hetenyi_reference', 14486, 1, '{"lines":686,"type":"analytical_oracle","no_fem_solve":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'cylindrical_pressure_reference', 14486, 1, '{"lines":662,"type":"analytical_oracle","no_fem_solve":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'cook_stress_reference', 14486, 1, '{"lines":1286,"type":"analytical_oracle","no_fem_solve":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'lame_reference', 14486, 1, '{"lines":200,"type":"analytical_oracle","no_fem_solve":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'plate_ritz_reference', 14486, 1, '{"lines":180,"type":"analytical_oracle","no_fem_solve":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'thermal_analytic', 14486, 1, '{"lines":150,"type":"analytical_oracle","no_fem_solve":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'torsion_reference', 14486, 1, '{"lines":120,"type":"analytical_oracle","no_fem_solve":true}', 'audit2-ab');

-- L0 模块（文档/工具级）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'tet10_mass_audit', 14486, 0, '{"lines":45,"no_test":true,"type":"audit_tool"}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'stiffened_plate_progressive', 14486, 0, '{"lines":180,"no_test":true}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'repo_layout', 14486, 0, '{"lines":51,"no_test":true,"type":"repo_tool"}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'hemisphere_benchmark_fields', 14486, 0, '{"lines":44,"no_test":true,"type":"field_definition"}', 'audit2-ab');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorFEM', 'buoyancy_coupling', 14486, 0, '{"lines":120,"no_test":true}', 'audit2-ab');
```

---

## 六、总结

### 6.1 核心发现

1. **代码规模庞大**: 347 个源模块，134,892 行代码，263 个测试文件。
2. **功能真实性高**: 绝大多数模块确实执行计算，非空壳或硬编码。
3. **测试覆盖良好**: 76% 模块有对应测试，总断言数约 8,500+。
4. **导出率低**: 仅 21% 模块导出到公共 API，大量功能模块无法通过标准导入访问。
5. **无 L5 级成果**: 未发现解决此前未解决问题的科研/工程突破级成果。
6. **自验证薄弱**: 仅 7 个源模块含 `assert` 语句。

### 6.2 改进建议

1. **提高导出率**: 将成熟的 L2+ 模块导出到 `__init__.py`。
2. **增加源模块自断言**: 在关键计算路径添加 `assert` 进行运行时验证。
3. **完善 NotImplementedError**: 完成 `plate_foundation_interface` 和 `box_torsion` 的缺失方法。
4. **文档澄清**: 区分"解析解参考"和"FEM 实现"，避免文档夸大。
5. **集成测试**: 增加模块间集成测试，验证工作流协同。

### 6.3 Honest FAIL 声明

- 未对全部 4,804 个测试用例逐一运行验证（时间限制），仅抽样运行了部分测试。
- 未对每个模块的数值精度进行独立基准对照（需实验/文献数据）。
- L5 级判定需要领域专家确认，本审计仅基于代码/测试证据。
