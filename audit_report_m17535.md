# 交差审计报告：XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION (m17535)

**审计人**: board-audit (m17579)  
**日期**: 2026-11-13  
**对象**: m17535 — joconnor22/LIFE implicit-IBM Aitken Δ² 松弛验证  
**仓库**: TensorFEM  
**裁定级别**: **L3（系统集成）**

---

## 1. 案例介绍

### 物理背景

在分区式流固耦合（FSI）强耦合方案中，流体和结构在每个时间步内通过子迭代求解以满足界面相容性。对于轻质结构（低质量比），附加质量效应导致不动点迭代发散，必须使用加速技术。Aitken Δ² 动态松弛法根据残差历史动态调整松弛参数 ω，是解决此问题的标准方法。

### 参考基准

| 参数 | 值 |
|------|-----|
| 锚点 | XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION |
| FSI-2 参考振幅 | 0.034 m |
| FSI-2 参考频率 | 2.0 Hz |
| Aitken 理论 ω_opt | 1/(1-λ) |
| 文献 | Küttler & Wall (2008); Irons & Tuck (1969) |

---

## 2. 审计方法

逐项核验 m17535 交付三件套（run_artifacts / benchmark ref vs got / repro_cmd），对照 `tensor_benchmark_cases` 基准裁判（该表不存在，如实记录），独立运行测试验证可复现性，审查源码确认实现真实性。

### 核验清单

| 核验项 | 方法 | 结果 |
|--------|------|------|
| run_artifacts 存在性 | ls + file 命令 | ✅ 3 PNG + 1 JSON 存在且为真实文件 |
| fsi_strong_coupling artifacts | ls | ⚠️ 目录为空 |
| 测试可执行 | pytest 运行 | ✅ 14/14 passed (8.51s) |
| 源码真实性 | 代码审查 | ✅ AitkenRelaxation 为真实算法实现 |
| 集成调用 | grep 搜索 | ✅ StrongCouplingFSI2 + fsi_coupling.py |
| __init__.py 导出 | 检查 | ⚠️ 未导出 |
| tensor_benchmark_cases | 数据库查询 | ❌ 表不存在 |
| FSI-2 基准 vs 参考 | 数值比较 | ⚠️ 粗网格无法比较 |

---

## 3. 计算结果

### Aitken 算法验证（A1-A8）

| 测试 | 状态 | 关键数字 | 误差 | <15%? |
|------|------|----------|------|-------|
| A1: ω 正确性 | PASS | λ=-0.9 → ω=0.5263 | <0.1% | ✅ |
| A2: 加速效果 | PASS | Aitken 2步 vs 固定 20+步 | N/A | ✅ |
| A3: 多维残差 | PASS | 2-DOF 收敛 <1e-6 | N/A | ✅ |
| A4: reset | PASS | ω 恢复 0.8, r_prev=None | N/A | ✅ |
| A5: 裁剪 | PASS | ω ∈ [-2.0, 2.0] | N/A | ✅ |
| A6: 集成 | PASS | 40 个有效 ω 值 | N/A | ✅ |
| A7: 收敛速率 | PASS | 后期衰减率 <0.1 | <10% | ✅ |
| A8: 文献对比 | PASS | λ=-0.6 → ω=0.625 | <5% | ✅ |

### FSI-2 粗网格结果

| 量 | got | ref | 误差 | <15%? |
|----|-----|-----|------|-------|
| 振幅 | N/A | 0.034 m | N/A | ⚠️ 网格太粗 |
| 频率 | N/A | 2.0 Hz | N/A | ⚠️ 网格太粗 |
| 平均子迭代 | 2.0 | — | — | ✅ |
| 收敛率 | 100% | — | — | ✅ |

---

## 4. 结果图

三张图均为真实 PNG（已用 file 命令验证）：
- fig1_aitken_convergence.png (1800×600): Aitken vs 固定松弛收敛对比
- fig2_omega_vs_eigenvalue.png (1200×750): ω_opt 与特征值关系
- fig3_fsi2_omega_history.png (1500×1500): FSI-2 运行中 ω 历史

---

## 5. 注意点与发现

### 阻塞级发现（影响级别判定）

1. **FSI-2 基准验证不完整**: 8 cells/D 粗网格无法解析涡脱落，振幅/频率无法与参考值比较。报告如实声明，但 L4 级别要求完整算例+误差分析，因此降级至 L3。

2. **tensor_benchmark_cases 表不存在**: 无法按任务要求的 SQL 查询进行基准裁判。此前多个 verdict 均记录此问题。

### 非阻塞发现

3. **AitkenRelaxation 未在 __init__.py 导出**: 不属于公共 API，外部用户无法直接 `from tensorfem import AitkenRelaxation`。

4. **两套 Aitken 实现未统一**: `fsi_lbm_strong_coupling.py` 有类实现，`fsi_coupling.py` 有标量内联实现，逻辑重复。

5. **artifacts/fsi_strong_coupling/ 目录为空**: 预期有 FSI 耦合基准工件但无内容。

6. **implicit_plastic_tangent.py 与锚点无关**: 工作树中存在但属于 J2 塑性模块，非 Aitken 松弛。

---

## 6. 误差判定

| 项目 | 误差 | <15%? | 说明 |
|------|------|-------|------|
| A1: ω 正确性 | <0.1% | ✅ | 与理论值 1/(1-λ) 一致 |
| A7: 收敛速率 | <10% | ✅ | 后期衰减率 <0.1 |
| A8: 文献对比 | <5% | ✅ | ω 在 Küttler & Wall 范围内 |
| FSI-2 振幅 | N/A | ⚠️ | 粗网格无法比较，需 ≥15 cells/D |
| FSI-2 频率 | N/A | ⚠️ | 同上 |

**结论**: Aitken 松弛算法本身验证通过（误差 <1% vs 理论），达到 L3 级别。FSI-2 物理基准验证因网格分辨率不足而无法完成，阻止 L4 判定。

---

## 级别裁定

**L3 — 系统集成**

理由：
- ✅ AitkenRelaxation 为真实算法实现（非 MOCK/STUB）
- ✅ 集成于 StrongCouplingFSI2 主求解器
- ✅ 14 项测试全部通过，算法误差 <1% vs 理论
- ✅ 报告六段齐全，诚实声明粗网格局限
- ⚠️ FSI-2 基准验证不完整（粗网格无法解析涡脱落）→ 不满足 L4
- ⚠️ 未导出公共 API、两套实现未统一

**verdict 已落表**: tensor_maturity_verdicts id=43, level=3, verdict_by=board-audit
