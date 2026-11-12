# m15555 交差核验：CASE-SPHERE-PRESSURE（m15405）board-audit 裁定

**审计对象**：mission 15405（CASE-SPHERE-PRESSURE·均压球壳薄膜应力·TensorFEM，已 succeeded）
**审计人**：board-audit（本单 m15555）
**裁定**：**L3**（L1-L3 实测成立；L4 未记）
**verdict 落表**：`tensor_maturity_verdicts` id=342，gate_state=pending（平台 maturity_gate 自动裁定）
**报告**：tensor_reports id=14663

## 一、三件套核验（逐项）

| 项 | 判据 | 核验结果 |
|---|---|---|
| run_artifacts | 真实工件存在 | ✅ `benchmarks/verified/fem_sphere_pressure_membrane/result.json`（43850 B）+ `gates_verified.json` + `figures/{convergence,field_meridian,sensitivity}.png` 全部存在 |
| benchmark ref vs got | 数值在容差内 | ✅ σ_ref=125 Pa=pR/(2t)，got σ_θ=125.000046（facets=8），误差 +7.309e-05%，<3% 通过（余量 4 个量级） |
| repro_cmd | 可执行 | ✅ `python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py` —— **本单重跑 13 s 单进程 float64，结果与 result.json 一致（reproduced=True）** |

**测试**：`pytest tests/test_sphere_pressure_membrane_case.py -q` → **10 passed (8.36 s)**；`test_axisym_shell.py` → 8 passed（复用路径未扰动）。

**报告六段**：`docs/sphere-pressure-membrane-case.md` ①-⑥ 齐全，数字程序化注入零手抄。

## 二、基准裁判（tensor_benchmark_cases）

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-SPHERE-PRESSURE';
-- 命中 0 行（全表 18 行无此锚点）
```

**F1_no_registered_judge**：任务书指定的「唯一裁判」`tensor_benchmark_cases` 无 CASE-SPHERE-PRESSURE 注册行 → 判据降级为 mission 自身 goal 的 `<3%` 与机器推导 ref。

## 三、判级（L0-L5）

| 级 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元验证 | ref 机器推导 + 单元复现 | ✅ sympy 两路独立静力平衡（轴向+Laplace）残差恒零 + Lamé 厚壁极限交叉锚；8 单元 27 DOF 复现膜态 7.3e-05% |
| L2 独立功能实现 | 真代码：输入→计算→输出 | ✅ run_verification.py（31267 B）：ref 推导+收敛+3 条敏感性+2 对照+3 图+gates |
| L3 系统集成 | 接入主求解器/平台 | ✅ 算例作为调用方接入生产 AxisymShell 路径（tensorfem.axisym_shell_benchmarks.sphere_pressure），PR #704 已合入 main |
| L4 基准验证 | 完整算例+误差分析+可复现 | ⚠️ 完整算例✅/误差分析✅/可复现✅（本单重跑 reproduced=True）但 **external_anchor 无手册原表原页（WITHHELD）+ tensor_benchmark_cases 无该锚点注册行** → 未记 |
| L5 科研工程成果 | — | 未达（L4 未记） |

**裁定 L3**。补齐 `tensor_benchmark_cases` 注册行 + 手册原表原页即可升 L4。

## 四、发现（findings）

- **F1_no_registered_judge**：tensor_benchmark_cases 无该锚点注册行（0/18）→ 平台裁判行缺位。
- **F2_external_anchor_withheld**：报告 §① 手册原表原页未核对（WITHHELD，诚实标注）；ref 由 sympy 机器推导 + Lamé 交叉锚承载（独立、可复核，非自指）。
- **F3_clean_report**：报告六段齐全、数字零手抄、gates 全通过（G8b 证伪留档未改判据）——无 m15313 式数字错标。

## 五、checks_not_run

- 集群 loong64 复跑（单机 13 s 已完成，未上集群）
- GitHub Actions x86 CI 复跑
- Roark 手册原表数值核实（无外网、平台知识库无 Roark 源）
