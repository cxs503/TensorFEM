# Board Audit: m16067 CASE-FEM-SCORDELIS-LO-ROOF 交差核验

**审计席位**: board-audit (m16129)
**审计对象**: m16067 (CASE-FEM-SCORDELIS-LO-ROOF · TensorFEM · succeeded)
**审计日期**: 2026-11-12

---

## ① 取证：三件套核验

### 1.1 run_artifacts ✅ 存在

| 文件 | 大小 | sha256(前16) |
|---|---|---|
| `artifacts/scordelis_lo_1964/result.json` | 4711B | 5b732b827241474c |
| `artifacts/scordelis_lo_1964/convergence.png` | 58161B | — |
| `artifacts/scordelis_lo_1964/deflection_field_16x16.png` | 62765B | — |
| `artifacts/scordelis_lo_1964/README.md` | 2148B | — |

### 1.2 benchmark ref vs got 数值核验

**基准裁判**: Scordelis & Lo (1964) ACI Journal 61(3):539–550, DOI 10.14359/7796
**参考值**: 0.3024 in (自由边中点法向挠度)

| 网格 | got (in) | 参考 (in) | 误差% | <5%? |
|---|---:|---:|---:|:--:|
| 4x4 | 0.08929123 | 0.3024 | 70.47% | ❌ |
| 8x8 | 0.17228505 | 0.3024 | 43.03% | ❌ |
| 16x16 | 0.25329636 | 0.3024 | **16.24%** | **❌ FAIL** |
| 32x32 | 0.30101200 | 0.3024 | 0.46% | ✅ |

**任务门限**: `|w(16x16)| within 5% of 0.3024 in` → **FAIL** (16.24% > 5%)

### 1.3 repro_cmd 可执行性 ✅ 已验证

```bash
cd /root/tp-agent/work/m16067 && PYTHONPATH=src python3 benchmarks/scordelis_lo_1964.py
```

审计窗口内重跑结果与 result.json 完全一致（数值逐位匹配）。

### 1.4 测试 ✅ 通过

`tests/test_scordelis_lo_verification.py`: 4 tests, all pass (17.66s)
- `test_flat_facet_shell4_fails_accuracy_gate` — PASS
- `test_cylindrical_shell4_passes_accuracy_gate` — PASS
- `test_cylindrical_shell4_high_accuracy` — PASS
- `test_verification_artifacts_exist` — PASS

---

## ② 判级（L0-L5）

| 级别 | 标准 | 判定 | 证据 |
|---|---|---|---|
| L0 | 文档成果 | ✅ | README.md + result.json |
| L1 | 数学/单元验证 | ✅ | 收敛研究（4网格序列，观测阶 0.71→1.41→5.14） |
| L2 | 独立功能实现 | ✅ | `cylindrical_shell4.py` (108L): 输入→刚度矩阵→求解→输出 |
| L3 | 系统集成 | ✅ | 接入主求解器路径；`cylindrical_shell_benchmarks.py` 调用生产单元 |
| L4 | 基准验证（完整算例+误差分析+可复现） | ❌ | **门限 FAIL**：16x16 误差 16.24% > 5%；且平台 `tensor/bench/l4-verdict/16067` 返回 CLAIMED_NOT_PROVEN（无 tensorbench_results 行） |
| L5 | 科研工程成果 | ❌ | 纯基准验证，非完整科研贡献 |

**最终级别: L3**

---

## ③ 关键发现

### F1: 仓库既有 verified 算例张冠李戴 ⚠️

`benchmarks/verified/fem_shell_scordelis_lo/` 目录名为 "Scordelis-Lo"，但实际参数：
- L=50, R=25, half-angle=40° (80° total), t=0.25, E=4.32e8, ν=0, p=90
- 边界: diaphragm ends (v=w=0)

这是 **MacNeal & Harder (1985)** 标准问题（DOI 10.1016/0168-874X(85)90003-4），
**不是** Scordelis & Lo (1964) 原始问题（L=3000, R=2400, 45°, h/R=1/800, 自重）。

两者目标值同为 0.3024 是 MH 标准集的构造巧合，不能互相替代。
m16067 正确识别并报告了这一问题。

### F2: 16x16 门限 FAIL — honest-FAIL ✅

任务判据要求 16x16 网格误差 <5%，实际 16.24%。
原因：薄壳 (h/R=1/800) 下 Q4 单元前渐近区较长，16x16 仍处于前渐近收敛阶段（观测阶≈1.4）。
32x32 进入渐近区 (0.46%)，证明单元能力真实存在。

m16067 如实报告了 FAIL，未粉饰，符合 honest-FAIL 纪律。

### F3: 平台 L4 裁定 CLAIMED_NOT_PROVEN

平台 API `/tensor/bench/l4-verdict/16067` 返回：
- `status: "CLAIMED_NOT_PROVEN"`
- `failed_checks: ["execution_success", "source_cluster_job", "cluster_job_bound", "ref_got_error_present", "within_tolerance"]`
- `reason: "no tensorbench_results row for this mission"`

说明该 benchmark 结果未经平台独立复算验证，为自声明结果。

### F4: 边界条件处理正确

m16067 正确识别了近刚体模态问题：仅 w=0+u=0 时条件数≈4.8e19，
必须额外约束 y 平移和绕竖轴转动。这是薄壳分析的常见陷阱。

---

## ④ 落表 verdict

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFEM', 'm16067 CASE-FEM-SCORDELIS-LO-ROOF', 16129, 3, '<JSON>', 'board-audit');
```

evidence JSON 见下方 `verdict_evidence.json`。

---

## ⑤ 结论

m16067 完成了 Scordelis-Lo 1964 圆柱屋盖壳弯曲基准的**真实实现**：
- 三件套齐全且可复现
- 收敛研究完整（4 网格序列）
- 诚实报告 16x16 FAIL (16.24%)，32x32 PASS (0.46%)
- 正确识别仓库既有 verified 算例实为 MacNeal-Harder 1985

**但**：
- 任务门限网格 (16x16) 未通过 5% 容差 → L4 不成立
- 无平台独立复算 (tensorbench_results 无记录) → L4 不成立
- 级别定为 **L3**（系统集成级）

**maturity_gate 建议**: `reopened`（门限 FAIL 需返修或调整门限网格）
