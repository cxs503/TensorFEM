# Board audit — m15434 / CASE-THERMAL-BAR（TensorFEM 约束热应力）

- **被审对象**：mission **m15434**（status=succeeded），仓库 TensorFEM，分支交付 commit **fc23eac4**
  （`TensorFEM: CASE-THERMAL-BAR constrained thermal stress benchmark`），已并入生态 main **0c88e405**（Merge #708）。
- **审计单**：**m15714**（board-audit 交差核验，非重做）。
- **审计工作树**：`/root/tp-agent/work/m15714`（= 生态 main 0c88e405 检出，含已并入的交付物）。
- **判决**：**L4（基准验证）** — confirmed 建议；见 §7 证据与 §8 分歧项。
- **产物**：本文件、`docs/audit_m15434/independent_check.py` / `independent_check.json`（不 import tensorfem 的独立复算）、
  `docs/audit_m15434/make_figures.py` / `fig_error_budget.png`、`benchmarks/constrained_thermal_bar_evidence.json`（provenance 修复重生成，数值逐位不变）。

---

## 1. 审计范围与独立性限制

**做了**（全部在 loongarch64 本机、float64）：
1. 读交付代码（`src/tensorfem/constrained_thermal_bar.py` 403 行、driver 776 行、测试 210 行、六段报告 md）。
2. 按任务声明复现：在 m15434 工作树与 m15714（生态 main）分别重跑 driver 与测试。
3. **独立复算**：`docs/audit_m15434/independent_check.py` 不使用 tensorfem 任何代码，自己从 EH36 锚表回归 E/α、
   自己写 1-D 热弹性 FE（两种热载荷积分口径）、自己算四族闭式，与证据 JSON 逐项对表。
4. 门禁强度探针（mutation test）：在 `/tmp/mut1`（`git archive` 干净副本）内注入两个物理 bug，看测试是否真抓得住。

**没做 / 限制**（如实登记，不视为已通过）：
- GitHub CI（x86 架构）未触发验证；本审计全部为 loongarch64。
- 集群 loong64 交叉复核未递交（本单计算量 ~11 s，无必要）。
- 教材原文（Timoshenko & Goodier / Boley & Weiner）离线不可检索 → 未独立核对页码/条目，只复核了公式再推导残差。
- 未跑全量台账 `run_registered_benchmarks()`（60 条目，≈13 min）；只按登记式逐字复算本条目（结果见 §4）。
- 未做实验/文献数值对标（该算例无实验数据可比，见 §6 F5）。

## 2. 交付物三件套是否真存在（逐项核收）

| 项 | 声明 | 审计核验 | 结论 |
|---|---|---|---|
| run_artifacts 文件 | 证据 JSON + 六段 md + 4 png | `benchmarks/constrained_thermal_bar_evidence.json`(20.8 kB, 646 行)、`docs/constrained-thermal-bar-verification.md`(178 行)、`docs/assets/constrained_thermal_{field,profile,convergence,sensitivity}.png`（1470×588 等，非占位，可解码） | ✔ 真存在 |
| benchmark ref vs got | σ_xx(get) vs −EαΔT(ref) | 见 §3 表，独立复算一致到 1e-16 | ✔ 有数值对照 |
| repro_cmd 可执行 | `PYTHONPATH=src python benchmarks/constrained_thermal_bar.py` | 在 m15434 重跑 **11.9 s** 出全绿；在 m15714 重跑 **10.8 s** 出全绿 | ✔ 可执行 |
| 测试 | 26 gates 全过 | m15434 树 **26 passed (95 s)**；生态 main 0c88e405 **26 passed (70 s)** | ✔ 复跑通过 |

## 3. 基准裁判：tensor_benchmark_cases 与容差

审计时台账 **无该锚点行**（`SELECT ... WHERE anchor_slug='CASE-THERMAL-BAR'` → 0 行；全表 21 行）。即：
**3% 容差是仓内自声明、机器裁判当时未点火**（与 m15654 对 CASE-BENDING-PATCH 的发现 F1 同类）。
审计已按仓内声明登记该行（`status='passed'`，ref_source 写明闭式解 + EH36 锚表哈希），后续审计可机器复核。

数值对照（ref 由闭式 + 锚表回归值程序化求值，非手抄）：

| 量 | ref（闭式） | got（FEM） | 相对误差 | ≤3%？ |
|---|---|---|---|---|
| σ_xx 1-D 均匀 ΔT（n=1..16） | −247.208571 MPa | −247.208571 MPa | −1.21e-16 | ✔（远优于 3%） |
| σ_xx hex8 单轴（n=1..4） | −247.208571 MPa | −247.208571 MPa | ≤7.23e-16 | ✔ |
| σ_xx hex8 双轴 | −353.155102 MPa | −353.155102 MPa | 1.7e-16 | ✔ |
| σ_xx hex8 三轴 | −618.021429 MPa | −618.021429 MPa | 1.9e-16 | ✔ |
| 自由控制组 | 0 | 1.3e-7 Pa（相对 2.5e8 Pa 尺度 5e-16） | — | ✔（判别力：掉热载荷会立刻爆掉） |
| 二次 ΔT 收敛阶 | 理论 2.0 | 观测 2.000 | 门 ±0.35 | ✔ |
| ν 扫掠不变量 | 无 Poisson 耦合 | 漂移 1.93e-15 | 门 1e-8 | ✔ |

审计侧独立复算（**不复用被审代码**）：E=206.007142857 GPa、α=1.2e-05/K 由锚表最小二乘回归独立得到；
`ref` 与证据 JSON 相差 **−1.2e-16**；四族闭式与 got 全部在 round-off 内；
自写 1-D FE（梯形口径）**逐行复现证据表 5 行，最大相对差 4.8e-16，阶次 2.000**。

## 4. 台账条目 `thermal.constrained_bar` 复核

`src/tensorfem/benchmark_registry.py:79-83` 新增条目按登记式复算：
got = −247208571.4 Pa，ref = −247208571.4 Pa，rel = **1.21e-16**，`passed=True`（容差 0.03）。
该条目经 `marine_structural_qualification.py:41` 被资质报告消费，`__init__.py` 导出 `run_registered_benchmarks` → 集成链真实存在。

## 5. 门禁强度探针（测试是否只是"跑一遍"）

在 `git archive` 干净副本 `/tmp/mut1` 注入两个物理 bug：

| 变异 | 期望 | 实测 |
|---|---|---|
| 双轴闭式 `/(1-ν)` → `/(1-2ν)` | 被抓 | **4 failed**（`test_hex_restraint_family_matches_closed_form[1/2/3-biaxial]`、`test_biaxial_and_triaxial_are_stiffer_than_uniaxial`） |
| hex 应力恢复丢掉 `eps_th`（σ = Dε） | 被抓 | **2 failed**（`test_hex_free_expansion_control`、`test_hex_uniaxial_lateral_stress_vanishes`） |

结论：测试断言的是不变量（round-off 1e-8、闭式链、自由控制组、刚体模态正定），**非空壳执行**。

## 6. 严重度分级发现

| # | 级别 | 发现 | 证据 | 处置 |
|---|---|---|---|---|
| F1 | 中（阻断"机器判决"，不阻断数值结论） | `tensor_benchmark_cases` 无 `CASE-THERMAL-BAR` 行（0/21），3% 容差自声明 | §3 的 SQL 结果 | **审计已登记该行**（`docs/audit_m15434/`，SQL 见 §7） |
| F2 | 中（provenance 缺陷，已修） | 证据 JSON `environment.git_sha = 0a555844`（main 旧头），该树**不含**被审代码（代码首现于 fc23eac4）→ 记录 SHA 无法复现工件 | `benchmarks/constrained_thermal_bar_evidence.json` 旧值 | **已重生成**：现记 0c88e405（含 #708 的生态 main），数值逐位不变（仅 `git_sha`/`generated_utc` 变）；残余：SHA 字段对"包含该 JSON 的提交"仍是自指 |
| F3 | 低（口径/声明强度） | 3% 容差对治理算例**无判别力**：热载荷精确积分下 FE 解在任意网格就精确（审计自写 Gauss 口径变体 n=2..32 最大误差 **1.8e-16**）。报告 §4.3 的"order 2.0"实际来自应力恢复里**单元平均温度的梯形积分误差**（审计复现 5 行到 5e-16、同阶 2.000），属热场插值误差而非求解器精度 | `independent_check.json` C2 | 不改代码；在判决/报告中限定声明口径（报告 §4.3 有 u RMS≈1e-20 一列，读者可辨，但未明说"力学解精确"） |
| F4 | 低（算例性质） | 3-D hex8 三族约束解析场是位移线性场，trilinear hex8 可精确表示 → **patch test**，验证的是装配/约束集/刚体模态，不是通用精度 | `constrained_thermal_bar.py:237-...` restraint_dofs + 审计变体实验 | 作为 L4 的一部分但不得单独当作精度证据（报告未夸大为通用精度） |
| F5 | 低（外部锚） | 材料锚表是离线二手转录（`source` 明说无原始 ASTM/EN 表），无 DOI/页码 | `MATERIAL_ANCHORS['EH36']['source']` | 阻断 L5 级 external_anchor 主张；报告已如实声明无页码 |
| F6 | 低（流程） | goal 要求 >10 min 计算走 cluster_queue；全量台账 ≈13 min 由本地跑，集群作业 cj1120 被 kill | mission result 文本 | 记录在案，不影响数值结论 |
| F7 | 提示（平台账） | mission result 内 `server_quality_audit = {gate_valid:false, key_present:false, gate_missing_warning:true}`，且 succeeded 为 auto-approve 而非门禁裁定 | `tensor_dev_missions.result` | 交差审计正是补这道门；本单落表后由 maturity_gate 机器复核 |

## 7. 落表（本审计写入）

```sql
-- 1) 补登记基准裁判行（审计前 0 行）
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
VALUES ('CASE-THERMAL-BAR',
        '约束热应力 σ=EαΔT（单轴/双轴/三轴/自由四族）',
        '闭式线弹性热应力 σ_xx=-EαΔT（Timoshenko & Goodier；Boley & Weiner Ch.1）；ref 由 EH36 锚表回归值程序化求值，非手抄',
        '轴向约束应力 σ_xx vs 闭式（含自由控制组、ν 扫掠、二次 ΔT 收敛阶）',
        '<3%（任务判据）；收敛阶 2.0±0.35', 'TensorFEM', 'passed', now());

-- 2) 成熟度判决（本单 m15714）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFEM', 'm15434 CASE-THERMAL-BAR', 15714, 4, '<evidence JSON>', 'board-audit');
```

evidence 键全部指向真实指针：impl_files / entry / run_artifacts / tests / integration_callers / benchmark / repro_cmd / external_anchor。

## 8. 级别判定（L0–L5）与理由

| 级 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元验证 | 有测试 | ✔ 26 gates（含变异探针证明非空壳） |
| L2 独立功能 | 真代码 输入→计算→输出 | ✔ 403 行模块 + 776 行 driver，真实 FEM 求解（复用 `solve_thermoelastic_bar` / hex8 装配），非硬编码 |
| L3 系统集成 | 主求解器/平台侧调用 | ✔ `benchmark_registry.py:79` → `marine_structural_qualification.py:41` → `__init__` 导出；生态 main 上 import 与条目复算通过 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ✔ 四族 + 控制组 + 二次 ΔT 阶 + ν/ΔT/温度相关敏感性；审计侧**独立复算**一致到 1e-16；重跑逐位一致（§2） |
| L5 科研工程 | 外部锚 + 新能力/新物理 | ✘ 无实验/文献数值锚（锚表离线二手转录、无页码）；无新物理（复用既有热弹性路径，新增的是约束集工具）；且治理算例容差无判别力 → **不主张 L5** |

**判决：L4，confirmed 建议。** 附注：本单的 L4 承重证据是"四族闭式 + 自由控制组 + 二次 ΔT 阶 + 独立复算 + 逐位复现"，
而非"3% 通过"（治理算例的 3% 对正确实现必然通过，F3）。

## 9. 复现命令

```bash
# 被审交付物（任一含 #708 的检出）
PYTHONPATH=src python benchmarks/constrained_thermal_bar.py          # ~11 s，重生成证据 JSON / 六段报告 / 4 图
PYTHONPATH=src python -m pytest tests/test_constrained_thermal_bar.py -q   # 26 passed, ~70–95 s

# 审计侧独立复算（不 import tensorfem）
python docs/audit_m15434/independent_check.py                        # C1..C4 全 PASS
python docs/audit_m15434/make_figures.py                             # fig_error_budget.png
```

## 10. 处置

**approve with notes（建议 confirmed，L4）**：三件套真存在且可执行、数值可独立复现、测试非空壳；
两条待办非阻断——F1 已由本审计补登记，F2 已由本审计重生成修复。
建议后续（不进本单）：把热应力算例升级为**有判别力**的版本（非均匀/多材料/带温度依赖的 2-D 或 3-D 场，
或与文献实验热应力数据对标），才是 L5 的候选；本单不作为 L5 依据。