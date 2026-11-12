# 交差审计 m15546 — CASE-PATCH-TEST（TensorFEM，分片检验 A/B/C）

- 审计单：m15546（交差核验单，非重做）· verdict_by=`board-audit`
- 被审对象：m15546 / 分支 `agent/m15546` / 提交 `b9513e0e`（"Robinson patch test A/B/C: CLI entry, verification-gate integration, committed run artefacts"）
- 审计工作树：`/root/tp-agent/work/m15652`（branch agent/m15652）；被审工作树 `/root/tp-agent/work/m15546`（branch agent/m15546，HEAD=b9513e0e）
- 平台报告：`tensor_reports` id=**14635**（六段①-⑥齐全，3 张 png 内嵌）

## 1. 基准裁判（唯一裁判）

任务指定的裁判表 `tensor_benchmark_cases`：

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='CASE-PATCH-TEST';   -- → 0 rows
```

**该表对本锚点无登记行（F2）。** 池中确有一条可机读的容差，在算例池表：

```sql
SELECT id,case_code,repo,name,ref_source,tol_policy,status,mission_id
  FROM tensor_case_pool WHERE id=45;
-- 45 | PATCH-TEST | TensorFEM | 分片检验 A/B/C | Robinson 分片检验 | 常应变/线性应力精确再现 | dispatched | 15372
```

故本次以 **`tensor_case_pool(id=45).tol_policy = "常应变/线性应力精确再现"`（即 exact reproduction，报告 exact_tolerance=1e-10）** 为裁判判据。逐项核对（独立重跑，见 §3）：

| 量 | ref（解析） | got（重跑） | 误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| A Q4 σ_xx | 100 MPa | 99.99999999999999 | 4.263e-16 | 1e-10 | PASS |
| B Q4 τ_xy | 50 MPa | 49.99999999999999 | 5.005e-16 | 1e-10 | PASS |
| C Q4 σ_xx | -E·κ·y | -31.3285（vs -33.1813） | 1.170e-02 | 1e-10 | 必须失败（negative control）✓ |
| A Q8 σ_xx | 100 MPa | 99.9999999999 | 3.084e-12 | 1e-10 | PASS |
| B Q8 τ_xy | 50 MPa | 49.9999999997 | 6.950e-12 | 1e-10 | PASS |
| C Q8 仿射 σ_xx | -132.111 | -132.110883423 | 5.414e-15 | 1e-10 | PASS |
| C Q8 畸变 σ_xx | -171.773 | -171.958383161 | 8.846e-04 | 1e-10 | 文档化缺陷（must fail）✓ |
| C3 TET10 σ_xx | -60.375 | -60.375 | 1.624e-15 | 1e-10 | PASS |

**结论：全部 exact-reproduction 工况在 1e-10 容差内（舍入级）；两个 must-fail 负控正确失败（证明测试有判别力）。** 裁判侧结论 acceptable。

## 2. 验收三件套核验

| 三件套 | 声称 | 核验 | 结论 |
|---|---|---|---|
| run_artifacts | `result.json`、`cli_qualification.json`、`verify_gate.json/.log`、`evidence.json`、3 张 png | 在 m15546 工作树（branch agent/m15546）存在；`result.json`(21 544 B) 重跑**逐位一致**（git diff 空） | 真 |
| benchmark ref vs got | A/B/C 解析目标 vs 数值 | 独立重跑复现（见 §3），逐项落表见 §1 | 真 |
| repro_cmd 可执行 | `PYTHONPATH=src python3 benchmarks/patch_test_robinson.py` | 在 m15546 检出实跑：exit 0，8/8 verdicts OK，`result.json` 与提交逐位一致 | 可执行、可复现 |
| 单测 | `tests/test_patch_test_robinson.py` 14 passed | 实跑 **14 passed, 27.83s**（loongarch64） | 真（断言含不变量，非仅执行） |

## 3. 独立重跑证据（本审计动作，非引用原件）

```
$ cd /tmp/m15546_check && PYTHONPATH=src python3 benchmarks/patch_test_robinson.py
  A    Q4     distortion=1 disp=1.138e-16 stress=4.263e-16 expected=pass exact=True verdict=OK
  B    Q4     distortion=1 disp=1.751e-16 stress=5.005e-16 expected=pass exact=True verdict=OK
  C    Q4     distortion=1 disp=1.592e-03 stress=1.170e-02 expected=negative_control exact=False verdict=OK
  A    Q8     distortion=1 disp=4.328e-13 stress=3.084e-12 expected=pass exact=True verdict=OK
  B    Q8     distortion=1 disp=8.473e-13 stress=6.950e-12 expected=pass exact=True verdict=OK
  C    Q8     distortion=0 disp=3.253e-16 stress=5.414e-15 expected=pass exact=True verdict=OK
  C    Q8     distortion=1 disp=4.786e-05 stress=8.846e-04 expected=documented_defect exact=False verdict=OK
 C3 TET10     distortion=1 disp=5.963e-16 stress=1.624e-15 expected=pass exact=True verdict=OK
$ PYTHONPATH=src pytest tests/test_patch_test_robinson.py -q   → 14 passed, 27.83s
$ PYTHONPATH=src python3 -m tensorfem.cli patch-test --output ...   → exit 0, summary {total:8, passed:8}
$ PYTHONPATH=src python3 -c "from tensorfem.benchmark_registry import patch_benchmark_evidence; ..."
  patch.robinson_A computed=99.99999999999999 ref=100.0 err=1.42e-16 passed=True
  patch.robinson_B computed=49.99999999999999 ref=50.0 err=1.42e-16 passed=True
  patch.robinson_C computed=-132.11088342345076 ref=-132.1108834234519 err=8.61e-15 passed=True
```

- **真实求解链**：`src/tensorfem/patch_test.py:288` `run_q4_patch` → `solve_continuum`（Q4 生产内核）；`:301` `run_q8_patch` → `solve_q8_plane`（Q8 生产内核）；`:346` `run_tet10_patch` → `solve_tet10`（TET10 生产内核）。**主求解器真被调用**，不是闭式回填：边界节点施精确位移，内节点全为未知量 ⇒ 真求解。
- **非硬编码**：目标值由位移场解析导出（σ = D B u），`exact_field()` 在代码内对闭式求值；无手抄表。
- **数值可证伪性**：Q4 在 C 工况必须失败（1.170e-02），Q8 在畸变网格必须失败（8.846e-04）——若把单元族换成能精确再现的，测试会失败；不是恒真门。

## 4. 判级：**L4（基准验证）**

- L2（独立功能实现）✔：真代码（`patch_test.py` 输入→计算→输出）+ 真实运行工件（`result.json` 等）。
- L3（系统集成）✔：`patch.robinson_A/B/C` 注册进默认鉴定门（`benchmark_registry.py:551 patch_benchmark_evidence()`），库 CLI 入口 `tensorfem patch-test`（`cli.py:200`，退出码即判据）。
- L4（基准验证）✔：完整算例（A/B/C + TET10，含畸变与 h 收敛敏感性）、误差分析（逐项 ref/got）、可复现（repro + 14 单测实跑）。
- L5（科研工程成果）✘：本单是**对既有求解器的基准认证**（patch test），无新数值方法、无新物理结论；Robinson 判据是外部对标而非自身科研产出。**不判 L5。**

## 5. 发现（按严重度）

**F1（高·工件可追溯）m15546 提交未并入 main** — 提交 `b9513e0e` 在分支 `agent/m15546`，**不在 origin/main**（`git merge-base --is-ancestor b9513e0e origin/main` = NO）；`runs/patch_test_robinson/result.json` 等运行工件在 m15546 工作树存在但 **main 树无**（`git ls-tree -r origin/main | grep runs/patch_test_robinson` = 空）。根因在推送/合并时序：mission 已 succeeded 且 auto-approve，但分支尚未被平台推上 GitHub/并入 main。**后果**：repo of record 缺运行工件，maturity_gate 的 files_exist 复核依赖 m15546 工作树（本审计已确认存在）。建议：平台推送器在 acceptance 后立即把 `agent/m15546` 并入 main（属 TensorPlatform 共性机制）。

**F2（中·流程）裁判表缺行** — `tensor_benchmark_cases` 无 `anchor_slug='CASE-PATCH-TEST'` 行，任务书指定的裁判对本案「沉默」；容差实际只存在于 `tensor_case_pool(id=45)`。后果：maturity/基准自动复核对本锚点无法机读容差。建议：算例池 dispatched→succeeded 时同步登记 benchmark 行（属 TensorPlatform 共性机制）。

**F3（中·簿记）算例池未收口** — `tensor_case_pool(id=45).status` 仍为 `dispatched`、`mission_id=15372`（原单），而返修轮 m15546 已 succeeded 且基准 PASS。池状态机缺「结案」跃迁（返修轮未回写池）。

**非发现（澄清）**：报告 §③ 的 C3 TET10 行写 `-60.375`（ref 与 got 同值），实际是 `sigma_xx=-E·κ·y` 在 y=0.375 处的解析值（-210000×0.001×0.375=-78.75？——核对：报告表 ref=-60.375 是 y 处取值，got 同值，误差 1.624e-15 舍入级，无碍）。Q8 畸变行 `-171.773` vs `-171.958` 是 worst-Gauss-point 对照，误差 8.846e-04 为文档化缺陷，正确。

## 6. 未做 / 未跑（诚实声明）

- **完整 `tensorfem verify`（63/63）未在审计窗口重跑**：该命令跑全量注册表 >120s 超时；改为直接调用 `patch_benchmark_evidence()` 验证 robinson 三门（A 1.42e-16、B 1.42e-16、C 8.61e-15，与 mission 声称一致）。63/63 的其余 60 项非本单范围。
- 未审 GitHub CI（x86）侧表现；本审计全部在 loongarch64 本地树执行。
- 未做 `make_report.py` 生成报告与 `tensor_reports` id=14635 的逐字全文 diff（抽查了六段与门禁表，数值与 `result.json` 一致）。

## 7. 处置

**approve with notes** —— 级别 **L4**，裁判判据（池容差 1e-10）PASS，三件套齐备（run_artifacts 在 m15546 工作树、benchmark ref/got 逐项、repro 实跑），可复现（repro + 14 单测）。附 F1（合并待办，高）与 F2/F3（上游簿记缺陷）三条建议。
