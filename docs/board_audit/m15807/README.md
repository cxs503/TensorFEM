# m15807 · CASE-SPHERE-PRESSURE 交差审计（m15703 · TensorFEM）

**结论：approve at L4。** 交付真实、可复现、已并入主线树；7 条备注为记账/口径问题，无正确性缺陷。

| 项 | 值 |
|---|---|
| 被审单 / 提交 | m15703 / `a5408f06`（branch `agent/m15703`，2026-11-12T02:05:33Z） |
| 合入审计分支 | `46262ef9`（解 registry/cli/verified result.json 三处纯格式冲突） |
| verdict 落表 | `tensor_maturity_verdicts.id=373`（repo=TensorFEM, scope=`m15703 CASE-SPHERE-PRESSURE`, mission_id=15807, **level=4**, verdict_by=board-audit） |
| 机器裁判行 | `tensor_benchmark_cases.id=23`（anchor_slug `CASE-SPHERE-PRESSURE`, tol `<3%`, status passed）——审计前该表 22 个 anchor 中**无**本算例 |
| case pool | `tensor_case_pool.id=46` 的 `linked_anchor` 补为 `CASE-SPHERE-PRESSURE` |
| 报告 | `tensor_reports.id=14935` |

## 判据（ref vs got，容差 3%）

| 量 | ref | got (FEM) | 相对误差 |
|---|---|---|---|
| σ_θ vs pR/(2t) | 125.0 Pa | 125.000045626 Pa | 7.31e-07 |
| σ_φ vs pR/(2t) | 125.0 Pa | 125.000150482 Pa | 1.20e-06 |
| w vs pR²(1-ν)/(2Et) | 1.2820513e-05 m | 1.2820593e-05 m | 6.26e-06 |
| 各向异性 | 0 | 1.91e-04 Pa | 1.52e-06 |

四项均 <3%（最差 6.26e-06）；10/11 门 PASS，唯一 FAIL 是预注册后被证伪的 `G8b`（如实保留，`all_gating_pass` 不含它）。

## 本审计做了什么

1. `git archive a5408f06` 干净树重跑 → 1419 个 JSON 叶子与记录工件逐项比对，仅 `wall_clock_s` 不同（位同复现）。
2. 独立探针 `probe_sphere_case.py`（不 import 被审比较代码）：
   - 平衡残差 `||Ku-f||/||f||` = 2.34e-15 ⇒ 真在解 `K u = f`（27 DOF）；
   - 能量恒等 `½u·f` vs 独立推导膜应变能 `N²(1-ν)/(Et)·A` 相对差 3.44e-07；
   - 只用节点位移反算 `σ=E·w/(r(1-ν))` = 124.99984 Pa（1.27e-06），与库报 7.31e-07 同量级；
   - 反证：弦线经线 rel error 1.98（+198%）、全积分剪切结果不同 ⇒ 报出的误差随单元物理变化，非解析值回帖。
3. 三件套：run_artifacts 非空且位同；ref vs got 见表；`repro_cmd` 实跑 rc=0（14.3 s）；`benchmark_registry.py:131` 的 `quick_check()` 进程内实调通过。
4. `git merge-base --is-ancestor a5408f06 main` 原为 **false**（交付未进 main）⇒ 合入审计分支并在合并树重跑 16 项测试 + 整算例。

## 备注（详见 ⑤ 节与 verdict.evidence.findings）

- F3 口径收窄：常膜力态该单元几乎精确可表示，生产路径 3% 判据**非判别性**；判别力在负对照（弦线 207.9%、平面面元 Shell4 133.2×）与观测阶（3.4–4.0）。
- F2 `run.log` 记的是父提交 `0c88e405`（算例比提交早 20 s 跑）；良性，已由干净树重跑闭合。
- F7 `recorded_error_curve_percent` 是文档转录（已标注），与同块新算 133.2× 口径不同，不可并列比较。
- 不判 L5：无新物理/新单元（复用 `AxisymShell`）；手册原页（Roark Table 13.1 case 1）未核 = `external_anchor` WITHHELD。
- 未跑：GitHub CI(x86)、集群 loong64 跨架构复跑（14 s 工作量本地）、`run_registered_benchmarks()` 全表端到端（约 13 min，改进程内实调 3 条新条目）。

## 文件

- `verdict_evidence.json` —— 落表 evidence 原文（8 个契约键 + 探针/发现/未跑项）
- `probe_sphere_case.py` / `probe_output.json` —— 独立探针脚本与输出
- `build_audit_report.py` / `audit_sphere_pressure.png` —— 报告生成器与图（数字程序化注入，零手抄）