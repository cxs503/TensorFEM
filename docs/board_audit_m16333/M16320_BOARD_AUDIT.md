# 交差审计 · m16320 · CASE-FEM-CLAMPED-PLATE-UNIFORM (TensorFEM)

审计单 m16333 · 对象 m16320（status=succeeded，anchor=CASE-FEM-CLAMPED-PLATE-UNIFORM）
裁定：**L4（基准验证）** — 已落表 `tensor_maturity_verdicts.id=469`，maturity_gate 机器复核 `gate_level=4`（`reasons=[]`）。

## 1. 范围与独立性

| 项 | 值 |
|---|---|
| 被审工作树 | `/root/tp-agent/work/m16320`（HEAD `53ceae97`，分支 `agent/m16320`） |
| 被审 commit | `53ceae97cc703f5401fef75c7770c1d10aa418f6` "round 2: cross-family verification" |
| 审计工作树 | `/root/tp-agent/work/m16333`（HEAD `6c4683e1`，分支 `agent/m16333`） |
| 独立性 | 审计者未参与 m16320 实现；复算在 `/tmp/m16320copy`（被审树字节拷贝，去掉 `.git`）内独立执行，数值比对由本单脚本 `artifacts/audit_m16320/build_verdict.py` 完成 |
| 未做 | 集群 loong64 全量回归（被审单同样未跑，`cj1298` 仍排队）；`oss_inventory` 横向调研（角色不可读）；`tests/test_case_clamped_plate_uniform_crossfamily.py` 本审计树不存在，其"8 passed"未被复核（unknown / not_run） |

## 2. 三件套取证（run_artifacts / benchmark / repro_cmd）

| 项 | 结论 | 证据指针 |
|---|---|---|
| run_artifacts | ✅ 存在且非空 | `benchmarks/artifacts/CASE-FEM-CLAMPED-PLATE-UNIFORM/case_result.json` 18771 B（sha256 `8d77cbe2…`）、`case_result.md` 4888 B、`docs/assets/case-clamped-plate-uniform/clamped_plate_uniform.png` 173835 B（sha256 `f9f24531…`）；字节副本见 `artifacts/audit_m16320/byte_copies/` + `sha256_manifest.txt` |
| benchmark ref vs got | ✅ 在容差内 | 注册裁判 1 行：`alpha = w_max D/(q a^4)`，`tolerance=<3%`，`status=passed`；got `0.0012642046539125982` vs ref `0.00126` → **0.334%** |
| repro_cmd 可执行 | ✅ 独立复跑通过 | `cd /root/tp-agent/work/m16320 && PYTHONPATH=src python benchmarks/case_clamped_plate_uniform.py` 在被审树拷贝内重跑：exit 0、18/18 闸门 PASS、**382/382 JSON 数值叶逐位相同**（仅 `wall_seconds` 151.7s → 191.0s 不同），PNG **字节相同**（sha256 `f9f2453103…` 双方一致） |
| 报告 | ✅ 六段齐全 | `/tensor/reports` id 15413（published，2026-11-12T12:40:19Z）：①案例介绍 ②怎么做 ③计算结果（程序化注入）④结果图（内嵌 `clamped_plate_uniform.png`）⑤注意点 ⑥误差判定；文档镜像 `docs/CASE-FEM-CLAMPED-PLATE-UNIFORM_ROUND2.md` §1–§7 |

明细：`artifacts/audit_m16320/referee.json`、`repro.json`、`verdict_evidence.json`。

## 3. 判级（L0–L5，级别=功能实质）

| 级 | 判据 | 裁定 | 依据 |
|---|---|---|---|
| L0 文档 | 报告/文档 | ✅ | `docs/CASE-FEM-CLAMPED-PLATE-UNIFORM_ROUND2.md` 7 节；报告 15413 六段 |
| L1 数学/单元验证 | 单元/格式正确性 | ✅ | BFS C1 补丁检验相对误差 6.8e-16；SS 板 vs 精确 Navier 1.1e-6；离散平衡 `|R|-qa²` 1.3e-12 |
| L2 独立功能 | 真代码 输入→计算→输出 | ✅ | `benchmarks/case_clamped_plate_uniform.py` 35765 B（自建网格、Q4 Mindlin 求解、Ritz 参考、Richardson、锚点判定、工件写出）；`src/tensorfem/plate_bfs.py` 24433 B（BFS 装配+补丁检验）；均无 `NotImplementedError`/placeholder |
| L3 系统集成 | 接入主求解器/平台 | ✅（有保留，见 F2） | 算例驱动的是库内生产求解器 `plate_uniform_load.solve_uniform_square_plate` 与 `clamped_uniform_ritz`；同一对函数被产品侧验证面调用：`src/tensorfem/structural_verification_face.py:94-97`（import）+ `:632` 起的 plate arm 每次运行都算该量 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ✅ | 注册裁判 <3% 通过（0.334%）；收敛序列（Q4 4→32 网格，误差 -3.66%→+0.55%）+ Richardson 极限 0.00126786（阶 1.99）；BFS 10/20/40 自收敛阶 2.99；跨族差 0.196%；**审计独立复跑逐位复现** |
| L5 科研工程 | 新方法/工程贡献 | ❌ | 教科书标准算例；BFS 为 Bogner–Fox–Schmit 既有单元（本仓新增实现但非新方法） |

## 4. Findings（按严重度）

- **F1 [中·已修复前提下不阻断] 交付未合入 main。** `53ceae97` 只是 `agent/m16320` 分支头；`git merge-base --is-ancestor 53ceae97 main` 为假。main 上已有 `src/tensorfem/plate_bfs.py`、`tests/test_plate_bfs.py`（早前已并），但 **`benchmarks/case_clamped_plate_uniform.py` 与两个 `test_case_clamped_plate_uniform*.py` 仅在分支**。影响：能力未进产品面；不影响本次对 commit 内容的判级。修复=按平台流程合并该分支。
- **F2 [中·记录在案] 算例未注册到任何产品路径。** 全仓 grep 无 `structural_verification_face` / `benchmark_cases` 对 case 模块的引用；注册裁判里的 `<3%` 表值门只存在于 benchmark 脚本与 `tensor_benchmark_cases` 行。审计采信 L3 的依据是"算例站在生产求解器+同一验证面算子上"，而非"算例被产品调用"——这是本裁定的主要保留项。
- **F3 [低·已由审计补足] 平台 L4 端点判 `CLAIMED_NOT_PROVEN`。** `/tensor/bench/l4-verdict/16320`：`rows=0`、`failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]`，理由"无 tensorbench_results 行=自声明"。本审计以独立复跑（382/382 叶相同 + 图字节相同）替代该行，故 L4 的"可复现"按审计侧证据成立；平台侧复算行仍缺（建议后续补 `tensorbench_results`，属机制债不属本单功能缺陷）。
- **F4 [低] 18 闸门中有 2 条非精度闸。** `mindlin_shear_raises_thick_plate_deflection` 容差 100%（仅判方向：a/t=10 挠度上升 18.4%），`bfs_patch_constant_curvature_exact` 容差 0（判 6.8e-16 级舍入）。真正承重的是注册裁判 3% 门（got 0.334%）；"18/18 PASS" 的表述不虚但含弱闸，读者不应把 18 当 18 个独立精度证明。
- **F5 [低·参考值下限] 表值只有 3 位有效数字。** ref 0.00126 自带 ±0.40% 舍入带；got 差 0.334% 已在该带内，不能再宣称更小误差。被审单如实写出（`reference_rounding_band_relative=0.003968…`、Ritz N=10→12 截断带 0.216%），未粉饰。
- **F6 [中·历史对照] 本锚点前两轮审计的 evidence 指针是幽灵路径。** `tensor_maturity_verdicts` 中 m16203 / m16245 的 L4 裁定被 gate 以"entry/run_artifacts 工件缺失"重开：其 evidence 指向 `src/tensorfem/case_clamped_plate_uniform.py`（不存在）、`artifacts/CASE-FEM-CLAMPED-PLATE-UNIFORM/…`（不存在）。本次 evidence 每个指针都实际存在（gate `files_exist`/`impl_real`/`callers_verified` 全绿，`gate_level=4`）。

## 5. 结论与处置

- **Disposition：approve with notes.** 功能实质达 L4（完整算例 + 逐项误差分析 + 独立可复现），不进 L5。
- 消账条件已满足：`gate_level=4 ≥ 2` → gate 侧 confirmed。
- 遗留动作（非本单职责）：①合并 `agent/m16320` 分支（F1）；②case 注册进产品验证面/基准注册表（F2）；③补 `tensorbench_results` 平台复算行（F3）。
- 未跑清单（不得记为 passed）：集群 loong64 全量回归、`crossfamily` 8 测试、`oss_inventory` 横调研。