# 交差审计 m15342（CASE-NAFEMS-LE11 椭球壳内压 · TensorFEM）

审计单 m15388 · 席位 board-audit · 2026-11-11 · verdict #317

## 1. 对象与锚点

- 被审单：m15342（status=succeeded，PR #698 merged，head=4c36c816），返修轮1，原单 m15261。
- 锚点：任务卡写 `CASE-NAFEMS-LE1`（椭圆膜应力集中）——系原单沿用的误标。m15342 实际锚点
  **CASE-NAFEMS-LE11**（3D 椭球壳体应力，task_key=case-nafems-le11-rework1）；LE1 真实载体为
  m15238（verdict#283 F1_scope_mislabel 已记）。本审计按 m15342 实际交付（LE11 椭球壳）核验。
- 前史：verdict#283（scope 'm15261 CASE-NAFEMS-LE1'）gate 判 L1 → 自动开返修 m15342；
  verdict#273（scope 'm15261 CASE-NAFEMS-LE11'）曾 confirmed L4（同案先例）。

## 2. 取证与复跑（全部本单实测，2026-11-11，worktree /root/tp-agent/work/m15342）

| 检查 | 命令/来源 | 结果 |
|---|---|---|
| 一键复现 | `PYTHONPATH=src python3 benchmarks/nafems_le11/run.py` | rc=0，verdict=PASS，wall=28.6s |
| 数值一致性 | 新跑 results.json vs 入库件 | 除 wall_seconds 外逐位一致（五项误差 0.005/0.003/0.077/2.646/2.307%） |
| 证据脚本 | `PYTHONPATH=src python3 benchmarks/nafems_le11/maturity_evidence.py` | rc=0，live_verdict=PASS，registry_ids=4 |
| 回归测试 | `pytest tests/test_ellipsoid_shell.py tests/test_nafems_le11_case_pool.py -q` | 18 passed in 12.98s |
| CLI 集成面 | 工件 cli_verify_report.json | total 59 / passed 59（含 4 项 shell.spheroid.*） |
| 工件存在 | benchmarks/nafems_le11/{results.json 17370B, report.md 3251B, evidence.json 6531B, cli_verify_report.json 23676B} | 全部存在 |
| 平台报告 | tensor_reports id=14434 | 六段齐全（①案例 ②方法 ③结果 ④收敛 ⑤注意点 ⑥误差判定）+图+工件 |
| 基准裁判表 | `SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug IN ('CASE-NAFEMS-LE1','CASE-NAFEMS-LE11')` | **0 行**（缺陷 F2） |
| 容差背书 | tensor_case_pool id=40 case_code=NAFEMS-LE11 tol_policy='<5%' + mission goal 预声明 <5% | 成立 |
| gate 预裁 | 直接调用 /root/tp-agent/maturity_gate.py::judge(4, evidence, 'TensorFEM', 15388) | gate_level=4，why=[] |

## 3. 基准裁判（ref vs got，报告网格 256 facets / 771 dof，容差 <5%）

| 量 | ref | got | 相对误差 | <5% |
|---|---|---|---|---|
| N_phi_pole | 0.999991 | 0.999945 | 0.005% | ✓ |
| N_theta_pole | 0.999974 | 0.999944 | 0.003% | ✓ |
| N_phi_equator | 0.500001 | 0.499614 | 0.077% | ✓ |
| N_theta_equator | -0.999992 | -0.973530 | 2.646% | ✓ |
| w_equator | -115.000000 | -112.346557 | 2.307% | ✓ |

最差 2.646% < 5%（且 <3%）。ref 出处：闭式薄膜静力（帽段轴向平衡 N_φ=pR2/2 + 法向平衡
N_θ=pR2(1−R2/(2R1))，精确平衡方程）+ 独立 Rayleigh-Ritz 膜位移解（自检复现闭式膜力至 1e-9）。
本审计独立重推闭式值（极部 N=1.0、赤道 N_φ=0.5、N_θ=−1.0）与工件一致；收敛表显示 FEM 随网格
单调收敛至 ref（w: 63.3%→2.3%，N_θ: 68.1%→2.6%），ref 与 FEM 相互印证。

## 4. 判级

- L2（impl 非占位 + entry + run_artifacts）：✓ 全部真实（ellipsoid_shell.py 379 行 /
  axisym_shell.py 366 行 / integration_callers.py 89 行 / run.py 278 行；工件入库非 runs/）。
- L3（integration_callers）：✓ benchmark_registry 注册 4 项进 `tensorfem.cli verify`（59/59），
  `run_nafems_le11_case()` 被 maturity_evidence.py 与测试真调用。
- L4（benchmark ref vs got + repro_cmd）：✓ 五项数值对照全过 + 复现命令本单实跑通过。
- L5：✗ external_anchor 非文献/实验值引用（手册不可得，见 F3）。
- **裁定：L4**（与 verdict#273 先例一致；附百页报告不改变级别，级别=功能实质）。

## 5. 发现（findings）

- **F1（info）锚点误标**：任务卡 'CASE-NAFEMS-LE1' 沿袭原单误标，实际 LE11；scope 串按任务卡
  写法落表供对账。
- **F2（medium）裁判表缺行**：tensor_benchmark_cases 无 CASE-NAFEMS-LE11 登记行（0 行），
  基准裁判无行可判；容差由 tensor_case_pool id=40 + mission goal 背书。建议算例池侧补登记
  （7 项指标 tol <5%）。
- **F3（medium）ref 非手册转录**：NAFEMS LE11 手册沙箱不可得（nafems.org 可达但文档 403），
  ref 降级为闭式精确静力 + 独立 Ritz；results.json provenance_note / docs §6 / evidence.json
  external_anchor 三处如实披露，非虚标；拿到手册后须补手册目标值为新增判定项。
- **F4（low）质量门告警**：mission result server_quality_audit gate_valid=false /
  key_present=false（finish 时质量门告警），不影响 maturity gate 证据复核。

## 6. 落表

`INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFEM', 'm15342 CASE-NAFEMS-LE1', 15388, 4, '<evidence.json>', 'board-audit')`
→ verdict id=317，gate_state=pending；gate 守护（30min 周期）自动裁定：
confirmed=消账 / reopened=返修。evidence 键=impl_files/entry/run_artifacts/tests/
integration_callers/benchmark/repro_cmd/external_anchor，值=真实指针（见 evidence.json）。
