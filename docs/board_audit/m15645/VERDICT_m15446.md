# 交差核验：m15446 · REPRO-FEM-HILLERBORG1976（TensorFEM）

审核单 m15645（board-audit）。被审单 m15446（succeeded）。
本报告全部数字由 `scripts/audit_m15446_hillerborg.py` 程序化抽取到 `docs/board_audit/m15645/verify_output.json`，零手抄。

## ① 案例介绍

锚点论文：Hillerborg, Modéer & Petersson (1976), *Cem. Concr. Res.* **6** 773-781 —— 虚构裂缝模型（FCM）：
裂缝面为张力软化面 `sigma = f(w)`，由 `f_t`、`G_F` 表征，本体线弹性；数值实现=影响函数（柔度）法。
算例：单边切口三点弯（SENB），几何相似系列 `S/W = 4`、`B/W = 1`、`a0/W = 0.2`；
材料点 `E = 30 GPa, nu = 0.2, f_t = 3 MPa, G_F = 100 J/m^2`，`l_ch = E G_F/f_t^2 = 0.3333 m`；
`W/l_ch` 系列 {0.25, 0.5, 1, 2, 4, 8, 16}，裂纹通道节点 48（收敛对照 96），CMOD 控制 40 步。

## ② 怎么做的（核验流程）

1. 取证：读 m15446 工作树 `artifacts/hillerborg1976/`、`src/tensorfem/fictitious_crack.py`、
   `benchmarks/hillerborg1976_repro.py`、`tests/test_fictitious_crack.py`、`docs/repro-hillerborg1976.md`，并取 sha256 存档：
   {"repro.json": "de3e1f2de48f7411", "fig_size_effect.png": "48cdfe15b6922edd", "fig_load_cmod.png": "731d5819a7c517e3"}。
2. 基准裁判：`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='REPRO-FEM-HILLERBORG1976'`
   —— **返回 0 行**（该锚点未登记基准算例）。同论文不同锚点 `XVAL-FEM-TPB1976` 存在（TensorFEM，pending），
   其判据是 LEFM 极限 `P_cr` vs 闭式 <10%，**不是**本单交付的软化/尺寸效应判据，不能替代。
3. 判据独立复算：不采信工件内 `passed` 标志，直接从原始数字 vs 工件文档字符串里**预先声明**的容差重算（M2 纪律）。
4. 集成核验：全仓 grep `fictitious_crack` 的 import 方；`src/tensorfem/__init__.py` 是否导出。
5. 复现核验：`python3 benchmarks/hillerborg1976_repro.py --quick --out /tmp/...` 实跑（60 s）；
   全量复跑递交集群作业 cj1165（repo=TensorFEM, ref=agent/m15446）。

## ③ 计算结果

### 判据逐条（复算，4 PASS / 6 FAIL，与工件自报标志分歧 0 条）

| 判据 | 量 | got | ref | 容差 | 复算 |
|---|---|---|---|---|---|
| K2_law_area | int f(w)dw vs G_F (rel.) | 1.019e-06 | 0.0 | 0.0001 | PASS |
| K1_elastic | K_I from model compliance vs Srawley-Gross | 11.8 | 0.0 | 5.0 | FAIL |
| K3_balance | worst |W_ext-U_solid-U_if-D|/W_ext | 2.926e+33 | 0.0 | 0.01 | FAIL |
| K4_lefm | P_peak/P_LEFM at W/l_ch=8 | 0.4638 | 1.0 | 5.0 | FAIL |
| K5_strength | P_peak/P_strength at W/l_ch=0.25 | 0.5877 | [0.75, 1.05] | 0.0 | FAIL |
| K6_mesh | P_peak drift nodes_y 48->96 | 1.319 | 0.0 | 2.0 | PASS |
| K7_penalty | P_peak drift over k0 1e2..1e4 | 0.5011 | 0.0 | 1.0 | PASS |
| K8_aW | P_peak(a0/W) monotone decreasing | 1 | True | 0.0 | PASS |
| K9_master | size-effect master-curve collapse deviation | 36.67 | 0.0 | 3.0 | FAIL |
| K10_gf | G_F back-calculated vs 100 J/m^2 (rel.) | 0.8027 | 0.0 | 0.05 | FAIL |

### 尺寸效应序列（工件原值）

| W/l_ch | P_peak [N] | P/P_LEFM | P/P_strength | G_F 反算 [J/m²] | 能量平衡 | 收敛步 |
|---|---|---|---|---|---|---|
| 0.25 | 1958.9 | 0.221 | 0.588 | 86.96 | 3.84e-01 | 38/40 |
| 0.50 | 6682.2 | 0.266 | 0.501 | 86.61 | 4.77e-01 | 38/40 |
| 1.00 | 22375.5 | 0.315 | 0.420 | 85.77 | 3.22e-03 | 40/40 |
| 2.00 | 73058.6 | 0.364 | 0.342 | 82.71 | 6.29e-02 | 36/40 |
| 4.00 | 234687.2 | 0.414 | 0.275 | 69.87 | 4.75e-01 | 29/40 |
| 8.00 | 744305.6 | 0.464 | 0.218 | 49.57 | 6.16e+32 | 33/40 |
| 16.00 | 2546824.3 | 0.561 | 0.187 | 19.73 | 2.93e+33 | 35/40 |

关键判决点：`K10_gf` 反算 G_F 最坏相对误差 **0.803**（容差 0.05）；
`K4_lefm` 在 W/l_ch=8 处 `P_peak/P_LEFM = 0.464`（LEFM 渐近应为 1.0，偏差 53.6%）；
`K3_balance` 最坏能量不平衡 **2.926e+33**（容差 1e-2）——量级荒谬，属数值失稳而非舍入。

## ④ 结果图

1. 尺寸效应归一化曲线 vs LEFM/强度渐近（K4/K5 判定依据）；
2. 逐条判据 实测偏差/容差 比值条图（>1 即 FAIL）；
3. 耗散反算 G_F vs 输入 100 J/m²（K10 判定依据）。

## ⑤ 注意点（核验中发现的东西）

- **run_artifacts 未随主分支落库**：m15446 的 `artifacts/hillerborg1976/*` 只存在于任务工作树与任务分支
  （commit `bc5d61a8` 有 repro.json），落到 main 的 `af10148d` 自述「4 个文件被门禁滤除」——即 main 上**没有**运行工件。
  这直接威胁 L2 机器复核（gate 的 files_exist 在 main/worktree 找文件），也威胁复现链。审计已把三件工件 sha256 存档于
  `docs/board_audit/m15645/artifacts_m15446/`。
- **判据设计瑕疵（不改变判级）**：K4 要求在 W/l_ch=8 就 5% 逼近 LEFM 渐近，与 Bažant 尺寸效应律（该 β 下仍远低于 1）
  不符，属**预声明阈值本身不合理**；K5 用 `(2/3) f_t B (W-a0)^2/S` 作强度极限对韧性试件亦非紧上界。
  两条 FAIL 部分是判据问题，但按「预声明即裁判」纪律仍记 FAIL，且 K3/K10 的 FAIL 是真数值/建模缺陷。
- 无 1976 原文图版对照（交付自述限制），故「对标文献值」实为对标闭式解——外部锚点强度弱。
- 全量运行 368 s/单核；quick 模式（60 s）判据结果与全量不同（K5 通过、K6/K7 不过），CI 若用 quick 会误读。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 软化曲线包面积 ∫f dw | G_F=100 J/m² | 1.02e-06 | 是 |
| 弹性 K_I（柔度导数） | Srawley–Gross 手册 | 11.80% | **否** |
| P_peak 网格收敛 | 自对照 48→96 | 1.32% | 是 |
| P_peak 罚因子敏感性 | 自对照 k0 1e2..1e4 | 0.50% | 是 |
| P_peak(W/l_ch=8) | LEFM 渐近 1.0 | 53.62% | **否** |
| P_peak(W/l_ch=0.25) | 强度极限带 [0.75,1.05] | 低 21.6% | **否** |
| 主曲线塌缩 | 同 l_ch 缩放材料 | 36.67% | **否** |
| G_F 耗散反算 | 输入 100 J/m² | 80.27% | **否** |
| 能量平衡 | 守恒应≈0 | 2.9e33（量纲≈无量纲比值） | **否** |

结论：**核心物理量未达 3%（亦未达各判据预声明容差），不构成「论文复现成功」**；可复现为真，精度未立。

## 判级（L0–L5）

| 档 | 机器判据 | 本单 |
|---|---|---|
| L1 | tests 非空 | ✅ `tests/test_fictitious_crack.py` 15 passed（本地实跑 15.62 s） |
| L2 | impl_files + entry + run_artifacts | ✅ 1147 行真实 Q4 影响矩阵/软化律/CMOD 控制 + driver + repro.json（sha 存档） |
| L3 | 另需 integration_callers | ❌ 主求解器 0 调用方（grep 命中 0；`__init__` 未导出；唯一调用方是 benchmark driver 与测试） |
| L4 | 另需 benchmark(ref vs got)+repro_cmd | ❌ 锚点未登记基准算例（0 行）；且 10 条自声明判据 6 条 FAIL |
| L5 | 另需 external_anchor | ❌ 无原文图版/实验值锚点，仅闭式解 |

**裁定：L2**（独立功能实现成立，系统集成与基准验证均缺）。
owner 榜令的「基准判据是唯一裁判」在本单落空：裁判席位空缺（未登记），因此不能记 L4；
平台内已有同族弱点——`REPRO-FEM-HILLERBORG1976` 应有 benchmark 行（建议补登记：
metric=`P_peak/P_LEFM @W/l_ch=8`, `G_F 反算`, `能量平衡`，容差取预声明值），否则复现类锚点永远无法被判。

诚实说明：本审计不重跑 368 s 全量（quick 已证可执行 + 集群 cj1165 补跑），对「工件数字是否可被同 commit 重算
复现到 1e-9」只做了弱核验（quick 与全量判据结论方向一致、量级一致）；这是本单的核验局限。
