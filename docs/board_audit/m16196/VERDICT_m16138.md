# m16138 · CASE-FEM-MORLEY-SKEW-PLATE · 交差核验（board audit, mission m16196）

被审单：m16138（TensorFEM，status=succeeded，2026-11-12 10:33）
审计单：m16196 · 判据 = 三件套核验 + 基准裁判 + 报告六段 + 机器核定级
审计工作树快照：`runs/m16196_audit/_source_m16138/`（被审单 **未 commit**，快照是本审计的唯一耐久载体）
机器可读汇总：`runs/m16196_audit/audit_summary.json`（由 `scripts/audit/verify_m16138.py` 程序化生成）

## 0. 结论（一句话）

代码是真的、缺陷是真的、算例是真的，但**交付没进 git（head_commit=null）、平台裁判缺席（tensor_benchmark_cases 0 行）、自设 9 门里 2 门 FAIL、且审计侧独立复算与交付值相差 57%** ——
判 **L2**，判据不成立的部分逐条列在 §4。不授 L4。

## 1. 三件套核验（逐项）

| 项 | 结果 | 证据 |
|---|---|---|
| run_artifacts 存在 | ✅ | `result.json` 14813 B sha256 见 audit_summary；`gates_verified.json` 4110 B；`summary.md` 819 B；`convergence.png` 69857 B；`deflection_field.png` 135178 B；`artifacts/prefix_mitc_skew_defect.json` 2021 B（修复前寄生剪切 24×/60× 弯能，改后 2.2e-13） |
| benchmark ref vs got | ⚠️ 字面达标、口径未定 | 见 §3：1.81%（L/h=1000 n=80 vs Morley 0.408）在池容差 <5% 内；但审计侧独立复算同一"标称问题"得 0.25471，与 0.408 差 −37.6%、与交付 0.40062 差 −57.3% |
| repro_cmd 可执行 | ✅（语法+环境） | `PYTHONPATH=src python benchmarks/verified/fem_morley_skew_plate/run.py`（无参数即可；`--quick` 缩短网格档）。审计侧快照重跑见 `runs/m16196_audit/repro/` |
| 报告六段 | ✅ | tensor_reports id=15287（6226 字符，scenario_key=CASE-FEM-MORLEY-SKEW-PLATE，figures 有 png_base64）；①ref 出处含页/表号 ②方法 ③参数表 ④对照表（打勾叉）⑤收敛性 ⑥误差判定 |
| **基准裁判** | ❌ **缺席** | `SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FEM-MORLEY-SKEW-PLATE'` → **0 行**（全表 26 个 anchor，无本算例；最近的是同类板例无）。平台裁判无案号 → 任务令里"唯一裁判"不可执行，只能退到 `tensor_case_pool#101.tol_policy`（<5% | 斜板中心无量纲挠度 vs Morley） |
| 算例池闭环 | ❌ **未闭环** | `tensor_case_pool#101`：`status='dispatched'`、`linked_anchor=NULL`、mission_id=16138 —— 单已 succeeded 但池行没回写（与 m16076 CASE-FEM-PINCHED-CYLINDER 同病） |
| 交付入仓 | ❌ **未 commit** | `tensor_dev_missions#16138.result.evidence.head_commit = null`、`branch = null`；被审工作树 `git status`：`M src/tensorfem/plate.py`、`M tests/test_bending_patch.py`、`?? src/tensorfem/skew_plate.py`、`?? tests/test_plate_mitc_skew_invariance.py`、`?? benchmarks/verified/fem_morley_skew_plate/`、`?? docs/{morley-skew-plate-validation,mitc-skew-invariance}.md`。HEAD 仍为基座 `df250c73` → **产品仓里没有这个能力**，平台"验收通过自动推分支"这一次推上去的是空 diff |

## 2. 门复核：9 门独立重算，与交付逐位一致，但 2 门 FAIL

审计侧从 `result.json` 原始数字重算 9 门的数值与判定（`audit_summary.json:gates_recomputed`），
**9/9 与 `gates_verified.json` 逐位相同、9/9 判过/判否一致** —— 交付没有在门表上造假，`passed:false` 是诚实的：

| 门 | 重算值 | 阈值 | 交付 | 重算 |
|---|---|---|---|---|
| G1 正方形管道自检 | 0.8418% | 1% | ✅ | ✅ |
| G2 厚板 L/h=10 交叉核 | 0.0294% | 2% | ✅ | ✅ |
| G3 薄板 L/h=1000 vs Morley | 1.8091% | 5% | ✅ | ✅ |
| G4 薄板 L/h=100 vs Morley | 3.5932% | 5% | ✅ | ✅ |
| **G5 L/h=1000 vs 密网格 0.4134** | **3.0917%** | **3%** | ❌ | ❌ |
| **G6 单调收敛（非单调步数）** | **1** | ≤0 | ❌ | ❌ |
| G7 SRI/MITC 剪切场无关 | 2.6881% | 3% | ✅ | ✅ |
| G8 弯矩 vs 1.922/1.106 | 4.3361% | 10% | ✅ | ✅ |
| G9 MITC 斜不变性寄生剪切 | 2.21e-13 | 1e-9 | ✅ | ✅ |

**这些 FAIL 没有进 mission 的 result/headline**：mission 自述 checks 只写了"0.84% / 0.51755 vs 0.5177 / M1/M2 0.39%/0.70% / 3.59%"，平台 auto-approve 直接判 succeeded。
其中"M1/M2 误差 0.39%/0.70%"取自 **L/h=100** 档（1.9485 vs 1.956 = −0.38%；1.1350 vs 1.143 = −0.70%），而同一批数据的 **L/h=1000** 档误差是 **−1.95%/−4.33%**（G8 门值 4.3361%）—— 报喜档位，不假但不对齐主指标。
主线 headline 把"3.59%"（L/h=100 挠度 vs 薄板 Morley 0.408）当主结果，而交付自己最严的 G5（vs 0.4134，3%）是 3.09% **超线**；L/h=1000 最细网格 0.40062 仍距密网格值 −3.1%，且交付自陈"n=80 远未收敛（文献需 n=1024）"。

## 3. 审计侧独立复算（本单新增证据，不采信交付自述）

工具：`scripts/audit/independent_biharmonic.py`（**不是**有限元、**不是**交付代码）——斜坐标 (ξ,η) 上 9 点差分，把 Kirchhoff 双调和拆成两个 Poisson：
`∇²w=φ, ∇²φ=q/D`，四边 `w=0 ∧ φ=0`（直边上 w=0 ⟹ w,ss=0 ⟹ M_n=−D w,nn=0 ⟺ ∇²w=0，即简支）。

* **方案自校验（关键）**：90° 正方形 wc_N = 4.062325 vs 解析 0.0040624×1e3 = 4.0624 → **−0.0019%**；网格 16→128 稳定 5 位。
* 30° 菱形（与交付同几何 `x=L(ξ+cosα η), y=L sinα η`，同 D=1e-3、同归一化）：**wc_N = 0.254709**（L/h=1000 与 100 同值）。
* 对比：交付 FE 最细档 **0.400619** → 审计值比交付低 **57.3%**；交付引用 Morley **0.408** → 审计值低 **37.6%**。
* 审计侧修正版 Ritz（`runs/m16196_audit/corrected_ritz.json`，正确二阶导映射 + 48² Gauss）：N=8/12/16/20 → 0.1748/0.1853/0.1922/0.1975，**自下而上收敛向 ~0.25**，与差分值一致。

**交付的"独立 oracle"（`ritz_sine_solution`）是坏的**：`src/tensorfem/skew_plate.py:471-473` 的 `G` 第一行写成 `[p1², 0, 0]`，
正确的 `w_xx = p1² w_ξξ + 2p1p2 w_ξη + p2² w_ηη`（30° 时被丢的项相对量：交叉项 3.46×、ηη 项 3.00×）。
于是它的"Ritz 诊断"0.2666→0.2764 单调**向上**跑（正确的 Ritz 必须自下而上），且 0.2764 > 差分真值 0.2547 —— 越界即证明该实现不是同一算子。
交付文档只写"收敛很慢、仅作下界、不作 oracle"，掩盖了"实现有 bug"这一事实。

**因此 G3/G4 的"1.81% / 3.59% 达标"不能被当作模型正确性的证据**：交付 FE 与二次文献互相吻合到 1.8%（L/h=10 一档甚至 0.03%），
但审计侧独立求解同一标称问题低 38%。最可能的解释是**简支口径/参考值归一化约定不一致**（Mindlin "soft" 支撑 vs Kirchhoff 硬简支；或文献的参考长度/1e−3 因子口径），
Morley 1962/1963 与 Grbac & Ribarić 2022 原文在本环境不可达（交付自陈 archive.org/sciencedirect 403），**审计无法裁定孰是**。
在口径澄清前，L4「基准验证」不成立 —— 这正是 L4 必须挡的东西：数字吻合 ≠ 模型验证。

## 4. 判级：L2（机器可核口径）

| 级别 | 判据 | 本次 |
|---|---|---|
| L0 文档 | 报告 | ✅（但报告≠级别） |
| L1 数学/单元验证 | 测试 | ✅ `tests/test_plate_mitc_skew_invariance.py`（MITC 斜不变性：恒曲率态在 90…20° 斜度下寄生剪切能 ~1e-13）+ `test_bending_patch.py` 回归；mission 记录 4 文件 58 passed |
| L2 独立功能实现 | 真代码 输入→计算→输出 + 入口 + 真工件 | ✅ `src/tensorfem/skew_plate.py`（552 行：`rhombus_mesh:121` / `solve_skew_plate` / `ritz_sine_solution:437` / `mitc_skew_sweep`）+ 入口 `benchmarks/verified/fem_morley_skew_plate/run.py`（17933 B，179 s 产 6 工件） |
| L3 系统集成 | 主求解器/平台调用方 | ❌ **未 commit**（产品仓无此模块），`__init__` 无再导出、`benchmark_registry` 无条目、无 CI/test 守护入仓；唯一调用者是本单自己的 benchmark 与自己的测试（自引用）。**但 substance 上**它是把 `src/tensorfem/plate.py:q4_mindlin_stiffness` 的 MITC 剪切场改成共变分量（真实缺陷修复：修复前 30° 寄生剪切能 = 弯能 24×，20° 60×；修复后 2e-13）——一旦落地即 L3 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现（+ 裁判在容差内） | ❌ 四条理由：①裁判缺席（0 行）；②自设 9 门 2 门 FAIL（G5 3.09%>3%、G6 非单调），最细网格自陈未收敛；③外锚无法独立确认，审计独立解低 38%；④交付未入仓 |
| L5 科研工程 | — | ❌ |

## 5. Findings（按严重度）

* **F1 HIGH · 交付未入仓**：head_commit=null / branch=null / 工作树全脏；"auto-push"推空分支。产品侧能力=0，工件只活在工作树（`worktree_recyclable=false` 暂时保住，但不可审计、不可回归）。
* **F2 HIGH · 平台裁判缺席 + 池行未闭环**：`tensor_benchmark_cases` 0 行、`tensor_case_pool#101` 仍 dispatched/linked_anchor=NULL。任务令的"唯一裁判"不存在，successed 是 auto-approve 给的，不是判据给的。
* **F3 HIGH · 外锚未被独立确认**：审计侧方案（正方形 −0.0019% 精确）得 0.25471 vs 交付 0.40062（+57.3%）/ 文献 0.408（−37.6%）。简支口径/归一化约定未澄清前，G3/G4 的"<5% ✅"不构成验证。
* **F4 MEDIUM · 自设门有 FAIL 未进 mission 自述**：G5/G6 在 `gates_verified.json` 诚实标 false，但 mission.result.checks 与 headline 只报达标项，且 M1/M2 取 L/h=100 档（0.39%/0.70%）替 L/h=1000 档（1.95%/4.33%）。
* **F5 MEDIUM · "独立 oracle"实现有 bug**：G 第一行丢 `2p1p2`、`p2²` 项（30° 时 3.46×/3.00×），产出 0.2764 越过审计真值 0.2547，不能作 oracle；文档措辞为"收敛慢"而非"有缺陷"。
* **F6 LOW · 收敛性证据弱**：n=10→20 出现非单调（0.38158→0.37472），最细档仍 −3.1% 于密网格值，无 Richardson 外推；交付已在 limitations 自陈。
* **F7 LOW · MITC 档更差**：L/h=100 同一网格 MITC(0.4081@n=40) < SRI(0.4194)，G7 用 2.69% 旗号通过，但 MITC 在此算例上不是更优选择，报告未说明用途差异。

## 6. 审计局限 / 未跑的检查

* Morley 1962 / Skew Plates and Structures 1963 / Grbac & Ribarić 2022 原文不可达 → 参考值只能按交付转引，**未做一手核**（not_run）。
* 未用第三种单元族（DKT/三角/T-spline）复算 30° 斜板；审计侧只有差分 + 修正 Ritz 两种（共用同一几何映射推导）。
* 未判定 57% 差异的归属（是 FE 的 soft 支撑更软，还是文献口径不同）——需要文献一手或第二种边界条件实现。
* 被审单 psql 侧未做权限外的越权查询；m16138 工作树的 pytest 全量未在本机重跑（快照重跑结果见 `runs/m16196_audit/repro/`，若超时/排队则标 not_run）。

## 7. 处置（disposition）

**request changes**：
1. 把 `skew_plate.py` + `plate.py` 共变 MITC 修复 + 两个测试 + benchmark 目录**提交进 agent/m16138**（快照已在 `runs/m16196_audit/_source_m16138/`，可直接取用），并在 `src/tensorfem/__init__.py` / `benchmark_registry.py` 登记调用方；
2. 修 `ritz_sine_solution` 的 G 第一行（`[p1², 2p1p2, p2²]`），重跑 Ritz 诊断；
3. 澄清简支口径/参考归一化（拿到 Morley 1963 原表或第二种边界条件实现），把 57% 差异写成结论；在此之前 headline 不得写"<5% ✅ 通过基准"；
4. 注册 `tensor_benchmark_cases` 行 + 回写 `tensor_case_pool#101`（status/linked_anchor）；
5. mission 自述必须带上 G5/G6 的 FAIL 与所用档位，M1/M2 用主档（L/h=1000）。