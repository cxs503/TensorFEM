# m20186 交差核验：m20143 / CASE-FEM-HEAT-L2（TensorFEM）

审计方：board-audit（m20186）｜对象：已 `succeeded` 的交差单 m20143｜性质：核验（非重做）
裁判优先级：`tensor_benchmark_cases`（唯一裁判） > 仓库可复核事实 > 报告自述

## 0. 结论（一句话）

**m20143 对仓库的交付为零**：实现/测试/工件/报告只存在于 `/root/tp-agent/work/m20143` 的**未提交工作树**
（`git for-each-ref --contains` 无任何 ref 含这些文件，`agent/m20143` == `main`@f634c7c5，`main` 树内 `heat` 匹配 0 个文件）；
`tensor_reports` 无 mission_id=20143 行（报告未发布）；`tensor_benchmark_cases` 无 `anchor_slug='CASE-FEM-HEAT-L2'` 行（**无裁判行 → L4 结构上不可能**）。
同一锚点的实质能力由 **m20054** 交付（commit `8867c2a4`，branch `agent/m20054`，m20108 已给 L4，verdict id=828），
m20143 是其**平行重复实现**，且其判据更弱。

## 1. 取证：三件套逐项核验

| 三件套 | 声明 | 核验结果 |
|---|---|---|
| run_artifacts | `benchmarks/verified/fem_heat_conduction_l2/{result.json,gates_verified.json,fig_*.png}` | **文件真存在**（工作树内，3244+1501+2252 B + 2 PNG，共 ~194 KB）；但**不在任何 git ref**：`git log --all -- benchmarks/verified/fem_heat_conduction_l2` → 空 |
| benchmark ref vs got | ref=制造解 `T=sin(πx)sinh(πy)/sinh(π)`，got=4 级网格 L2 | ref 存在（解析式，程序化求值，非手抄）；**got 数值可 bit 级复现**（见 §2）；但**无 DB 裁判行**，容差 3%/1.8 为作者自定 |
| repro_cmd | `OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_heat_conduction_l2/run.py` | **可执行**：`pytest tests/test_heat_conduction_l2.py -q` → **8 passed in 6.39 s**；独立驱动复算 4 级全部命中（相对偏差 ≤1.2e-16） |

补充：报告 `docs/case_fem_heat_l2_report.json` 六段齐全（①②③④⑤⑥）、2 图内嵌、tags 6 个；
但 `SELECT ... FROM tensor_reports WHERE mission_id=20143` → **0 行** ⇒ 报告纪律（"报告未发=本单未完成"）未满足。

## 2. 独立复算（不用对方 run.py）

脚本 `docs/board_audit/m20186/verify_heat_l2.py`（只复用被测模块，误差度量自行重写）：

| 网格 | 声明 L2_rel | 复现 L2_rel | 相对偏差 | n_dof 一致 |
|---|---|---|---|---|
| 4×4 | 2.2854545e-2 | 2.2854545e-2 | 0 | ✔ 9 |
| 8×8 | 6.5170398e-3 | 6.5170398e-3 | 0 | ✔ 49 |
| 16×16 | 1.7658990e-3 | 1.7658990e-3 | 0 | ✔ 225 |
| 32×32 | 4.6179077e-4 | 4.6179077e-4 | 1.2e-16 | ✔ 961 |

⇒ 数值**没有硬编码/编造**，是真求解输出（torch.float64 直接解）。

## 3. 关键实质缺陷（按严重度）

### F1【阻塞·判据】"L2 收敛率 ≥2 阶" 未被交付数据支持
- 声明（报告标题/门 2）：≥2 阶；实际交付的 3 个观测阶 = **1.810 / 1.884 / 1.935，均值 1.876 < 2**；
- 门限被作者自降到 **1.8**（`heat_conduction_l2.py:order_threshold=1.8`；`gates_verified.json:criteria`），即**锚点要求 2、自定门限 1.8**，属"事后放宽阈值"；
- 扩大到 64×64（本审计新增一级）：全部节点口径 1.966、自由 dof 口径 2.038 → **趋势指向 2 但 4 级梯子内从未达到 2**。

### F2【根因·度量】L2 范数把精确 Dirichlet 边界节点算进分母/分子，系统性压低观测阶
- m20143：`l2_rel = RMS(T_h−T)/RMS(T)` 在**全部节点**上（含边界节点，边界误差恒为 0）；
  边界点占比随 h 变化（32×32 边界 124/1089），误差被稀释的比例随网格改变 → 观测阶被压到 2 以下；
- 本审计按**自由 dof 口径**复算：2.392 / 2.173 / 2.080 / 2.038（→2 从上方收敛）；
- 交叉证据：m20054（同一锚点、已交差、L4）用**独立 Gauss 单元积分**∫e²/∫T² 口径，finest 阶 **1.998~2.000**（commit `8867c2a4`：`src/tensorfem/heat_l2_convergence.py:112-113,233`）⇒ 1.876 是**度量口径缺陷**，不是 Q4 核缺陷。
- 附带：报告 §② 称"Richardson 外推计算观测阶"，实际代码是 log-log 斜率（`heat_conduction_l2.py:157-163`），无外推 → 方法学描述失真。

### F3【门 3 是同义反复】"heat_balance_residual < 1e-10" 不构成守恒校验
`heat_conduction_l2.py:107-111` 计算的正是线性解算 `K_ff·T_f − f_f` 的残差——任何线性求解器解完都 ~1e-15。
它不是独立守恒不变量（不是净热流出/入积分），所以该门通过是**构造上必然**，不提供信息；测试 `test_heat_conduction_l2_heat_balance`
同样只断言了这个恒真量。

### F4【交付缺失】工作树未提交 + 报告未发 + 分支与 main 同点
`git status` → `M src/tensorfem/__init__.py` + 4 个 `??`；`git log main..agent/m20143` → 空。即 m20143 的全部产出**没有进入任何 ref**，
平台"验收后自动推分支"对它永不触发；`tensor_reports` 无 20143 行；`tensor_maturity_verdicts` 无 20143 行（本单为首次落表）。

### F5【治理】同一锚点双重实现，且非法版本更弱
全局 grep：`main` 树内 `heat` 匹配 **0** 个文件；`refs/heads/agent/m20054` 含完整实现
`src/tensorfem/heat_l2_convergence.py`（720 行）、`benchmarks/heat_l2_convergence.py`、`tests/test_heat_l2_convergence.py`、`artifacts/CASE-FEM-HEAT-L2/evidence.json`、
`docs/CASE-FEM-HEAT-L2-steady-conduction.md`（commit `8867c2a4`）。m20143 与 m20054 是同一锚点的两份平行实现，
m20143 版度量口径更弱（F1/F2）、且未进入仓库 → 应判为**重复劳动作废**，锚点能力记账归 m20054。

## 4. 判级（L0–L5）

**level = 0**。
- L2（独立功能实现）在**工作树层面**确实存在（真实求解→真实误差→真实收敛率，可复现），
  但"级别=功能实质"针对的是**进入仓库/交付的实质**：m20143 贡献的 ref 内文件数 = **0**，报告行 = 0，裁判行 = 0；
- L3 无（唯一"集成"是未提交的 `__init__.py` 导出，且平台侧无任何 caller）；
- L4 **结构上不可能**：`tensor_benchmark_cases` 该锚点 0 行（125 行 / 全表），无容差即无裁判；且锚点已有 m20054 的 L4（verdict 828），重复授予会重复记账；
- 故 L0 + 差异披露：m20143 若把工作树提交，上限也只是 **L2**（无裁判行、无集成 caller），且其 "≥2 阶" 头条与其自身数据冲突（F1）。

## 5. 处置

`reopened`（返修）：① 报告发布到 `tensor_reports`（mission_id=20143）；② 若仍主张该锚点，改用单元积分 L2 口径并提交 ref；
③ 否则**明确作废**并与 m20054 去重（该锚点能力已由 m20054/`8867c2a4` 交付并经 m20108 判 L4）；
④ `tensor_benchmark_cases` 缺 `CASE-FEM-HEAT-L2` 裁判行需补登（否则任何 L4 主张都无机器裁判，属平台侧欠账）。

## 6. 未做（unknown / not_run）

- 未审 m20054 实现本身的正确性（只交叉引用其口径与 commit，属 m20108 审计范围）；
- 128×128 扩展阶梯：本地超 120 s 已递交计算门（cj2547），若未在窗口内回收则记 not_run，结论以 64×64 为准；
- 未测试非 Cartesian/剪切网格（报告 §⑤ 亦自承未覆盖）。
## 7. 执行记录（可复核）

- 独立复算：`PYTHONPATH=/root/tp-agent/work/m20143/src LADDER=4x4,8x8,16x16,32x32,64x64 python verify_heat_l2.py`
  → `docs/board_audit/m20186/verify_heat_l2.json`（本地 stdout 同内容）
- 交付测试：`PYTHONPATH=/root/tp-agent/work/m20143/src pytest tests/test_heat_conduction_l2.py -q` → 8 passed / 6.39 s
- 图：`python make_figures.py` → `fig_m20186_convergence.png`, `fig_m20186_claim.png`
- 落表：`python insert_verdict_m20186.py` → `tensor_maturity_verdicts` **id=851**, level=0, verdict_by='board-audit'
- 报告：`python docs/board_audit/m20186/post_report.py` → `tensor_reports` **id=18400**（scenario_key=CASE-FEM-HEAT-L2, mission_id=20186）
- 计算门：cj2547（stage/PYTHONPATH 配置错，rc=1，已作废）、cj2563（64×64+128×128 扩展阶梯）**仍在排队，未回收 → not_run**；
  128×128 结论记为未完成，§3/§6 的结论以 64×64 级为准（该级已足以表明全节点口径 orders<2、自由 dof 口径 >2）。
