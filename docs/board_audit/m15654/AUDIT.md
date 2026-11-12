# 交差核验：m15427 — CASE-BENDING-PATCH（TensorFEM）

**审计单** m15654 ｜ **对象** m15427（succeeded，PR #706 / commit `3b12fbe0`，merged `177457f1`）
｜ **锚点** `CASE-BENDING-PATCH` ｜ **裁定级别** **L4** ｜ **处置** 见 §8

本单是**核验**，不是重做：不修改被审工件，只把 m15427 的每条主张拉回原始证据复核。
所有数字由 `audit_bending_patch.py` / `poisson_crosscheck_rederived.py` 程序化注入，报告内无手抄数字。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

常量曲率 patch test：Q4 Mindlin-Reissner 板单元在「无横向剪切的常曲率状态」下必须精确复现该状态。
误差来源只有插值能力，因此该算例是**单元一致性（consistency）必要条件**，不是工程精度算例。

| 项 | 值 |
|---|---|
| 材料 | E = 210000 MPa，ν = 0.3，t = 1 mm |
| 曲率状态 | kx=0,ky=0,kxy=0.001, kx=0.001,ky=0,kxy=0, kx=0.001,ky=0.0006,kxy=0.0004 |
| 无量纲数（程序注入） | D/Et³ = 0.0915751，Ds/Et = 0.320513，弯曲/剪切刚度比 = 0.285714 |
| 网格族 | affine / skew / distorted（27 个 patch 用例 + 8 行细化阶梯） |
| 载荷驱动对照 | 悬臂板条 20 例（力矩边界，答案未给定；对称胞 = 柱面弯曲 D，自由边 = 梁刚度 D_b=E t³/12） |
| 基准出处 | 常量曲率闭式场（Irons & Ahmad 1980 ch.8；Zienkiewicz & Taylor Vol.2 §10.4）；板条 M L²/(2D)（Timoshenko & Woinowsky-Krieger Art.10） |

## 2. 怎么做的（核验方法 / 工具链）

1. **三件套取证**：在 main（PR #706 后）与 m15427 工作树双向比对 impl/entry/run_artifacts/tests/docs/4 张图，
   sha256 逐文件比对（10 个文件，全部存在且**逐字节相同**）。
2. **裁判面**：查 `tensor_benchmark_cases`（SQL 由 `runuser -u postgres` 直读，见 §5 F1）。
3. **ref 独立性**：不 import tensorfem，用纯数学把每个 ref 独立重算（D、D_b、M=D_b κ、w=M L²/(2D)、θ=-M L/D）。
4. **复现**：干净副本 `/tmp/audit_bend`（去 .git）重跑 `run.py` 与 `pytest`，与提交的 `result.json` 逐标量 diff。
5. **门自检**：17 个 gate 逐条按各自 operator/threshold 重判，并核两份工件里的 criteria 是否一致。
6. **收敛阶重算**：对 ladder 行独立做 log-log 最小二乘，比对声称的 order。
7. **恒等性检查（tautology）**：判断哪些"精确"是结构性的、哪些是真实精度信息。
8. **F-CRIT 定点复核**：`poisson_crosscheck` gate 的 measured 比值到底取自计算挠度还是闭式参考值。

## 3. 计算结果（ref vs got，逐项）

### 3.1 闭式 ref 的独立复现（不 import tensorfem）

- 弯矩 ref 独立重算 M = D_b κ：`[22.69230769230769, 17.307692307692307, 2.6923076923076925]` vs got `[22.69230769230769, 17.307692307692307, 2.6923076923076925]`，
  最大相对偏差 **0**。
- 板条挠度 ref：声明 vs 独立重算最大相对偏差 **0**；
  转角 ref **0**。→ **ref 数字不是代码自身输出**，ref 独立性成立。

### 3.2 复现（干净副本重跑）

| 项 | 结果 |
|---|---|
| pytest | 21 passed（日志 `/tmp/bend_pytest.log`） |
| run.py 墙钟 | 74.2 s（m15427 自报 74.3 s；日志 `/tmp/bend_run.log`） |
| 17 gates | passed = True，failed = [] |
| result.json 逐标量 diff（排除 wall_seconds/timestamp） | **0 处差异**（共 1369 个标量键） |

### 3.3 非恒等证据（真实精度信息）

- distorted patch（精确二次场落在双线性 w 空间**之外**）：
  w = 1.54%，M = 0.751%，
  E = 0.165%（容差 2%）。
- 载荷驱动板条（答案未给定）：对称胞 tip w 误差 2.18e-09，
  自由边 1.76%。
- 锁定负控（必败门）：patch 能量误差 2.11% ≥ 2%，
  板条挠度误差 0.979 ≥ 0.5 —— 负控真的在失败侧，gate 有鉴别力。

### 3.4 h 收敛（ladder 独立重算）

| n | h | 剪切处理 | w 误差 | M 误差 | 能量误差 |
|---|---|---|---|---|---|
| 32 | 0.0312 | mitc | 0.0179% | 0.0185% | 0.00328% |
| 16 | 0.0625 | mitc | 0.0613% | 0.0527% | 0.0121% |
| 8 | 0.1250 | mitc | 0.211% | 0.355% | 0.0627% |
| 4 | 0.2500 | mitc | 0.912% | 0.25% | 0.165% |
| 32 | 0.0312 | sri | 0.0503% | 0.0163% | 0.00117% |
| 16 | 0.0625 | sri | 0.173% | 0.0453% | 0.00504% |
| 8 | 0.1250 | sri | 0.887% | 0.136% | 0.0285% |
| 4 | 0.2500 | sri | 1.54% | 0.161% | 0.113% |

| 量 | 重算阶 | 单调下降 |
|---|---|---|
| displacement_rel_error/sri | 1.717 | yes |
| moment_rel_error/sri | 1.151 | yes |
| energy_rel_error/sri | 2.228 | yes |
| displacement_rel_error/mitc | 1.879 | yes |
| moment_rel_error/mitc | 1.403 | no |
| energy_rel_error/mitc | 1.933 | yes |

声明 order（`result.json.orders`）= {"w_sri": 1.7173691653735683, "w_mitc": 1.8788532865102572, "M_sri": 1.1511895023986738, "M_mitc": 1.4030702028463193}；仅 w 架了下限门（['order_w_min']，下限 1.5）。

## 4. 结果图

`fig_bending_patch_audit.png`（三面板，全部由工件注入）：
(1) distorted patch 的 h 收敛（唯一非恒等的 patch 证据）＋ slope-2 参考线；
(2) 17 个 gate 的余量（threshold/value），红色 = 恒等门（F-CRIT），橙色 = 余量 < 1.5×；
(3) Poisson 交叉核验：声明值 0.0 vs 由**计算挠度**重算的比值（F-CRIT 的可视化）。

## 5. 注意点（发现，按严重度）

**F-CRIT｜`poisson_crosscheck` 是恒等门，报告把它写成 "measured"（证据虚高，非数值造假）**
`run.py:542` 取 `w_exact_kirchhoff(free) / w_exact_kirchhoff(symmetry)` —— 两个操作数**都是闭式解**
（M L²/2D 与 M L²/2D_b），比值代数上恒等于 1/(1-ν²)，所以 `rel_error = 0.0`
是构造出来的，该门永远看不到元素的 Poisson 耦合缺陷。用**计算挠度** `w_tip` 重算的真实比值见下表
（门槛仍全部 < 2%，但真实误差是 1.76% 量级，不是 0）：

| t | 剪切 | n_x | w_free (计算) | w_symmetry (计算) | 比值 | 相对误差 | 2% 门 |
|---|---|---|---|---|---|---|---|
| t=0.01 | sri | 4 | 28224.555776 | 25999.999999 | 1.085560 | 1.21% | pass |
| t=0.01 | sri | 8 | 28368.410148 | 26000.000000 | 1.091093 | 0.711% | pass |
| t=0.01 | sri | 16 | 28424.344939 | 26000.000000 | 1.093244 | 0.515% | pass |
| t=0.01 | sri | 32 | 28441.379201 | 26000.000057 | 1.093899 | 0.455% | pass |
| t=0.01 | mitc | 4 | 28069.244483 | 25999.999999 | 1.079586 | 1.76% | pass |
| t=0.01 | mitc | 8 | 28165.685699 | 26000.000000 | 1.083296 | 1.42% | pass |
| t=0.01 | mitc | 16 | 28203.097179 | 25999.999996 | 1.084735 | 1.29% | pass |
| t=0.01 | mitc | 32 | 28214.113776 | 26000.000011 | 1.085158 | 1.25% | pass |

对照：同一报告 §4.3 写 "Measured ratio ... (error 0.0%) — this is the physical cross-check"，
"measured" 一词与实现不符。**建议修法**：把两个操作数换成 `w_tip`（一行改动，物理结论不变，
因为重算后仍全部通过）。这是**可修的证据缺陷，不改变元素数值结论**。

**F1（平台侧，high）｜锚点无机器裁判行**
`tensor_benchmark_cases` 共 19 行，`anchor_slug='CASE-BENDING-PATCH'` → **0 行**；
`%BEND%`/`%PATCH%` 亦无命中。因此锚点判据「ref: 文献基准，容差 <2%」在本仓只以
`result.json.criteria.case_pool_tolerance = 0.02` 的**自声明**形式存在（由被评的 `run.py` 自己写）。
"基准判据是唯一裁判"的机制对本案无法自动点火，本裁定依赖审计侧独立复算。建议补登记行。

**F2（major）｜affine/skew 的 8e-16 是恒等，不是精度**
精确场落在 Q4 双线性 w 的插值空间内、边界自由度直接给精确节点值，所以 w 精确 = 插值恒等式
（各网格族最大 w 误差：{"affine": "6.1e-16", "skew": "8.03e-16", "distorted": "0.0154"}）。
它证明**一致性**（必要），不证明粗网格精度。m15427 自己的 `gates_verified.json.limitations`
已如实写明这一点（诚实），但任何下游把它读成"精度 1e-16"都是误读，本裁定的 benchmark 记分不用它。

**F3（major）｜自由边 2% 门余量薄且收敛慢**
`strip_free_w` = 1.76% vs 门 2%（余量
1.14×），且自由边误差拟合阶
sri = 0.471、
mitc = 0.161（n=4→32 只降
1.41×）。
这是 Mindlin 自由边边界层 + 梁刚度闭式的建模差，不是元素缺陷，但"自由边 <2%"是**网格依赖**的通过，
更稳的写法是取内部弯矩/曲率或加大长宽比。

**F4（minor）｜两份工件的 criteria 块不一致**
`result.json.criteria` = {"case_pool_tolerance": 0.02, "exact_tolerance": 1e-12}；
`gates_verified.json.criteria` = {"case_pool_tolerance": 0.02, "exact_tolerance_affine": 1e-12, "observed_order_min": 1.5, "locking_control_min": 0.5} —— 键集相同？False。
只读 `result.json` 的机器消费者看不到 `observed_order_min` / `locking_control_min`。

**F5（minor/info）｜集成方向是"harness 消费生产元素"，不是"求解器调用 harness"**
`bending_patch.py` 导入 `from .plate import _shape, q4_mindlin_stiffness`；仓内 `bending_patch`
只被 tests/benchmarks 引用；`q4_mindlin_stiffness` 有 68 处 src 使用。
故 L3 成立是靠"被测的就是生产元素"，如实记录方向，不夸大成"接入主求解器"。

**F6（major，交付缺口）｜第 2 轮"选型沉淀落 oss_solutions"未落库**
`oss_solutions WHERE feature_slug='CASE-BENDING-PATCH'` → **0 行**（做法：agent 无 INSERT 权限，
SQL 暂存在 `artifacts/oss_solutions/CASE-BENDING-PATCH_oss_solutions_insert.sql`，4 条文献/方法条目）。
审计侧不代执行（避免审计者为自己评的对象造证据），如实记缺口，建议平台侧执行。

**F7（info，诚实项）｜未跑项自报诚实**
m15427 result 明确写：集群 oss-cache 全库 grep（cj1115 仍排队）按 unknown/not_run 登记、
未扩到 n=64；本审计复核认为这两条确实未完成，**没有**被写成通过。

## 6. 误差判定（是否 <3%）

| 关键量 | ref | got | 误差 | <3%? |
|---|---|---|---|---|
| 弯矩 M = D_b κ（独立复算） | 22.692308 | 22.692308 | 0 | 是（机器精度） |
| distorted patch w | 闭式常曲率场 | — | 1.54% | 是（1/2 容差内） |
| 板条对称胞 tip w（载荷驱动） | M L^2/(2D), D = E t^3/(12(1-… | — | 2.18e-09 | 是 |
| 板条自由边 tip w（载荷驱动） | 梁刚度闭式 | — | 1.76% | 是（余量仅 1.14×） |
| Poisson 交叉（**计算值**重算） | 1/(1-ν²)=1.098901 | 最差 1.093899 | 1.76% | 是 |
| Poisson 交叉（**声明门值**） | 同上 | 同上（闭式除闭式） | 0.0（恒等，**不计为精度证据**） | 不适用 |

**结论**：全部**非恒等**关键量误差均 <3%（最大 1.76%），
但其中 1.76% 一项余量薄、阶低，且 17 门里有 1 门恒等 —— 详见 F-CRIT/F3。

## 7. 判级（L0–L5）

**L4**。级别=功能实质，不因报告篇幅改变：

- **L1 数学/单元验证** ✓ 21 个测试实跑全绿（干净副本，审计侧复现），断言不变量而非只跑通道。
- **L2 独立功能** ✓ 真代码 `src/tensorfem/bending_patch.py`（装配 + Dirichlet 求解 + 曲率/能量/弯矩后处理，
  无 placeholder），工件 10 个全部存在、sha256 与 m15427 工作树逐字节一致。
- **L3 系统集成** ✓ 被测刚度来自生产元素 `tensorfem.plate.q4_mindlin_stiffness`（68 处 src 使用），
  即被测面就是生产面；方向为 harness→生产（F5）。
- **L4 基准验证** ✓ ref vs got 逐项在案（§3）、ref 可**不 import tensorfem** 独立复现（0 /
  0）、干净副本重跑 **逐标量 0 差异**、`repro_cmd` 可执行（本审计实跑）、
  含收敛阶与负控。**扣分不掩**：锚点无 registry 行（F1）、1/17 恒等门（F-CRIT）、自由边余量薄（F3）。
- **L5 科研工程成果** ✗ 不记。无外部数值锚点/实验值（文献仅作方法出处，`gates_verified.json.limitations`
  亦声明教材离线不可检索、目标值全部由闭式解在代码内求出）；且 patch test 属一致性验证，
  本身不构成工程科学成果。

## 8. 处置（disposition）

**approve with notes → L4，confirmed 方向**（L4 的机器判据六键齐备且审计侧独立复现），
同时把下列两项作为**返修项**入账（不影响级别，但须在锚点后继单里消账）：

1. F-CRIT：`run.py:542` 的 `poisson_crosscheck` 操作数改为 `w_tip`（重算后仍全部 <2%，纯证据修正）。
2. F6：`oss_solutions` 4 条选型条目落库（平台侧执行暂存 SQL）。
3. F1：补 `tensor_benchmark_cases` 锚点行（`metric=tolerance` 0.02），否则本锚点永远无法机器自动裁定。

**未跑的检查（unknown / not_run，不记通过）**：GitHub CI（x86）未跑；集群 loong64 跨架构复跑未跑（本审计全部在
loongarch64 本地）；n=64 网格未跑；教材原文离线不可检索（沿用 m15427 的声明，未独立核对页码/条目）；
`poisson_crosscheck` 修复后未回跑（属后继单）。

**审计独立性限制**：本审计在同一物理机、同一 torch 2.7.1 上复跑，数值逐位一致**不能**排除
"环境特异的确定性 bug"；作者与审计者同为平台租户，非外部第三方；F-CRIT 系读源码定位，
已用计算挠度重算量化，但仍属**代码阅读 + 复算**证据，非受控实验。

---

### 附：三件套清单（sha256，审计时点）

| 文件 | bytes | sha256 | 与 m15427 工作树一致 |
|---|---|---|---|
| `src/tensorfem/bending_patch.py` | 26538 | `e82ac7f257271a10…` | yes |
| `benchmarks/verified/fem_bending_patch/run.py` | 34480 | `5c0ed3ba6a58373d…` | yes |
| `benchmarks/verified/fem_bending_patch/result.json` | 49750 | `1dd55e1d74fc2f7b…` | yes |
| `benchmarks/verified/fem_bending_patch/gates_verified.json` | 6102 | `85700659341f1f4d…` | yes |
| `tests/test_bending_patch.py` | 6019 | `c02228b0c1ecbeb9…` | yes |
| `docs/bending-patch.md` | 13183 | `1ff1bdd742cb7ba6…` | yes |
| `docs/assets/bending_patch/patch_meshes.png` | 97620 | `398fef57135b283b…` | yes |
| `docs/assets/bending_patch/error_vs_h.png` | 75098 | `255e5d66870ba7be…` | yes |
| `docs/assets/bending_patch/error_budget.png` | 44371 | `658188020b1e5e58…` | yes |
| `docs/assets/bending_patch/strip_bending.png` | 66665 | `cc5f170cda4b3c34…` | yes |

报告 14500：六段齐全 = True，markdown 13165 字符，内嵌图 4 张
（patch_meshes.png, error_vs_h.png, error_budget.png, strip_bending.png）。

工件：`verify_output.json`、`poisson_crosscheck_rederived.json`、`fig_bending_patch_audit.png`、
`audit_bending_patch.py`、`poisson_crosscheck_rederived.py`、`make_figure.py`、`make_audit_md.py`。
