# 验证欠账审计：两个壳单元锚点的收敛判据缺陷与渐近精度重判

**锚点**：`CASE-FEM-PINCHED-CYLINDER`（主，feature_ledger `open`）、`CASE-FEM-SCORDELIS-LO-ROOF`（同族同缺陷，`open`）
**EDIT_WHITELIST**：`src/`、`tests/`、`benchmarks/verified/`、`docs/`
**分类**：defect_fix（验证面）+ 一条被推翻的既有结论（pinched cylinder 的 ≤3% 精度声明）

---

## 1. 案例介绍（物理背景与参数）

两个经典壳单元基准，均已在仓内有锚点目录与 `result.json`，此前被当作"已验证"：

| 基准 | 几何 / 材料 | 载荷与边界 | 参考量 | 出版参考值 |
|---|---|---|---|---|
| MacNeal–Harder 夹持圆柱（pinched cylinder） | R=300 in, L=600 in, t=3 in；E=3×10⁶ psi, ν=0.3 | 中跨一对对径单位力；端部刚性隔膜，对称八分之一模型 | 加载点径向位移 | −1.8248×10⁻⁵ in（MacNeal & Harder 1985） |
| Scordelis-Lo 圆柱屋盖 | L=50, R=25, 半角 40°, t=0.25；E=4.32×10⁸, ν=0 | 竖向均布 q=90；曲端刚隔膜 | 自由边中点竖向位移 | −0.3024（Scordelis & Lo 1960） |

单元：`Cylindrical Shell4`（Reissner–Mindlin，精确柱面度规，2×2 膜+弯、1 点选择性减缩剪切）；Scordelis-Lo 另比对 `Shell4` 平面片（flat facet）公式。

## 2. 怎么做的

1. **复现**：两锚点驱动脚本各跑一遍，逐位复现仓库内 `result.json`（确定性，差异 0）。
2. **独立重算**：不采信脚本自报，改用仓内既有权威模块 `tensorfem.convergence_verification`（Richardson 观测阶 / 外推极限 / Roache GCI）重算整条阶梯。
3. **新增判据层** `src/tensorfem/refinement_gate.py`：把上述权威算术封装为 **PASS / FAIL / WITHHELD** 三态门（WITHHELD = 证据不足，绝不等于通过），并把"精度裁决"与"收敛研究裁决"分离（`accuracy_verdict` / `study_verdict`）。
4. **修两条驱动**：删掉各自手写的观测阶估计器，改为调用权威实现；证据文件由脚本重新生成（零手抄）。
5. **加回归测试**（`tests/test_refinement_gate.py`，10 项）与只读审计工具（`benchmarks/verified/asymptotic_accuracy_audit.py`，生成图 `docs/assets/shell_anchor_asymptotic_accuracy.png`）。

## 3. 发现的缺陷（每条附证据）

### D1 观测阶公式错误（Scordelis-Lo 驱动，已修）
原实现 `order = (e_coarse/e_fine − 1)/(r − 1)`——不是对数，不是阶。产出物理不可能的 **p = 14.11 与 81.12**（线性壳单元），并被写进 `result.json`。改为权威 `richardson_observed_order` 后为 **p = 3.801、2.276**（前者为前渐近段的超收敛读数，见 D3）。

### D2 观测阶估计器忽略第三个参数 → 假 PASS（pinched cylinder 驱动，已修）
原 `observed_order(d1,d2,d3)` 只用到达 `d3` 之前的两个误差、且把 Aitken/外推式套在"误差相对参考值"上。对阶梯 2.86%→3.35%→0.17%→1.82%，它给出 **p = 4.325 ≥ 1.0 → G3 PASS**，而同一批数据用正确公式为 **p = 1.0928**。虽然 1.09 仍过 1.0 的门，但 4.33 是伪值，且掩盖了数据非单调这一事实。

### D3 只按最细网格误差判精度 → 精度被系统性高估（本次核心）
| 锚点 | 最细网格误差（原声称） | 外推极限值 | **认证误差（Richardson 极限 vs 参考）** | 原结论 → 现结论 |
|---|---|---|---|---|
| pinched cylinder（32×32） | 1.8163% | −1.884506×10⁻⁵ | **3.2719%（>3%）** | G1 PASS / overall FAIL → **accuracy FAIL / overall FAIL** |
| Scordelis-Lo 柱面单元（32×32） | 0.4147% | −0.303962 | **0.5167%（<3%）** | "PASS"（且文档自相矛盾：G4 FAIL 却总判 PASS）→ **accuracy PASS / study WITHHELD / overall WITHHELD** |
| Scordelis-Lo 平面片（32×32） | 49.4018% | −0.463239 | **53.1876%** | FAIL（结论不变，但数字由最细网格改为渐近值） |

pinched cylinder 的阶梯 4×→8× 位移**背离**参考（−1.7727e−5 → −1.7637e−5），此后才单调下降；最细网格 1.82% 是"跨参考值来回摆"的抵消假象。**外推极限 3.27% 才是该离散的真实精度**，故 ≤3% 的精度声明不成立（仍远优于平台 15% 红线，但"PASS at 1.82%"属于过度声明）。另：最细网格真误差 1.816% 落在 GCI 带 1.787% **之外**，说明阶梯尚未进入渐近区。

### D4 证据链的代码出处未钉扎（已修）
`benchmarks/verified/fem_pinched_cylinder/run_verification.py` 原写 `sys.path.insert(0, parents[2])`——即把 `benchmarks/` 目录插入路径，**并未指向本仓 `src/`**。实际 `import tensorfem` 落到环境里另一份 editable 安装（审计时实测解析到 `/root/tp-agent/work/m14687/src/tensorfem`），即该锚点一直在验证**另一棵工作树**的代码。本次改为按所在检出钉扎 `parents[3]/src` 并加 provenance 断言（解析到检出外即 `SystemExit`）。这条缺陷被"新增模块后 import 失败"当场暴露。

## 4. 计算结果（程序化生成，零手抄）

`gates_verified.json` / `result.json` 均由驱动脚本重跑生成：

- pinched cylinder：`G1_accuracy_finest=PASS(0.018162)`、`G2_accuracy_asymptotic=FAIL(0.032719)`、`G3_monotone_qoi=FAIL`、`G4_observed_order=PASS(1.09279)`、`G5_asymptotic_range_consistency=WITHHELD`、`G6_coarsest_sanity=PASS(0.028573)` → `overall_verdict=FAIL`，`certified_accuracy=3.2719e-2`。
- Scordelis-Lo：柱面单元 `accuracy=PASS(0.5167%) study=WITHHELD(仅 asymptotic_range_consistency 未证) overall=WITHHELD`；平面片 `accuracy=FAIL(53.1876%) study=FAIL overall=FAIL`。

图：`docs/assets/shell_anchor_asymptotic_accuracy.png`（三条阶梯的逐级误差、各自认证误差水平线、3% 容差线）。

## 5. 注意点（坑与适用边界）

- **最细网格误差不是精度**：任何"跨参考值振荡"的阶梯都可能给出虚假小误差；锚点必须以 Richardson 极限（或 GCI 保守带内）裁决。
- **非单调阶梯不能认证任何阶**：`evaluate_convergence_study` 对非单调三连返回 blocked，新判据层把它映射为 WITHHELD —— 缺证据既不给 PASS 也不给 FAIL。
- **参考值有效位数是硬上限**：−0.3024 只有 4 位有效数字（相对不确定度 ~0.02%），Scordelis-Lo 16×16 的 0.023% 已在该噪声量级——那正是超收敛假象，不足以"认证优于 0.1%"。要认证亚 0.1% 需更高精度参考值 + 第 5 层网格（本次未做，见 §7）。
- **不重算=不吃算力**：本审计全部基于已存证据重算 + 两次锚点重跑（最大 32×32），在 load15≈116/128 的饱和机况下仍为轻量作业。

## 6. 误差判定（逐项对基准，显式回答是否 <3%）

| 关键量 | 基准 | 认证误差 | <3%？ |
|---|---|---|---|
| pinched cylinder 加载点位移（渐近） | −1.8248×10⁻⁵ in（MacNeal & Harder 1985） | 3.27% | **否**（3.27% > 3%；最细网格读数 1.82% 不构成证据） |
| Scordelis-Lo 自由边中点位移（柱面单元，渐近） | −0.3024（Scordelis & Lo 1960） | 0.52% | **是** |
| Scordelis-Lo 同一量（平面片单元，渐近） | −0.3024 | 53.19% | **否**（平面片公式对曲壳不适用，结论与旧版一致） |

注：平台 15% 红线三者中仅平面片越线；pinched cylinder 3.27% 仍优于红线，但与锚点自我声明的 ≤3% 判据冲突，故按 `FAIL` 记账（honest FAIL，不覆盖）。

## 7. 未完成 / 后续（诚实申报）

- pinched cylinder 的 3.27% 究竟是单元固有偏置还是对称八分之一模型的探针口径问题，本次未定性——需加 64×64 层或改用整体模型对照（算力受限，本机 load15 已 >100，未起重算）。
- Scordelis-Lo 柱面单元的 `withheld` 只能靠"第 5 层网格"或"更高精度参考值"解除，本单未做。
- `asymptotic_accuracy_audit.py` 打印的阶区间标签未填 level label（`->: p=...`），数值正确、显示待改善。
- D4 的污染面可能不止本锚点：建议全仓扫 `sys.path.insert(0, ...parents[2])` 类写法（本单只修了命中的 1 处 + 1 处错误深度）。