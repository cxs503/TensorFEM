# 壳锚点证书重基线（m19702，#823 重开第 1 轮）

## ① 案例介绍

两个壳单元验证锚点（`benchmarks/verified/`）在 main 上失修：元素在 #907(m18206) 被改成
MITC4（膜 2×2 全积分 + Dvorkin–Bathe 假定剪切，清除 5 个伪机构），但两个锚点的
`result.json` / `gates_verified.json` 没有跟着重出，且分级链本身在"前渐近阶梯"上抛异常。

| 案例 | 几何/参数 | 参考值 | 判据 |
|---|---|---|---|
| MacNeal–Harder 夹持圆柱 | R=300 in, L=600 in, t=3 in, E=3e6 psi, ν=0.3, 对称八分域，跨中一对反向单位力 | u_ref = −1.8248e−5 in（MacNeal & Harder 1985） | 误差 ≤3%，阶 ≥1，最粗网格 ≤50% |
| Scordelis–Lo 圆柱屋面 | R=25 ft, L=50 ft, 半开角 40°, t=0.25 ft, 自重 90 lb/ft²，两端隔板 | 自由边中点 w = −0.3024 ft（Scordelis & Lo 1960） | 同上（柱壳 + 平面壳两条阶梯） |

## ② 怎么做的

1. **发现真缺陷（核心模块，已声明）**：`src/tensorfem/convergence_verification.py` 在
   观测阶非正（增量不随加密收缩）时，`grid_convergence_index` 的 `r**p − 1 ≤ 0` 抛
   `ValueError`，而 `evaluate_convergence_study` 未捕获 → 分级链整体崩溃，锚点、证书审计、
   测试全部连带失败。修复：非正阶判为"不可外推"，记 `status=blocked` + `blocked_reason`，
   `gci/richardson_value` 置 None，新增 `unevaluable_checks` 记录不可评估的检查（粗对
   的收敛带合同不再与退化阶比较），`_reference_block` 增加 `extrapolation_available`，
   杜绝把最细网格读数当作"认证精度"。
2. **重出两个锚点的工件**：`run_verification.py` 重跑，`result.json` / `gates_verified.json`
   改为 `tensorfem.verified-gates.v1` 全字段（status/operator/threshold/actual，可机器复核）；
   夹持圆柱发射器补 `operator` 字段（消除审计缺陷 E_PASS_WITHOUT_CRITERION）。
3. **消除手写结论**：Scordelis 脚本的 `conclusion` 原为硬编码"柱壳在 Richardson 极限复现
   参考值"，数据变了仍留着 → 改为由分级结果程序化生成（零手抄）。
4. **台账机器重生成**：`tools/verified_anchor_audit.py --write` → `anchor_certificates.json`、
   `anchor_certificate_debt.json`（收缩 −14 行：`fem_inverse_identification`、
   `fem_winkler_plate_k_sweep` 已修复不再欠账，`fem_thermal` 免债项漂移更正）、
   `docs/verified-anchor-certificate-table.md`。
5. **测试去冻结、改契约**：`tests/test_refinement_gate.py` 原先把"旧元素的数字"
   （1.82%/3.27%）写死；改为**可复现契约**——`gates_verified.json` 必须等于对
   `result.json` 阶梯的重新分级（逐 gate 的 status/operator/threshold/actual + 总体判决），
   并新增 `TestPreAsymptoticLadder` 两条回归（非正阶不得抛异常；粗对非正阶不得参与比较）。

## ③ 计算结果（程序化注入自 result.json，零手抄）

**夹持圆柱**（当前元素，误差 / 观测阶）：

| 网格 | u (in) | 误差 |
|---|---:|---:|
| 4x4 | -2.656166e-06 | 85.444% |
| 8x8 | -6.260528e-06 | 65.692% |
| 16x16 | -1.107445e-05 | 39.311% |
| 32x32 | -1.533341e-05 | 15.972% |

细网格误差 15.97%，Richardson 极限
163.14%（观测阶 p≈0.177 的外推，已明显偏离参考）。

夹持圆柱门（`gates_verified.json`，整体 FAIL）：

| gate | 状态 | 算子 | 阈值 | 实际 |
|---|---|---|---|---|
| `G1_accuracy_finest` | FAIL | <= | 0.03 | 0.159721 |
| `G2_accuracy_asymptotic` | FAIL | <= | 0.03 | 1.631444 |
| `G3_monotone_qoi` | PASS | <= | 0.0 | 0.0 |
| `G4_observed_order` | FAIL | >= | 1.0 | 0.176709 |
| `G5_asymptotic_range_consistency` | PASS | <= | 1.0 | 0.059943 |
| `G6_coarsest_sanity` | FAIL | <= | 0.5 | 0.854441 |

**Scordelis–Lo 柱壳**：4×4→32×32 误差
93.91%, 88.68%, 78.22%, 54.47%；
自由边位移 -0.137674 ft（参考 −0.3024 ft）。
柱壳阶梯整体 FAIL，认证精度
None（无 Richardson 极限 → WITHHELD）。

| gate | 状态 | 算子 | 阈值 | 实际 |
|---|---|---|---|---|
| `flat_facet.accuracy_finest` | FAIL | <= | 0.03 | 0.494018 |
| `flat_facet.accuracy_asymptotic` | FAIL | <= | 0.03 | 0.531876 |
| `flat_facet.monotone_qoi` | FAIL | <= | 0.0 | 1.0 |
| `flat_facet.observed_order` | PASS | >= | 1.0 | 1.166495 |
| `flat_facet.asymptotic_range_consistency` | WITHHELD | <= | 1.0 | 15.596373 |
| `flat_facet.coarsest_sanity` | FAIL | <= | 0.5 | 0.629753 |
| `cylindrical.accuracy_finest` | FAIL | <= | 0.03 | 0.54473 |
| `cylindrical.accuracy_asymptotic` | WITHHELD | <= | 0.03 | n/a |
| `cylindrical.monotone_qoi` | PASS | <= | 0.0 | 0.0 |
| `cylindrical.observed_order` | FAIL | >= | 1.0 | -1.18206 |
| `cylindrical.asymptotic_range_consistency` | WITHHELD | <= | None | n/a |
| `cylindrical.coarsest_sanity` | FAIL | <= | 0.5 | 0.939074 |

## ④ 结果图

两张误差阶梯图（场量/曲线）见附件：`pinched_ladder.png`、`scordelis_ladder.png`，均含
"当前元素 vs 修复前元素"对照与 3% 判据线。

## ⑤ 注意点

* **这不是"过 3% 门"的交付**：两个锚点都是 honest-FAIL。把它们记成通过才是错的。
* **元素精度回归是真实存在的**（本单用同 harness 重跑旧积分方案对照，见 ④ 灰线）：
  修复前 4×4→32×32 为 2.86/3.35/0.17/1.82%，修复后 85.44/65.69/39.31/15.97%；
  Scordelis 由 28.33/1.88/0.02/0.41% 退到 93.91/88.68/78.22/54.47%。
  说明旧元素的"高精度"依赖膜单点积分造成的 5 个伪机构（过度柔性），
  而全积分 + 假定剪切在 R/t=100 的 4 节点曲壳上仍严重过刚（近似锁定）。
  结论：精度问题在**单元公式**（需假定膜应变/高阶单元），不在锚点 harness；
  本单不触碰 `cylindrical_shell4.py`（热文件、与其它 PR 重叠，已在报告中声明）。
* 观测阶 p≈0.18 < 1，且粗对 p<0 ⇒ 阶梯处前渐近区，任何"最细网格误差"都不是证据。
* 复核命令：`PYTHONPATH=src python benchmarks/verified/fem_pinched_cylinder/run_verification.py`、
  `.../fem_shell_scordelis_lo/run_verification.py`、`tools/verified_anchor_audit.py --check`。

## ⑥ 误差判定：是否 <3%？

| 关键量 | 基准 | 本单 got | 误差 | <3%? |
|---|---|---:|---:|:--:|
| 夹持圆柱 u @32×32 | −1.8248e−5 in | -1.5333e-05 in | 15.97% | **否** |
| 夹持圆柱 Richardson 极限 | −1.8248e−5 in | 163.14% 偏离 | 163.14% | **否** |
| Scordelis 柱壳 w @32×32 | −0.3024 ft | -0.137674 ft | 54.47% | **否** |
| 证书自洽性（审计重算全部 gate） | 0 矛盾 | 0 矛盾（gates_recomputed=431） | 0 | 是（结构性判据） |

**显式回答：否。** 位移精度未达 3%（15.97% / 54.47%），锚点保持 honest-FAIL；
达标的结构性部分（证书可复算、无伪机构账、无手抄数字、无静默豁免）已交付。

