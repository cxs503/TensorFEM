# HONEST-FAIL · GAP-PP-3-ICE-STRUCT-PLASTIC-SET — 3 网格极限载荷证据（验收未达成）

锚点：`GAP-PP-3-ICE-STRUCT-PLASTIC-SET`（板格低温钢弹塑性 + 永久挠度，大斑载荷下的真实失效模式）
本文件记录 **3 网格 FE 验证（cj4436，mission m22855，rc=0）** 的实测结果：**验收判据未达成**。
这是诚实 FAIL，不覆盖、不粉饰。仓：TensorFEM，分支 `agent/m23116`。

## 1. 案例（程序化注入，非手抄）

| 项 | 值 |
|---|---|
| 板 | a=2.0 m, b=1.0 m, t=0.020 m（无加筋板格，aspect=2.0, t/b=0.02） |
| 材料 | DH36，T = −20 °C（设计温度；DBTT = −65 °C，塑性可用） |
| 流变应力 | σ_y(−20 °C) = 374.2 MPa（`low_temperature_flow_stress`） |
| 硬化 | 等向硬化 H = 1000 MPa |
| 载荷 | 中心斑压 0.5 m × 0.5 m（斑边与单元边重合，`patch_plate_mesh` 强制校验） |
| 边界 | 简支 w=0 + `inplane_restrained=True` |
| 解析屈服压力 | p_y = 0.7125 MPa（`analytic_first_yield_pressure`，Navier 双正弦） |
| 上限解（屈服线/斑） | p_UB = 2.0528 MPa（`yield_line_patch_upper_bound`） |
| 单元 | 分层 Mindlin Shell4（`layered_shell4_plasticity`），layers = 9 |
| 网格族 | 8×4 / 16×8 / 24×12（对齐族，nx%8==0, ny%4==0，ratio 2） |

## 2. 实测结果（3 网格，同一几何/材料/斑）

| 量 | 8×4 | 16×8 | 24×12 |
|---|---|---|---|
| 同网格弹性偏置 `elastic_bias` | −15.57 % | −3.23 % | −1.38 % |
| FE 首次屈服误差 `first_yield_error` | **+29.70 %** | **+20.66 %** | **+20.44 %** |
| 路径实测极限（切线交点）`limit_load_measured` | 1.4248 MPa | 0.5094 MPa | **0.0724 MPa** |
| 极限 vs 屈服线上界 p_UB=2.0528 MPa | −30.6 % | −75.2 % | **−96.5 %** |
| 永久挠度 `permanent_deflection` | 25.9 mm | 14.6 mm | 23.3 mm |
| 永久集有效 `permanent_set_valid` | **True** | **True** | **False** |

## 3. 验收判据逐项判定

| 判据 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 弹性→塑性过渡点 vs 解析屈服载荷 | <5 % | 原始误差 ~20 %，**不随网格收敛**（8×4→16×8 降，16×8→24×12 平台） | **未达成** |
| 塑性极限载荷 vs Ilyushin/上限解 | <10 % | 极限载荷随细化**塌缩**（1.42→0.51→0.072 MPa），最细网格 −96.5 % | **未达成** |
| 3 档网格收敛 | 判据须在网格极限达成 | 过渡误差平台、极限载荷发散 | **未达成** |
| 永久挠度后处理 | 报告量 | 最细网格 `permanent_set_valid=False`（卸载未收敛） | **未达成** |

## 4. 关键缺陷（可复现）

### D1（阻塞）极限载荷提取「切线交点」随网格细化塌缩
`limit_load_from_path` 的 `intersection_of_tangents` 在 8×4 给 1.42 MPa、16×8 给 0.51 MPa、
24×12 给 0.072 MPa —— 最细网格比粗网格低 20 倍。这不是收敛，是**发散**。切线交点对
路径末段曲率极敏感，细网格上路径在截面门附近被截断，切线斜率趋平 → 交点被推到极低载荷。
`limit_spread=None`（未给出可接受性门），`collapse_is_limit_point=False`（加载终点非极限点）。

### D2（阻塞）过渡误差不收敛到 <5 %
原始 `first_yield_error` 在 16×8 后平台于 ~20 %。`raw_error_convergence` 的
`observed_order=0.026`（近零）→ Richardson 外推 `extrapolated_error=-2.6e-15` 是**假收敛**：
误差序列平坦时外推极限被推到 0，`met_in_mesh_limit=true` 是假通过（与 docs 发现 N1/N7 同源）。

### D3（阻塞）最细网格永久集无效
24×12 卸载 3 步后 `permanent_set_valid=False`，永久挠度不可信。

## 5. 处置与建议

- **本单不宣称验收通过**。锚点 `GAP-PP-3-ICE-STRUCT-PLASTIC-SET` 维持 `open`。
- 建议新条目（slug+量化验收+所属域）：
  `GAP-PP-3-ICE-STRUCT-PLASTIC-LIMIT-EXTRACTOR`（TensorFEM）——
  修复 `limit_load_from_path` 的切线交点极限提取：要求极限载荷在 8×4/16×8/24×12 上
  网格稳定（最细两档漂移 <10 %），且 `permanent_set_valid=True` 全网格成立；
  验收：3 网格极限载荷漂移 <10 % + 过渡误差 Richardson 极限 <5 % + pytest≥10 绿。
- 证据：`outputs/gap_pp3_3grid_results.json`（cj4436 实测，程序化注入）。

## 2026-11-16 · D1 修复（m23449）

**D1（limit-load 提取器塌缩，604% 漂移）已修复**：`limit_load_from_path` 现在在
`k1/k0 < 3.0`（路径被截面门截断、未形成真实塌缩支）时**扣留** `intersection_of_tangents`
（其分母 `k1-k0→0` 数值不稳定），改用**全路径定义**的 `stiffness_criterion_0.3` 为主。

修复后 3 网格极限载荷（Pa）：
| 网格 | 旧 intersection | 新 limit | vs Ilyushin sine UB |
|---|---|---|---|
| 8×4  | 1.4248e6 | 1.4248e6 | −16.5% |
| 16×8 | 0.5094e6 | 1.4273e6 | −16.3% |
| 24×12| 0.0724e6 | 1.4086e6 | −17.4% |

**最细两网格漂移 = 1.31%（<10% 判据，原 604%）** —— `test_limit_estimate_stable_between_finest_meshes` 转绿。

**仍开放**：D2（首屈服误差平 ~20%，Richardson 未收敛 <5%）、D3（最细网格永久挠度无效）。
