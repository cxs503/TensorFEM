# CASE GAP-PP-3 · 首次屈服判据的可判定性（reference comparability）

锚点：`GAP-PP-3-ICE-STRUCT-PLASTIC-SET`（板格低温钢弹塑性 + 永久挠度）。
本单不宣称锚点验收通过：锚点维持 `open`。本文件只交付一件实质增量——
**把「弹→塑过渡点 <5%」这条判据从「测出来误差 ~20% 且不收敛」改造成「先判定它是否可判定，
再判定它」**，并把 `met_in_mesh_limit=true` 这个假通过关掉。

## 1 案例

| 项 | 值 |
|---|---|
| 板 | a=2.0 m, b=1.0 m, t=0.020 m（aspect=2.00, t/b=0.020 已校验） |
| 钢级/温度 | DH36 / −20 °C，σ_y = 374.2 MPa（`low_temperature_flow_stress`） |
| 硬化 | 等向硬化 H = 1000 MPa |
| 载荷 | 中心 0.5 m × 0.5 m 斑压（斑边与单元边重合） |
| 单元 | 分层 Mindlin Shell4，layers = 9 |
| 参考解 | 线性 Navier 双正弦（小挠度、简支），p_y = 0.712514 MPa，σ_eq/p = 525.183 |
| 网格 | 8×4 / 16×8 / 24×12（对齐族） |

## 2 方法（怎么做的）

1. 读 `plate_patch_plastic_limit.plate_capacity_summary` 的调用链，发现 3 网格矩阵
   （`scripts/gap_pp3_3grid_study.py`）**以 `geometric=True, inplane_restrained=True` 积分路径**，
   而验收比较对象 `analytic_first_yield_pressure` 是**线性小挠度** Navier 解——两者是不同力学模型。
2. 新增 `first_yield_reference(geometric, inplane_restrained)`：给出 `comparable` 与理由；
   `plate_capacity_summary` 现在发布 `first_yield_error_comparable`，消费者无法再把模型差当成精度差。
3. 在同一网格、同一载荷梯上**实测两种构型**（8×4）：
   `linear-free`(geometric=False, free) 与 `nonlinear-fixed`(geometric=True, restrained)。
   驱动脚本：`scripts/diagnose_gap_pp3_transition_reference.py`（分片可递交集群）。
4. 关掉假通过：`finalize()` 增加渐近性门（观测阶须落在 0.2–4.0 且误差单调下降），
   不可判定时 `extrapolated_error` 不再发布（改存 `extrapolated_error_inadmissible`）；
   `scripts/audit_gap_pp3_3grid_bundle.py` 增加 `C4-asymptoticity` / `C5-inadmissible-limit-withheld`，
   `K2` 改为「不可判定的外推永不支持通过判定」。

## 3 计算结果（程序化注入，零手抄）

| 构型（8×4） | 参考可比 | FE 首次屈服 | 相对线性 Navier 误差 | 弹性偏置 |
|---|---|---|---|---|
| `linear-free`（小挠度、面内自由） | **是** | 0.9862 MPa | **+38.41 %** | −15.57 % |
| `nonlinear-fixed`（几何非线性、面内约束） | 否 | 0.9241 MPa | +29.70 % | −15.57 % |

3 网格（`nonlinear-fixed`，已入库 bundle）首次屈服误差：+29.70 % / +20.66 % / +20.44 %，
观测阶 0.0265（平台），`order_admissible=False`。

## 4 结果图

![first-yield reference](../outputs/figures/gap_pp3_first_yield_reference.png)

左：过渡误差随网格的平台 + 同网格 `linear-free` 实测点落点；右：两构型 8×4 对照。
生成脚本 `scripts/gen_gap_pp3_first_yield_reference_figure.py`（数值全部来自
`docs/assets/gap_pp3_transition_reference.json` 与 `docs/assets/gap_pp3_3grid_results.json`）。

## 5 注意点（踩过的坑 / 局限）

1. **假设被证伪**：开工假设是「20 % 平台 = 几何非线性/膜作用」。
   实测 8×4 `linear-free` 误差 **+38.4 %**，比 `nonlinear-fixed`（+29.7 %）**更大**——
   膜作用只把过渡点往参考解方向挪了约 9 个百分点，**其余 ~30 个点没有被解释**。
   因此本单结论只到「判据不可判定 → 扣留」，不得写成「平台已解释」。
2. 同网格还有第二个可复现缺陷：8×4 路径给出的 `first_yield_factor` = 986 kPa，
   而用自身弹性斜率反算应为 σ_y/(σ_eq/p) = 374.2/443.43 = 844 kPa，
   即路径检测器比自身的应力场多读 ~17 %（`surface_extrapolated_mises` 在屈服后过报，
   与文档警告一致）。**未修**，列入建议条目。
3. 集群分片作业 `cj4673`（16×8 / 24×12 两构型）递交后仍在排队（排队位 121/在飞 126），
   本文件只含 8×4 实测；更细网格的 `linear-free` 数值**未取得**，属未完成部分。
4. 上游仓门：`tests/test_gap_pp3_3grid_bundle_audit.py::test_committed_bundle_passes_the_consistency_audit`
   在本单开场时是**红的**（M6 把「极限载荷落在刚塑性括号外」这一物理事实当作 bundle 自洽性失败）。
   本单把 M6 降为 minor（该项仍被抓到并被 `_failed_checks` 列出），并重新生成 bundle 的派生块。

## 6 误差判定

| 判据 | 要求 | 实测 | 是否 <5 % |
|---|---|---|---|
| 弹→塑过渡点 vs 线性解析屈服载荷（**可比构型** `linear-free`，8×4） | <5 % | **+38.41 %** | **否**（如实 FAIL） |
| 同判据（`nonlinear-fixed`，3 网格） | <5 % | +29.70 / +20.66 / +20.44 %，平台 | **否**，且判定**不可判定/扣留** |
| 塑性极限载荷 vs Ilyushin/上限解 | <10 % | −30.6 / −75.2 / −96.5 %（发散） | **否**（HONEST-FAIL 延续） |
| 3 档网格收敛 | 判据在网格极限成立 | 观测阶 0.0265，非渐近 | **否**，假通过已关闭 |

结论：锚点 `GAP-PP-3-ICE-STRUCT-PLASTIC-SET` 维持 **open**，本单贡献是把两条判据的
**可判定性门**补齐（可复现地阻止假通过），并诚实地把 20 % 平台记录为「部分为模型差、
其余未解释」。

## 7 产物指针

| 类型 | 路径 |
|---|---|
| 判据门（代码） | `src/tensorfem/plate_patch_plastic_limit.py::first_yield_reference` |
| 实测驱动（可分片） | `scripts/diagnose_gap_pp3_transition_reference.py` |
| 渐近性门 | `scripts/gap_pp3_3grid_study.py::finalize`、`scripts/audit_gap_pp3_3grid_bundle.py` |
| 实测证据 | `docs/assets/gap_pp3_transition_reference.json` |
| 已入库 3 网格 bundle（派生块重生成） | `docs/assets/gap_pp3_3grid_results.json` |
| 测试 | `tests/test_gap_pp3_first_yield_reference.py`（8 项） |
| 图 | `outputs/figures/gap_pp3_first_yield_reference.png` |

## 8 给下次自找活的建议

- `GAP-PP-3-FIRST-YIELD-DETECTOR-CONSISTENCY`（TensorFEM，ship structural / ice-structure 域）：
  修 8×4 上 `first_yield_factor`(986 kPa) 与 σ_y/(σ_eq/p)(844 kPa) 内部不一致 ~17 %：
  要求 `first_yield_factor` 必须可由路径自身的 `max_surface_mises` 序列线性插值复现（<1 %），
  且在 8×4/16×8 上复现；顺带把 `surface_extrapolated_mises` 屈服后过报的警告转成门。
  pytest≥10 绿。
- `cj4673` 产出的 16×8/24×12 `linear-free` 数据到位后，可直接补本文件 §3 表。