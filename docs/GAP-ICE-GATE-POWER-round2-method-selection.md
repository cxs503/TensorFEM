# GAP-ICE-GATE-POWER 第 2 轮：开源横向调研 + 方法谱系 + 选型沉淀

> 本文件由 `benchmarks/ice_gate_power/make_doc.py` 与证据 JSON 一同生成；
> 数字与 `docs/GAP-ICE-GATE-POWER-round2.md` 同源。

## 1. 开源横向调研（feature_slug ≈ '%POWER%'）

**执行**：集群 `~/oss-cache` 清点（`run_compute` 作业 `cj4035`，见 §5 状态）+ 本机缓存清点。

本机 `/root/oss-cache` 仅有 LBM 生态（`Maarten-vd-Sande/lbm`、`hschwane/pylbm`、`pylbm`）；
**没有** 2-D 附加质量/势流边界元/一致性判定规则的同类实现可复用。结论：

| 候选开源实现 | 是否命中 | 处置 |
|---|---|---|
| LBM 系（pylbm 等） | 命中缓存，但域为格子玻尔兹曼，**不提供** 2-D 附加质量参考 | 不迁移 |
| 本仓既有 `added_mass_bem2d`（单层位势 BEM，含矩形/圆/椭圆生成器） | **命中且可用** | **直接复用**作为参考计算核（不新写求解器） |
| 本仓既有 `hydro_added_mass`（偶极子、Rayleigh 活塞） | 命中 | 仅作概念参照（量纲/场景不同：有限障板活塞 vs 自由面截面），本单不重复调用 |

> 说明：`oss_inventory` 表的 `SELECT` 在沙箱角色下为 `permission denied`，故横向调研以
> 缓存目录清点 + 仓内实现清单替代；`cj4035` 的集群侧清点结果见本文件 §5（若在被收割前返回）。

## 2. 方法谱系（文献/教科书标准做法）

| 方法 | 出处 | 在本单的角色 | 取舍 |
|---|---|---|---|
| 单层位势边界元（常值强度面板）求 2-D 附加质量 | 本仓 `added_mass_bem2d`（v8 交叉验证，圆/椭圆/薄板极限自证） | **参考值计算核** | **采用**（可算、自证、可测观测阶） |
| 自由面镜像技巧（偶/奇延拓） | Lamb, *Hydrodynamics*；Newman, *Marine Hydrodynamics*（2-D 截面水动力系数） | 把半潜截面问题折为无界流加倍体 | **采用**（两条件各给一条路线，差即模型带） |
| 深浸圆柱 `ρπa²` | Lamb art. 73 | 路线自证精确解 | **采用**（两路线均钉住） |
| strip theory / Lewis form 半经验式 | Newman (1977) 类公式 | 上一轮 m22156 的做法 | **不采用**：与其自设 15% 容差相差 +92.5%（m22176 F2），且文献式号不可追溯 |
| 一致性判定：护带规则（guard band） | ILAC-G8:09/2019；JCGM 106:2012；ISO/IEC 17025:2017 §7.8.6 | **门限方法** | **采用**（把容差与测量不确定度显式挂钩，并给出检出率/误接受率） |
| 统计功效（power）分析 = 检出率 | 计量学/假设检验标准做法（JCGM 106 二元判定风险） | 判据 (b) 项：门限↔检出率 | **采用**（解析 + 蒙特卡洛双算） |
| Vugts (1968) 矩形截面附加质量实验/理论表 | Int. Shipbuilding Progress | 理想外部文献锚 | **未采用（缺口）**：本单无法取得表值，避免转抄风险 |

## 3. 选型结论与迁移方案

**选型**：`护带二元判定 + 可算参考（复用 added_mass_bem2d 的两条自由面镜像路线）
+ 检出率量化 + 真实数值负控 + 退化闸元检`。

**迁移方案（生态内可复用）**

1. 任何"声明值 vs 计算参考"的验收判据，把 `|dev| ≤ tol` 换成
   `|dev| ≤ tol − k·u` / `|dev| ≥ tol + k·u` 三分判定，`u` 用参考路线实测残差。
2. 判据交付必须附：参考路线自证（精确解）、观测阶或检出率、误接受率、
   至少一个**真实数值**负控（禁止 `ref×常数`）。
3. 元闸：`tol` 过宽（对已知失败无检出）或过窄（无法接受合规值）一律判 `informative=False`。
4. 本轮结论：`u = 0.029` 时 15% 容差只有 `>25.57%`
   偏差可 95% 检出；要把容差收紧必须先降 `u`（表 ④-3 给出 `u ↔ δ_95%` 曲线）。

**落库**：`oss_solutions`（repo='tensorfem', feature_slug='GAP-ICE-GATE-POWER',
tag='method-selection'）的幂等 INSERT 已生成于
`docs/oss/GAP-ICE-GATE-POWER-oss_solutions.sql`；沙箱角色对 `oss_solutions` 的
`INSERT/SELECT` 均被拒（`permission denied for table oss_solutions`），故由平台侧应用。

## 4. 与前轮失败的差异（负面知识核对）

| 前轮死法 | 本单对策 | 证据 |
|---|---|---|
| m22156：被判量量纲错（3-D kg vs 2-D kg/m）、几何错（T=8 vs 6） | 声明值由链路 `ShipModel` 程序读取并除以 `L`；几何取自链路（T=6, C_b=0.70） | `test_declared_value_is_read_from_the_chain`、`test_chain_geometry_is_the_audited_one`、`test_judged_quantity_is_two_dimensional` |
| m22156：负控是 `ref×2.65` 恒等式 | 负控全部为真实数值（链路垂荡/涌浪通道、符号翻转、锁死零、被否的 m22156 参考值） | 表 ④-4、`test_negative_controls_are_real_numbers_not_scaled_synthetics` |
| m22156：验收命令 deselect 掉唯一数值项、文档称 "skipped" 实为 deselected | 测试 19 项全跑（含数值项），CI 命令即 `pytest tests/test_ice_gate_power.py -q` | 本单 `checks` |
| m22213：未留 fail_stage/failure_reason | 本单 `finish` 显式写 checks 与未完成项（`tensor_benchmark_cases` 注册、`oss_solutions` 落库两处平台侧缺口） | 本单 finish |
| m22176 F4：无基准裁判行 | 提供 `metric/tolerance/ref_source` 三元组（见 §3 落库 SQL 的 evidence 字段），可被 `tensor_benchmark_cases` 直接采用 | `docs/oss/GAP-ICE-GATE-POWER-oss_solutions.sql` |

## 5. 调研执行状态

* 本机 `~/oss-cache` 清点：**已执行**（3 个 LBM 包，无同类实现）。
* 集群 `~/oss-cache` 清点：已递交 `run_compute` 作业 `cj4035`（repo=TensorFEM, 1 core,
  `ls ~/oss-cache` + `grep -li "added_mass|added mass|radiation mass"`）；递交时排队位次 22，
  本文件生成时状态见 finish 记录 —— **若队列未清空则该结果记为 not_run**，不作为通过计。
* `oss_inventory` / `oss_solutions` 直连：**不可用**（沙箱角色 `permission denied`），已按上表替代并落 SQL 文件。
