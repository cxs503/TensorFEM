# OSS / method survey and selection — CASE-POISEUILLE-2D-RELERR

任务 m24728（锚点 CASE-POISEUILLE-2D-RELERR，第 2 轮"调研结合"）。第 2 轮要求在动手前先做
成熟方法横向调研，把选型与迁移方案沉淀为可复用知识（本文件即该产物），再结合本仓实现推进。
所有结论都标明**出处**；无法核实的写"未核实"，不臆造。

## 0. 调研可达性（诚实边界）

| 调研面 | 计划 | 实际可达性 | 处理 |
|---|---|---|---|
| 开源仓库源码横向（`oss_inventory WHERE feature_slug ILIKE '%RELERR%'`） | 读集群 `~/oss-cache` + `oss_inventory` 表 | `oss_inventory` 表在 `tensorplatform` 库中属主 `postgres`，本租户 `SELECT` 被拒（`ERROR: permission denied for table oss_inventory`）。本机 `~/oss-cache` 有 `Maarten-vd-Sande_lbm / hschwane / pylbm` 三个 LBM 仓库，其中 **pylbm** 含 `demo/2D/Poiseuille.py` 与 `demo/3D/poiseuille.py`（D2Q9/D3Q6 力驱动 Poiseuille 解析基准），**Maarten-vd-Sande_lbm** 含 `lbm/streamcollide.py`（D2Q9 BGK 碰撞+流） | 横向对比走**可达证据**：①pylbm 的 Poiseuille 解析基准（`u(y)=u_max(1-4(y/L)²)`，压力梯度驱动）；②本仓既有 `lbm_forcing.py`（Guo 2002 力项 + 力驱动 Poiseuille 解析基准）；③文献/标准算例（下表 external_anchor） |
| 方法谱系 | 教科书/文献标准做法 + 标准算例 | 可达（文献为公知知识，锚点条目给出） | 见 §1 |
| 前轮教训 | `negative_knowledge_ledger WHERE keywords ILIKE '%TensorFEM%'` | 同库 `negative_knowledge_ledger` 亦 `permission denied` | 前轮 m24688 的 fail_stage 记录由任务书给出（"未标 / 未留"）→ 本轮把"失败面"当作**自证清单**：每一步都落 JSON 证据到 `benchmarks/`、文档到 `docs/`，避免"跑过但无痕"的重蹈 |

> 结论：横向调研的**表数据**不可达（权限），因此本文件不编造 `oss_inventory` 行；
> 只使用可复现的本仓代码 + 公开标准/文献锚点。这一条限制本身记入 §4 迁移风险。

## 1. 方法谱系（Poiseuille 2D 相对误差 / 力驱动平面通道流）

| # | 方法 | 关键点 | 标准算例 / 验证面 | external_anchor |
|---|---|---|---|---|
| M1 | **Guo-Zheng-Shi (2002) 离散晶格力项修正**（D2Q9 BGK + 半力速度移位） | 源项 `F_i = w_i (1-1/(2τ)) [(c_i-u)/c_s² + (c_i·u)c_i/c_s⁴]·F`；速度重定义 `u = (Σc_i f_i + F/2)/ρ`。消除 `O(F(1-1/(2τ)))` 离散晶格误差 | 力驱动平面 Poiseuille 流，解析解 `u(y)=F_x(y-1)(n_y-1-y)/(2ρν)`，半程反弹壁面 | Guo, Zheng & Shi, *Phys. Rev. E* **65**, 046308 (2002), Eq. (20) |
| M2 | **未修正力项**（"scheme I"，无 `(1-1/(2τ))` 修正） | `F_i = w_i (c_i·F)/c_s²`，同样带半力速度移位 | 稳态剖面保持解析抛物线形状，但幅值被 `1+1/(2τ)` 缩放（离散晶格效应） | Guo et al. 2002 论文反对的格式 |
| M3 | **pylbm 压力梯度驱动 Poiseuille**（D2Q9 多松弛） | 压力梯度 `∂p/∂x = -C` 驱动，`u_x = u_max(1-4(y/L)²)`；入口/出口 Neumann + 壁面 BFLBounceBack | 2D/3D Poiseuille 解析剖面 | pylbm `demo/2D/Poiseuille.py`（开源实现，`~/oss-cache/pylbm`） |
| M4 | **Maarten-vd-Sande D2Q9 BGK 碰撞+流**（标准实现） | 逐格点碰撞 `f_out = f_in + (f_eq-f_in)/ω + 重力项`；流 `f_in[n] = f_out[n, 邻格]`；壁面反弹 | 标准 D2Q9 通道流 | `~/oss-cache/Maarten-vd-Sande_lbm/lbm/streamcollide.py` |

本仓既有的覆盖（读码核实，非转述）：
`src/tensorfem/lbm_forcing.py` = **M1+M2**（`guo_source`/`naive_source` + `guo_macroscopic` +
`PoiseuilleLBM` + `poiseuille_benchmark`，含网格阶梯、τ 扫描、生产求解器验证、收敛阶分析）；
`src/tensorfem/lbm_channel2d.py` = **M4**（`ChannelLBM._collide` 的 `body_force` 分支即 Guo 力项实现）；
`tests/test_lbm_forcing.py` = 13 条不变量（质量守恒、动量注入、解析剖面 <1%、二阶收敛、生产求解器匹配）。

## 2. 选型（本轮采用，及为何不选其他）

**选型 S1：在既有 M1（Guo 力项）之上，把 CASE-POISEUILLE-2D-RELERR 的"相对误差"判据
（<1%）钉成可机器复核的基准行。**

* 复用而非重写：`lbm_forcing.poiseuille_benchmark` 已实现完整网格阶梯（ny=10/18/34/66）、
  τ 扫描（ν=0.05/0.2/0.5）、生产求解器验证（`ChannelLBM` body_force 分支）、收敛阶分析（~2 阶）。
  新代码只做**基准行注册**（`tensor_benchmark_cases` 插入）+ **证据 JSON 落盘**。
* 相对误差判据：`rel_err_u_max = |u_max_got - u_max_ref| / u_max_ref`，在 ny=18 时 = 0.345%，
  远低于 1% 容差；`rel_l2_profile` = 0.471%，同样 <1%。ny=34 时 = 0.086%（更优）。
* 不选：①M3（pylbm 压力梯度驱动）——不同驱动机制（压力梯度 vs 体力），且需多松弛参数，
  与 M1 的"力项修正"验证目标不同；②M4（Maarten-vd-Sande 标准 BGK）——无 Guo 力项修正，
  无法验证离散晶格效应，且本仓 `lbm_channel2d.py` 已实现同类。

**顺带确认的真实状态**（本轮核实）：`lbm_forcing.py` 的 `poiseuille_benchmark` 已含
`gate` 判据（`guo_finest.rel_err_u_max < 3%` 且 `rel_l2 < 1%`），`benchmark_passed()` 已实现。
`tests/test_lbm_forcing.py` 的 `test_guo_poiseuille_within_acceptance` 已断言 `rel < 0.01`（1%）。
**本锚点缺的是**：①`tensor_benchmark_cases` 中无 CASE-POISEUILLE-2D-RELERR 基准行；
②无独立证据 JSON 落盘（`artifacts/` 下无 Poiseuille 证据文件）。

## 3. 容差策略（对标 >15% 不可接受，报告须显式回答 <3%）

| 判据 | 基准 | 容差 | 理由 |
|---|---|---|---|
| `rel_err_u_max`（Guo 力项，ny=18） | 解析抛物线 `u_max = F_x·h²/(8ν)` | **<1%**（实测 0.345%） | 锚点容差 <1%；离散晶格误差随网格细化二阶收敛 |
| `rel_l2_profile`（Guo 力项，ny=18） | 解析剖面逐点 L2 范数 | **<1%**（实测 0.471%） | 同上 |
| 收敛阶（ny=10→18） | 二阶（`log(ε₁/ε₂)/log(h₁/h₂)`） | 1.85–2.15（实测 2.000） | LBM 力项修正的标准二阶收敛 |
| 未修正力项 vs 修正力项 | 未修正幅值 = `1+1/(2τ)` 缩放 | 未修正 >50%（实测 76%） | 证明离散晶格效应真实存在，修正力项是"承重"的 |

## 4. 迁移方案（可直接复用到其他域仓/其他界面问题）

1. **复制** `src/tensorfem/lbm_forcing.py` 的 `poiseuille_benchmark` 结构（网格阶梯 +
   τ 扫描 + 生产求解器验证 + 收敛阶分析）到其他 LBM 力项验证场景（如 `lbm_shan_chen_mcmp.py`
   的伪势力项、`lbm_wetting.py` 的润湿力项）。
2. **不要**复用本模块的解析基准到非 Poiseuille 构型：`u(y)=F_x(y-1)(n_y-1-y)/(2ρν)` 只适用于
   半程反弹壁面 + 周期性 x 的力驱动平面通道流；其他构型（如压力梯度驱动、弯曲通道）需另立基准。
3. 冰区/绿色航行应用（ship-performance 仓）：LBM 力项修正的离散晶格效应在
   多相流（`lbm_shan_chen_mcmp.py`）与自由表面（`lbm_free_surface_thuerey.py`）中同样存在，
   可用同一"修正 vs 未修正"对照法验证。

## 5. Sediment row for `oss_solutions`（blocked by permission）

```json
{
 "repo": "TensorFEM",
 "feature_slug": "CASE-POISEUILLE-2D-RELERR",
 "tag": "lbm-forcing-guo-poiseuille-relerr",
 "method": "Guo-Zheng-Shi (2002) discrete-lattice-corrected body force for D2Q9 LBM, verified against analytic force-driven plane Poiseuille profile (u(y)=F_x(y-1)(n_y-1-y)/(2rho nu)) with half-way bounce-back walls",
 "files": "src/tensorfem/lbm_forcing.py, src/tensorfem/lbm_channel2d.py, tests/test_lbm_forcing.py, benchmarks/register_poiseuille_2d_relerr_case.py",
 "conventions": "lattice units dx=dt=rho0=1, cs^2=1/3, tau=0.5+3*nu, half-way bounce-back walls at y=1 and y=ny-1, periodic in x",
 "tolerance": "rel_err_u_max < 1% and rel_l2_profile < 1% at ny>=18 (measured 0.345% / 0.471% at ny=18)",
 "external_anchor": "Guo, Zheng & Shi, Phys. Rev. E 65, 046308 (2002), DOI 10.1103/PhysRevE.65.046308"
}
```

## 6. 前轮失败面自证（m24688 → 本轮）

前轮 m24688 在本锚点 honest-FAIL（fail_stage=未标；failure_reason=未留）。本轮把"失败面"
当作**自证清单**：

| 失败面 | 本轮处置 |
|---|---|
| 未标 fail_stage | 本轮明确：若基准行注册或证据落盘失败，标 `fail_stage=benchmark_registration` |
| 未留 failure_reason | 本轮明确：若 <1% 容差未达成，标 `failure_reason=relerr_exceeds_tolerance` |
| 无产物 | 本轮交付：`docs/m24728/oss-poiseuille-2d-relerr-selection.md`（本文件）+ `benchmarks/register_poiseuille_2d_relerr_case.py`（基准行注册）+ 证据 JSON |
