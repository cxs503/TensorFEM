# GAP-FEM-3D-COHESIVE · 第 2 轮：调研 + 选型 + 迁移方案

任务 m15001（锚点 GAP-FEM-3D-COHESIVE，第 2 轮"调研结合"）。第 2 轮要求在动手前先做
成熟方法横向调研，把选型与迁移方案沉淀为可复用知识（本文件即该产物），再结合本仓实现推进。
所有结论都标明**出处**；无法核实的写"未核实"，不臆造。

## 0. 调研可达性（诚实边界）

| 调研面 | 计划 | 实际可达性 | 处理 |
|---|---|---|---|
| 开源仓库源码横向（`oss_inventory WHERE feature_slug ILIKE '%COHESIVE%'`） | 读集群 `~/oss-cache` + `oss_inventory` 表 | `oss_inventory` 表在 `tensorplatform` 库中属主 `postgres`，本租户 `SELECT` 被拒（`ERROR: permission denied for table oss_inventory`；且该表实际没有 `feature_slug` 列，任务给的 SQL 与库结构不一致）。本机 `~/oss-cache` 只有 `Maarten-vd-Sande_lbm / hschwane / pylbm` 三个 LBM 仓库，**无 cohesive/delamination 同类实现**。`/api/v1/oss/*` 端点返回 401（凭据不匹配），ouraccess 到此为止 | 横向对比改走**可达证据**：①同仓既有内聚实现（`tensorfem.cohesive`(2D)、`tensorfem.mixed_mode_cohesive`(BK)、`tensorfem.cohesive_bending`(2D CZM 三点弯)、`benchmarks/cohesive3d_delamination.py`(m14686，单宽度单元)）；②文献/标准算例（下表 external_anchor）；③开源实现的**方法层**特征按文献通行做法记录（明确标注"未读源码"） |
| 方法谱系 | 教科书/文献标准做法 + 标准算例 | 可达（文献为公知知识，锚点条目给出） | 见 §1 |
| 前轮教训 | `negative_knowledge_ledger WHERE keywords ILIKE '%TensorFEM%'` | 同库 `negative_knowledge_ledger` 亦 `permission denied`；`/api/v1/tensor/negative-knowledge/search` 需 token，本会话 `/api/v1/auth` 用 `agent.env` 凭据登录失败（401） | 前轮 m14987 的 fail_stage 记录由任务书给出（"未标 / 未留"）→ 本轮把"失败面"当作**自证清单**：每一步都落 JSON 证据到 `benchmarks/`、文档到 `docs/`，避免"跑过但无痕"的重蹈 |

> 结论：横向调研的**表数据**不可达（权限），因此本文件不编造 `oss_inventory` 行；
> 只使用可复现的本仓代码 + 公开标准/文献锚点。这一条限制本身记入 §4 迁移风险。

## 1. 方法谱系（内聚区模型 / 3D 界面单元）

| # | 方法 | 关键点 | 标准算例 / 验证面 | external_anchor |
|---|---|---|---|---|
| M1 | 本征型双线性牵引-分离律（intrinsic bilinear TSL），历史变量 `κ=max(δ_eff)` 驱动不可逆损伤 | 初始罚刚度 `k_n`、峰值 `t_n0`、失效开口 `δ_f=2G_c/t_n0`；压缩不损伤（Macaulay 括号） | 单元级：能量守恒、起裂判据、切线 vs AD | Ortiz & Pandolfi, *Int. J. Numer. Methods Eng.* 44 (1999) 1267；Alfano & Crisfield, *IJNME* 50 (2001) 1701 |
| M2 | 零厚度界面单元 COH3D8 拓扑（Abaqus 一致） | 8 节点、中面 2×2 Gauss、`B^T t dA` 内力 + `B^T D B dA` 切线；参考构型（小应变）或共旋框架 | 补丁测试、刚体旋转客观性、网格客观性 | Abaqus Theory Manual §4.5.5 "Cohesive elements"（公知拓扑）；本仓 `docs/cohesive-3d-element.md` 已比对 11 条闭式判据 |
| M3 | 混合模式：二次名义应力起裂 + Benzeggagh–Kenane 传播准则 | `G_c = G_Ic + (G_IIc-G_Ic)(G_s/G)^η`，方向敏感性进切线（否则 Newton 只收敛线性率） | 纯 I/II/III 面积 = `G_c`；混合模式面积 | Benzeggagh & Kenane, *Composites Sci. Tech.* 56 (1996) 439 |
| M4 | DCB 解析对标：梁理论 + 根部旋转修正（ASTM D5528 修正梁理论） | `C = 4a³/(Ebh³) + a/(kGbh)`，`G_I = 6P²a²/(Eb²h³)`，`a → a+Δ`，`Δ = h√((E/11G)(3−2(Γ/(1+Γ))²))`，`Γ=1.18√(E/G)` | 柔度、峰值载荷、R 曲线（载荷-裂纹长度） | ASTM D5528-13（修正梁理论 CBBM/CBT）；Williams, *J. Strain Analysis* 24(4) (1989) 207 |
| M5 | 破坏/断裂能全局能量守恒校核 | `W_ext − U_solid − W_interface ≈ 0`，且 `W_interface/ΔA → G_Ic` | 能量平衡、界面功/失效面积 | 内聚区模型能量一致性（Ortiz–Pandolfi 1999 同源），本仓 `cohesive_bending.py` 已用同法（2D） |
| M6 | 试样级验证算例族：DCB（I 型）/ ENF（II 型）/ MMB（混合） | 裂纹前缘由损伤场决定，不预置；位移控制 Newton + 步长自适应 | 与 M4 闭式解 & VCCT 双路互证 | ASTM D5528（DCB）、D7905（ENF）、D6671（MMB）；VCCT: Rybicki & Kanninen, *Eng. Fract. Mech.* 9 (1977) 931 |

本仓既有的覆盖（读码核实，非转述）：
`src/tensorfem/cohesive3d.py` = M1+M2+M3（738 行，含 corotational 变体）；
`benchmarks/cohesive3d_verification.py`（11 条单元级闭式判据，`docs/cohesive-3d-element.md`）；
`benchmarks/cohesive3d_delamination.py` = M6，但 `MESH={"nx":100,"nz":3}` 且**宽度只有 1 个单元**
（平面应力式理想化），且仓内**没有提交它的证据 JSON**（`benchmarks/cohesive3d_delamination.json` 缺失）——
这正是"3D 单元 + 验证算例"仍记为缺口的原因：试样级证据是 quasi-2D 且无落盘工件。

## 2. 选型（本轮采用，及为何不选其他）

**选型 S1：在既有 M1–M3 单元之上，补 M4+M5+M6 的"真 3D（宽度分辨）"试样级验证面。**

* 复用而非重写：单元/律/切线一律调用生产实现 `cohesive3d.cohesive_hex8_response` 与
  `BKCohesive3DLaw`；新代码只做**装配 + Newton + 解析基准**（`src/tensorfem/cohesive3d_dcb.py`）。
* 宽度分辨：`ny` 个 HEX8 穿过宽度，界面层节点**复制**（真内聚界面，两臂不共节点），
  这样才在张量应力状态下检验单元；`ny=1` 与 `ny=3`、`b/h=1` 与 `b/h=6` 两组对照给"3D 客观性"。
* 解析基准不循环：柔度/峰值/裂纹前缘三处都用**闭式解**（M4）或**全局平衡**（M5），
  不与代码自身比对；并显式给出 Williams 根旋转带来的 ± 带（见 §3 容差策略）。
* 性能：原逐单元装配 10 ms/单元（实测），试样级网格不可行 → 结构化界面片是平移不变的，
  参考构型下所有 COH3D8 的 `B`、旋转阵、Gauss 面积相同，故用**批量 einsum 装配**，
  并用测试断言"批量装配 ≡ 逐单元生产函数"（机器精度），保证不改物理。
* 不选：①VCCT-only 路线（Rybicki–Kanninen）——只给 SERR，不给损伤演化，且需要预置裂纹前缘；
  ②相场断裂——另一套理论，属于别的题；③壳/梁内聚单元——2D 层面已有，补不了 3D 缺口。

**顺带修的真实缺陷**（本轮发现，属 M1 实现层）：`BKCohesive3DLaw.update` 的
docstring 声明"one (or many) integration points"，但批量输入时切线组装两处缺 `[..., None]`
（`k * h`、`k * delta_eff * hp`），任何 `(...,3)` 批量调用直接抛 `RuntimeError`；
生产路径逐 Gauss 点调用所以没暴露。已最小修复 `tangent = diag_embed(k * h[..., None] * jac_clamp)`、
`coef = (k*delta_eff) * hp[..., None]`，并由 `tests/test_cohesive3d_dcb.py::test_batched_law_tangent_matches_per_point_calls`
把"批量 == 逐点"钉成回归。

## 3. 容差策略（对标 >15% 不可接受，报告须显式回答 <3%）

| 判据 | 基准 | 容差 | 理由 |
|---|---|---|---|
| 完好试样柔度 | 梁理论（含剪切）+ Williams 根旋转 | 落在"无修正"与"根旋转修正"两闭式之间，且与根旋转式 ≤10% | 根旋转本身是近似（Kanninen 弹性地基 vs Williams），DCB 尖端截面 5–8% 不确定性是**理论侧**而非单元侧；给区间比给单点更诚实 |
| 峰值载荷 | LEFM + 根旋转 | ≤5% | 同上；并给出无修正值作为第二参照 |
| 双裂纹长比值 `P(a1)/P(a2)` | `(a2+Δ)/(a1+Δ)` | ≤2% | 根旋转在一阶上抵消 → 对理论近似不敏感，是**强判据** |
| 全局能量平衡 | `W_ext−U−W_coh=0` | ≤1% | 离散恒等式级，可判装配/历史提交错误 |
| 界面功/失效面积 | `G_Ic` | ≤10% | 末步部分损伤区未计入失效面积，上界偏差 |
| R 曲线 | `P_LEFM(a_front)`（a_front 取自 0.9 损伤前沿） | ≤10% | 前沿判读有 ±1 单元分辨 |
| 宽度客观性 | `ny=3` vs `ny=1`（平面应变模量换算）| ≤8% + 前沿直线度 ≤2 单元 | `b/h=6` 仍处过渡区，anticlastic 效应真实存在 |

## 4. 迁移方案（可直接复用到其他域仓/其他界面问题）

1. **复制** `src/tensorfem/cohesive3d_dcb.py` 的 `DCBModel.reference_frames()`
   （结构化界面片 → 一次求参考标架，`B`/面积缓存）与 `solve_dcb()`（位移控制 Newton、
   `κ` 只在收敛后提交、步长折半/×1.2 自适应）。这两个函数与 DCB 几何**解耦**，
   只需替换 `build_*_model()`（ENF/MMB/冰-结构粘附层/层合板分层都适用）。
2. **不要**复用本模块的解析基准到非 DCB 构型：ENF/MMB 的闭式解在
   `benchmarks/cohesive3d_delamination.py` 里（`enf_compliance`, `mmb_linear_prediction`），
   且必须走 VCCT 双路互证。
3. 冰区应用（ship-performance 仓 / TensorPhysics 冰区语义）：冰-结构粘附层用同一单元，
   但强度/能量参数来自冰力学（Lindqvist/IACS 锚），几何为界面片而非直裂纹，
   迁移时把 `DCBGeometry` 换成"界面片 + 载荷工况"，`reference_frames()` 需检查界面片是否**等面积**
   （本模块显式校验，不等面积会抛 `ValueError` 而不是静默给错装配）。
4. **风险（本轮实测）**：逐单元装配 10 ms/单元 → 试样级网格（>100 单元 × >100 步）不可行，
   必须用批量装配；批量装配的前提是"界面片平移不变"，一旦做**graded/非结构界面**就必须回退逐单元
   （`reference_frames()` 已按此抛错）。

## 5. 本轮产物与运行入口

* `src/tensorfem/cohesive3d_dcb.py` — 宽度分辨 3D DCB 装配 + Newton + 解析基准（复用生产单元）。
* `benchmarks/cohesive3d_dcb_3d.py` — 判据执行器：写 `benchmarks/cohesive3d_dcb_3d.json`，
  并**程序化生成** `docs/cohesive3d-dcb-3d-verification.md`（零手抄）与
  `docs/assets/cohesive3d-dcb-3d/*.png`。
* `tests/test_cohesive3d_dcb.py` — 拓扑/装配等价性/批量律/根旋转闭式/柔度夹逼/损伤单调性。
* 计算路径：本机控制面对单进程 >120 s 即收割，故 `run_cluster submit`（stage_dir=本工作树）执行，
  证据 JSON 取回后入库。

## 6. 未完成 / 未核实（诚实申报）

* `oss_inventory` / `negative_knowledge_ledger` 两类表**读不到**（权限），开源横向调研只剩
  "本仓 + 文献"两条腿；如需真横向源码对比，需平台授 `oss_inventory` 只读视图或提供 `~/oss-cache` 的 FEM 快照。
* 未做：ENF/MMB 的宽度分辨版（本轮聚焦 I 型；`cohesive3d_delamination.py` 仍是 quasi-2D）。
* 未做：大变形（共旋）DCB、裂纹面接触/摩擦、循环退化 → 见 `docs/cohesive3d-dcb-3d-verification.md` §限制。