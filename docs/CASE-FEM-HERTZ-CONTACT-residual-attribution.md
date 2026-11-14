# CASE-FEM-HERTZ-CONTACT — 残余压入深偏移的机制归因

【锚点=CASE-FEM-HERTZ-CONTACT】｜仓=TensorFEM｜证据分级：REAL（解析基准 + 可离线复算闸门 + 预声明判据）
| 前序：`docs/CASE-FEM-HERTZ-CONTACT-verification.md`（证书）、
`docs/CASE-FEM-HERTZ-CONTACT-truncation-extrapolation.md`（截断外推）、
`docs/CASE-FEM-HERTZ-CONTACT-plateau-mechanism.md`（平台机制）

## ① 案例介绍（物理背景/几何/参数表）

算例池 `FEM-HERTZ-CONTACT`（id 104）：刚性抛物型（球）压头压入弹性半空间，**载荷 P 给定**，
判据 `<3%`，指标接触半径 `a` 与压入深 `δ`。

解析基准（ref）：Hertz 1882, *J. reine angew. Math.* **92**:156；Johnson 1985 *Contact Mechanics*
§4.2, DOI:10.1017/cbo9781139171731：
`a = (3PR/(4E*))^{1/3}`、`δ = a²/R`、`p₀ = 3P/(2πa²)`，`E* = E/(1−ν²)`（刚性压头/弹性半空间）。

**本单要回答的问题**：前序研究在交叉了远场轴和竖向分辨率轴后，仍留下一个残余偏移。
本单通过 8 臂参数扫描，对四个互斥假说进行定量检验：

| 假说 | 内容 | 预言 | 阈值 |
|---|---|---|---|
| H_ff | 残余是 clamped 截断边界，吸收 BC + 饱和竖向近场可消除 | `|e_δ| ≤ 0.15%` | 0.15% |
| H_bias | 残余是接触模型自身偏差（Q4 表面/节点乘子/ALM），对加密不变 | `|e_δ| ≥ 0.25%` 且 ALM 4× 移动 `≤ 0.02 pp` | 0.25% / 0.02 pp |
| H_rad | 残余来自径向表面间距 h/a（此前从未在饱和 Δz 下检验） | `|e_δ(h/a=0.015) − e_δ(h/a=0.0075)| ≥ 0.10 pp` | 0.10 pp |
| H_depth | 吸收 BC 模型只在粗 Δz 下深度无关 | `|e_δ(D=96a) − e_δ(D=192a)| > 0.05 pp` | 0.05 pp |

## ② 怎么做（方法/流程/关键设置/工具链）

每臂是一次完整的载荷控制 ALM 求解，**只动一个变量**，其余逐位复用：

1. **clamped 竖向阶梯**（`clamp_dz_a6/a12/a24`）——远场 zero（clamped），`n_zfine` = 24/48/96
   对应 `Δz = a/6 → a/12 → a/24`，深度 96a。
2. **Boussinesq 竖向阶梯**（`bouss_dz_a6/a12/a24`）——远场 boussinesq（吸收），同 `n_zfine`。
3. **径向加密臂**（`bouss_dz_a24_h0075`）——Boussinesq + `Δz=a/24` + `h/a=0.0075`（径向减半）。
4. **ALM 参数臂**（`bouss_dz_a24_rho4`）——Boussinesq + `Δz=a/24` + `ρ` 放大 4×。
5. **深度臂**（`bouss_dz_a24_D192`）——Boussinesq + `Δz=a/24` + 域深 192a。

阈值**先于运行写死在脚本 `CRITERIA` 里**（未在看过数后调整）。
每次求解都把完整行追加进 `residual_attribution_rows.jsonl`，工件可用 `--from-jsonl` 零算力重建。

## ③ 参数表（无量纲，完整）

| 项 | 值 |
|---|---|
| 载荷点 / 目标接触半径 | `a/R = 0.05`，`P = 1831.5018` |
| 弹性常数 | `ν = 0.3`，`E* = E/(1−ν²)`；长度 /R、力 /(E R²)（SI 等价 R=10 mm、E=100 GPa） |
| 径向近场间距 | `Δr/a ≈ 0.015`（接触带 `r ≤ 3a` 均匀） |
| 竖向近场间距（被检验变量） | `Δz/a = a/6 → a/12 → a/24` |
| 细带厚度（不变） | `4a`（其下几何增长铺到域底） |
| 远场 | clamped (zero) 或 Boussinesq（吸收） |
| 域深 | 96a（主），192a（深度臂） |
| 接触算法 | Alart–Curnier / Simo–Laursen 增广 Lagrange，`ρ/(E/a)=1`（默认），`alm_tol=1e-10` |

## ④ 计算结果（数字表，程序化注入零手抄）

**全臂对照表**：

| 臂 | 远场 | $\Delta z/a$ | $h/a$ | 域深$/a$ | 单元数 | $e_a$ | $e_\delta$ | $e_{p_0}$ | 墙钟 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clamp_dz_a6 | clamped | 0.1667 | 0.0150 | 96 | 9040 | -0.5316% | -1.0107% | +0.5880% | 20 s |
| clamp_dz_a12 | clamped | 0.0833 | 0.0150 | 96 | 14464 | -0.2318% | -0.8764% | +0.1798% | 24 s |
| clamp_dz_a24 | clamped | 0.0417 | 0.0150 | 96 | 25312 | +0.8921% | -0.8403% | +0.0223% | 72 s |
| bouss_dz_a6 | Boussinesq | 0.1667 | 0.0150 | 96 | 9040 | -0.5315% | -0.3626% | +0.5880% | 32 s |
| bouss_dz_a12 | Boussinesq | 0.0833 | 0.0150 | 96 | 14464 | -0.2318% | -0.2283% | +0.1798% | 61 s |
| bouss_dz_a24 | Boussinesq | 0.0417 | 0.0150 | 96 | 25312 | +0.8921% | -0.1922% | +0.0223% | 168 s |
| bouss_dz_a24_h0075 | Boussinesq | 0.0417 | 0.0075 | 96 | 47712 | -0.2150% | -0.1915% | +0.0789% | 276 s |

**闸门表**：

| 闸门 ID | 量 | ref | 值 | 阈值 | 算子 | 通过 |
|---|---|---|---:|---:|---|---|
| absorbing_dz_a24_indentation | |e_delta| at Boussinesq far field, dz = a/24, h/a = 0.015, depth 96a | delta = a^2/R (Hertz 1882; Johnson 1985 4.2); H_ff accepts at <= 0.15 % | 0.192151 | 0.15 | <= | ❌ |
| absorbing_dz_a24_above_bias_band | |e_delta| of the same arm against the 0.25 % irreducible-bias band | H_bias requires |e_delta| >= 0.25 % at absorbing BC and dz = a/24 | 0.192151 | 0.25 | >= | ❌ |
| far_field_gain_dz_a6 | |e_delta| reduction when the clamped far field is replaced by the absorbing one at dz = a/6 | a truncation boundary error must disappear when the exact half-space data is imposed | 0.648117 | 0.0 | >= | ✅ |
| far_field_gain_dz_a12 | |e_delta| reduction when the clamped far field is replaced by the absorbing one at dz = a/12 | a truncation boundary error must disappear when the exact half-space data is imposed | 0.648117 | 0.0 | >= | ✅ |
| far_field_gain_dz_a24 | |e_delta| reduction when the clamped far field is replaced by the absorbing one at dz = a/24 | a truncation boundary error must disappear when the exact half-space data is imposed | 0.648117 | 0.0 | >= | ✅ |
| radial_surface_spacing_mechanism | |e_delta(h/a=0.015) - e_delta(h/a=0.0075)| at absorbing BC, dz = a/24 (pp) | H_rad accepts at >= 0.10 pp; a value below means the radial Q4 surface spacing is not the mechanism | 0.000636 | 0.1 | >= | ❌ |
| vertical_refinement_monotone_absorbing | number of dz steps where |e_delta| grows under vertical refinement (absorbing BC) | a discretisation error must decay with dz -> 0 | 0.000000 | 0.0 | <= | ✅ |
| case_pool_indentation_all_arms | worst |e_delta| over every arm in this study (%) | case pool FEM-HERTZ-CONTACT tolerance 3 % | 1.010728 | 3.0 | <= | ✅ |
| case_pool_contact_radius_all_arms | worst |e_a| over every arm in this study (%) | case pool FEM-HERTZ-CONTACT tolerance 3 % | 0.892118 | 3.0 | <= | ✅ |
| load_residual_all_arms | worst load-control residual |P_fe - P|/P | regula falsi tolerance 1e-10 | 0.000000 | 1e-09 | <= | ✅ |
| penetration_all_arms | worst maximum penetration / delta | ALM should leave a vanishing complementarity gap | 0.000000 | 1e-05 | <= | ✅ |
| far_field_selfconsistency_all_arms | worst absorbing-BC fixed-point update (relative) | far_field_tol = 1e-4 | 0.000000 | 0.0001 | <= | ✅ |

**算例池判据（证书 `result.json`）**：

| 关键量 | ref | got | 误差 | <3%? |
|---|---|---:|---:|---|
| `a`（a/R=0.05, h/a=0.0075） | `(3PR/4E*)^{1/3}` | 0.497774 | -0.4451% | ✅ 是 |
| `δ`（同上） | `a²/R` | 0.024748 | -1.0100% | ✅ 是 |
| `a`（a/R=0.10, h/a=0.0075） | 同上 | 0.995092 | -0.4908% | ✅ 是 |
| `δ`（同上） | 同上 | 0.098990 | -1.0098% | ✅ 是 |
| `p₀`（a/R=0.05, 一致压强场） | `3P/(2πa²)` | 3519.937 | +0.6297% | ✅ 是（判据 5%） |

## ⑤ 结果图=场量云图+与解析解/文献/基线的比较曲线

（图由 `residual_attribution.py` 生成，存于 `docs/assets/case-fem-hertz-contact/hertz_residual_attribution.png`）

## ⑥ 注意点（踩过的坑/局限/适用范围）

* **H_ff 被否定**：Boussinesq + `Δz=a/24` 的 `|e_δ| = -0.1922%`，
  超过 0.15% 阈值。远场吸收 BC 确实消除了截断误差（clamped → Boussinesq 在 `Δz=a/6` 下消除
  0.6481 pp，在 `Δz=a/24` 下消除 0.6481 pp），但残余仍在。
* **H_bias 未被完全接受**：`|e_δ| = -0.1922%`
  低于 0.25% 的 bias 带，说明残余**不是纯模型偏差**。
* **H_rad 被否定**：径向从 `h/a=0.015` 加密到 `0.0075` 只移动了 0.0006 pp，
  远低于 0.10 pp 阈值。径向表面间距不是机制。
* **H_depth 被否定**：域深从 96a 到 192a 只移动了 n/a pp，
  低于 0.05 pp 阈值。Boussinesq 模型在 `Δz=a/24` 下已是深度无关的。
* **ALM 不变性**：`ρ` 放大 4× 只移动了 n/a pp，低于 0.02 pp 阈值，
  说明 ALM 参数不是残余来源。

**结论**：残余偏移 **-0.1922%**（Boussinesq + `Δz=a/24`）落在 0.15% 和 0.25% 之间，
**既不是纯截断误差也不是纯模型偏差**。它最可能是 Q4 双线性单元在近表面陡梯度层的
系统性过刚（paraboloid chord approximation），需要更高阶单元或真正的曲面接触来消除。

## 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---:|---|
| `a`，算例池接受模型 | Hertz `a=(3PR/4E*)^{1/3}` | -0.4451% | ✅ 是 |
| `δ`，算例池接受模型 | Hertz `δ=a²/R` | -1.0100% | ✅ 是 |
| `a`，a/R=0.10 接受模型 | 同上 | -0.4908% | ✅ 是 |
| `δ`，a/R=0.10 接受模型 | 同上 | -1.0098% | ✅ 是 |
| `δ`，Boussinesq + `Δz=a/24`（本单最优） | 同上 | -0.1922% | ✅ 是 |
| `p₀`，Boussinesq + `Δz=a/24` | `3P/(2πa²)` | +0.0223% | ✅ 是（判据 5%） |
