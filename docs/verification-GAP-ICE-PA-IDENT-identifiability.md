# GAP-ICE-PA-IDENT 压强-面积律可辨识性：把物理指数从内生性地板里分出来

自动生成：`benchmarks/ice_coupled_6dof/run_w31_pa_identifiability.py` -> `benchmarks/ice_coupled_6dof/pa_identifiability.json`。表中每个数字都由同一次运行注入，无手抄。

## 1. 案例介绍

冰-船接触设计律 `p = C_R (A/A_0)^ex`。W22/W23 链把该律当作**压力源**，对其自身样本拟合只能回到输入指数（闭合、无判别力）；W24 改用**无面积指数的速率机制**（`p = p_ref (epsdot/epsdot_ref)^n`、`epsdot = V_close/(4w)`、`w = A/h`）作处理臂，并以「每浮冰随机强度、无面积依赖」作负控，结果处理臂 `ex ~ -0.10` 与负控同量级，物理信号落在自己的零假设内。本单回答：地板到底是什么、群体指数是什么、机制指数能否辨识。

臂清单（全部在冻结的 W24 实验定义上运行：浓度 [0.3, 0.5, 0.7]、种子 [1101, 1102, 1103, 1104, 1105]、时长 40 s、dt = 0.01 s）：

| arm | 角色 | n |
|---|---|---|
| `C1_iso_area_law` | closure (imposed ISO area law as pressure source) | n/a |
| `C0_random_strength` | negative control (no area dependence) | n/a |
| `C0_null_strength` | negative control with zero scatter: p = 1.9 MPa exactly, no area dependence -- deterministic zero check of the estimator | n/a |
| `T_rate_n015` | treatment (rate-dependent crushing, n = 0.15) | 0.1500 |
| `T_rate_n033` | treatment (rate-dependent crushing, n = 0.33) | 0.3300 |
| `T_rate_n050` | treatment (rate-dependent crushing, n = 0.5) | 0.5000 |
| `T_rate_n028` | declared sensitivity: rate-dependent crushing with n equal to |declared reference exponent| (not a calibration) | 0.2800 |

## 2. 方法

处理臂压力律取对数即为线性：

```
ln p = const + n ln V_close + n ln h - n ln A
=> ex_pop = -n + n (beta_h + beta_V) = n (beta_h + beta_V - 1)
   beta_h = dln h/dln A ,  beta_V = dln V_close/dln A
```

`ex_pop` 是**群体（设计口径）指数** = 材料指数 x 群体协方差结构；若面积变化主要来自冰厚（`A = wh`、`w` 固定、`beta_h -> 1`），两项相消、`ex_pop -> 0`，**对任何 n 成立** —— 这是内生性地板的代数来源。

估计器：固定效应（within）回归，按混杂量分箱、箱内去均值，只用**宽度驱动**的面积变化。变体矩阵：

| variant | 箱子 (冰厚 x 闭合速度) | 解释 |
|---|---|---|
| `pop` | 1 x 1 | 原始群体斜率（不条件化） |
| `h_only` | 16 x 1 | 条件化 |
| `V_only` | 1 x 16 | 条件化 |
| `both` | 16 x 16 | 条件化 |

**条件化本身是可证伪的**：负控（无面积依赖）在同一估计器下必须保持在地板附近；若某个条件化变体把负控推离 0，说明它条件化在**接触力的下游变量**上（闭合速度是 6DOF 响应的后代），该变体不可用于物理结论。本单把每个变体都在**所有臂**上跑一遍，因此这个检验是机器可查的。

另外增设 `C0_null_strength`（`cv = 0`，压力为常数）：此时真实斜率精确为 0，任何非零返回值都是估计器自身的偏置 —— 这是估计器的确定性零点检验。

## 3. 计算结果

### 3.1 变体矩阵（每臂中位 over 3 个浓度）

| arm | n | 机制 -n | pop | h_only | V_only | both | both vs -0.28 | both vs -n | 最小 sigma vs 地板 |
|---|---|---|---|---|---|---|---|---|---|
| `C1_iso_area_law` | n/a | -0.280 | -0.2800 | -0.2800 | -0.2800 | **-0.2800** | 0.00 % | n/a | 4.84 |
| `C0_random_strength` | n/a | 0.000 | -0.1205 | -0.1348 | -0.1273 | **-0.1429** | 48.95 % | n/a | 0.00 |
| `C0_null_strength` | n/a | 0.000 | 0.0000 | -0.0000 | -0.0000 | **0.0000** | 100.00 % | n/a | 8.51 |
| `T_rate_n015` | 0.1500 | -0.150 | -0.0394 | -0.0448 | -0.0868 | **-0.1168** | 58.27 % | 22.11 % | 1.68 |
| `T_rate_n033` | 0.3300 | -0.330 | -0.0970 | -0.1088 | -0.1970 | **-0.2606** | 6.93 % | 21.03 % | 3.91 |
| `T_rate_n050` | 0.5000 | -0.500 | -0.1734 | -0.1932 | -0.3159 | **-0.4054** | 44.80 % | 18.91 % | 8.30 |
| `T_rate_n028` | 0.2800 | -0.280 | -0.0795 | -0.0889 | -0.1672 | **-0.2206** | 21.21 % | 21.21 % | 1.99 |

### 3.2 逐浓度分离检验（primary = `both`, 16x16 格）

| arm | 浓度 | 冲击数 | pop ex | both ex | se(冲击级/种子 boot 取大) | 地板 ex | sigma | >=3sigma |
|---|---|---|---|---|---|---|---|---|
| `C1_iso_area_law` | 0.3 | 762 | -0.2800 | -0.2800 | 0.0000 | n/a | 4.84 | 是 |
| `C1_iso_area_law` | 0.5 | 1267 | -0.2800 | -0.2800 | 0.0000 | n/a | 9.07 | 是 |
| `C1_iso_area_law` | 0.7 | 1776 | -0.2800 | -0.2800 | 0.0000 | n/a | 10.89 | 是 |
| `C0_random_strength` | 0.3 | 762 | -0.1260 | -0.1785 | 0.0210 | -0.1785 | 0.00 | 否 |
| `C0_random_strength` | 0.5 | 1267 | -0.1174 | -0.1429 | 0.0151 | -0.1429 | 0.00 | 否 |
| `C0_random_strength` | 0.7 | 1776 | -0.1205 | -0.1401 | 0.0129 | -0.1401 | 0.00 | 否 |
| `C0_null_strength` | 0.3 | 762 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.51 | 是 |
| `C0_null_strength` | 0.5 | 1267 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.46 | 是 |
| `C0_null_strength` | 0.7 | 1776 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.90 | 是 |
| `T_rate_n015` | 0.3 | 762 | -0.0353 | -0.1194 | 0.0056 | n/a | 2.73 | 否 |
| `T_rate_n015` | 0.5 | 1267 | -0.0394 | -0.1155 | 0.0047 | n/a | 1.73 | 否 |
| `T_rate_n015` | 0.7 | 1776 | -0.0436 | -0.1168 | 0.0052 | n/a | 1.68 | 否 |
| `T_rate_n033` | 0.3 | 762 | -0.0864 | -0.2743 | 0.0126 | n/a | 3.91 | 是 |
| `T_rate_n033` | 0.5 | 1267 | -0.0970 | -0.2574 | 0.0123 | n/a | 5.87 | 是 |
| `T_rate_n033` | 0.7 | 1776 | -0.1068 | -0.2606 | 0.0124 | n/a | 6.74 | 是 |
| `T_rate_n050` | 0.3 | 762 | -0.1573 | -0.4162 | 0.0195 | n/a | 8.30 | 是 |
| `T_rate_n050` | 0.5 | 1267 | -0.1734 | -0.4007 | 0.0149 | n/a | 12.15 | 是 |
| `T_rate_n050` | 0.7 | 1776 | -0.1878 | -0.4054 | 0.0210 | n/a | 10.77 | 是 |
| `T_rate_n028` | 0.3 | 762 | -0.0731 | -0.2281 | 0.0134 | n/a | 1.99 | 否 |
| `T_rate_n028` | 0.5 | 1267 | -0.0795 | -0.2179 | 0.0094 | n/a | 4.22 | 是 |
| `T_rate_n028` | 0.7 | 1776 | -0.0879 | -0.2206 | 0.0094 | n/a | 5.06 | 是 |

地板（负控 `both`）逐浓度：

| 浓度 | 地板 ex (`both`) | se | 地板 ex (`pop`) |
|---|---|---|---|
| 0.3 | -0.1785 | 0.0210 | -0.1260 |
| 0.5 | -0.1429 | 0.0151 | -0.1174 |
| 0.7 | -0.1401 | 0.0129 | -0.1205 |

确定性零点检验（`C0_null_strength`, cv = 0）：最大 |ex| = 0.000000000000（6 个值）-> **PASS**

### 3.3 恒等式核对与衰减

| arm | |残差| 中位（`ex_pop` vs `n(beta_h+beta_V-1)`） |
|---|---|
| `C1_iso_area_law` | n/a |
| `C0_random_strength` | n/a |
| `C0_null_strength` | n/a |
| `T_rate_n015` | 0.0000000000 |
| `T_rate_n033` | 0.0000000000 |
| `T_rate_n050` | 0.0000000000 |
| `T_rate_n028` | 0.0000000000 |

收敛（`T_rate_n033`, 变体 `both`）：

| 箱子 | ex 中位 |
|---|---|
| 1x1 | -0.0970 |
| 2x2 | -0.1318 |
| 4x4 | -0.1673 |
| 8x8 | -0.2168 |
| 16x16 | -0.2606 |

收敛（负控 `C0_random_strength`, 变体 `both`）：

| 箱子 | ex 中位 |
|---|---|
| 1x1 | -0.1205 |
| 2x2 | -0.1408 |
| 4x4 | -0.1402 |
| 8x8 | -0.1455 |
| 16x16 | -0.1429 |

### 3.4 反演：-0.28 在速率机制下需要什么

* 群体协方差（`T_rate_n033` 合并样本）：`beta_h = 0.1195`、`beta_V = 0.5799`、`beta_h + beta_V = 0.6994`
* 反演速率指数 `n_implied = -0.28 / (beta_h + beta_V - 1) = 0.9315`
* 声明性脆性压碎窗口 `n in [0.2, 0.4]` 可复现 -0.28 所需协方差窗口：`beta_h + beta_V in [0.3000, -0.4000]`

## 4. 判据结论

| id | 判据 | 状态 | 明细 |
|---|---|---|---|
| I0 | deterministic zero check: with a constant contact pressure (cv = 0) the estimator must return exactly zero | **PASS** | max_abs_slope=0.0000; n_values=6 |
| I1 | treatment exponent separates from the negative-control floor by >= 3 combined sigma at every concentration (primary estimator = cells of fixed thickness x closing speed, (16, 16)) | **PASS** | min_sigma=3.9118; treatment_median=-0.2606; floor_median=-0.1429 |
| I2 | treatment exponent recovered by the conditioned estimator is within 15 % of the declared reference -0.28 | **PASS** | deviation=0.0693; threshold=0.1500; value=-0.2606; caveat=the estimator is attenuated towards zero at finite conditioning; the bias-free limit is -n (= -0.33), which is 17.9 % away from -0.28 -- see I3 |
| I3 | bias-corrected statement: the recovered exponent equals the arm's own mechanism exponent -n (attenuation accounted for via the convergence sweep) | **FAIL** | deviation_from_mechanism=0.2103; mechanism=-0.3300; treatment_median=-0.2606; convergence=[{'resolution': [1, 1], 'slope_median': -0.09695568850031562, 'n_concentrations': 3}, {'resolution': [2, 2], 'slope_median': -0.13184030222675444, 'n_concentrations': 3}, {'resolution': [4, 4], 'slope_median': -0.16734514036960613, 'n_concentrations': 3}, {'resolution': [8, 8], 'slope_median': -0.21682500545375896, 'n_concentrations': 3}, {'resolution': [16, 16], 'slope_median': -0.26059337679531885, 'n_concentrations': 3}]; reading=|ex| grows monotonically with the conditioning resolution while the control stays put: the limit is the mechanism exponent, not -0.28 |
| I4 | population (unconditioned) exponent of the treatment sits inside the negative-control floor -- the W24 FAIL mode | **CONFIRMED** | treatment_pop_median=-0.0970; floor_pop_median=-0.1205; note=CONFIRMED reproduces the W24 outcome and explains it by ex_pop = n (beta_h + beta_V - 1) |
| I5 | control: when the mechanism really is an area law (closure arm) the same conditioned estimator returns that law's exponent, so the conditioning does not destroy a genuine area dependence | **PASS** | value=-0.2800; declared_reference=-0.2800; deviation=0.0000 |
| I6 | the declared reference exponent -0.28 is reproducible by an indentation-rate mechanism on this population only if the implied rate exponent falls inside the declared brittle-crushing window | **FAIL** | n_implied=0.9315; window=[0.2, 0.4]; sum_betas=0.6994; admissible_beta_sum_window=[0.29999999999999993, -0.40000000000000013] |

## 5. 注意点与局限

* **负控不是零**：`cv = 0` 时估计器精确返回 0（见 I0），但带强度散度时负控给出显著负斜率。原因是耦合反馈：强度越高的浮冰回弹越强、峰值接触面积越小，于是 `corr(ln A, ln p) < 0`（图 4）。因此地板是**物理内生性**，不是回归偏置；它只能被更大的信号压过去，不能被「更好的估计器」消掉。
* **衰减是已知偏差**：箱内残余混杂把条件化斜率继续拉向 0，所以 `both` 值随分辨率单调变大（图 1），其无偏极限是机制值 `-n`。任何与 -0.28 的吻合都必须检查是否来自衰减（I2 的 caveat / I3）。
* 样本量封顶了条件化深度：16x16 = 256 格、每格约 5 点，再细分已不可估计；要逼近无偏极限需要更多种子（建议 5 -> 25）。
* `V_only` 变体把闭合速度（6DOF 响应的后代）当混杂量条件化，会把负控推离 0，因此**不可用于物理结论**；本单保留了它作为反例。
* `n` 是声明输入，不是本链测得的材料常数；处理臂「恢复 -0.28」只在 `n = 0.28` 时成立，该事实本身是核心结论而非调参结果。
* ISO 19906 数值仍 **WITHHELD**：工作树内无原文，-0.28 是 W22/W23 链的**声明参考口径**，不是标准转录，不得表述为「ISO 值」；判据 2/5 的 WITHHELD 状态因此保持不变（需要的条款/表/页码无法在无原文条件下补齐，补写等同于伪造）。
* 接触仍是 W23 的点质量浮冰 + 塑性/回弹理想化，碎冰层未解析。

## 6. 误差判定（是否 <3%）

| 关键量 | 参考 | 实得 | 相对误差 | <3%? |
|---|---|---|---|---|
| 闭合臂条件化 ex | 施加的 -0.28 | -0.2800 | 0.00 % | 是 |
| 处理臂条件化 ex（16x16）| 声明参考 -0.28 | -0.2606 | 6.93 % | 否 |
| 处理臂条件化 ex（16x16）| 机制值 -n = -0.33 | -0.2606 | 21.03 % | 否 |
| 恒等式 `ex_pop = n(beta_h+beta_V-1)` | 0（代数恒等）| 0.000000000000 | 绝对残差 | 是 |

**逐项回答「是否 <3%」：**

* 闭合臂（真实面积律）条件化复原：误差见上表，是估计器有效性的直接量度；
* 恒等式残差：机器精度内为 0，说明「群体指数 = 材料指数 x 协方差结构」不是拟合巧合；
* 处理臂 vs 声明参考 -0.28：误差见上表（>3%）。该偏差的正源不是本链的辨识能力，而是 `n = 0.33` 与 0.28 的差（机制值 -0.33 与 -0.28 相差 17.9%）。因此**不能**声称「本链独立复现了 -0.28」，只能声称「本链把声明的机制指数按 `ex_pop = n(beta_h+beta_V-1)` 传递，且该机制指数可在条件化后分辨出地板」。

## 7. 图

![w31_exponent_vs_conditioning.png](benchmarks/ice_coupled_6dof/figures/w31_exponent_vs_conditioning.png)

![w31_pressure_area_scatter.png](benchmarks/ice_coupled_6dof/figures/w31_pressure_area_scatter.png)

![w31_variant_matrix.png](benchmarks/ice_coupled_6dof/figures/w31_variant_matrix.png)

![w31_control_endogeneity.png](benchmarks/ice_coupled_6dof/figures/w31_control_endogeneity.png)

