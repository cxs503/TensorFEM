# W22 · 船体梁 ULS 衡准：IACS CSR 逐字核销尝试与 Smith 渐进崩溃 M_u

> 仓库 TensorFEM · 任务 mission 14512 · 学科域：结构（船体梁总纵强度）  
> 工件：`docs/w22_hull_girder/uls_utilization.json`（sha256 由 runner 记录）、`benchmarks/anchors/iacs-csr-w22.json`  
> 运行标识：git `5fb5781f02f1` · module sha256 `a9114d8fa194cef1` · anchor sha256 `a9e1cf367dee5e32`  
> runner `benchmarks/w22_hull_girder_uls.py` sha256 `c72d3c9a684aa08e`

## 0. 结论摘要（先给判决，再给证据）

| 判据 | 量值 | 阈值 | 判决 | 证据指针 |
|---|---|---|---|---|
| C1 锚件完整率 | 0%（0/6 条 VERBATIM） | 100% | **FAIL** | `benchmarks/anchors/iacs-csr-w22.json` |
| C2 ≥3 剖面利用率 ≤1.0 | n=4，max U=0.627 | n≥3 且 U≤1.0 | **PASS** | `docs/w22_hull_girder/uls_utilization.json` |
| C3 M_u 收敛（fiber/步数加倍） | 0.18% | <2% | **PASS** | `docs/w22_hull_girder/uls_utilization.json` |
| C4 与解析式交叉验证 | 单元级 Perry-Robertson 1.13%；剖面级 vs 限值应力模型 67.21%（补充项） | <15% | **PASS**（附补充项 FAIL，见 §5.3） | `docs/w22_hull_girder/uls_utilization.json` |

**直答科研问题**：IACS CSR 船体梁 ULS 衡准在本环境**不能**逐字核销——规则文本（CSR BC&OT 2019 Pt.1 Ch.5 / UR S7）无法取得，6 条因子全部 BLOCKED（§2）。能核销的是**不依赖规则文本的那一半**：Smith 渐进崩溃法算得的 M_u 与 CSR 包络 (M_sw + M_wv) 的比值。四个船中剖面在**单位分项安全因子**下利用率 0.394–0.627（全部 ≤1.0），对应的可容许分项安全因子 γ_adm = M_u/(M_sw+M_wv) 为 1.60–2.54；即只有当 CSR 实际因子乘积 ≤ 该值时才合规。这个数**不是** CSR 合规结论（γ 值缺锚）。

## 1. 案例介绍：物理背景、几何与参数

船体梁总纵强度是船舶总体性能的“骨架”学科：把整船当作变截面薄壁梁，中剖面在总纵弯曲下由甲板/船底板架与纵骨先弹性、再屈曲、最后崩溃。IACS CSR 的 ULS 衡准把这个过程压缩成一个不等式 `γ_s·M_sw + γ_w·M_wv ≤ M_u`（弯矩），屈服衡准则为 `σ = M/Z ≤ σ_y/γ_R`。本单要回答两件事：(a) 该衡准能否逐字核销；(b) 用真正的 Smith 渐进崩溃法（构件曲线会软化、每增量步保持轴力平衡）算出的 M_u 与 CSR 允许值差多少。

中剖面参数（**声明型工程输入**，不是锚值；Cb 由截面面积曲线数值反算校验）：

| case | L_pp (m) | B (m) | D (m) | T (m) | Cb (输入) | Cb (积分) | L/B | B/T | W_deck (m³) | W_bottom (m³) | t_deck (mm) | UR S11 W_min (m³) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kvlcc2 | 320.00 | 58.00 | 30.00 | 20.80 | 0.8095 | 0.809500 | 5.52 | 2.79 | 96.38 | 132.62 | 27.9 | 96.38 |
| capesize_bc | 290.00 | 45.00 | 25.00 | 18.50 | 0.8300 | 0.830000 | 6.44 | 2.43 | 62.06 | 86.38 | 24.3 | 62.06 |
| aframax_tanker | 250.00 | 44.00 | 22.00 | 14.90 | 0.8200 | 0.820000 | 5.68 | 2.95 | 44.36 | 67.10 | 18.0 | 43.46 |
| panamax_bc | 225.00 | 32.26 | 20.00 | 14.00 | 0.8300 | 0.830000 | 6.97 | 2.30 | 30.48 | 43.50 | 18.0 | 25.24 |

几何校验：全案例 Cb 反算误差 ≤ 1.37e-16（阈值 1e-3），甲板板厚按 UR S11 最小中剖面模数闭合（其余板厚按 √(L/320) 从 KVLCC2 模板缩放，缩放律见模块文档）。材料统一 σ_y = 235 MPa（模板值，缓和钢）。

## 2. 外部锚：逐字核销尝试（C1 的证据）

锚件 `benchmarks/anchors/iacs-csr-w22.json`（sha256 `a9e1cf367dee5e32`）状态 **NOT_OBTAINED**，完整率 0%。需要逐字抄录的 6 条因子与获取尝试：

| 因子 | 角色 | 条款定位（任务书指派，未核验） | 状态 |
|---|---|---|---|
| `gamma_s` | partial safety factor on the still-water bending moment M_sw | IACS CSR BC&OT (2019) Pt.1 Ch.5, hull-girder ULS load combination | **BLOCKED** |
| `gamma_w` | partial safety factor on the vertical wave bending moment M_wv | IACS CSR BC&OT (2019) Pt.1 Ch.5, hull-girder ULS load combination | **BLOCKED** |
| `gamma_R` | resistance factor in the yield criterion sigma = M/Z <= sigma_y/gamma_R | IACS CSR BC&OT (2019) Pt.1 Ch.5, yield / permissible-stress check | **BLOCKED** |
| `M_u` | definition of the ultimate vertical bending moment and how the rule expects it to be obtained | IACS CSR BC&OT (2019) Pt.1 Ch.5, ultimate strength; IACS UR S7 (2010) minimum longitudinal strength standard | **BLOCKED** |
| `M_wv` | rule vertical wave bending moment (hogging and sagging) and its distribution factors | IACS CSR BC&OT (2019) Pt.1 Ch.5 wave-load formula (equivalently IACS UR S11) | **BLOCKED** |
| `W_min` | minimum midship section modulus used to close the scantlings before the ULS check | IACS UR S7 (2010) minimum longitudinal strength standard | **BLOCKED** |

获取尝试（真实命令与真实结果，全部记录在锚件 `acquisition_attempts`）：

- **local filesystem corpus** — `find / -maxdepth 6 -iname '*csr*' -o -iname '*iacs*' (excluding /proc, /sys and this worktree)` → no rule text; hits are the repository's own hull-girder code (src/tensorfem/hull_girder_csr_chain.py, docs/hull-girder-csr-bending-chain.md, docs/assets/hull-girder-csr/) and unrelated csrc directories
- **local PDF corpus** — `find / -maxdepth 5 -iname '*.pdf'` → no IACS publication among the PDFs found (tooling manuals and unrelated papers only)
- **repository / platform corpus** — `grep -rIl -i 'common structural rules' /root /opt /srv /data (excluding this worktree)` → no match: no verbatim text, table extract or page locator for CSR Pt.1 Ch.5 or UR S7
- **network** — `urllib GET https://iacs.org.uk/ , https://iacs.org.uk/resolutions/unified-requirements , https://www.iacs.org.uk/download/10936 (timeout 6 s); direct curl to external hosts is blocked by the execution sandbox policy` → HTTP 403 Forbidden from the sandbox proxy for every IACS URL; no retrievable full text. IACS rule text is a controlled publication, so even a successful anonymous fetch would not be an authorised copy for verbatim transcription
- **operator-supplied copy** — `checked benchmarks/anchors/sources/ for an operator-deposited transcription` → only sname_tr_bulletin_3-37_4-7.transcription.json (different topic); no CSR deposition

为什么不用记忆值：IACS CSR 的 γ_s/γ_w/γ_R 属受控出版物，不同版本、不同船型（BC vs OT）、不同载荷组合的表值不同；无授权副本时任何“记忆中的数”都会让利用率看起来像合规结论。因此本单**不写任何 γ 值**，改报 γ_adm = M_u/(M_sw+M_wv) ——把“规则说了算”的那一半留给规则文本，把“算得出来”的那一半交给 Smith 求解器。仓库内已有 `CSR_GAMMA_M = 1.43`、`CSR_PERMISSIBLE_STRESS = 220 MPa` 等常量，已登记为 `UNCITED_CANDIDATE`（不得用于合规声明），本单未使用。

## 3. 怎么做的：方法、流程、关键设置

1. **几何先行**：每个案例用 `HullForm` 反算 Cb（误差 1.4e-16），通过后才进入载荷与强度。
2. **剖面板厚闭合**：从仓库已文档化的 KVLCC2 中剖面模板按 √(L/320) 缩放板厚与纵骨面积，甲板板厚二分闭合到 UR S11 最小模数（记录闭合历史）。
3. **构件本构（Smith 第 1 要素）**：每个板架组拆成板格 `PlateStrip`（含后屈曲软化）与纵骨柱 `TeeStiffener`（Perry-Robertson 柱 + 局部屈曲），按**面积加权**合成面板曲线，并以较弱机制为上限——`tensorfem.hull_girder_smith_collapse.interaction_law` 的逐点带符号最小值实际保留了**较强**机制（§5.1 缺陷），故本单自行实现 `panel_component_law`。
4. **增量剖面求解（Smith 第 2 要素）**：用仓库既有引擎 `solve_moment_curvature`（每增量步对中和轴做轴力平衡二分，曲率网格二次加密到 8κ_y），得到可峰值后下降的 M-κ 曲线，取峰值即 M_u（每个方向独立算 sagging/hogging）。
5. **能量账本（Smith 第 3 要素）**：引擎同时给 ∫M dκ 与 ΣA·U 的残差，作为路径自洽性检查。
6. **载荷侧**：M_sw 由“均质满载”重量分布 + 浮力 + 吃水/纵倾平衡 → 简支梁双积分取得中剖面弯矩；M_wv 取本仓库实现的 UR S11 闭式（0.19/0.11 系数），并用物理设计波（h_w=13.5 m，Smith 修正）做量级互检。
7. **收敛与交叉验证**：纤维数 8→16、曲率步 320→640 各自独立加倍；单元级与解析 Perry-Robertson 对比；剖面级与仓库另一套“限值应力”增量模型对比；另做“关闭软化”的消融与纵骨几何 ±25% 敏感度。

关键设置表：

| 设置 | 值 |
|---|---|
| strips | 8 |
| curvature_steps | 320 |
| convergence_strips | 16 |
| convergence_steps | 640 |
| maximum_curvature_ratio | 8.0 |
| yield_stress_Pa | 235000000.0 |
| 构件组数（每剖面） | 6 组，深构件按 8（收敛 16）条离散 |
| 曲率网格上限 | 8.0 κ_y（含峰后段） |

## 4. 计算结果（数字全部由工件注入）

### 4.1 ULS 利用率（单位分项安全因子，显式标注）

| case | M_sw (GN·m) | M_wv,sag (GN·m) | M_wv,hog (GN·m) | M_u,sag (GN·m) | M_u,hog (GN·m) | U_sag | U_hog | γ_adm,sag | γ_adm,hog | 判决(≤1.0) |
|---|---|---|---|---|---|---|---|---|---|---|
| kvlcc2 | 0.545 | 10.601 | 9.820 | 22.373 | 15.670 | 0.498 | 0.627 | 2.007 | 1.596 | PASS |
| capesize_bc | 0.321 | 6.827 | 6.397 | 14.780 | 10.443 | 0.484 | 0.613 | 2.068 | 1.633 | PASS |
| aframax_tanker | 0.186 | 4.780 | 4.454 | 11.079 | 7.548 | 0.448 | 0.590 | 2.231 | 1.695 | PASS |
| panamax_bc | 0.105 | 2.776 | 2.601 | 7.314 | 5.204 | 0.394 | 0.500 | 2.539 | 2.001 | PASS |

说明：M_sw 为单一均质满载工况的中剖面弯矩（有符号，正=中拱），包络按同号叠加取 max(M_sw,0)+M_wv,sag 与 max(−M_sw,0)+M_wv,hog；UR S11 闭式下 M_wv,sag > M_wv,hog（0.11(Cb+0.7) > 0.19Cb 当 Cb < 0.96），这是系数本身的性质，不是符号错误。

### 4.2 屈服衡准（σ = M/Z，γ_R 缺锚故报可容许值）

| case | σ_deck, hog (MPa) | γ_R,adm = σ_y/σ | σ_bottom, sag (MPa) | γ_R,adm = σ_y/σ |
|---|---|---|---|---|
| kvlcc2 | 101.9 | 2.306 | 84.0 | 2.796 |
| capesize_bc | 103.1 | 2.280 | 82.7 | 2.840 |
| aframax_tanker | 100.4 | 2.340 | 74.0 | 3.175 |
| panamax_bc | 85.4 | 2.753 | 66.2 | 3.549 |

### 4.3 收敛（C3）

| case | 纤维加倍 ΔM_u/M_u | 曲率步加倍 ΔM_u/M_u | max |
|---|---|---|---|
| kvlcc2 | 0.14% (hog) / 0.18% (sag) | 0.00% (hog) / 0.02% (sag) | 0.18% |
| capesize_bc | 0.09% (hog) / 0.16% (sag) | 0.00% (hog) / 0.09% (sag) | 0.16% |
| aframax_tanker | 0.09% (hog) / 0.15% (sag) | 0.00% (hog) / 0.04% (sag) | 0.15% |
| panamax_bc | 0.10% (hog) / 0.17% (sag) | 0.03% (hog) / 0.00% (sag) | 0.17% |

### 4.4 交叉验证与消融（C4）

| case | Smith 合成峰值 vs 解析 PR | 解析 PR vs TeeStiffener 实现 | 剖面级 vs 限值应力模型 (hog) | 同参数柱限值模型 (hog) | 关闭软化后 ΔM_u (hog) | M_u/M_p (sag / hog) | 纵骨几何 ±25% 摆幅 (hog) |
|---|---|---|---|---|---|---|
| kvlcc2 | 0.98% | 0.00% | 66.53% | 79.39% | 24.78% | 0.773 / 0.541 | 17.56% |
| capesize_bc | 0.92% | 0.00% | 62.97% | 76.76% | 25.11% | 0.782 / 0.552 | 14.21% |
| aframax_tanker | 1.13% | 0.00% | 67.21% | 79.38% | 24.83% | 0.804 / 0.548 | 13.26% |
| panamax_bc | 0.95% | 0.00% | 64.67% | 77.45% | 26.01% | 0.767 / 0.546 | 14.58% |

能量账本残差与峰值形态：

| case | ledger 残差 (sag/hog) | 峰在内部 (sag/hog) | 首次屈服 M/M_u (hog) | 设计波/规则波弯矩比 |
|---|---|---|---|---|
| kvlcc2 | 1.46e-05 / 1.50e-05 | True / True | 0.462 | 0.955 |
| capesize_bc | 1.92e-05 / 1.42e-05 | True / True | 0.465 | 0.976 |
| aframax_tanker | 4.12e-05 / 1.31e-05 | True / True | 0.463 | 1.027 |
| panamax_bc | 1.71e-05 / 1.44e-05 | True / True | 0.473 | 1.046 |

### 4.5 结果图

![moment_curvature_and_utilization.png](figures/moment_curvature_and_utilization.png)

*Smith incremental moment-curvature for the four midship sections (top) and the unit-factor ULS utilization against the 1.0 limit (bottom).*

![capacity_comparison.png](figures/capacity_comparison.png)

*Ultimate capacity per case: Smith M_u (sagging/hogging), the repo's limiting-stress incremental model (hogging) and the fully plastic moment M_p.*

![component_curves.png](figures/component_curves.png)

*Smith component stress-strain curves of the KVLCC2 midship panel groups: post-buckling softening in compression, elastic-perfectly plastic in tension.*

## 5. 注意点：踩过的坑、缺陷与局限

### 5.1 已登记缺陷：`interaction_law` 保留“较强”机制（非保守）

`tensorfem.hull_girder_smith_collapse.interaction_law` 取各曲线的逐点**带符号最小值**。压缩侧应力为负，最小值=压缩幅值最大者，即它保留了**较强**的机制、丢掉了柱屈曲上限；而且板格律按板面积归一、柱律按板+纵骨面积归一，两者直接逐点取极值量纲不一致。
实测（deck 板架，`tests/test_hull_girder_csr_w22.py::test_interaction_law_defect_is_reproduced`）：板格峰值 234.1 MPa、柱限值 159.0 MPa，`interaction_law` 合成峰值 234.1 MPa —— 柱限值被完全抹掉（本单修正后 158.4 MPa）。本单因此自行实现面积一致、以弱机制为上限的 `panel_component_law`；该缺陷对仓库既有 ISSC2009 双底算例同样生效（其 girder web 用到该函数），登记为待修项，未在本单改动共享代码以免静默改变他人基线。

### 5.2 软化是 M_u 的主导项，不可省略

同一求解器、同一构件强度，仅把压缩幅值改为“峰值平台”，M_u 抬高 24.8–26.0%。这正是仓库既有“限值应力”增量模型（无软化分支）比 Smith 高 63–67% 的原因：**该模型对 M_u 非保守**。本单据此不把剖面级差异写成“验证通过”。

### 5.3 C4 的判决口径（偏差声明）

判据原文允许“与解析/简化公式（如 CSR 附录法或 Perry-Robertson 单筋）交叉验证差异 <15%”。本单把**具名方法（Perry-Robertson 单筋）的单元级比对**作为判据量：全案例偏差 ≤1.13%，判 PASS；剖面级与另一套模型差 63–67%，作为**补充项**单列并明确标 FAIL。若评审认定剖面级比对才是判据，则 C4 = FAIL，结论“Smith M_u 显著低于限值应力模型”不变。此口径在工件 `criteria[3].deviation_disclosure` 中披露。

### 5.4 其他局限

- CSR Pt.1 Ch.5 partial safety factors gamma_s/gamma_w/gamma_R not transcribed in this environment: utilization reported at unit factors plus the admissible factor M_u/(M_sw+M_wv).
- Scantlings are a parametric CSR-consistent reconstruction scaled from the documented KVLCC2 template; not a vessel's approved drawings.
- Single homogeneous-loaded still-water condition used for all cases (tanker weight fractions); a bulker loading matrix is out of scope.
- All members at sigma_y = 235 MPa (mild steel template value); high-tensile deck plating would raise M_u.
- Plane sections, no shear lag / torsion / warping / tripping.
- 纵骨以等效 T 形代表（模板只给面积不给型材），几何 ±25% 摆幅使 M_u 变化 12–18%，是 M_u 的主要模型不确定度来源之一。
- Aframax / Panamax 两案例甲板板厚落在 18 mm 下界（下界已满足 UR S11 最小模数），因此其实际板厚可能偏保守。

## 6. 误差判定（是否 <3%？逐项回答）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| Smith 面板合成峰值压缩应力 | 解析 Perry-Robertson（单筋，同 η） | 1.13% | 是 |
| 解析 PR 实现 vs 模块 Perry-Robertson | 同一闭式 | 0.00% | 是（同式自洽，非独立验证） |
| M_u（纤维/步数加倍） | 自身加密解 | 0.18% | 是 |
| 静水平衡（力/矩残差） | 0 | 6.4e-15 / 3.8e-14 | 是 |
| Cb（几何闭合） | 输入 Cb | 1.4e-16 | 是 |
| M_u | 仓库限值应力增量模型（无软化） | 62.97%–67.21% | **否**（模型差异，非数值误差；见 §5.2） |
| M_u 绝对值 | 实验/规范参考 M_u | **无可用基准（WITHHELD）** | 无法判定——本环境无 CSR 文本、无船体梁崩溃试验锚 |

诚实结论：**数值/实现层面全部 <3%（单元级 1.13%、收敛 0.18%、平衡 ~1e-14）**；**物理量绝对精度无法判定**，因为既没有规则文本给出的 M_u 允许值，也没有可抄录的崩溃试验锚——这部分按 WITHHELD 记，不用“看起来合理”冒充精度。

## 7. 复现

```bash
python benchmarks/anchors/iacs_csr_w22_anchor.py     # 重写锚件（0% verbatim，6 条 BLOCKED）
python benchmarks/w22_hull_girder_uls.py            # 4 案例研究 -> docs/w22_hull_girder/uls_utilization.json + figures
python benchmarks/w22_hull_girder_report.py         # 由工件渲染本报告（零手抄）
python -m pytest tests/test_hull_girder_csr_w22.py -q
```

## 8. 待办（本单之外）

1. **取得 CSR 文本后**：把 6 条 BLOCKED 因子逐字抄入锚件（VERBATIM + 条款/页码/印刷值串），C1 才能转 PASS，利用率才能称为 CSR 合规值。
2. **修 `interaction_law`**：改为面积加权 + 以弱机制为上限（本单 `panel_component_law` 可直接复用），并同步更新 ISSC2009 算例的基线数字（需独立任务与对照基线）。
3. **载荷侧升级**：用 CSR 的 M_sw 包络（多装载工况）替代单一均质满载工况；纵骨型材用真实 HP/T 型材库替代等效 T 形。
4. **锚定 M_u 绝对值**：接入公开的船体梁崩溃试验（如 ISSC 2009 双底、或已发表箱形梁崩溃试验），把 M_u 从“自洽”升级为“有外部真值”。
