# 锚点 `fem_cylinder_radiation`：圆柱模态辐射效率的独立验证证据（m16372）

本页由 `scripts/gen_cylinder_radiation_doc.py` 从锚点证据文件程序化生成（零手抄）。

- 生成源：`benchmarks/verified/fem_cylinder_radiation/result.json`、`gates_verified.json`
- 闸门总数：15，PASS 15，锚点裁定 `passed = true`
- 环境：n/a / python n/a / torch n/a

## 闸门明细（引用式：每条闸门的基准公式与阈值见 manifest）

| gate | 参考式 | 值 | 阈值 | 判定 |
|---|---|---|---:|---|
| `G1_piston_law` | sigma -> (pi/2) k R  (Junger & Feit 1992, low-ka piston law) | 0.000521904 | 0.005 | **PASS** |
| `G2_high_ka_plateau` | sigma_0 -> 1, sigma_n -> 1/2  (high-ka plateau, <cos^2 n theta> = 1/2) | 0.00520109 | 0.02 | **PASS** |
| `G3_evanescent_cutoff` | evanescent decay for k_x > k (Hankel exponential decay) | 3.00398e-16 | 0.001 | **PASS** |
| `G4_window_closure` | window closure: Pi = (1/2pi) int G_n(k_x)\|V(k_x)\|^2 dk_x | 3.99603e-07 | 0.02 | **PASS** |
| `G5_window_convergence` | sampling 2001->8001, band 8->16 half-waves | 1.18756e-08 | 0.005 | **PASS** |
| `G6_esm_pulsating_sphere` | Pi = 2 pi a^2 rho c V^2 (ka)^2 / (1 + (ka)^2)  (pulsating sphere, exact) | 1.67284e-05 | 0.01 | **PASS** |
| `G7_compact_body_asymptote` | monopole + dipole asymptote, dipole = 0 for breathing sphere | 0.01 | 0.02 | **PASS** |
| `G7b_dipole_null` | D = 0 by symmetry (v_n uniform) | 6.04212e-17 | 1e-12 | **PASS** |
| `G8_rigid_body_null` | rigid body radiates nothing | 0 | 1e-30 | **PASS** |
| `G9_velocity_squared_scaling` | Pi ~ \|V\|^2 (linearity of the exterior problem) | 0 | 1e-09 | **PASS** |
| `G10_axial_spectrum_closed_form` | q (1 - (-1)^m e^{-i k_x L}) / (q^2 - k_x^2),  q = m pi / L | 2.03741e-11 | 0.0001 | **PASS** |
| `G11_mode_identification` | Fourier decomposition isolates the injected mode | 1 | 0.95 | **PASS** |
| `G11b_dominant_pair` | planted harmonic (2, 3) | 0 | 0 | **PASS** |
| `G12_grid_lateral_area` | lateral surface area of the cylinder | 2.82716e-16 | 1e-12 | **PASS** |
| `G12b_grid_caps_zero` | caps carry zero area weight | 0 | 1e-14 | **PASS** |

## 关键量的收敛/一致性数字（result.json 原值）

| 量 | 值 |
|---|---:|
| `window_closure_ratio` | 1.0000003996 |
| `window_coarse_over_fine` | 1.00000001188 |
| `velocity_scaling_ratio` | 4 |
| `mode_share` | 1 |
| `dominant_pair` | (2, 3) |
| `piston_law_errors` (ka = 1e-4, 1e-3, 1e-2) | 9.826e-08, 7.524e-06, 5.219e-04 |
| `esm_sphere_ratios` (ka = 0.5, 1, 2, 3) | 0.999983272, 0.999985474, 0.999991307, 0.999998770 |

## 图

| 文件 | 内容 |
|---|---|
| `figs/efficiency_vs_ka.png` | 各周向阶 `n` 的效率 `sigma(ka)` vs 活塞律/平台渐近 |
| `figs/window_closure_convergence.png` | 窗口闭合的谱采样收敛（sigma/sigma(8001)） |
| `figs/esm_pulsating_sphere.png` | ESM 等效源 vs 脉动球解析功率 |
| `figs/evanescent_decay.png` | `k_x > k` 倏逝段电导塌陷（含 `k_x = k` 截断点） |

## 本轮修复（m16372）

- `infinite_cylinder_conductance` 在**精确轴向截断** `k_x = k` 处原先返回 `nan`（`H_n^(1)(0)` 与 `kappa = 0` 的 0/0），并使 `evanescent_decay` 图静默带一个 `nan` 点；现按物理极限（掠射、无辐射功率）返回 `0.0`，与 `k_x > k` 侧连续。回归测试 `tests/test_verified_cylinder_radiation_anchor.py::test_axial_cutoff_returns_zero_not_nan`。
- 锚点自此具备完整证据链（`result.json` + `gates_verified.json` + 图 + README + 独立复算测试），不再是「只有 driver、无证据」的半成锚点。

## 注意点与适用范围

- 空气介质（`rho = 1.225`、`c = 343`）：未含流体加载/附加质量，结构-声耦合的湿模态另见 `fsi_wet_modal`。
- 验证对象是**给定表面速度场**的辐射算子，不是壳体特征值问题本身。
- 窗口闭合是轴向谱一维求积，继承无限长柱闭合；有限长带端盖柱的端部效应对低 `ka` 或低阶 `m` 仍是开口问题（`docs/reports/fem-structural-acoustics.md` 的 C3/C4 诚实 FAIL 即此）。
