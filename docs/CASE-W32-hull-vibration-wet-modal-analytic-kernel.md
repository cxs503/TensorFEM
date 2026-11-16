# CASE W32 · 湿模态附加质量 vs 解析（hull-vibration-analytic-kernel）

**领域**：振动噪声 · **锚**：A11（四边简支板 Navier 一阶频率，`anchors.json` 直读）
**工件**：`artifacts/w32-hull-vibration-analytic-kernel/`（`wet_modal.json` 为主工件；论文线归档副本
`papers/data/w32-2026-11-16/vibration/wet_modal.json`）
**生成脚本**：`scripts/w32_wet_modal_analytic_kernel.py`（判据数字全部由 anchors.json 与本次实跑输出直读，零手抄）

## 新增能力

`src/tensorfem/wet_plate_fe_modal.py`

* `solve_wet_plate_fe_modal` —— **耦合**湿模态：干 Mindlin Q4 特征对 + 非局部 Rayleigh 单层势
  流体算子（`hydro_added_mass.dipole_added_mass_matrix`）在干模态基下组成广义特征问题
  `(diag(omega_i^2 g_i) - omega^2 [diag(g_i) + A]) x = 0`，`A = Phi^T M_hydro Phi`。
  湿频率是解出来的，不预设振型、不预设解析附加质量；同时返回干/湿振型与板面流体模态反力场，
  并给出 `mu_modal_diag`（对角附加质量比）与 `collapsed_modes`（模态穿越计数）。
* `piston_added_mass_per_area` —— 平板/等效活塞（ellipsoid-piston）近似核。
* `prolate_spheroid_transverse_added_mass` —— 长椭球横向附加质量（椭球近似核）。
* `wet_plate_modal_convergence` / `plate_fluid_mesh_convergence` —— 结构网格与流体板网格分别收敛。

## 结论要点（数值见工件与判据表）

1. 板干湿频率比：耦合 FE vs 精确挡板 Rayleigh 积分在全部 24 个模态样本上一致（主判据达标）；
   模态附加质量对角项与精确解析 mu 同量级一致，湿/干振型 MAC 接近 1，说明
   「湿振型≈干振型 + 只取对角附加质量」在板类结构上成立。
2. `rho_f/k_mn` 渐近律在低阶板模态（k_mn·a = O(1)–O(10)）显著低估计加质量，不可用于换算。
3. 平板/活塞近似核与模态无关，只在基阶模态附近可用，高阶模态误差迅速放大。
4. 圆柱壳：Junger & Feit 解析附加质量与有限差分势流独立解一致；长椭球近似系统性偏低
   （细长极限为二维精确值的 2/3），只能作量级估计。
5. 已知限制：`a/h = 1000`（h = 0.01 m）的板会触发 `plate_vibration` 的机制检测
   （SRI / MITC 均触发），因此 FE 湿模态系综取 h ∈ {0.02, 0.04, 0.08} m；
   该厚度由解析锚 A11 单独覆盖。此为库级限制，已记录在工件报告的「注意点」中。

## 验证

* `pytest tests/test_wet_plate_fe_modal.py`（14 项）：椭球/活塞解析极限、干 FE vs Navier、
  对角附加质量 vs 精确 Rayleigh 积分、湿/干频率比 vs 单模态换算、流体网格无关性。
* 工件内判据表（`criteria_table.{json,csv,md}`）逐条机判 PASS / honest FAIL，未达标项如实记录。