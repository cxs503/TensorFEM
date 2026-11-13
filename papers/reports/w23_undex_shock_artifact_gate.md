# TensorFEM#774 重开第3轮：UNDEX 冲击链制品完整性门（数值可复现性 + 两处溯源缺口记账）

# TensorFEM #774 重开（第 3 轮）：UNDEX 冲击链构件制品的完整性门

## ① 案例介绍

对象：水下爆炸（UNDEX）冲击下船体板架的水弹性响应链——**Cole 环境 → 板 FSI FEM → 冲击响应谱(SRS)**。
本单不改物理模型，交付的是这条链**已入库制品**（`case_matrix.csv` / `chain_metrics.json` / `srs_comparison.json` / 6 张图）的**可复现性闸门**（REAL 级门禁，非 MOCK）。

几何与材料（`benchmarks/undex_shock` 全链固定）：板 1.5 m × 1.0 m × 10 mm，钢 E=206 GPa、ν=0.3、ρ=7850 kg/m³（面密度 78.5 kg/m²，Kirchhoff f11=35.17 Hz）；海水 ρ=1025 kg/m³、c=1500 m/s（ρc=1.5375e6 Pa·s/m）；网格三级 (nx,ny,steps/θ)=(12,8,40)/(18,12,60)/(27,18,90)，r=1.5，Fs=1.25。
参数（Cole 经验律 `p_m=K(W^(1/3)/R)^A`、`p_i=p_m e^(-t/θ)`、`θ=K_θ W^(1/3)(W^(1/3)/R)^n`、`T=K_b W^(1/3)/(H+10.33)^(5/6)`）：
K=5.240e7 Pa、A=1.130、K_θ=8.4e-5 s、n=−0.24、K_b=2.11 s·m^(5/6)·kg^(−1/3)、大气头 10.33 m。锚定证据状态 **WITHHELD**（Cole 1948 原文不可达，参数是仓库自身声明，非抄录）。

制品当前覆盖的工况（锚声明 6 例，制品仅 4 例——见 ⑤）：

| case | W [kg] | R [m] | H [m] | W^(1/3)/R | Cole p_m [MPa] | theta [us] | T_b [s] | Cole 冲量 [Pa·s] |
|---|---|---|---|---|---|---|---|---|
| C1 | 5.0 | 12.0 | 40.0 | 0.142498 | 5.7961 | 229.27 | 0.137747 | 1328.89 |
| C2 | 10.0 | 18.0 | 40.0 | 0.119691 | 4.7593 | 301.22 | 0.173551 | 1433.57 |
| C3 | 25.0 | 25.0 | 40.0 | 0.116961 | 4.6368 | 411.08 | 0.235545 | 1906.10 |
| C4 | 50.0 | 30.0 | 40.0 | 0.122801 | 4.8993 | 511.91 | 0.296768 | 2507.98 |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **读代码而非读结论**：先核对 `scripts/w23_undex_shock_chain_run.py` 的产物流水线（`cole_case_environment` → `panel_fsi_shock` → `srs_pair/srs_band_table` → `make_figures` → `report`），确认 `case_matrix.csv` 的每一列都能由**当前**库代码重算。
2. **建门禁**：新增 `tests/test_undex_chain_artifacts.py`（14 个断言组，12 passed + 2 xfail），不重跑 FEM，而是从当前代码/当前锚**反向推导**每一条记录：
   - 覆盖：矩阵/ SRS 的 case 集合 = `cole_anchor.json` 的 case 集合；`chain_metrics.json:rows` = CSV 行；
   - 数值：`reduced_range / p_m / θ / T_b / 冲量` 逐例用 `tensorfem.undex_chain` 重算（容差 1e-9 相对）；
   - 标签：`criterion1_pass`、`gci_pass`、`wall_peak_over_cole≡2`、`wall_peak_rel_dev_vs_cole≡1` 必须由原始偏差推出（禁止手改 PASS）；
   - 求解器轴：Taylor 闭式解、离散 1-D 声学 FE 腿（`taylor_fsi_fem`/`taylor_analytic`，120 元，dt=θ/60）**现场重跑**并与入库列比对 1e-9；
   - SRS：6 带定义、带内偏差 = |Sa_fem−Sa_ref|/Sa_ref 自洽、频率网格 1 Hz–10 kHz 单调；
   - 版图：面板时程（挠度/加速度/壁压/能量残差）必须能复原矩阵中的标量列；
   - 溯源：`anchor_sha256`、`script_sha256`、`parameters_used`（K/A/K_θ/n/T_b、大气头、5/6 指数）；
   - 图：6 张 PNG 均为真 PNG（校验签名 + IHDR 尺寸）、无孤儿图。
3. **把缺口记账而不是粉饰**：门禁把两处**真实缺口**写成 `benchmarks/undex_shock/artifact_integrity.json` 的 FAIL 条目，测试断言「缺口必须在台账里」才允许 xfail——台账被删或与事实不符，CI 立刻红。
4. **重计算走集群**：6 案例 × 3 级 FEM 重生成（≤10 min/节点）递交 `run_cluster`（作业 **cj2046**，stage_dir=本工作树），未占用 99 控制面。

## ③ 计算结果（程序化注入，零手抄）

**A. 数值可复现性（本单核心结论）**：制品覆盖的全部案例，其 Cole 侧五列（reduced range、p_m、θ、T_b、冲量）由当前库重算的最大相对偏差 = **0.0**（≤1e-9 容差内严格通过）；SRS 带内偏差由自身峰值对重推一致；离散流体腿三列（界面压力/速度/预空化冲量）现场重跑比对一致（1e-9）。**制品数字与当前 main 代码一致，未发生计算漂移。**

**B. 入库存档的求解器与守恒量**

| case | 壁压峰值 [MPa] | 峰值/2p_m(恒等式) | Taylor 压力误差 | Taylor 速度误差 | 离散流体腿压力误差 | 能量残差 | SRS 带内最大偏差 |
|---|---|---|---|---|---|---|---|
| C1 | 11.5922 | 2.0 | 2.88% | 4.44% | 50.60% | 5.25e-04 | 2.94% |
| C2 | 9.5185 | 2.0 | 2.91% | 4.18% | 50.46% | 6.89e-04 | 3.17% |
| C3 | 9.2736 | 2.0 | 2.95% | 3.96% | 50.34% | 9.41e-04 | 3.42% |
| C4 | 9.7985 | 2.0 | 2.92% | 3.77% | 50.27% | 1.17e-03 | 3.38% |

**C. 网格收敛（GCI，上限 10%）**

| case | GCI 挠度 | GCI 加速度 | GCI 预空化冲量 | GCI 壁压 | GCI 判定 |
|---|---|---|---|---|---|
| C1 | 0.00000 | 0.00412 | 0.09899 | 0.00000 | True |
| C2 | 0.00978 | 0.00412 | 0.00000 | 0.00000 | True |
| C3 | 0.01280 | 0.00412 | 0.00000 | 0.00000 | True |
| C4 | 0.01817 | 0.00412 | 0.00000 | 0.00000 | True |

**D. 制品完整性台账（`artifact_integrity.json`）**

| finding | status | detail |
|---|---|---|
| numeric_reproduction | PASS | every covered case re-derives from the current library code: reduced range, peak pressure, decay time, bubble period and impulse agree to <=1e-9 relative with tensorfem.undex_chain… |
| coverage_vs_anchor | FAIL | case_matrix.csv and srs_comparison.json cover C1-C4 only, while cole_anchor.json declares 6 cases (C5: W=2 kg R=25 m, C6: W=100 kg R=20 m, added by the chain-3 revision of the anch… |
| anchor_sha256 | FAIL | chain_metrics.json records anchor_sha256 = cf8159f22306a6b9b5d6cceb4d50a3b52c6c5102c9ce513c068cccaf6e3b34ce, but the committed cole_anchor.json hashes to 63391daf1c80dbf82d0de609a1… |
| script_sha256 | FAIL | chain_metrics.json records script_sha256 = 4f2133984ff9106868f5fa6ca482d3917538a29e3d90417c5e698f6b2f4e0dae, but the committed scripts/w23_undex_shock_chain_run.py hashes to c47a4e… |
| figures | PASS | all six recorded figures exist in docs/assets/w23_undex_shock, are valid PNGs with real IHDR dimensions and are named in the mission report; none is orphaned. |

## ④ 结果图（均为入库制品的原图内嵌，非新绘）

- `fig1_cole_environment.png`：6 例（制品 4 例）Cole 脉冲 p_m·e^(−t/θ) 与 40 m 深气泡周期；
- `fig2_wall_pressure.png`：C2 壁压 vs **Taylor 解析解**、刚性壁 2p_i、Cole 自由场——与解析解比较曲线；
- `fig4_srs.png`：C2 SRS（ζ=2%）vs Taylor 刚性板 SRS，竖线为 6 个频带边界；
- `fig5_energy_ledger.png`：入射/反射/板能量与相对残差（离散能量恒等式）。

**为什么没有场量云图**：本单交付对象是「一维板-声学耦合链的制品完整性」，板响应量按中心点/面积积分归档（`srs_comparison.json:panel_fem_curves`），制品本身不含节点场数据；若需云图，须重生成面板全场（不在 10 文件预算内），故如实声明缺失而非用示意图充数。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **两处真实缺口（记账未清除）**：①制品只覆盖 C1–C4，锚已扩到 6 例（C5 W=2 kg/R=25 m，C6 W=100 kg/R=20 m），读者从链制品只能看到 2/3 的准则 2 证据；②`chain_metrics.json` 记录的 `script_sha256=4f2133…` 与仓库内脚本 `c47a4e0…` 不符（跑完后有过一次重排版提交），且 `anchor_sha256` 指向已被 chain-3 修订（schema 1.0.0→1.1.0）替换的旧锚 → 溯源链断了。清除方式已写进台账：重跑 `scripts/w23_undex_shock_chain_run.py`（已递交 cj2046，本分支收口时仍在排队），重生成后把台账翻成 FRESH、删掉两处 `pytest.xfail`。
2. **±15% 判定里有陷阱**：壁压峰值 vs Cole 自由场恒为 +100%（t=0 板未动，`p_wall=2p_i`），vs 2p_m 恒为 0%——后者是**同义反复**，绝不能当验证引用；真正能失败的是预空化冲量（16.5–27.4%，全 FAIL，属模型类别问题）与离散流体腿（50.3–50.6%，FAIL，属线性元色散）。
3. **锚是 WITHHELD**：K/A/K_θ/n/K_b 是仓库声明的实现参数，不是 Cole(1948) 的抄录；「重算一致」只证明抄录/算术自洽，不证明任何物理量值。
4. 声学耦合是局部阻抗、法向入射的精确解，无衍射/球面扩散/加强筋；空化未建模（壁压可到 −1.12 MPa 张应力，空化起始 99.4–119.4 μs 之后非物理）。
5. 门禁的 1e-9 容差只保证「制品 ↔ 当前代码」不漂移；**改代码必须重生成制品**，否则门禁按设计变红。

## ⑥ 误差判定（逐项 vs 基准，显式回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| Cole 环境 5 列（4 例） | 当前库重算 | 0.0 %（≤1e-9） | **是**（但仅实现一致性，非物理验证） |
| 壁压峰值（板 FEM） vs Taylor 闭式解 | 解析解 | 2.88–2.95 % | **是**（2.95% < 3%） |
| 中心速度 vs Taylor 闭式解 | 解析解 | 3.77–4.44 % | **否**（>3%，4 % 量级；<15 % 可接受带内） |
| 刚性极限耦合冲量（加强板 vs 2×Cole 入射） | 解析恒等式 | 0.22–0.63 % | **是** |
| 离散 1-D 声学 FE 腿 界面压力 | Taylor 闭式解 | 50.27–50.60 % | **否**（已如实记录 FAIL：线性元色散） |
| 壁压峰值 vs Cole 自由场 p_m | 自由场 p_m | 100.0 %（结构恒等式） | **否**（不构成求解器判据） |
| 预空化冲量 vs Cole 入射冲量 | Cole 入射 | 70.66–84.52 % | **否**（模型类别：空化截断 + 柔性反射） |
| SRS 带内最大偏差 vs Taylor 刚性板 | 解析解 | 2.94–3.42 % | **否**（柔性效应，非求解误差；部分带 <3%） |

**结论**：求解器轴（板 FEM vs Taylor 解析解压力 2.88–2.95%）满足 <3%；速度误差、离散流体腿、自由场/冲量三条不满足，且均已在制品与 README 中如实标为 FAIL 并给出物理原因，无粉饰。本单新增能力是**制品可复现性闸门**：它把「数字是否还属于这段代码」变成 CI 可判定的问题，并把两处溯源缺口钉进台账。


## Figures

* `docs/assets/w23_undex_shock/fig1_cole_environment.png` -- 图1 Cole 脉冲与气泡周期（6 例锚，制品覆盖 4 例）
* `docs/assets/w23_undex_shock/fig2_wall_pressure.png` -- 图2 C2 壁压 vs Taylor 解析解 / 刚性壁 2p_i / Cole 自由场
* `docs/assets/w23_undex_shock/fig4_srs.png` -- 图3 C2 SRS（ζ=2%）vs Taylor 刚性板，竖线为频带边界
* `docs/assets/w23_undex_shock/fig5_energy_ledger.png` -- 图4 C2 能量账：入射/反射/板能量与相对残差
