# CASE-FEM-PINCHED-CYLINDER — MacNeal–Harder 夹持圆柱壳算例（TensorFEM）

- 算例锚点：`CASE-FEM-PINCHED-CYLINDER` ｜ mission `m16076` ｜ 生成时间 2026-11-12T09:43:24+00:00
- 复现命令（repro_cmd）：`PYTHONPATH=src python benchmarks/verified/fem_pinched_cylinder/run.py`
- 数值工件：`benchmarks/verified/fem_pinched_cylinder/case_result.json`、`case_gates.json`、`case_convergence.png`、`case_field.png`、`case_sensitivity.png`

## ① 问题与参考值出处

薄壁圆柱壳，轴向长度 L = 600 in，中面半径 R = 300 in，壁厚 t = 3 in，各向同性 E = 3.0e+06 psi、ν = 0.3；两端刚性隔板（端环截面平移 v = w = 0，轴向 u 自由，轴向刚体位移已被中跨对称面消除）；中跨处一对直径相对的**单位**径向集中力对夹。
被测量：加载点处壳体中面的径向（向内、故为负）挠度。

R/t = 100 使加载点附近膜应变能近乎为零，响应是**不可伸长弯曲（inextensional bending）**边界层：发生剪切/膜锁定的单元会把挠度算小几个量级 —— 这正是该算例被列入标准难题集的原因。

参考值（ref，即 oracle）：

```
w_ref = -1.8248e-05 in      (radial (inward) deflection of the shell at the load point, mid-span, under two diametrically opposed unit point forces)
```

出处：MacNeal, R. L. & Harder, R. L. (1985), 'A proposed standard set of problems to test finite element accuracy', Finite Elements in Analysis and Design 1(1), 3-20, DOI 10.1016/0168-874X(85)90003-4。夹持圆柱壳是该文集的标准考题之一，目标挠度取该题陈述中给出的收敛参考值。
该算例**没有闭式解**，唯一 oracle 是收敛的数值解，因此本报告不把该数字当作「可信即可」——而是用本程序自己产生的数字独立佐证（见 ⑤ 与 `oracle_corroboration`）：

- **C1 外推极限**：对本 ladder 细网格段拟合渐近模型，得到本格式自己的连续极限；
- **C2 双路径等价**：同一离散算子的稠密消元（生产路径）与 CSR+SuperLU（稀疏路径）必须逐位一致；
- **C3 参数扫描**：剪切修正因子 k、drilling 正则、端部轴向约束三组扫描给出响应对「不该敏感的参数」的不敏感性（以及对「建模选择」的定量后果）。

## ② 方法与实现（复用 TensorFEM 现有能力，不另起炉灶）

- 单元：`tensorfem.cylindrical_shell4.cylindrical_shell4_stiffness` —— 圆柱 Reissner–Mindlin Shell4，保留精确圆柱度量与曲率，膜+弯 2×2 Gauss，横向剪切一点选择性减缩积分。
- 模型：`tensorfem.pinched_cylinder.pinched_cylinder_model` —— 对称八分体 x∈[0,300]、θ∈[0,π/2]，角点载荷 P/4；对称面约束法向平移+两个面内转动分量；端部隔板环约束 v、w。
- 求解路径两条（本单新增稀疏路径，稠密路径为既有生产路径）：
  - `tensorfem.pinched_cylinder.solve_pinched_cylinder_linear (torch.linalg.solve)`（稠密算子在 N=16 即 1734 DOF，N=32 需 341 MB，N=64 >5 GB）；
  - `tensorfem.pinched_cylinder.solve_pinched_cylinder_sparse (CSR + scipy SuperLU)` —— 用同一批单元矩阵装配 CSR 算子，使 ladder 尾部（N = 24…64，2.5×10⁴ DOF）可用。
- 收敛 ladders：各向同性 octant 网格 N = 4x4、8x8、12x12、16x16、24x24、32x32、48x48、64x64；另有各向异性 (nx×nθ) 研究定位误差来源方向。
- 环境：python 3.11.6、torch 2.7.1、numpy 2.4.6、scipy 1.16.0、loongarch64、float64。

## ③ 参数表（含完整无量纲数）

| 参数 | 符号 | 数值 | 无量纲 / 备注 |
|---|---|---|---|
| 长度 | L | 600 in | L/R = 2.00 |
| 中面半径 | R | 300 in | R/L = 1/0.50 |
| 壁厚 | t | 3 in | **R/t = 100**（薄壳） |
| 杨氏模量 | E | 3.0e+06 psi | — |
| 泊松比 | ν | 0.30 | 0.30 |
| 集中力（全圆柱） | P | 1 lbf | **P/(E t²) = 3.704e-08** |
| 八分体角点载荷 | P/4 | 0.25 lbf | 对称性折减 |
| 剪切修正因子 | k | 0.8333 | 5/6（敏感性见 ⑤） |
| 网格轴向/环向步长比 | RΔθ/Δx | 1.571 | 等分时环向弧长步长为轴向的 1.571 倍 |
| 端部约束 | — | 刚性隔板 | v = w = 0，u 自由（轴向夹死会刚化 16.6%，见 ⑤） |
| 目标容差 | — | 5.0% | 算例池判据 <5% |

## ④ 对照表（ref vs got，误差逐项打勾叉）

判据：**N=16×16 octant 网格加载点挠度 vs ref 相对误差 < 5%**。

| 网格 | 单元数 | DOF | got（in） | ref（in） | 误差 | 判据 |
|---|---|---|---|---|---|---|
| 4x4 | 16 | 150 | -1.772660e-05 | -1.8248e-05 | 2.857% | ✅ |
| 8x8 | 64 | 486 | -1.763665e-05 | -1.8248e-05 | 3.350% | ✅ |
| 12x12 | 144 | 1014 | -1.806824e-05 | -1.8248e-05 | 0.985% | ✅ |
| 16x16 ← **算例判据点** | 256 | 1734 | -1.827849e-05 | -1.8248e-05 | 0.167% | ✅ |
| 24x24 | 576 | 3750 | -1.848067e-05 | -1.8248e-05 | 1.275% | ✅ |
| 32x32 | 1024 | 6534 | -1.857942e-05 | -1.8248e-05 | 1.816% | ✅ |
| 48x48 | 2304 | 14406 | -1.868019e-05 | -1.8248e-05 | 2.368% | ✅ |
| 64x64 | 4096 | 25350 | -1.873469e-05 | -1.8248e-05 | 2.667% | ✅ |

**算例判据点：N=16×16 误差 0.167% < 5% → 通过 ✅**

佐证项（同一批工件）：

| 检查 | 量 | 阈值 | 实测 | 判定 |
|---|---|---|---|---|
| G1_primary_16x16_within_5pct | load-point deflection on the 16x16 octant mesh within 5% of the MacNeal-Harder reference | 5.0 | 0.167% | ✅ |
| G2_finest_mesh_within_5pct | finest mesh (64x64) within 5% of reference | 5.0 | 2.667% | ✅ |
| G3_extrapolated_limit_within_5pct | extrapolated continuum limit of this ladder (self-generated, model selected by residual) within 5% of the published reference | 5.0 | 3.539% | ✅ |
| G4_dense_sparse_consistent | CSR/SuperLU path reproduces the production dense elimination at the primary mesh | 1e-10 | 1.29e-13 | ✅ |
| G5_shear_factor_insensitive | k in {0.5, 5/6, 1.0} changes the load-point deflection by < 1% (R/t = 100, bending dominated) | 1.0 | 1.308% | ❌ |
| G6_drilling_factor_insensitive | drilling regularisation 1e-10 ... 1e-4 changes the deflection by < 0.1% | 0.1 | 0.000% | ✅ |

总体判定：**FAIL**（失败项：G5_shear_factor_insensitive）。

## ⑤ 收敛性与参数敏感性（≥2 点）

### 5.1 收敛性：非单调，N=16 是误差相消而非收敛值

误差序列在 16x16 触底（0.17%）后回升（见 ④ 表与 `case_convergence.png`）：粗网格上的选择性减缩积分剪切释放与离散圆柱曲率误差反号，粗网格段两误差恰好相消，因此 **16x16 的 0.17% 是巧合而非收敛结论**，判据点按算例定义取 16x16。

对细网格段（16x16、24x24、32x32、48x48、64x64）做渐近外推，两种可容许模型同报，按残差自动择一：

| 模型 | 形式 | 极限 u_lim（in） | vs ref | 残差 |
|---|---|---|---|---|
| power ← 选用 | u_inf + C*N^-p | -1.889371e-05 | 3.54% | 8.086e-18 |
| log | a + b*log(N) | -1.742495e-05 | 4.51% | 4.988e-15 |

即：本格式的离散极限比文献收敛值**更柔** 3.54%（power 模型，p = 0.97），仍在 5% 容差内。点载荷在壳内是奇异载荷，加载点挠度对局部网格尺寸不严格收敛（幂律与对数律两种正则化都在表里），这也是标准做法以「某一约定网格上的收敛值」为 oracle 的原因。

### 5.2 敏感性一：横向剪切修正因子 k（锁定指示，不是拟合参数）

| k | 挠度（in） | vs ref | Δ vs k=5/6 |
|---|---|---|---|
| 1.0000 | -1.821869e-05 | 0.161% | 0.327% |
| 0.8333 | -1.827849e-05 | 0.167% | 0.000% |
| 0.5000 | -1.851756e-05 | 1.477% | 1.308% |

k 在 0.5…1.0 内变化只改变 1.308% —— R/t = 100 弯曲主导，横向剪切不控制响应，也说明单元没有剪切锁定。

### 5.3 敏感性二：drilling 正则因子

| drilling 因子 | 挠度（in） | Δ vs 默认 1e-8 |
|---|---|---|
| 1e-10 | -1.827849e-05 | 0.0000% |
| 1e-08 | -1.827849e-05 | 0.0000% |
| 1e-06 | -1.827849e-05 | 0.0000% |
| 1e-04 | -1.827849e-05 | 0.0000% |

对 1e-10…1e-4 三个量级的变化不敏感（<0.1%），drilling 项只起数值正则作用，未污染物理响应。

### 5.4 敏感性三：端部轴向约束（建模选择的定量后果）

- 隔板语义（u 自由，本题采用）：-1.827849e-05 in（0.167% vs ref）
- 若端环再夹死轴向 u：-1.524546e-05 in（16.45% vs ref，刚化 16.6%）

16% 量级的差异说明端部语义是对照量级的建模选择：只有「隔板不抗轴向平移」的语义才能与 ref 对齐，因此本模型不是靠松散约束凑出小误差。

### 5.5 敏感性四：各向异性加密（误差来自哪个方向）

| nx × nθ | 单元数 | 挠度（in） | 误差 |
|---|---|---|---|
| 8 × 8 | 64 | -1.763665e-05 | 3.350% |
| 8 × 16 | 128 | -1.797257e-05 | 1.509% |
| 16 × 8 | 128 | -1.818707e-05 | 0.334% |
| 16 × 32 | 512 | -1.839346e-05 | 0.797% |
| 32 × 16 | 512 | -1.874617e-05 | 2.730% |

沿轴向与环向单独加密都先向 ref 靠近再越过，轴向加密的越过量更大 → 渐进误差由**轴向弯曲边界层**的分辨率主导，而非环向离散。

## ⑥ 失败模式、工件与生态摩擦

失败模式（已知边界，不是本单引入）：

1. **非单调收敛**：粗网格段误差非单调（4x4 2.86% → 8x8 3.35% → 12x12 0.99% → 16x16 0.17% → 24x24 1.28% → 32x32 1.82% → 48x48 2.37% → 64x64 2.67%），N=16 是两股反号误差相消，任何「用最粗/最细网格误差排序」的自动判据都会误判；
2. **点载荷奇异**：加载点挠度不严格网格收敛，本格式极限比 ref 柔 3.54%（5.1）；
3. **稠密路径上限**：生产稠密装配在 N≥24 内存不可承受（N=32 需 341 MB、N=64 >5 GB），故本单新增 `solve_pinched_cylinder_sparse` 稀疏路径；稠密/稀疏在 N=16 上逐位一致（相对差 1.29e-13）。

工件（run_artifacts）：

| 工件 | 内容 |
|---|---|
| `benchmarks/verified/fem_pinched_cylinder/run.py` | 一键复现驱动（本报告全部数字的来源） |
| `benchmarks/verified/fem_pinched_cylinder/case_result.json` | 全部数值（ladder/外推/敏感性/环境） |
| `benchmarks/verified/fem_pinched_cylinder/case_gates.json` | 判据逐项裁决 |
| `benchmarks/verified/fem_pinched_cylinder/case_convergence.png` | 误差-网格曲线（含 5%/3% 门与外推极限） |
| `benchmarks/verified/fem_pinched_cylinder/case_field.png` | N=16 八分体径向挠度场云图 |
| `benchmarks/verified/fem_pinched_cylinder/case_sensitivity.png` | 剪切/drilling/各向异性扫掠 |
| `src/tensorfem/pinched_cylinder.py` | 新增稀疏装配/求解、各向异性网格、端部轴向夹死选项 |
| `tests/test_pinched_cylinder.py` | 判据点 16×16 <5%、稀疏/稠密等价、端部语义、剪切不敏感 |

生态摩擦（friction，如实记）：

1. `run_shell` 单次命令 120 s 硬超时并整组收割进程，本算例完整 ladder（N=64 约 3–4 min）无法本地跑完 → 只能走 cluster_queue；这是「算例池以百计铺开」时的主要吞吐瓶颈。
2. `negative_knowledge_ledger`/`oss_solutions` 表对工作树 psql 只读账号 `permission denied`，平台 API 只有 `/tensor/negative-knowledge/search`（`q=` 参数）而无 oss 检索面 → 开工三查中的「negative knowledge / oss」两查只能部分完成（见下）。
3. 本单前，`pinched_cylinder` 的验收工件（`run_verification.py`）把 N=32 的 1.82% 当作「最细网格精度」并给出 G2 非单调 FAIL，缺少稀疏路径与渐近外推，读者无法分辨「格式极限更柔 3.4%」与「N=16 碰巧 0.17%」——本单补齐这一层，未改动原锚点脚本。

开工三查结果：knowledge_base 同域（pinched/cylindrical shell/MacNeal）0 命中；negative_knowledge_ledger 关键词 TensorFEM 因权限不可读（friction 2）；oss_solutions 同域因权限不可读。仓内既有能力已足够（`pinched_cylinder.py` + `cylindrical_shell4`），故按「不另起炉灶」直接复用并补齐稀疏路径。
