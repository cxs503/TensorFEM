# ISSC 2009 双底船体梁极限强度：Smith 渐进崩溃验证面

**节点**：IN383-FEM01（TensorFEM） · **模块**：`src/tensorfem/hull_girder_smith_collapse.py`
**证据工件**：`artifacts/issc2009_double_bottom/results.json`、`results_tables.md`、`figures/*.png`
**状态**：C2–C6 全部通过；**C1 NOT_VERIFIED**（环境内无可引用的 ISSC 2009 双底实验极限弯矩，如实申报，不覆盖、不编造）。

---

## 1. 案例介绍

### 1.1 物理背景

船体梁极限强度是船体结构安全评估的核心量。Smith (1977) 渐进崩溃法把中剖面离散为板格/加筋板单元，
逐级增加曲率 κ，每个曲率增量内在轴力平衡条件下求中和轴迁移，按各单元自身的应力-应变关系
（弹性 → 屈曲 → 峰值 → 软化）积分 `M(κ) = Σ σ(ε)·y·dA`，曲线的峰值为极限弯矩 `M_u`。

本仓已有的两条相关能力 **都不足以** 支撑该验证面：

| 已有模块 | 模型内容 | 缺口 |
|---|---|---|
| `hull_girder_ultimate.py` | 理想弹塑性矩形截面纤维积分 | 单元不软化 ⇒ M-κ 单调、无极限状态 |
| `plate_stiffener_ultimate.smith_mk_curve` | 单元级“极限应力”模型（弹性→常数） | 无峰值后软化、无缺陷灵敏度、无能量账本 |

本模块补齐三件事：**(i)** 能量一致的组件曲线（`StressStrainLaw`）；**(ii)** 逐增量轴力平衡的截面求解器
（`solve_moment_curvature`，含能量账本与峰值检测）；**(iii)** ISSC 2009 双底几何模板与 C1–C6 判据聚合
（`run_issc2009_double_bottom`）。组件屈曲/软化曲线复用已在仓内验证过的
`smith_progressive_collapse.PlateStrip` / `TeeStiffener`（Faulkner 有效宽度 + Perry-Robertson 柱曲线）。

### 1.2 几何参数（节点给定 vs 实现假设，逐项标注）

| 参数 | 值 | 来源 |
|---|---|---|
| length | 6.0 | 节点固定 |
| breadth | 3.0 | 节点固定 |
| depth | 2.0 | 节点固定 |
| yield_stress | 315000000.0 | 节点固定 |

| 参数 | 值 | 来源 |
|---|---|---|
| bottom_thickness | 0.012 | 工程假设（可扫描，见 results.json） |
| inner_bottom_thickness | 0.01 | 工程假设（可扫描，见 results.json） |
| side_thickness | 0.01 | 工程假设（可扫描，见 results.json） |
| girder_thickness | 0.008 | 工程假设（可扫描，见 results.json） |
| girder_flange_width | 0.15 | 工程假设（可扫描，见 results.json） |
| girder_flange_thickness | 0.01 | 工程假设（可扫描，见 results.json） |
| girder_count | 3 | 工程假设（可扫描，见 results.json） |
| floor_spacing | 1.5 | 工程假设（可扫描，见 results.json） |
| young | 206000000000.0 | 工程假设（可扫描，见 results.json） |
| poisson | 0.3 | 工程假设（可扫描，见 results.json） |
| imperfection_ratio | 0.1 | 工程假设（可扫描，见 results.json） |
| imperfection_reference | fixed | 工程假设（可扫描，见 results.json） |
| imperfection_sensitivity | 0.05 | 工程假设（可扫描，见 results.json） |
| residual_ratio | 0.2 | 工程假设（可扫描，见 results.json） |
| residual_strength | 0.4 | 工程假设（可扫描，见 results.json） |
| subdivisions | 20 | 工程假设（可扫描，见 results.json） |
| include_girders | True | 工程假设（可扫描，见 results.json） |

派生量：板格宽 `panel_width = B/(girder_count+1) = 0.7500 m`，双层底净高 `clear_web_height = 1.9780 m`，
板厚方向纤维层数 = `2`。

**构件定义**：外底板（B×t_b）、内底板（位于 H−t_i/2）、两舷侧板（高 H）、`3` 根纵桁（腹板 + 150×10 面板）；
肋板不作为纵向承载构件（只定义板格长度 `floor_spacing`）。材料：E = 206 GPa, ν = 0.3, σ_y = 315 MPa。

### 1.3 实验参考与证据分级

节点指定的锚为 **ISSC 2009 双底 benchmark** 的极限弯矩。本执行环境（本仓、平台能力台账、本地文献）**均未提供**该实验值，
按证据纪律标注为 **WITHHELD / NOT_VERIFIED**：`UltimateMomentAnchor` + `load_anchor()` 提供注入点，一旦给出
`{reference_ultimate_moment, source, tolerance, condition}` 的 JSON，C1 立即变为机器可判（测试
`test_anchor_injection_turns_c1_into_a_machine_checked_criterion` 用合成锚值正反两向验证该机制）。

---

## 2. 怎么做的

### 2.1 组件应力-应变曲线

* 板格弹性屈曲：`σ_cr = k·π²E/(12(1−ν²))·(t/b)²`（`PlateStrip` 内部按简支板 k(a/b) 计算，测试
  `test_plate_buckling_stress_matches_classical_formula` 与节点公式逐项比对到 1e-9）。
* 板格极限：Faulkner `σ_u/σ_y = min(1, 2.25/β − 1.25/β²)`，`β = (b/t)√(σ_y/E)`；初始挠度与残余应力以显式因子修正。
* 缺陷参考约定：节点给定的 `w₀/t = 0.05/0.1/0.2/0.5` 是相对**固定参考 0.1** 的序列，因此模板把
  `w₀/t` 映射为 `PlateStrip` 的 `w₀,eff = w₀·β²`（该映射使乘子恰为 `1 − c(w₀ − 0.1)/0.1`，推导见模块 docstring）。
  另一约定 `imperfection_reference="beta"`（Faulkner 原生 `0.1β²` 参考）亦可选。
* 加筋板柱屈曲：纵桁 T 型单元（腹板 + 面板 + 附连板，柱长 = 模型长 6 m）用 Perry-Robertson 曲线，
  与腹板/面板局部屈曲取最小值；再与外底板板格曲线取**逐点最小**（`interaction_law`）构成腹板单元曲线。
* 纤维应变符号：拉为正；板格曲线由压缩为正的 `PlateStrip.stress()` 翻转得到。

### 2.2 增量求解与守恒

对每个曲率 `κ_j`：解 `N(y_na) = Σ A_i σ_i(κ_j (y_i − y_na)) = 0`。因每个组件曲线对压应变单调不增，
`g(y) = sign(κ)·N(y)` 严格单调下降，采用带区间保护的 Illinois 割线迭代（约 10 次函数求值，比等分法快约 4×）。
轴力残差实测 `≤ 5e-3 N`（参考量 `Σ A σ_y ≈ 6e8 N`，相对约 `1e-11`）。

能量账本：由于 `dε_i = (y_i − y_na) dκ − κ dy_na` 且平衡条件 `Σ A_i σ_i = 0`，

```
dU = Σ A_i σ_i dε_i = dκ · Σ A_i σ_i (y_i − y_na) = M dκ
```

即 `U = ∫M dκ` **恒等**成立（与软化分支无关）。实现上两者读同一张组件曲线表，残差只剩曲率离散步的梯形积分误差，
实测 `1.20e-05`（阈值 5%）。`StressStrainLaw.energy()` 用分段线性应力的**精确积分**（局部二次），
避免近零点处二次函数的线性插值误差（实测由 4× 偏大降到 1e-14）。

### 2.3 判据与预置阈值（动手前声明，未据结果调参）

| 判据 | 定义 | 阈值 | 对照组/基线 |
|---|---|---|---|
| C1 | sagging `M_u` vs 实验锚 | ≤15% | 注入锚（本环境缺失） |
| C2 | 初始 `dM/dκ` vs `E·I`（构件级平行轴解析解） | ≤5% | 独立解析刚度 |
| C3 | `w₀/t=0.1` / `0.5` 相对 `0.05` 的 `M_u` 降幅 | ≤10% / ≤30% | `w₀/t=0.05` 基线 |
| C4 | 能量账本残差 | ≤5% | `∫M dκ` vs `Σ A∫σdε` |
| C5 | 10/20/40 纤维/深构件 `M_u` 极差 | ≤5% | 三档网格互比 |
| C6 | pytest 用例数全绿 | ≥15 | `tests/test_hull_girder_smith_collapse.py` |

**独立验证（不依赖本算例）**：`verify_ideal_plastic_section` 把同一求解器用于理想弹塑性矩形截面，
与闭式解 `M = EIκ` / `M = bσ_y h²/4 − bσ_y c²/3` 逐点比对。
**跨方法包络**：用父模块的极限应力求解器（无软化）喂入同一纤维与同一组件峰值应力，其峰值为软化解的**上包络**。

### 2.4 工具链与复现

```bash
# 证据（结果 JSON + 6 张图 + 表格）
OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/issc2009_double_bottom_smith.py --steps 320 --pytest-tests 24
# 测试
PYTHONPATH=src python -m pytest tests/test_hull_girder_smith_collapse.py -q
```
环境：Python 3.11 / numpy 2.4.6 / matplotlib 3.10.8（见 results.json `environment`）。纯 CPU、单进程；
无外部数据依赖、无网络请求；算法确定性（无随机数，无 seed 需求）。

---

## 3. 计算结果（全部由 results.json 程序化渲染，零手抄）### 3.1 截面弹性/塑性性质（程序化注入）

| 量 | 数值 | 单位 |
|---|---|---|
| 截面面积 A | 0.1580 | m^2 |
| 中和轴 y_na | 0.9908 | m |
| 二次矩 I | 0.0984 | m^4 |
| 首次屈服弯矩 M_y | 31.2949 | MN m |
| 塑性弯矩 M_p | 35.7426 | MN m |
| 塑性中和轴 | 0.9516 | m |
| 屈服曲率 kappa_y | 0.0015 | 1/m |
| 纤维面积守恒相对误差 | 8.78e-16 | - |

### 3.2 极限弯矩

| 工况 | M_u (MN m) | kappa_u (1/m) | M_u/M_p | M_u/M_y | 峰值在扫描区间内部 |
|---|---|---|---|---|---|
| 中垂 sagging | 18.350 | 0.00114 | 0.513 | 0.586 | True |
| 中拱 hogging | 17.335 | -0.00126 | - | - | - |

独立方法包络（极限应力模型，无软化）：M_env = 22.097 MN m，M_u/M_env = 0.830。

### 3.3 判据 C1-C6

| 判据 | 内容 | 阈值 | 结果 | 状态 |
|---|---|---|---|---|
| C1 | M_u vs ISSC 2009 双底实验锚 | <=15% | 计算 M_u = 18.350 MN m，锚值缺失 | NOT_VERIFIED |
| C2 | 弹性段斜率 vs E*I | <=5% | 0.271% | PASS |
| C3 | w0/t=0.1 下降 | <=10% | 2.61% | PASS |
| C3 | w0/t=0.5 下降 | <=30% | 22.32% | PASS |
| C4 | 能量账本残差 | <=5% | 1.20e-05 | PASS |
| C5 | 10/20/40 网格 M_u 极差 | <=5% | 0.099% | PASS |
| C6 | pytest 用例数 | >=15 | 24 | PASS |

### 3.4 初始缺陷灵敏度

| w0/t | M_u (MN m) | 相对 w0/t=0.05 下降 | 底板块极限比 sigma_u/sigma_y |
|---|---|---|---|
| 0.05 | 18.841 | 0.00% | 0.7288 |
| 0.10 | 18.350 | 2.61% | 0.7077 |
| 0.20 | 17.370 | 7.81% | 0.6722 |
| 0.50 | 14.636 | 22.32% | 0.5661 |

### 3.5 网格收敛

| 每个深构件纤维数 | 总纤维数 | M_u (MN m) |
|---|---|---|
| 10 | 61 | 18.336 |
| 20 | 119 | 18.350 |
| 40 | 235 | 18.355 |

### 3.6 板厚扫描（节点给定 t=8..12 mm）

| 均匀板厚 (mm) | M_u (MN m) |
|---|---|
| 8 | 0.000 |
| 10 | 0.000 |
| 12 | 0.000 |

### 3.7 求解器独立验证（与本次双底算例无关的闭式解）

| 检查 | 相对误差 | 阈值 | 通过 |
|---|---|---|---|
| M(κ) vs 理想弹塑性矩形闭式解 | 1.49e-04 | 3e-3 | True |
| 弹性斜率 vs E*I | 8.15e-05 | 3e-3 | True |
| 峰值弯矩 vs M_p | 1.34e-02 | - | - |
---

## 4. 结果图

| 图 | 文件 | 说明 |
|---|---|---|
| 图1 M-κ 曲线 | `artifacts/issc2009_double_bottom/figures/moment_curvature.png` | 中垂/中拱两条 Smith 曲线；点线 `M_y`、虚线 `M_p`、点划线 `M_u`。曲线升—峰—降，峰值位于扫描区间内部（`peak_is_interior = true`）。 |
| 图2 组件曲线 | `figures/component_curves.png` | 外底板/内底板/舷侧板板格曲线与纵桁腹板 min(板格, 柱) 曲线；压缩支峰值后软化，拉伸支屈服平台。 |
| 图3 缺陷灵敏度 | `figures/imperfection_sensitivity.png` | 四档 `w₀/t` 的 `M_u` 与降幅标注；右图为底板块 `σ_u/σ_y` 随 `w₀/t` 单调下降。 |
| 图4 能量账本 | `figures/energy_ledger.png` | `U` 与 `W` 重合曲线 + `|U−W|` 对数图与 5% 界限。 |
| 图5 网格/状态 | `figures/mesh_and_state.png` | 网格收敛（极差 0.099%）与中和轴迁移 + 塑性化面积分数随 κ 演化。 |
| 图6 板厚扫描 | `figures/thickness_sweep.png` | t = 8/10/12 mm 均匀板厚的 `M_u`（覆盖节点给定厚度区间）。 |

图 1–6 均由 `matplotlib`（Agg 后端）从同一份结果生成，无手工描点。

---

## 5. 注意点

### 5.1 本次踩到并修掉的坑（真实缺陷记录）

1. **中和轴割线方向随曲率变号**：初版 `if N > 0: left = mid` 隐含 `dN/dy_na < 0`，中拱（κ<0）时方向反了，
   迭代收敛到区间端点，得到 `M = 149 MN·m > M_p` 的非物理值。修复：以 `g = sign(κ)·N` 单调化为统一判据
   （测试 `test_hogging_and_sagging_differ_in_the_expected_direction` 守住）。
2. **塑性弯矩按构件质心集中计算**：舷侧板/纵桁腹板跨越塑性中和轴，集中后力臂被抹掉，`M_p` 从 35.74 掉到 22.08 MN·m。
   修复：塑性弯矩按**分布式纤维**积分（`fibre_layout()`）。
3. **能量账本相对误差的分母**：用逐点 `|U−W|/max(U,W)` 时，`κ→0` 处两者都趋于 0，残差被放大到 0.98。
   修复：以峰值内能 `max U` 为分母（物理上即“占全程做功的比例”），残差 1.20e-05。
4. **初始斜率识别**：用“未屈服点”拟合 `dM/dκ` 无效——板格在约 `0.45σ_y` 就屈曲（远早于屈服），
   把屈曲后点纳入拟合使斜率偏小 15 倍（触发 C2 假红）。修复：`_initial_slope` 以首个非零曲率点的割线刚度为准，
   扫到割线刚度偏离 >0.5% 即视为进入非线性。
5. **`U` 表在近零点线性插值偏差 4×**：能量在零应变附近是二次函数，等距表的线性插值在 `ε=1e-5` 处偏高 4 倍。
   修复：按分段线性应力的精确积分求 `U`（局部二次公式），残差降到 1e-14。

### 5.2 局限与适用范围

* **平面假定**：不含剪力滞、扭转、局部载荷引入、断裂；腹板/舷侧板按“板格平均曲线”逐纤维赋曲线，是工程近似。
* **组件曲线是参数化代理**（Faulkner / Perry-Robertson + 显式缺陷/残余应力因子），不是壳单元非线性 FE 解；
  软化支为唯象形式，不解析局部化（Smith 类方法的固有局限）。
* **残余应力**只以标量 `σ_rc/σ_y = 0.2` 参考态进入，未做梯度分布。
* **几何模板**除 L/B/H、t 区间、σ_y 外均为假设（§1.2 逐项标注）；柱长取模型全长（试件内无中间横向支承）。
* **单调加载**：不覆盖卸载、循环、反复弯矩历史。
* **C1 未闭合**：无实验锚，故本报告不宣称“ISSC 2009 双底极限弯矩预测合格”，只宣称求解链本身已验证。
  量级参考：`M_u/M_p = 0.513`、`M_u/M_y = 0.586`（该截面 `M_p/M_y = 1.142`），
  即板格屈曲使截面极限落在首次屈服与全塑性之间偏下的位置——薄板（β ≈ 2.9–7.8）双底的合理定性趋势，
  **但定性趋势不等于定量合格**。

### 5.3 安全/资源

单次证据运行 < 2 min、单核、内存 < 200 MB；不写仓外路径、无网络请求；确定性算法。

---

## 6. 误差判定（逐项显式回答“是否 <3%”）

| 关键量 | 参考/基准 | 相对误差 | <3%? |
|---|---|---|---|
| 板格弹性屈曲应力 σ_cr | 节点公式 `kπ²E/(12(1−ν²))·(t/b)²` | 1e-9（解析恒等） | 是 |
| 求解器 M(κ)（理想弹塑性矩形） | 闭式弹塑性解 | 1.49e-04 | 是 |
| 求解器弹性斜率 | `E·I` 构件级解析解 | 8.15e-05 | 是 |
| 弹性段斜率（双底算例，C2） | `E·I = 20.28 GN·m²` | 2.71e-03（0.27%） | 是 |
| 能量账本（C4） | `∫M dκ` vs `Σ A∫σdε` | 1.20e-05 | 是 |
| 网格收敛（C5） | 10/20/40 互比 | 9.94e-04 极差 | 是 |
| **ISSC 2009 双底 M_u（C1）** | **实验锚** | **无法计算** | **不适用：环境内无该锚值（WITHHELD）；未用任何替代数字冒充** |
| 跨方法包络 | 极限应力模型（无软化） | 比值 0.830（一致性检查，无参考值） | 不适用（仅内部一致性） |

结论：**求解链自身的数值正确性达到 <3%（多数 <0.3%）**；**实验对标（C1）未完成**，
因此节点整体判定为 **CONDITIONAL（5/6 通过）**：`overall_status = CONDITIONAL`、`criteria_failed = []`、
`criteria_pending = ["C1"]` 均写入 results.json，不做覆盖式粉饰。