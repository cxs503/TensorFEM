# CASE — GAP-PP-3 transition detector: the +20 % plateau is a ladder artefact, not an element error

锚点：`GAP-PP-3-ICE-STRUCT-PLASTIC-SET`（板格低温钢弹塑性 + 永久挠度，TensorFEM）
仓：`TensorFEM` · 分支 `agent/m23289` · 复算脚本 `scripts/gap_pp3_transition_detector.py`
证据：`docs/assets/gap_pp3_transition_detector.json`（程序化生成）、
`docs/assets/gap_pp3_transition_detector.png`；对照 `docs/assets/gap_pp3_3grid_results.json`（已入库矩阵）
回归：`tests/test_gap_pp3_transition_detector.py`（9 项，全绿）

## 1. 案例（物理背景与参数）

冰排对船体板格施加的中心矩形斑压力。弹性阶段由 Navier 双正弦级数给出；超过首屈后由塑性储备
控制。验收判据第 1 条是「弹性→塑性过渡点与解析屈服载荷误差 <5 %」。

| 参数 | 符号 | 值 |
|---|---|---|
| 板长×宽×厚 | a×b×t | 2.0 × 1.0 × 0.020 m |
| 斑尺寸 | u×v | 0.5 × 0.5 m（网格对齐：nx%8=0, ny%4=0） |
| 钢种 / 温度 | — | DH36 / −20 °C（高于 DBTT −65 °C，塑性塌陷可用） |
| 流变应力 | σ_y | 374.2 MPa（线性拟合 σ_y(20 °C)=355 + 0.96 MPa/K×20） |
| E / ν / H | — | 206 GPa / 0.3 / 1000 MPa（近理想塑性） |
| 层数 | n | 9（分层壳，厚度方向 2×2 Gauss） |
| 解析首屈压力 | p_y,Navier | 0.712514 MPa（centre Mises 525.18 Pa/Pa） |

## 2. 方法（怎么做的）

同一个网格上并存两个「过渡点」估计量，此前被混为一谈：

* **路径交叉量** `PatchPath.first_yield_factor`：在载荷阶梯上对
  「表层 von Mises − σ_y」做线性插值。锚定阶梯是 `0.45, 0.90, 1.50, 2.10, 2.70 × p_y`，
  交叉点必然落在最后一个**弹性**阶梯（0.90 p_y）与一个已进入**饱和分支**的阶梯
  （1.50 p_y，表层应力被钉在 ≈1.13 σ_y）之间。跨饱和段插值 → 过渡点被读高。
* **弹性场估计量** `first_yield_factor_elastic = σ_y / (表层 Mises 每 Pa)`：一次弹性求解、
  与阶梯无关；只要屈服前响应是线性的就精确。新增于
  `plate_capacity_summary`（字段 `first_yield_factor_elastic` /
  `first_yield_error_elastic` / `first_yield_basis`，原 `first_yield_error` 保留为
  `first_yield_error_path`）。

网格家族 `8x4 / 16x8 / 24x12`（对齐家族，h = b/ny = 0.25 / 0.125 / 0.0833 m），
每个网格做：(i) 0.20 p_y 的弹性探针，(ii) 同载荷下 `geometric=True+restrained` 的对照探针，
(iii) 在 8x4 上用「屈服前保持弹性」的密集阶梯 `0.30…1.35 p_y` 跑真实载荷路径，取路径交叉量。
解析参考：Navier 双正弦（简支、小挠度），与 `geometric=False, inplane_restrained=False`
路径是同一力学模型；锚定矩阵用的是 `geometric=True, inplane_restrained=True`（为了有极限点），
两者在 0.20 p_y 处的差别为 1e-13 量级（探针可通用于两配置，见下表 `config_delta`）。

## 3. 计算结果（数字全部由脚本程序化注入 JSON）

弹性场过渡点误差（`error_elastic = p_y,elastic / p_y,Navier − 1`）：

| 网格 | 单元 | h [m] | 表层 Mises [Pa/Pa] | 弹性偏置 | 过渡误差 | config_delta |
|---|---|---|---|---|---|---|
| 8x4 | 32 | 0.2500 | 443.43 | −15.57 % | **+18.44 %** | −1.2e−14 |
| 16x8 | 128 | 0.1250 | 508.20 | −3.23 % | **+3.34 %** | −2.3e−13 |
| 24x12 | 288 | 0.0833 | 517.96 | −1.38 % | **+1.39 %** | n/a（本地窗口） |

收敛规则（`_rules`：p = ln(e_f/e_c)/ln(h_f/h_c)，Richardson 极限）：
**观测阶 p = 2.16**（二阶），序列单调、渐近，**外推极限 −3.5e−18**。
判据「<5 %」：**8x4 不满足（18.4 %），自 16x8 起满足（3.34 % / 1.39 %）**，网格极限 → 0。

两个估计量在同一网格上的一致性（8x4，密集阶梯路径）：

| 估计量 | 值 [Pa] | 误差 vs p_y,Navier |
|---|---|---|
| 密集阶梯路径交叉 `first_yield_factor` | 843 884.605638 | +18.44 % |
| 弹性场 `σ_y / (Mises/Pa)` | 843 884.605638 | +18.44 % |
| 相对差 | **1.8e−13** | — |

对照锚定 3 网格矩阵（已入库证据，粗阶梯 + 几何非线性 + 面内约束）：
路径交叉误差 +29.7 % / +20.7 % / +20.4 %，而同一网格的弹性场误差是
+18.4 % / +3.34 % / +1.39 %。即：**矩阵的「平台」比网格真实误差高 9–19 个百分点，
差额来自阶梯插值跨过饱和段（以及几何非线性使交叉点后移），不是单元误差。**

## 4. 图

`docs/assets/gap_pp3_transition_detector.png`：左图 = 过渡误差 vs h 的双对数图
（含二阶参考斜率与 5 % 判据线）；右图 = 同网格上「密集阶梯路径交叉」与「弹性场估计」的
误差棒对比（两者重合于 18.4 %，5 % 线仅 16x8 起跨过）。

## 5. 注意点（坑与适用范围）

* **不要**用 `first_yield_error`（路径交叉）做验收数字：它依赖阶梯，粗阶梯跨饱和段会系统性读高
  （实测 8x4：粗 924 096 Pa vs 密 843 885 Pa，同一网格差 9.5 个百分点）。
* 弹性场估计量的精确性以「屈服前响应线性」为前提。`geometric=True` 时屈服前已有轻微膜效应，
  该估计量变成近似值；此时应同时看 `first_yield_reference()["comparable"]`（本锚定矩阵为
  `False`），不得据线性 Navier 判定非线性路径。
* 8x4 网格**不满足** 5 % 判据，这是真实结论，不得用归一化误差掩盖（归一化误差恒 ≈0，无信息量）。
* 24x12 只跑了线性/自由探针（本地 120 s 窗口限制），锚定对照列留空并在 JSON 中注明理由；
  低载探针的配置无关性已在 8x4、16x8 两档验证到 1e-13。
* 本文件不涉及极限载荷（第 2 条判据）：已知 `limit_load_measured` 在锚定矩阵里随网格崩塌
  （1.42 → 0.51 → 0.07 MPa），仍是独立未决缺口，不在本单范围。

## 6. 误差判定（逐项回答是否 <3 %）

| 关键量 | 基准 | 误差 | <3 %？ |
|---|---|---|---|
| 24x12 过渡点 p_y | Navier 0.712514 MPa | 1.39 % | **是** |
| 16x8 过渡点 p_y | Navier | 3.34 % | 否（3.34 % > 3 %），但满足 5 % 判据 |
| 8x4 过渡点 p_y | Navier | 18.44 % | 否 |
| 网格极限（Richardson） | Navier | <1e−15 | 是 |
| 密集阶梯路径交叉 vs 弹性场 | 同网格弹性解 | 1.8e−13 | 是 |
| 表层 Mises/Pa 24x12 | Navier 525.18 | 1.38 % | 是 |

结论：判据第 1 条（过渡点 <5 %）在 **≥16x8** 上成立且以二阶收敛到 0；**不是**所有网格都成立。
`first_yield_error` 的 +20 % 平台是检测器（阶梯插值）伪影，已定位、已量化、已给出无阶梯的替代量。