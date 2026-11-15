# 冰压斑方位角对加筋板架极限强度的影响（斜置压斑能力 + 独立校核）

**锚点** `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE` · **仓** TensorFEM · **新模块** `src/tensorfem/grillage_oblique_patch.py` · **验证** `benchmarks/grillage_oblique_patch_verification.py`（12 道门 O1–O7）· **测试** `tests/test_grillage_oblique_patch.py`（15 项）

## 1. 案例介绍（物理背景/几何/参数表）

冰区舷侧冰带的冰压斑不是矩形对齐于骨架：冰缘走向与纵骨夹角任意。已审的半解析板架层（`grillage_ice_ultimate`）把压斑写成**轴对齐**矩形并夹到一个框架间距 × 一个肋距，独立复核把斜置压斑列为 not_run。本单补上这一几何维度，并给出它对极限冰压的量化影响。

| 参数 | 值 |
|---|---|
| 板厚 t | 20.0 mm |
| 纵骨间距 b / 横框架间距 a | 0.35 m / 2.10 m |
| 板格数 / 框架间距数 | 8 / 4 |
| 代表板架尺寸 (x × y) | 8.40 m × 2.80 m |
| 材料 | E=206 GPa, ν=0.3, σ_y=355 MPa（理想弹塑性）|
| 纵骨 / 横框架 | T250×10+80×12 / T550×10+120×14 mm（与已审算例同，见 `case_study_geometry`）|
| 压斑 | 长 × 宽 = 1.65 m × 0.55 m，方位角 θ 0…90°，面积恒定 0.9075 m² |
| 组合工况 | σ_x ∈ {53, 106, 160} MPa 与静水侧压 {0.05, 0.10, 0.15} MPa 三档 |

## 2. 方法（流程/关键设置/工具链）

1. **载荷系数（新）**：任意凸多边形压斑的 Navier 系数 `q_mn = (4/ab) integral p sin(m pi x/a) sin(n pi y/b) dA` 由结构化网格单元 + Sutherland-Hodgman 精确裁剪 + 5 次精度三角形/张量 Gauss 得到；整块内部单元走向量化快路径。θ=0 时**精确退化**为已审闭式系数（O1）。
2. **响应场**：与已审层同一 SSCC 正交各向异性 Navier 场（模态集 1…2·terms），输出峰值挠度/弯矩与模态载荷分配 f_x+f_y+f_t=1。
3. **构件载荷路径（新）**：把压斑多边形与纵骨/横框架的**支流条带**精确求交，得到每个构件的载荷份额；对轴对齐居中压斑它退化为已审的等分规则（可用 `member_allocation='base_convention'` 完整复现已审数），对斜置/偏心压斑它给出与位置连续的分载，而不是固定 2 根框架 50/50。
4. **极限判据**：容量律**未重写**，直接从已审层导入（板格塑性弯矩折减 sqrt(1−(σ_x/σ_y)²)、有效宽度梁柱 Perry-Robertson/Johnson-Ostenfeld + 精确 secant 放大、横框架刚塑性倒塌 8Mp/(L(1−c/2L))）；只换载荷。三条分支对冰压**线性**，故极限压力取闭式根 min_b(1−u0_b)/du_b，并保留二分交叉校核（O6b）。
5. **工具链**：numpy（场/积分）、matplotlib（图）、pytest（15 项回归）、本仓 `orthotropic_plate_fd`（独立 FD 参考）。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 方位角扫描：极限冰压 p_ult [MPa]（同面积压斑）

| θ [deg] | light-ice + ballast head | design-ice + design head | severe-ice + tank head | 构件控制算例 |
|---|---|---|---|---|
| 0 | 2.2476 | 2.1176 | 1.9241 | 0.4990 |
| 15 | 2.2510 | 2.1208 | 1.9270 | 0.5619 |
| 30 | 2.3180 | 2.1839 | 1.9844 | 0.5785 |
| 45 | 2.4414 | 2.3001 | 2.0899 | 0.5666 |
| 60 | 2.5634 | 2.4150 | 2.1944 | 0.5521 |
| 75 | 2.6485 | 2.4953 | 2.2673 | 0.5230 |
| 90 | 2.6788 | 2.5238 | 2.2932 | 0.5009 |

极差/θ=0：板格控制 19.184 %；构件控制 15.945 %。三档板格控制工况的极差相同（19.184 %），因为同一几何下三条分支的斜率比不随载荷变化。

### 3.2 验证门（12 道，全部 PASS）

| 门 | 内容 | 实测 | 容差 | 判定 |
|---|---|---|---|---|
| O1_theta0_vs_closed_form | θ=0 与已审闭式层（8 例，弯矩/分配/合力） | 0.000000 | 0.0 | PASS |
| O2_polygon_geometry | 旋转正方形面积、边界裁剪、整板合力恒等 | 0.000000 | 0.0 | PASS |
| O3_quarter_turn_invariance | 各向同性方板 90° 旋转 m_x↔m_y | 0.000000 | 0.0 | PASS |
| O4_fd_superposition | 独立 FD 条带叠加（45°，w 与 m_x） | 0.000392 | 0.1 | PASS |
| O5_tributary_conservation | 支流份额守恒/条带划分/镜像对称 | 0.000000 | 0.0 | PASS |
| O5_single_strip_patch | 小压斑只加载一个构件 | 0.000000 | 0.0 | PASS |
| O6_tieback_p_ult | θ=0 复现已审 ultimate_ice_pressure（8 例） | 0.000000 | 0.0 | PASS |
| O6_tieback_utilisations | 极限点三条分支与已审层一致 | 0.000000 | 0.0 | PASS |
| O7_orientation_spread_light | 方位角极差相对于 θ=0（要求有限且 <100 %） | 0.191838 | 1.0 | PASS |
| O7_orientation_spread_design | 方位角极差相对于 θ=0（要求有限且 <100 %） | 0.191838 | 1.0 | PASS |
| O7_orientation_spread_severe | 方位角极差相对于 θ=0（要求有限且 <100 %） | 0.191838 | 1.0 | PASS |
| O7_orientation_spread_member_governed | 方位角极差相对于 θ=0（要求有限且 <100 %） | 0.159448 | 1.0 | PASS |

### 3.3 独立 FD 交叉校核（45° 压斑，条带叠加收敛）

| 条带数 | FD 网格 nx | w_max 相对差 | m_x_max 相对差 |
|---|---|---|---|
| 16 | 96 | 0.117 % | 0.192 % |
| 32 | 96 | 0.060 % | 0.145 % |
| 32 | 192 | 0.029 % | 0.039 % |

级数峰值 m_x = 259348.8 N·m/m；FD 自身随 nx 与条带数单调收敛。

## 4. 结果图

![orientation](artifacts/grillage_oblique_patch/fig_orientation_sensitivity.png)

*图 1* 极限冰压随压斑方位角变化（三档组合工况，同面积）。θ=0（压斑长边沿纵骨）最低，θ=90° 最高。

![fields](artifacts/grillage_oblique_patch/fig_patch_fields.png)

*图 2* θ=0 与 θ=45° 的板格 m_x 场（压斑轮廓红线），峰值位置随方位角移动，说明峰值不能只在中点取样。

## 5. 注意点（坑/局限/适用范围）

* **θ=0 必须精确退化**：早期实现把板格分支放在 8.4 m × 2.8 m 全局板上算，弯矩虚高 9 倍；已审层用的是局部板格 2.1 m × 0.35 m + 各向同性 D + 压斑夹到板格，本模块已改为同一理想化（`bay_load_polygon`），并用 O6 把全链回接到 1e-7。
* **静水侧压的分配在全局板、弯矩在局部板格**：两者用不同域，复现已审约定；O6 覆盖了该组合。
* **构件分载约定差异**：默认 `tributary`（按面积连续分载）与已审 `base_convention`（等分/固定 2 框架）在构件控制算例上给出不同利用率——本单只把后者用于回接校验，未改动已审数字。
* **未做（unknown/not_run）**：非均匀压斑（压力-面积衰减）、压斑跨越多个板格时的多板格峰值统计、斜置压斑下的刚塑性系统倒塌重分配（仅弹性构件极限）、温度/棘轮路径、规范（IACS PC/CSR）外部数据对照。
* **适用**：小挠度板理论、理想弹塑性、无初始缺陷路径、无腹板局部屈曲/侧倾；结论限于本几何与本材料。

## 6. 误差判定（逐项，是否 <3 %）

| 关键量 | 参照 | 误差 | <3 % ? |
|---|---|---|---|
| q_mn / 响应场（θ=0，8 例） | 已审闭式 Navier 层 | 0.0000 % | **是** |
| 45° 压斑峰值 m_x | 独立 FD（条带叠加，nx=192） | 0.0392 % | **是** |
| 45° 压斑峰值 w | 同上 | 0.0285 % | **是** |
| p_ult 全链（θ=0，8 例） | 已审 ultimate_ice_pressure | 0.00003 % | **是** |
| 极限点三条利用率 | 已审层 | 0.00003 % | **是** |
| 象限旋转不变性 | 解析（θ vs θ+90°） | 0.00000 % | **是** |
| 支流份额守恒 | Σshare = 1 / Σarea = A | 0.00000 % | **是** |

**结论：全部关键量均 <3 %（最大为 FD 校核的 0.04 %）。** 方位角结论（15–19 % 极差）不是误差，而是被建模忽略的物理敏感性——它落在已审假设的安全侧（θ=0 最不利）。

## 7. 复现

```
PYTHONPATH=src python benchmarks/grillage_oblique_patch_verification.py core
PYTHONPATH=src python benchmarks/grillage_oblique_patch_verification.py tieback
PYTHONPATH=src python benchmarks/grillage_oblique_patch_verification.py orientation
PYTHONPATH=src python -m pytest tests/test_grillage_oblique_patch.py -q
```
