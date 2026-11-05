# 大角度复原力臂审计：墙舷式箱型船的精确 GZ 与线性模型的适用边界

本页由 `benchmarks/marine_large_angle_stability_audit.py` 程序化生成，正文中的每个数字
都取自 `benchmarks/marine_large_angle_stability_evidence.json`，不手工抄写。

## 1. 案例与验证欠账

`tensorfem.marine_hydrostatics` 把复原力臂写成 `GZ = GM*sin(phi)`，并把横倾角硬限在
10°；`docs/marine-hydrostatics.md` 声明「正弦式在声明范围内与
线性关系相差小于 3%」。该声明此前只与线性关系自身比较，因此任何实现错误都不可能被它发现。
本次审计用**独立路线**（浸水剖面几何）重建精确复原力臂，量化该声明在何处成立、在何处失效。

主案例（墙舷式箱型船，中横剖面，单位船长）：

| 量 | 数值 |
|---|---|
| 船长 L × 船宽 B × 型深 D | 90 m × 20 m × 8 m |
| 吃水 T / 重心高 KG | 6 m / 4.5 m |
| 密度 / 重力加速度 | 1025 kg/m³ / 9.80665 m/s² |
| KB / BM / KM / GM | 3.000000 m / 5.555556 m / 8.555556 m / 4.055556 m |
| 舷缘入水角 φ_deck = atan(2(D−T)/B) | 11.309932° |
| 底部出水角 atan(2T/B) | 30.963757° |

## 2. 方法

坐标系：右手系，y 指向右舷，z 自基线向上，沿船长取单位长度。船体横倾 φ 后，质点
(y, z) 的空间高度为 `h = z*cos(phi) - y*sin(phi)`，自由液面即 `h = h0`。

1. **平衡吃水不假设、而是解出来**：对给定 φ，二分求 h0 使浸水面积等于 `B*T`
   （浮力守恒）。墙舷式的闭式 `h0 = T*cos(phi)` 因此成为对求解器的独立校核，而不是它的输入。
2. **浸水剖面用精确多边形裁剪**：矩形剖面被半平面 `h <= h0` 切割（Sutherland–Hodgman，
   凸多边形无容差），再用鞋带公式求面积与形心，得到浮心 B(φ)。
3. **复原力臂取空间水平距离**：`GZ = Y_B - Y_G`，其中
   `(Y, Z) = (y*cos(phi) + z*sin(phi), z*cos(phi) - y*sin(phi))`，y_G = 0、z_G = KG。
   该路线完全不经过教科书公式。
4. **闭式对照**：`GZ_ws = sin(phi)*(GM + 0.5*BM*tan^2(phi))`；
   复原力矩 `ρ*∇*g*GZ`；动稳性积分闭式 `GM*(1-cos) + 0.5*BM*(sec + cos - 2)`。
5. **失效面**：任意剖面都能算出「几何浸水 GZ」，但超过 φ_deck 后甲板入水属于淹水而非稳性，
   因此把 φ_deck 作为本 oracle 的声明有效域上界（`ImmersedSection.deck_immersed` 显式标注）。

## 3. 计算结果

### 3.1 oracle 自身的独立锚点（收紧到 1e-9 量级）

| 锚点 | 描述 | 相对误差 | 门限 | 判定 |
|---|---|---|---|---|
| flotation_offset_vs_closed_form | bisected free-floating equilibrium versus the wall-sided closure h0 = T*cos(phi) | 4.594e-15 | 1e-09 | PASS |
| immersed_area_closure | unit-length displacement conservation over the solved heel range | 4.737e-15 | 1e-09 | PASS |
| waterplane_chord_vs_b_over_cos | waterplane breadth grows as B/cos(phi) while the vertical sides are cut | 1.776e-16 | 1e-09 | PASS |
| centroid_vs_trapezoid_integrals | 2-D polygon clip versus closed-form trapezoid integrals y_B = BM tan, z_B = KB + BM tan^2/2 | 1.214e-14 | 1e-09 | PASS |
| exact_gz_vs_wall_sided | geometric righting arm versus sin(phi)*(GM + BM tan^2(phi)/2) | 1.242e-14 | 1e-09 | PASS |
| dynamic_stability_vs_closed_form | Simpson integral of the exact arm versus GM(1-cos) + BM(sec + cos - 2)/2 | 1.145e-12 | 1e-08 | PASS |
| zero_arm_angle_vs_root | bisected GZ = 0 lurch angle of an unstable section versus atan(sqrt(-2 GM/BM)) | 8.929e-15 | 1e-09 | PASS |
| merged_hydrostatics_gm | upright KB/BM/GM of tensorfem.marine_hydrostatics versus the oracle | 0.000e+00 | 1e-12 | PASS |

直立 **KB/BM/GM** 与合并模块 `analyze_box_barge` 的结果逐项一致（相对误差
0.00e+00），说明大角度 oracle 与既有小角度模型
共享同一套直立静水力，不存在口径漂移。

### 3.2 主案例 GZ 表（截取一半采样点，全表见证据 JSON）

| 横倾 φ (°) | 精确 GZ (m) | 闭式 GZ_ws (m) | 线性 GM·sinφ (m) | 线性式误差 (%) |
|---|---|---|---|---|
| 0.49 | 0.034808 | 0.034808 | 0.034806 | 0.005 |
| 1.48 | 0.104455 | 0.104455 | 0.104408 | 0.045 |
| 2.46 | 0.174199 | 0.174199 | 0.173979 | 0.126 |
| 3.44 | 0.244102 | 0.244102 | 0.243499 | 0.247 |
| 4.43 | 0.314231 | 0.314231 | 0.312947 | 0.409 |
| 5.41 | 0.384650 | 0.384650 | 0.382303 | 0.610 |
| 6.39 | 0.455428 | 0.455428 | 0.451546 | 0.852 |
| 7.38 | 0.526632 | 0.526632 | 0.520656 | 1.135 |
| 8.36 | 0.598333 | 0.598333 | 0.589613 | 1.457 |
| 9.34 | 0.670603 | 0.670603 | 0.658396 | 1.820 |
| 10.33 | 0.743516 | 0.743516 | 0.726985 | 2.223 |
| 11.31 | 0.817151 | 0.817151 | 0.795360 | 2.667 |

### 3.3 线性模型的 3% 适用边界

线性式相对误差的解析形式为 `r/(1+r)`，`r = BM*tan^2(phi)/(2*GM)`；在 10° 处
`GM/BM >= 0.5026` 才满足 3% 门限。

| GM/BM | GM (m) | 10° 处线性式误差 (%) | < 3% ? |
|---|---|---|---|
| 0.100 | 0.5556 | 13.454 | no |
| 0.250 | 1.3889 | 5.854 | no |
| 0.400 | 2.2222 | 3.741 | no |
| 0.503 | 2.7944 | 2.998 | yes |
| 0.750 | 4.1667 | 2.031 | yes |
| 1.000 | 5.5556 | 1.531 | yes |

### 3.4 墙舷闭式自身的有效边界（两端都有限制）

闭式 `sin(phi)*(GM + 0.5 BM tan²φ)` 只在**自由液面切割两侧竖舷**时精确，因此有效域为
`min(φ_deck, atan(2T/B))`：右舷舷缘入水会破坏它，升高侧舭部出水同样会破坏它。
主案例（舷缘控制）：越过 φ_deck 后与几何精确解分离：

| 横倾 / φ_deck | 闭式相对几何解的偏差 (%) |
|---|---|
| 0.98x_deck_edge | 0.0000 |
| 1.02x_deck_edge | 0.0423 |
| 1.10x_deck_edge | 0.9322 |
| 1.25x_deck_edge | 4.8082 |

深窄剖面（B=6 m、T=5 m、
D=15 m，GM=1.100 m）由
**舭部出水**控制：φ_deck=73.301° 但
atan(2T/B)=59.036°，故有效域只剩
59.036°；越过该角闭式与几何解分离：

| 横倾 / 有效域上界 | 闭式相对几何解的偏差 (%) |
|---|---|
| 0.99x_envelope | 0.0000 |
| 1.05x_envelope | 1.2151 |
| 1.10x_envelope | 5.0575 |
| 1.25x_envelope | 42.8384 |

最后，近中性剖面（GM=-0.1000 m）的 `GZ = 0` 侧倾平衡角闭式
10.743404° 与几何解 10.743404° 相差
8.93e-15（相对），且该根落在有效域
30.964° 之内；若闭式根落到有效域之外（如深窄剖面），
闭式结论即失效，只能采用几何解。

## 4. 图

* `docs/assets/marine_gz_audit.png`：主案例与低 GM 案例的 GZ 曲线（精确 / 闭式 / 线性），
  有效域与舷缘入水角标注。
* `docs/assets/marine_gz_linear_envelope.png`：线性式误差随横倾角与 GM/BM 的变化，3% 等值线。
* `docs/assets/marine_section_geometry.png`：0°、5°、10°、φ_deck 的浸水剖面、自由液面、
  浮心 B 与复原力臂 GZ 的几何示意（本任务为解析几何，无场量云图；以剖面几何图替代）。

## 5. 注意点与局限

1. **仅墙舷式矩形剖面**：无舷弧、无外飘、无舭部圆弧；真实船型的大角度 GZ 需要任意多边形
   剖面与上层建筑/舱口进水建模，本模块不覆盖。
2. **φ_deck 之后不是稳性结论**：越过舷缘入水后水会漫上甲板，几何浸水体积不再是可恢复浮力，
   代码只作为几何量输出，并显式标记 `deck_immersed=True`。
3. **无自由液面修正、无纵倾、无进水/破舱、无风压倾侧力臂**；`KG` 视为定值（不含液舱晃动）。
4. **无量纲/单位强约束**：输入必须 SI，`0 < T < D`，|φ| < 90°，否则 fail-closed。
5. 结论不能用于入级或法规符合性判定。

## 6. 误差判定（逐项回答「是否 < 3%」）

| 判定对象 | 基准 | 相对误差 | < 3% ? |
|---|---|---|---|
| oracle 精确 GZ（几何路线） vs 教科书墙舷闭式 | 闭式 `sin(phi)(GM+½BM tan²)` | 1.24e-14 | 是 |
| 动稳性 Simpson 积分 vs 闭式积分 | `GM(1-cos)+½BM(sec+cos-2)` | 1.15e-12 | 是 |
| 浮心形心 vs 独立梯形积分 | `y_B = BM tan, z_B = KB + ½BM tan²` | 1.21e-14 | 是 |
| 平衡液面高度 vs `T*cos(phi)` | 墙舷体积闭合 | 4.59e-15 | 是 |
| **合并模块线性臂 @10°（既有工况剖面，GM/BM=0.730）** | 精确 GZ | 2.085% | 是 |
| **合并模块线性臂 @10°（低 GM 剖面，GM/BM=0.460）** | 精确 GZ | 3.269% | 否 |

结论：oracle 与被审计的线性模型之间的差距不是数值噪声，而是**模型误差**。合并模块在 10° 声明
边界内对出厂工况剖面满足 3%（2.09%），但对低 GM 剖面在 10° 处达到
3.27%（超门限），因此
`docs/marine-hydrostatics.md` 中「小于 3%」的表述只对该文档所附算例成立，不是通用保证。
建议（未在本单实施，属合并模块的 API 变更）在 `righting_arm` 上补墙舷闭式或把
`MAX_SMALL_HEEL_DEG` 与 `GM/BM` 耦合，并在返回值中携带模型误差界。
