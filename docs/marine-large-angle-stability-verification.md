# Wall-sided large-angle stability: closing the `marine_hydrostatics` 10-degree gap

验证对象：已合并的 `tensorfem.marine_hydrostatics`（box-barge 初稳性 + 小角复原力矩，
`|heel| <= 10 deg`，`GZ = GM sin(phi)`）。本报告回答两件事：

1. 矩形舷侧（wall-sided）驳船的**大角复原臂有精确闭式解**，模块声明的 10° 上限是
   实现范围限制，不是物理限制——现已补齐并做双重独立核验；
2. 原小角模型在**声明范围内的实际误差**是多少、在什么参数下会越过 3% 门槛。

## 1 物理背景与几何参数

矩形舷侧驳船（SI 单位，海水 1025 kg/m³，g = 9.80665 m/s²）。`L` 船宽 `B` 型深 `D`
正浮吃水 `T`，重心高 `KG`：

    KB = T/2,  I_wl = L B^3/12,  BM = I_wl / (L B T) = B^2/(12 T),  GM = KB + BM - KG

横倾 φ（右舷下沉为正）时，对矩形截面存在经典 wall-sided 复原臂闭式解
（Biran & López-Pulido, *Ship Hydrostatics and Stability*, 2nd ed.；Derrett,
*Ship Stability for Masters and Mates*）：

    GZ(φ) = sin φ · [ GM + ½ BM tan²φ ]                                        (1)

其杠杆分解形式为

    GZ = y_B cos φ + (KB′ − KG) sin φ,  y_B = BM tan φ,  KB′ = T/2 + ½ BM tan²φ  (2)

式 (1) 的适用区间由两个几何条件夹住：低舷甲板边不浸水 `tan|φ| < 2(D−T)/B`，
高舷龙骨角不出水 `tan|φ| < 2T/B`，先到者截止。**注意**：两者都是「水线固定为
`z = T + y tanφ`」下的判据；越过龙骨出水后浮体随吃水守恒而上浮，实际甲板浸水角更大
（见 §5 第 3 条）。

五个试验工况（`benchmarks/marine_large_angle_stability.py`）：

| barge | L | B | D | T | KG | GM | BM | 闭式解截止角 | 原因 |
|---|---|---|---|---|---|---|---|---|---|
| design_barge | 40 | 12 | 4 | 2.0 | 2.10 | 4.900 | 6.000 | 18.435° | 龙骨出水 |
| shallow_draft_barge | 40 | 12 | 6 | 2.0 | 2.50 | 4.500 | 6.000 | 18.435° | 龙骨出水 |
| small_gm_barge | 80 | 20 | 10 | 4.0 | 9.60 | 0.733 | 8.333 | 21.801° | 龙骨出水 |
| negative_gm_barge | 80 | 20 | 10 | 4.0 | 10.83 | −0.497 | 8.333 | 21.801° | 龙骨出水 |
| deck_limited_barge | 30 | 10 | 4 | 2.5 | 2.00 | 2.583 | 3.333 | 16.699° | 甲板边浸水 |

## 2 方法：两条完全独立的求解路径

- **闭式解路径**：直接实现式 (1)（`WallSidedBarge.gz_closed_form`），并在越界时
  fail-closed 抛错，不做静默外推。
- **独立几何路径**（oracle）：不经过式 (1)/(2) 的代数结果，而是把截面做成多边形
  `[−B/2,0]–[B/2,0]–[B/2,D]–[−B/2,D]`，用倾斜水线半平面做 Sutherland–Hodgman
  裁剪，shoelace 公式取面积与形心，得到 `y_B, KB′`；水线平移量（sinkage）由
  **位移守恒**（面积 = B·T）二分求解。复原臂由式 (2) 的形心表达式给出
  （`gz_numerical`）。该路径在龙骨出水区依然成立，因为 sinkage 在求解中被放松。
- **小角模型路径**：`gz_small_angle = GM sinφ`，即已合并模块的模型，仅在 10° 内可调用。

交叉核验：两条路径在五个工况、共 198 个倾角上逐点比较（表 B）。

## 3 计算结果（程序化生成，见 `docs/assets/marine_large_angle_stability/tables.md`）

表 A（参数与截止角）、表 B（闭式解 vs 多边形 oracle）、表 C（小角模型误差审计）、
表 D（越过龙骨出水后的外推偏差与上浮量）、表 E（初稳性消失角恒等式）均由
`benchmarks/marine_large_angle_stability.py` 直接写出，报告正文不手抄数字。

关键结果：

| 量 | 数值 | 判定 |
|---|---|---|
| 闭式解 − oracle 最大绝对偏差（5 工况 / 198 倾角） | 9.10e-15 m | 机器精度 |
| 同上，相对最大偏差 | 1.37e-13 | 机器精度 |
| 位移守恒残差（裁剪面积 − B·T） | ≤ 2.84e-13 m² | 收敛到底 |
| 倾角 19.047° 下的 GZ（负 GM 工况，oracle） | 6.22e-15 m | 恒等式闭合 |
| 初稳性消失角：闭式 19.047414° vs oracle 零点 19.047414° | 差 4.05e-13° | 恒等式闭合 |

图 1 `gz_vs_heel.png`：五个工况的 GZ–φ 曲线，闭式解（实线）、多边形 oracle（空心点）、
`GM sinφ` 小角模型（点线）与截止角（虚线）同图；图 2 `small_angle_error.png`：小角模型
相对误差随倾角变化（对数纵轴，3% 与 15% 门槛线与 10° 声明上限）。

## 4 误差判定（逐项，是否 < 3%）

| 关键量 | 基准 | 误差 | < 3%？ |
|---|---|---|---|
| GZ(φ)，闭式解 | 独立多边形/形心 oracle（本文给出） | ≤ 1.4e-13 相对 | 是（机器精度，远优于 3%） |
| GZ(φ)，多边形 oracle | 式 (1) 解析闭式（文献） | ≤ 9.1e-15 m 绝对 | 是 |
| 位移守恒（裁剪面积） | 正浮 B·T | ≤ 2.8e-13 m² | 是 |
| 初稳性消失角 | `tan²φ = −2GM/BM` | 4.1e-13° | 是 |
| 与已合并 `marine_hydrostatics` 在例case 8° 的一致性 | 同一 `GM sinφ` 模型 | 0（逐位相同） | 是 |
| **已合并小角模型 vs 精确 wall-sided GZ（≤10°，GM 充裕工况）** | 式 (1) | 1.87% / 2.03% / 1.97% | 是 |
| **已合并小角模型 vs 精确 wall-sided GZ（≤10°，`small_gm_barge`，BM/GM = 11.4）** | 式 (1) | **15.01%**（首次越过 3% 在 4.5°） | **否** |

最后一行是本单的核心欠账结论：模块声明的 10° 上限**并不能**保证 GZ 相对误差 < 3%。
误差可解析写出：`GZ/(GM sinφ) = 1 + ½ (BM/GM) tan²φ`，于是相对误差

    e(φ) = [½ (BM/GM) tan²φ] / [1 + ½ (BM/GM) tan²φ]

只由 `BM/GM` 与倾角决定。`GM` 充裕（`BM/GM ≈ 1.2–1.3`）时 10° 处误差 1.9–2.0%；
当 `KG` 接近 `KM`（宽浅驳船轻载，`GM` 小）时 `BM/GM` 可达 10 以上，10° 处误差越过 15%
门槛。该式已在测试中作为断言核对（`test_small_angle_model_debt_for_a_wide_shallow_low_gm_barge`）。

## 5 注意点与局限

1. **只对矩形舷侧精确**。式 (1) 依赖水线二阶矩随倾角不变（`I_wl = L B^3/12` 恒定）；
   有舷弧、外飘、首尾变宽的实船必须走横剖面数值积分（cross-curves），本模块**不能**
   替代，也不应外推到实船 GZ。
2. **截止角必须 fail-closed**。越过龙骨出水继续用式 (1) 会**高估**复原臂：表 D 中
   23.4°/28.4°/33.4° 处外推值分别高出 6.15%/19.97%/39.63%，且偏差单调放大。因此
   `gz_closed_form` 在越界时抛 `ValueError` 并指明是哪一条限界；诊断需要时用
   `strict=False` 显式接管。
3. **两个不同的甲板浸水角**。`deck_immersion_limit_deg` 是「吃水固定」的几何界
   （式 (1) 的适用范围终点），`equilibrium_deck_immersion_deg()` 才是等位移浮态的
   实际甲板浸水角；`shallow_draft_barge` 分别是 33.690° 与 36.870°，因为龙骨出水后
   浮体上浮 0.50 m（sinkage = −0.500 m）使甲板边更晚浸水。混用会低估可用倾角 3.18°。
4. **负 GM 工况的相对误差不可用**。`GM ≤ 0` 时 GZ 过零、以 GZ 为分母的相对误差失去
   意义（`negative_gm_barge` 的 35.3% 属此类，已在表 C 标注 n/a 口径）；该工况改用
   初稳性消失角恒等式验收（表 E）。
5. **甲板浸水之后不建模**。越过等位移甲板浸水角，水线上甲板、淹水与破舱稳性均超出
   本模块，`section_state` 抛错；不做外推。横倾之外的自由液面、纵倾、非矩形舱容、
   规则符合性全部不在范围内。
6. 计算量为 2D 多边形裁剪与二分，单工况 44 个倾角毫秒级，不占用 FE 算力。

## 6 复现

```bash
PYTHONPATH=src python3 benchmarks/marine_large_angle_stability.py   # 证据 + 表 + 图
PYTHONPATH=src python3 -m pytest tests/test_marine_large_angle_stability.py -q
```

测试共 31 项，含两条路径的逐点一致性、位移守恒、两条截止限界的 fail-closed、
`sinkage` 符号与量级、负 GM 消失角恒等式、与已合并 `marine_hydrostatics` 的一致性，
以及「证据 JSON 与当前代码不得漂移」的回归断言（
`test_evidence_matches_a_fresh_recomputation`）。