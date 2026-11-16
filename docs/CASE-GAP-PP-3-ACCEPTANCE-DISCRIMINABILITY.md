# CASE — GAP-PP-3 验收判据的可判定性（verifier 席复核）

锚点：`GAP-PP-3-ICE-STRUCT-PLASTIC-SET`（板格低温钢弹塑性 + 永久挠度，TensorFEM）。
本单只做一件事：把总账验收里那条 **「塑性极限载荷与 Ilyushin/上限分析解误差 < 10 %」**
从「待重述」推进到**可判定**——给出判据本身能否被满足的判决程序，并用一道实测数字判决它。
不伪造 FE 极限载荷数字（真实梯子作业仍在队列，见 §5）。

---

## 1. 案例与参数

| 项 | 值 |
|---|---|
| 板格 | a × b × t = 2.0 × 1.0 × 0.020 m，四边简支 |
| 材料 | DH36，−40 °C，`flow_stress_at_temperature` → σ_y = 400.0 MPa |
| 塑性弯矩 | m_p = σ_y t²/4 = 40 000 N·m/m |
| 冰斑载荷 | 中心 0.5 × 0.5 m 局部压力 p |

## 2. 方法与流程

1. 用已合并的 `plastic_bound_bracket()` 给出**可采纳区间**（下界 `strip_lower`；
   上界取 Johansen 机制与金字塔机制的较小者，含扭转项）。
2. 新增 `criterion_discrimination(upper, lower, tolerance)`：极限载荷只被知道落在
   `[lower, upper]` 内，因此**任何正确解**对上限的相对误差下界是 `1 - lower/upper`；
   当该下界 > tolerance 时，判据对任何实现都不可满足（不是实现的问题，是判据的
   问题）。可判定的充要条件是区间宽度 ≤ `1/(1-tolerance)`（10 % ⇒ 1.111）。
3. 新增离散 Johansen 机制 LP（`scripts/plate_mechanism_lp_bound.py`）：在三角化网格上
   优化全部分片线性机构 `min Σ m_p|θ|l / ∫_patch w dA`（`w=0` 于简支边界，因此每个
   返回值都是**严格上界**；网格加密只会向下收敛到真解）。
4. 判据层接进 `scripts/patch_plate_plastic_mesh_ladder.py --analyse`，acceptance 表里
   多一格 `criterion_discriminative`，由 artifacts 现算。

## 3. 计算结果（程序化注入，无手抄）

区间（MPa）：

| 量 | 值 |
|---|---|
| `lower_strip` 下界 | 0.6095 |
| `upper_yield_line` 金字塔上界 | 2.1943 |
| `upper_ilyushin_sine`（旧参考，非可采纳） | 1.8237 |
| `upper_johansen_twist` | 2.6211 |
| `tightest_admissible_upper` | 2.1943 |
| 区间宽度比（可采纳） | 3.600 |

判据判决（tolerance = 10 %）：宽度 3.600 vs 允许 1.111 ⇒ **not_discriminative**；
任何正确解对上限的最小相对误差 = **72.2 %**；要把判据变成对求解器的检验，区间还需收紧
**3.24 倍**。

机制 LP 校验：

| 场景 | LP 上界 | 精确/闭合解 | 判定 |
|---|---|---|---|
| 正方形板、满布载荷 n=6/8/12/16 | 24.0000 m_p/a² | 24 m_p/a²（精确 Johansen） | 复现精确解（误差 0.00 %） |
| 4 × 0.5 m 长板、满布载荷 16×4 | 37.565 m_p | 单向极限 32 m_p（a→∞），条带下界 32.5 m_p | 落在区间内，合法上界 |
| 锚点板格 0.5×0.5 斑 16×8 | 63.045 m_p = 2.5218 MPa | 闭合式金字塔 54.857 m_p = 2.1943 MPa | LP **更松**（见 §5） |

图：`artifacts/gap_pp3_acceptance_discriminability.png`（左：区间与两种上界；右：正方形板
LP 上界对精确解 24 的收敛）。数据：`artifacts/gap_pp3_acceptance_discriminability.json`。

## 4. 误差判定

* LP 对精确正方形板极限：24.0000 vs 24 → **0.00 %，< 3 % ✔**（这是对工具的校验）。
* 判据判决本身是算术恒等式，无基准误差（下界 `1 - lower/upper` 是集合意义上的下确界）。
* 本单**没有**给出任何新的 FE 极限载荷，因此不存在「FE vs 解析」的百分比；不声明。

## 5. 注意点 / 局限（honest）

* **LP 没能收紧区间**：金字塔机构的脊线是「斑角 → 板角」斜线，不在网格边上，因此
  可达分辨率下 LP 上界（63.3 m_p）比闭合式（54.9 m_p）**更松**。结论方向是
  「闭合式金字塔值仍是最佳可报告参考」，LP 的作用是**独立的正确性校验**（正方形板精确 24），
  而不是收紧工具。
* **开发中抓到并修掉的真实缺陷**：铰线转角若把两片三角的系数向量**相加**（而不是相减），
  对称机构上系数正好抵消，LP 会输出 `p_u = 0` 这种"零耗散机构"。`theta = |s1 - s2|` 的
  符号错是静默的，只有和独立算出的二面角逐铰比对才能抓住——该比对已固化为测试不变式。
* 判据 `not_discriminative` 的结论**不降低**锚点的能力要求：材料/几何/永久挠度后处理都在，
  缺的是**更紧的上界**（或换用与区间相容的判据：区间准入 + 收敛值 + 区间宽度报出）。
* 三段网格梯子（nx = 8/12/16）本单已递交 `cj4560` / `cj4562` / `cj4563`，交付时仍在
  `queued`：**3 档网格收敛 = not_run**，`塑性极限载荷 vs 上界` = **not_discriminative**
  （判据层面的否定结论，而不是"没做"）。

## 6. 复现

```bash
PYTHONPATH=src:scripts python -m pytest tests/test_plate_mechanism_lp_bound.py -q   # 14 passed
PYTHONPATH=src:scripts python scripts/gen_gap_pp3_discriminability_figure.py
PYTHONPATH=src:scripts python scripts/plate_mechanism_lp_bound.py
```