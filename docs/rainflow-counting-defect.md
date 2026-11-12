# Rainflow counter defect: closing cycles counted as half cycles

Status: **defect reproduced, not fixed** · module `tensorfem.marine_fatigue_advanced`
(`rainflow_cycles`) · consumer `tensorfem.rainflow_spectrum_reference`,
`tensorfem.weld_fatigue_ice`, `marine_benchmark_suite` qualification
`marine.fatigue_advanced.rainflow.triangle_count`
· reproduction `tests/test_rainflow_counting_defect.py` (2 strict-xfail + 3 green)
· date 2026-11-08 · verifier seat (tp-ver-1)

## 1. 案例介绍（背景）

`rainflow_cycles` 是 TensorFEM 全部疲劳数字的最前端：热点应力谱的 Miner 损伤、
焊缝筛选、冰激振动的循环计数都经它配对。它的输入是应力时序，输出是
`(range, mean, count=1.0|0.5)` 的循环集合。计数结构错了**不会报错、不会破坏总量
记账**，只会改变损伤值——属于典型的静默错误（silent error），必须用可解析判据
逐项核对。

本审计的锚点（全部可手算复核，见 §2）：
* 等幅块（amplitude a，n 个周期）→ 必须得到 **n 个 full cycle、range 2a**；
* 几何嵌套记录 `0,1,-1,2,-2,3,-3,0`（闭合）→ 按"闭合点插值"定义分解为
  `1,1,2,2,3,3` 六个 full cycle（记录从未回到 ±2、±3，所以不存在 range 4/5/6 的循环）；
* 三角波 `0,100,0` → 1 个 full cycle；
* 弧守恒律：把记录蒸馏成反转点后，`Σ count×range = ½ Σ|Δs|`（每条弧恰被消费一次）。
  该律对闭合/开记录都成立，是**唯一**的硬记账判据。

参数：全部为无量纲应力单位（代码内部 SI，判据与单位无关）；S-N 曲线
`SNLogLogCurve(100e6, 2e6, m=3.0)`（用于把循环结构换成损伤数值）。

## 2. 怎么做的（方法与工具链）

1. 只读代码：`rainflow_cycles`（三点栈算法）、`_reversals`（反转点蒸馏）、
   `rainflow_count_invariant`（记账判据）、下游 `spectrum_miner_damage`。
2. 独立构造上述锚点（手算推导写在 docstring 与本文件），与实现逐条对照；
3. 用弧守恒律对 300 条随机闭合记录做批量核对（程序化，无手抄数字）；
4. 试做一版修复（保留插值闭合层级 + 单调段合并），再用同一批判据回归——
   结论见 §5：**该修复被自己的判据否决**，因此没有提交；
5. 用 `pytest` 把"今天成立的不变量"钉成绿色测试，把缺陷钉成 `xfail(strict=True)`
   回归绊线（修好即 xpass→红，强制同步更新）。

命令：
```
python3 -m pytest tests/test_rainflow_counting_defect.py \
                 tests/test_marine_fatigue_advanced.py \
                 tests/test_marine_fatigue_spectrum_reference.py -q
# 28 passed, 2 xfailed  (2026-11-08, 本机 python3.11 / torch CPU)
```

## 3. 计算结果（数字表）

栈实现的判据是：栈顶三点 A,B,C，若 `|A-B| ≤ |B-C|` 则 A→B 闭环成立。
问题出在代码对 `len(stack)==3` 的特判：它不是记 `count=1.0` 的 full cycle，
而是记 `count=0.5` 并弹掉栈底元素——于是**只要一条记录靠前段落就闭环，
后面每一步都会被拖进这个分支**，整条记录退化成"相邻反转区间逐个记 half cycle"。

| 记录 | 期望（rainflow 定义） | 实测（当前实现） | 判定 |
|---|---|---|---|
| `0,10,0` | 1 full, range 10 | 2 half, range 10（合 1 等效循环） | 结构错、损伤等价 |
| `0,10,0,10,0,10,0` | 3 full, range 10 | **0 full**, 6 half, range 10 | 结构错、损伤等价 |
| `0,1,-1,2,-2,3,-3,0` | 6 full, range 1,1,2,2,3,3 | 7 half, range 1,2,3,4,5,6,3 | 结构错、**损伤错** |
| `0,5,0,5,0,20,0,20,0` | 2×5 + 2×20 full | 4 half 5 + 4 half 20 | 结构错、损伤等价 |
| 三角波 `0,100,0` | 1 full | 2 half | 结构错、损伤等价 |
| 40 周期正弦 | 40 等效循环 | 40 等效循环（全 half） | 仅结构错 |

损伤影响（m = 3，单位任意）：嵌套记录
* 期望：`1+1+8+8+27+27 = 72`
* 实测：`0.5×(1+8+27+64+125+216+27) = 234`
* **偏差 +225 %（实测/期望 = 3.25）**，远超平台"误差 >15 % 不可接受"的线。

关键量逐项判据（要求 <3 %）：

| 关键量 | 基准 | 实测 | 相对误差 | <3 %? |
|---|---|---|---|---|
| 等幅块的 full cycle 个数（3 周期） | 3 | 0 | 100 % | **否** |
| 嵌套记录的循环集合 | {1,1,2,2,3,3} | {1,2,3,4,5,6,3}÷2 | 集合不符 | **否** |
| 嵌套记录损伤（m=3） | 72 | 234 | 225 % | **否** |
| 弧守恒 `Σcount·range = ½Σ|Δ|`（随机闭合记录，300 条） | ½Σ|Δ| | 相对残差 max 3.9e-16 | ~0 | 是 |
| 三角波等效循环数 | 1 | 1 | 0 % | 是 |
| 40 周期正弦等效循环数 | 40 | 40 | 0 % | 是 |

结论：**记账（弧守恒）与等效循环数是对的，循环配对结构是错的**；错配对在
"等幅/嵌套/多级块"这类结构化载荷谱上把损伤抬到 2.25 倍，而在窄带随机谱上
影响很小（见 §5）。

## 4. 为什么现有测试没抓住

`tests/test_marine_fatigue_spectrum_reference.py::test_counting_identity_holds_for_closed_records`
用的判据是 `Σ count == (n_rev-1)/2`。这个所谓"恒等式"只在**不插值闭合点**的
变体里成立，是缺陷实现的副产品：它无法区分"3 个 full cycle"和"6 个 half cycle"，
也无法区分"正确的 1,1,2,2,3,3"和"错误的逐区间 1,2,3,4,5,6,3"（两者
`Σcount·range` 都等于 ½ 路径长度）。用弧守恒替代它才有判别力。

`test_rainflow_triangle_has_two_half_cycles_and_one_equivalent_cycle` 则把
"闭合三角=两个半循环"当成规格固定下来——这正是缺陷入口的合法化。

## 5. 试修复与它的失败（为什么本单不交付补丁）

已实现并回归过两版修复（保留插值闭合层级 + 单调段合并），在结构化锚点上全对
（等幅 → n full；嵌套 → 1,1,2,2,3,3；三角 → 1 full；300 条随机闭合记录弧守恒
残差 3.9e-16），但被另外两条判据否决：

| 判据 | 阈值 | 修复版实测 | 判定 |
|---|---|---|---|
| 窄带随机谱损伤 vs Rayleigh 解析律 | < 3 % | **72.9 %** | 否决 |
| 40 周期正弦等效循环数 | 40 | **80（每周期 2 个）** | 否决 |
| 记录起点平移后循环数变化 | ≤ 1 | **261** | 否决 |

失败原因（对后续修复者是关键情报）：一旦把"闭环时的插值闭合层级"留在栈上并在
后续复用，就会在**没有真实反转点的层级**上反复闭环——正弦（从 0 起振）每半周期
就闭一个 range 50 的循环，count 翻倍；而教科书答案是每周期 1 个 range 100 的循环。
正确的做法是按 **ASTM E1049 四点规则**（`|S3-S2| ≤ |S2-S1|` 且 `|S3-S2| ≤ |S4-S3|`
才闭内环），闭合点必须是路径上真实存在过的反转点；三点规则不足以区分"插值闭合"
与"开放端"。修复规格（三条判据，缺一不可）：

1. 结构化锚点：等幅 n full、嵌套 {1,1,2,2,3,3}、三角 1 full、两级块 {5×2,20×2}；
2. 窄带随机谱：`narrow_band_damage_case(...)["relative_error"] < 0.03`（现有门）；
3. 正弦 40 周期 → 40 等效循环、相邻反转区间不被当循环；起点平移损伤散布 < 2 %。

## 6. 注意点（局限与适用范围）

* 本单**没有**修复缺陷：`rainflow_cycles` 源码未改动，行为与 2026-11-08 之前一致；
  交付物是复现、量化、判据与失败路线的证据。两个 `xfail(strict=True)` 测试是
  修复绊线，修好后它们会红，逼迫同步更新本文件与判据。
* 结构化载荷谱（等幅块、嵌套、多级块）上的损伤偏差已量化（m=3 时 2.25 倍）；
  **窄带随机谱上偏差很小**（现有 3 % Rayleigh 门在当前实现下仍通过），所以下游
  已有窄带结论不必撤回，但"block program / 结构化谱"类结论不可用。
* 占位约定（未复核）：闭记录残留按半循环计（本实现与常见工程约定一致），
  我未能对照 ASTM E1049 原文图例核对该约定，故标 WITHHELD。
* 弧守恒律对当前实现**成立**，因此它不能单独作为"计数正确"的证据——必须与
  结构化锚点同时使用；这是本审计给后续验证者的方法论教训。

## 7. 证据指针

* 复现与绊线：`tests/test_rainflow_counting_defect.py`（2 xfail strict + 3 绿）
* 现有测试：`tests/test_marine_fatigue_advanced.py`、
  `tests/test_marine_fatigue_spectrum_reference.py`（28 passed / 2 xfailed）
* 代码位置：`src/tensorfem/marine_fatigue_advanced.py::rainflow_cycles`
  （`len(stack) == 3` 分支 + `del stack[-3:-1]`）
* 下游受影响：`spectrum_miner_damage`、`narrow_band_damage_case`、
  `weld_fatigue_ice`、`marine_benchmark_suite` 的
  `marine.fatigue_advanced.rainflow.triangle_count`（该例的等效循环数仍为 1，
  资格门不受影响）

## 8. 后续验证（2026-11-12, tp-ver-7 / mission 16502）：§5 的修复规格不可满足

独立复核 `docs/verification/rainflow-fix-spec-contradiction.md`（复现
`benchmarks/verified/rainflow_fix_spec/run.py`，绊线
`tests/test_rainflow_fix_spec_satisfiability.py`）的结论：

* §5 的 **C1（嵌套 → full 1,1,2,2,3,3）与 C3（40 周期正弦 → 40 循环、max 2A）互斥**：
  满足 C1 的唯一规则（在插值闭合层级闭环）必然把正弦每周期劈成两个 range=A 的循环，
  `Σcount` 翻倍到 80、`max_range` 减半到 50，并让工程门 C2 的 Rayleigh 误差从 0.91%
  恶化到 78.1%（损伤降至律的 0.44 倍）。
* §5 推荐的 **ASTM 四点规则亦不满足 C1**：嵌套记录上给出 full `{1,2,5}` 且弧守恒
  残差 −4（丢弧）。其残余配对约定是否合于标准原文，**本单未定论（WITHHELD）**。
* 因此 §5 的"三条判据缺一不可"实际上**没有解**；修复者应先改规格（用真实反转点
  闭环的教科书答案替换 C1 的嵌套期望），再改算法。缺陷本体仍未修复，两个
  `xfail(strict=True)` 绊线保持有效。
