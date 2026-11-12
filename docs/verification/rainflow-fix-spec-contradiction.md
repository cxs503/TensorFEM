# 雨流计数修复规格的可满足性审计：C1 与 C3 互斥

Status: **spec falsified, defect still open**（规格被证伪，缺陷仍未修复）·
对象 = `docs/rainflow-counting-defect.md` §5 的三条修复判据 ·
验证器席 tp-ver-7 / mission 16502 ·
复现 = `benchmarks/verified/rainflow_fix_spec/run.py`（结果 `result.json`，图 `rainflow_fix_spec.png`）·
绊线 = `tests/test_rainflow_fix_spec_satisfiability.py`（5 passed）

## 1. 案例介绍（背景）

`tensorfem.marine_fatigue_advanced.rainflow_cycles` 是 TensorFEM 全部疲劳数字的前端：
热点应力谱 → 循环配对 → Miner 损伤 / 焊缝筛选 / 冰激振动寿命。2026-11-08 的验证单
（tp-ver-1）已把它钉成**结构错但记账对**的缺陷，并在 §5 给出"修复规格（三条判据，
缺一不可）"，同时登记了两次修复尝试被否决的过程。本单的任务不是再试一版修复，而是
**先检验这份规格本身是否可满足**——因为一个不可满足的规格会让后续每一位修复者重复
同一圈失败（2026-11-09 owner 不重复令）。

物理背景：雨流计数（ASTM E1049-85 / Downing & Socie 1982）把一段应力-时间历程
分解为闭环滞回环（full cycle, count=1）与残余半环（half cycle, count=0.5），
损伤按 Miner 线性累积 `D = Σ count·N(S)^-1`。计数的**唯一硬记账律**是弧守恒
`Σ count·range = ½ Σ|Δs|`（反转点蒸馏后每条弧恰被消费一次）；除此之外的"结构"
判据（哪些环是 full、在哪个层级闭环）取决于**闭环层级是否允许插值**。

判据原文（§5 摘要）：

* **C1** 结构化锚点：等幅块 → n full；嵌套 `0,1,-1,2,-2,3,-3,0` → full 范围 `1,1,2,2,3,3`；
  三角 `0,100,0` → 1 full；两级块 → `5,5,20,20` full。
* **C2** 窄带随机谱：`narrow_band_damage_case(...)["relative_error"] < 0.03`（工程门）。
* **C3** 40 周期正弦 → 40 等效循环、最大范围 `2A`、相邻反转区间不得被当循环。

参数：应力单位无量纲（判据与单位无关）；正弦振幅 `A = 50`、40 周期、200 采样/周期；
窄带谱 `relative_bandwidth=0.05, cycles=2000`，S-N 曲线 `SNLogLogCurve(100e6, 2e6, m=3)`。

## 2. 怎么做的（方法/流程/工具链）

1. 三个候选计数器，全部按同一 `RainflowCycle(range, mean, count)` 契约实现于
   `benchmarks/verified/rainflow_fix_spec/run.py`：
   * `production` —— 合并版实现（三点栈 + `len(stack)==3` 特判，即缺陷本体）；
   * `four_point` —— §5 亲自推荐的 ASTM 四点规则（`|S3-S2| ≤ |S2-S1|` 且
     `|S3-S2| ≤ |S4-S3|` 才闭内环），闭记录循环扫描 + 残余端对端配对；
   * `remainder` —— 半环"余量"配对（pagoda/range-pair 形式）：在**插值闭合层级**
     闭环，未用完的部分留在栈上继续配对（这是唯一能复现 §5 嵌套期望的规则）。
2. 判据程序化评分：C1 要求 `full_ranges` 精确等于期望集合 **且** 弧守恒残差 `<1e-6`；
   C3 要求 `|Σcount−40|/40 < 5%` **且** `max_range ≈ 2A` **且** 弧守恒；
   C2 通过 monkeypatch `rainflow_spectrum_reference.rainflow_cycles`，让工程门
   （Rayleigh 窄带解析律）直接给每个候选打分，**不改动生产模块**。
3. 全部数字由 `run.py` 写入 `result.json`，本文件表格**程序化复制**（零手抄），
   图由 `make_figures.py` 读同一 json 生成。
4. 复现命令：`PYTHONPATH=src python benchmarks/verified/rainflow_fix_spec/run.py`
   （约 8 s，纯 CPU）；判据回归 `pytest tests/test_rainflow_fix_spec_satisfiability.py`。

## 3. 计算结果（数字表）

锚点结构与弧守恒（`half_path` = ½ 路径长，即 `Σcount·range` 的应收值）：

| 锚点 | half_path | production | four_point | remainder | C1/C3 期望 |
|---|---|---|---|---|---|
| 等幅块 3 周期 | 30 | 6 half ×10（Σcount 3） | **3 full ×10** | **3 full ×10** | 3 full ×10 |
| 嵌套 zigzag | 12 | 7 half {1,2,3,3,4,5,6}（Σcount 3.5） | full {1,2,5}，**弧残差 −4** | **full {1,1,2,2,3,3}** | full {1,1,2,2,3,3} |
| 三角 `0,100,0` | 100 | 2 half ×100 | **1 full ×100** | **1 full ×100** | 1 full |
| 两级块 `5/20` | 50 | 8 half | **full {5,5,20,20}** | **full {5,5,20,20}** | full {5,5,20,20} |
| 40 周期正弦 | 3998.43 | Σcount 40.5，max 100 | Σcount 40.5，max 100 | **Σcount 80，max 50** | 40，max 100 |

工程门（C2，Rayleigh 窄带解析律，容差 3%）：

| 计数器 | `relative_error` | `count_rate_error` | `damage_rate_ratio` | 循环数 | C2 |
|---|---|---|---|---|---|
| production | 0.0091 | 0.0031 | 1.0123 | 2006.5 | **PASS** |
| four_point | 0.0091 | 0.0031 | 1.0123 | 2006.5 | **PASS** |
| remainder | 0.7809 | 1.0058 | 0.4396 | 4012.0 | **FAIL** |

三条判据总判定：

| 计数器 | C1 | C2 | C3 | 三者同时 |
|---|---|---|---|---|
| production | ✗ | ✓ | ✓ | ✗ |
| four_point | ✗ | ✓ | ✓ | ✗ |
| remainder | ✓ | ✗ | ✗ | ✗ |

图 `rainflow_fix_spec.png`：左 = 嵌套锚点三计数器的环分解（虚线 = 弧守恒应收值）；
中 = 正弦等效循环数（虚线 = C3 的 40 与 2A=100 目标）；右 = Rayleigh 门误差（虚线 3%）。

## 4. 结果图与关键判读

* **C1 与 C3 互斥**：唯一满足 C1 嵌套期望的 `remainder` 恰好让正弦每半周期闭一个
  range=A 的环 → `Σcount` 翻倍到 80、`max_range` 减半到 50（图中柱高与标注）。
  机理不是实现细节而是定义本身：嵌套期望 `{1,1,2,2,3,3}` 要求**在插值闭合层级
  （记录起点 0，非反转点）闭环**；正弦的每个周期在同一个插值层级 0 上闭环时，
  必然把范围 2A 的环劈成两个 A。两个锚点对"闭环层级是否可插值"给出**相反**的要求。
* **§5 推荐的 four_point 规则也过不了 C1**：嵌套记录上给出 `full {1,2,5}` 且
  **弧守恒残差 −4**（丢弧）。丢弧是硬错误，逐区间记账法（production）反而恒为 0。
  该变体的残余配对规则是否与 ASTM 四点法一致，本单**未定论（WITHHELD）**：
  需对照 ASTM E1049-85 原文图例核对残余半环约定后才能判定是规则问题还是本实现的配对问题。
* **C2 是唯一保护生产数字的判据**：`remainder` 使窄带损伤降到 Rayleigh 律的 44%
  （`damage_rate_ratio=0.4396`），因为循环率翻倍且每环范围减半（m=3 → 损伤 ≈ 1/4×2）。
  即"按 C1 修好结构"会**直接推翻**现有窄带疲劳结论，与 §5 自述"修好即三项全中"矛盾。
* **弧守恒不足以判别**（与 §6 的方法论结论一致）：production 与 remainder 弧残差均
  为 0，结构却完全不同——本单的表格再次证明必须"记账 + 结构"两把尺子同时用。

## 5. 注意点（坑/局限/适用范围）

* 本单**没有修改** `rainflow_cycles`：生产行为与 2026-11-08 之前完全一致，两个
  `xfail(strict=True)` 绊线仍绿（22 passed / 2 xfailed，见验证命令），缺陷仍开着。
* `four_point` / `remainder` 是**审计用候选实现**，放在 `benchmarks/verified/` 内，
  不进入生产路径；它们的残余配对约定为本审计自定，未经 ASTM 原文图例核对（WITHHELD）。
* 只覆盖计数结构；`spectrum_miner_damage`、`narrow_band_damage_case` 的门阈值未改。
* 正弦锚点采样 200/周期，避免 §6 记录的采样损失污染 C3 判读。

## 6. 误差判定（关键量 vs 基准，逐项回答"是否 <3%"）

| 关键量 | 基准（来源） | 实测 | 相对误差 | <3%? |
|---|---|---|---|---|
| 窄带损伤 vs Rayleigh（production） | `S_range` Rayleigh 律（Bendat 1964；DNV-RP-C203 App. B） | 1.0123 × 律 | 1.23% | **是** |
| 窄带损伤 vs Rayleigh（remainder） | 同上 | 0.4396 × 律 | 78.09% | **否**（故 C2 否决） |
| 循环率 vs 上穿率（production） | 谱矩 `ν_up T` | 2006.5 vs 2000 | 0.31% | **是** |
| 循环率 vs 上穿率（remainder） | 同上 | 4012 vs 2000 | 100.6% | **否** |
| 弧守恒残差（production / remainder，全部锚点） | `½Σ|Δs|` | ≤1e-12 | ~0% | **是** |
| 弧守恒残差（four_point，嵌套） | `½Σ|Δs| = 12` | 8 | 33% | **否**（丢弧，硬错误） |
| 嵌套循环集合（remainder vs §5 期望） | §5 C1 | `{1,1,2,2,3,3}` | 0% | **是** |
| 嵌套循环集合（production vs §5 期望） | §5 C1 | 7 half | 集合不符 | **否** |

结论：规格的 C1 与 C3 在数学上互斥，C1 又有害于工程门 C2；production 在 C2/C3 上
已 <3%，其问题局限在"结构标注"（full/half 归属）与结构化载荷谱（等幅块/嵌套/多级块）
的损伤分配，而非窄带随机谱的损伤水平。

## 7. 给下一手的具体建议（可执行）

1. **改规格，不是再改算法**：把 C1 的嵌套期望换成**真实反转点闭环**的教科书答案
   （本单推导 + 数值核对：full `{2,4}` + half `{3,6,3}`，弧守恒 2+4+6=12；
   m=3 损伤 207 vs 现 234，偏差 13% 而非 §5 声称的 225%），并在缺陷文档中登记
   "插值闭环"与"反转点闭环"两种定义的分歧与各自适用面（block program 用前者、
   随机谱用后者）。该答案需先对照 ASTM E1049-85 原文图例核定后入册（当前 WITHHELD）。
2. 修复只在满足"弧守恒 + 窄带门 + 正弦门"三项后合入；结构标注（full vs half）可先
   改为"闭环层级不得低于相邻反转点"的单点改动，再逐锚点回归。
3. 建议登记新功能条目：`slug=GAP-FEM-RAINFLOW-REVERSAL-CLOSURE`，
   验收 = 本文 §6 表中 production 的 8 项中"结构"类 2 项转绿（嵌套集合、等幅 full
   计数）**且** C2 误差仍 <3% **且** 正弦 `Σcount∈[38,42]`、`max_range∈[98,102]`，
   所属域 = 疲劳/断裂（TensorFEM·marine fatigue）。