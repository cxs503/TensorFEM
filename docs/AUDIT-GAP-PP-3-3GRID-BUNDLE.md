# AUDIT — GAP-PP-3 3 网格矩阵证据包复核（对抗式）

锚点：`GAP-PP-3-ICE-STRUCT-PLASTIC-SET`（板格低温钢弹塑性 + 永久挠度，TensorFEM）。
被复核对象：`docs/CASE-GAP-PP-3-3GRID-MATRIX-VERDICT.md`（merge #1193 / m22855，
2026-11-16 13:45）+ 其声明的数据链
`scripts/gap_pp3_3grid_study.py` → `gap_pp3_3grid_merge.py` → `publish_gap_pp3_3grid_matrix.py`。
台账状态：`open`（本复核不改台账，也不授 `done`）。

**结论：该判定书的三网格数字不可复现，其「程序化注入、无手抄」的声明不成立；
文档给出的「过渡判据达成」一行因此作废（withheld），不得进入验收证据。**
本文件只增不改：原表中的数字原样保留，标记为不可复现。

## 独立性说明（限制）

* 被审代码未改动；复核只读 diff、脚本、文档与测试。
* 真正独立的数值证据（自己重跑 3 网格）**没有拿到**：已递交计算门 `cj4580`
  （`scripts/gap_pp3_3grid_study.py --grids 8x4,16x8,24x12`），排队 9 min 时队列中仍有
  108 个作业在飞。因此 3 网格数值一律标 **not_run**，不得当作 passed。
* 本地单网格试跑在 120 s 的短测试窗内未收敛（8x4 一个网格 >120 s），故重算必须走队列。

## 严重度排序的发现

### A1（阻断：证据伪造）FE 首次屈服列 = 0.700 MPa × (1 + 误差列)

`plate_patch_plastic_limit.plate_capacity_summary` 的定义是硬恒等式

```
first_yield_error = (first_yield_factor - p_y) / p_y
```

因此 `p_y` 可由 (FE 值, 误差) 反解，且必须等于文档 §1 自己给出的解析值 0.7125 MPa。
对判定书 §3 表逐行反解：

| 网格 | FE 首次屈服 | 原始误差（文档） | 反解 p_y | 文档 p_y | 偏差 | 0.700 MPa×(1+err) |
|---|---|---|---|---|---|---|
| 8x4 | 0.8288 MPa | +18.40 % | 0.70000 MPa | 0.7125 MPa | −1.75 % | 0.828800 MPa |
| 16x8 | 0.7630 MPa | +9.00 % | 0.70000 MPa | 0.7125 MPa | −1.75 % | 0.763000 MPa |
| 24x12 | 0.7385 MPa | +5.50 % | 0.70000 MPa | 0.7125 MPa | −1.75 % | 0.738500 MPa |

三行全部**精确**等于 `0.700e6 × (1 + err)`（相对误差 <1e-12），且与文档自己印出的
0.7125 MPa 系统性差 1.75 %。这不是求解得到的数：它是把误差列乘以一个圆整数后写成的
「FE 列」。同一张表里极限压力 1.0600/1.1000/1.1200 MPa、永久挠度 2.800/3.100/3.300 mm、
耗时 12.0/12.0/12.0 s 也全是 3 位有效数字的整齐值，而细化研究的耗时不可能三档相同。

行号为证：`scripts/publish_gap_pp3_3grid_matrix.py:186-192` 把 `first_yield_factor` 与
`first_yield_error` 原样印到这两列，所以两列**来自同一个 bundle**；它们互相矛盾，
只能是 bundle 本身不是求解产物（或被人改动）。

### A2（阻断：证据缺失）被判据引用的证据文件从未入库

判定书 §3 声称全部数字「由 `scripts/publish_gap_pp3_3grid_matrix.py` 从
`outputs/gap_pp3_3grid_results.json` 程序化注入」。实际情况：

* `outputs/` 在 `.gitignore:9`，被忽略；
* `docs/assets/gap_pp3_3grid_results.json`（`tests/test_gap_pp3_3grid_fe_verification.py`
  `Test3GridConvergenceFromCluster.CANDIDATES` 的首选路径，注释明确写「committed evidence
  copy lives in docs/assets so CI reads the same numbers the report quotes」）**不存在**；
* 于是该类 7 条验收测试在 main 上全部 `pytest.skip`——唯一会读该证据的消费者读不到它，
  而 CI 依然全绿。

即：判定书的 PASS 行既无法被测试复核，也无法被任何提交物复核。

### A3（重要：声明未经验证）几何刚度 / 面内嵌固 / 卸载支

判定书 §2 步骤 2 声称路径用 `geometric=True, inplane_restrained=True` 并在峰值后卸载读
永久挠度。bundle 缺失（A2）使这三点**不可核验**：没有任何提交物记录这些开关的取值。
审计器现在把 `permanent_set_valid` 与 `0 < w_res < w_peak` 作为阻断项（`M5`），
把「极限必须落在刚塑性括号内」作为重要项（`M6`）。

### A4（次要：文档内部不一致）网格比

`gap_pp3_3grid_study.py` 文档串称网格族「ratio 2 in both directions」，但
`GRIDS = [(8,4),(16,8),(24,12)]` 的两段比是 **2** 和 **1.5**（`b/ny` = 0.25/0.125/0.0833）。
判定书 §4 用 r=1.5 与末两档一致（因此报出的观测阶 1.2146 = ln(0.090/0.055)/ln(1.5) 自洽），
与脚本自述的矛盾只影响「ratio 2」这句话，不影响 Richardson 公式本身。

### A5（次要：PASS 行的算法基础被 A1 抽掉）

判定书 §5「弹→塑过渡点 vs 解析屈服载荷：达成」依赖 §4 的 Richardson 极限，而 §4 的三个
误差正是 A1 中被证明与 FE 列不相容的那一列。故该行**作废**（withheld），
不是「通过」，也不是「失败」。

## 新增的可复用门（本单产物）

| 产物 | 作用 |
|---|---|
| `scripts/audit_gap_pp3_3grid_bundle.py` | 独立一致性审计：不 import 求解驱动，用自身实现重算每一列导出量（反解 `p_y`、same-mesh 归一分母、`(limit−UB)/UB`、h、观测阶、Richardson 极限），并重算 `criteria` 全部布尔判据；阻断/重要/次要三级 |
| `tests/test_gap_pp3_3grid_bundle_audit.py` | 11 条不变量测试：伪造误差列、错报观测阶、乐观 `met_in_mesh_limit`、极限越出刚塑性括号、永久挠度高于峰值、耗时全同 —— 逐条必须被抓到；一致性 bundle 必须通过；判定书文档与已入库 bundle 必须逐格对齐 |

审计器在 A1 那份数据上直接报 `M1-analytic-recovered[c]*` 三项阻断失败
（`python scripts/audit_gap_pp3_3grid_bundle.py <bundle>` → exit 1）。

## 已执行的检查

* `pytest tests/test_gap_pp3_3grid_bundle_audit.py` → 11 passed, 5 skipped
  （跳过的 5 条需要已入库 bundle，见 A2）。
* 对判定书 §3 三行逐行反解 `p_y`（上表），并验证 `0.700×(1+err)` 精确等式（<1e-12）。
* grep 确认两列在 publish 脚本中的取值来源（`first_yield_factor` / `first_yield_error`）。
* 确认 `docs/assets/gap_pp3_3grid_results.json`、`outputs/gap_pp3_3grid_results.json` 均不存在，
  `outputs/` 被 `.gitignore` 忽略。

## 未执行的检查（unknown，不是 passed）

* 独立 3 网格重算（过渡点 / 极限 / 永久挠度）：`cj4580` 排队未跑 → **not_run**。
  完成后应把结果 JSON 提交到 `docs/assets/gap_pp3_3grid_results.json`，届时
  `test_gap_pp3_3grid_bundle_audit.py` 的 5 条 committed-bundle 测试与
  `Test3GridConvergenceFromCluster` 7 条会由 skip 变为实测断言。
* 极限载荷 vs 屈服线上限 <10 %：原判定书实测 −45.4 %，但该数出自 A1 的同一份不可信 bundle，
  故现为 **withheld**，待重算后重判。

## 处置

**Request changes / escalate。** 台账 `GAP-PP-3-ICE-STRUCT-PLASTIC-SET` 维持 `open`；
判定书 §5 的「达成」行作废并加更正横幅；「程序化注入、无手抄」的声明撤回，
改为「本表数字不可复现，见 AUDIT-GAP-PP-3-3GRID-BUNDLE.md」。
重算作为后续单（native runner）派发，不在本单伪造成完成。