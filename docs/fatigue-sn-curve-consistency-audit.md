# 疲劳 S-N 曲线跨模块一致性审计：两处实际缺陷 + 一条失效门禁的修复

锚点：【锚点=NEW:跨模块 S-N 曲线一致性与门禁独立性审计（TensorFEM 疲劳面 8 实现）】
（总账中 `GAP-FEM-*` 已全部 done/verified，本单是总账外新发现的验证缺口，建议新条目见文末）
Library: `TensorFEM`。新增: `src/tensorfem/fatigue/sn_curve_consistency.py`、`tests/test_fatigue_sn_curve_consistency.py`、
`benchmarks/fatigue_sn_curve_consistency.py`。工件: `artifacts/fatigue_sn_curve_consistency.json`、
图 `docs/assets/fatigue_sn_curves.png`。

## ① 案例介绍（物理背景 / 范围 / 参数）

船体焊接节点的疲劳寿命由 S-N 曲线给出：`log10 N = log10 a - m log10 Δσ`，DNV-RP-C203
curve D **in air** 的出版值为 **`log a1 = 12.164`（m1 = 3，N ≤ 1e7）、`log a2 = 15.606`（m2 = 5）**，
膝点 `N = 1e7` 对应 `Δσ = 52.642 MPa`（FAT90 @ 2e6 循环）。本库对同一条曲线有 **8 个独立声明**：

| 模块 | 角色 |
|---|---|
| `structural_fatigue_detail.DNVDualSlopeSN` | 细节疲劳成熟化（本单修复对象之一） |
| `weld_fatigue_ice.DNVSNCurve` | 冰区焊接节点疲劳 |
| `weld_fatigue_ice.IIWFatClass` | IIW FAT90 类别式 |
| `fatigue.weld_toe.ENVIRONMENTS[air / seawater_cp / seawater_free]` | 焊趾热点应力法三环境 |
| `propeller_ice_milling_fatigue_chain.dnv_d_curve_in_air` | 螺旋桨冰铣削疲劳链 |
| `ice_slamming_whipping.SNCurve` | 冰激振动/颤振疲劳 |

每个模块**自洽**（自己两段在膝点相接），所以单独看任何一篇报告都通过；跨模块漂移只能靠
独立对照发现——这正是本次审计的靶子。四个检查项：①两段在膝点连续性；②`log a1 / log a2`
对出版值；③**分支受控**的斜率恒等式 `N(σ)/N(2σ) = 2^m`；④数据库声明的常数与对象实际返回一致。

## ② 怎么做的（方法 / 流程 / 工具链）

1. 新模块 `sn_curve_consistency.py` 收集 8 个实现**自报的常数**（不复制、不硬编码进审计），
   对每个实现用出版表独立重算寿命 `reference_life()`，再与对象自己的 `cycles_to_failure()`
   比（`declared_vs_callable`，抓「文档写着 DNV、代码跑着别的」）。
2. 斜率恒等式一律**分支受控**：m1 用 `1.4σ_knee` 与 `2.8σ_knee`（都在膝上），m2 用 `0.9σ_knee`
   与 `0.45σ_knee`（都在膝下，且避开 IIW 的 `1e9` 循环耐久极限截断——踩坑见 ⑤）。
3. 复现 v1 的 m=5 协议（`100 MPa / 50 MPa`）作为**负控**，定量记录它测到了什么。
4. 回归测试 8 项 fail-closed；基准脚本把全部数字写进 JSON（零手抄），图由 JSON 直接绘制。

## ③ 计算结果（程序化注入，来源 `artifacts/fatigue_sn_curve_consistency.json`）

### 3.1 八实现逐项审计

| implementation | standard | log a1 | m1 | log a2 | m2 | knee [MPa] | knee continuity | N(100 MPa) | ΔN@100MPa | ΔN@30MPa |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `structural_fatigue_detail.DNVDualSlopeSN` | DNV-RP-C203 D air | 12.164000 | 3 | 15.606000 | 5 | 52.642 | 1.53e-03 | 1.458814e+06 | +0.000 % | +0.000 % |
| `weld_fatigue_ice.DNVSNCurve` | DNV-RP-C203 D air | 12.164000 | 3 | 15.606000 | 5 | 52.642 | 1.53e-03 | 1.458814e+06 | +0.000 % | +0.000 % |
| `weld_fatigue_ice.IIWFatClass` | IIW FAT 90 | 12.163758 | 3 | 15.606263 | 5 | 52.632 | 1.86e-15 | 1.458000e+06 | -0.056 % | +0.060 % |
| `fatigue.weld_toe.ENVIRONMENTS['air']` | DNV-RP-C203 D air | 12.164000 | 3 | 15.606667 | 5 | 52.642 | 3.35e-15 | 1.458814e+06 | +0.000 % | +0.154 % |
| `fatigue.weld_toe.ENVIRONMENTS['seawater_cp']` | DNV-RP-C203 D seawater_cp | 12.164000 | 3 | 16.273333 | 5 | 113.414 | 1.86e-15 | 1.876434e+06 | +28.627 % | +364.872 % |
| `fatigue.weld_toe.ENVIRONMENTS['seawater_free']` | DNV-RP-C203 D seawater_free | 11.764000 | 3 | 15.606667 | 5 | 83.432 | 2.33e-16 | 5.807644e+05 | -60.189 % | +0.154 % |
| `propeller_ice_milling_fatigue_chain.dnv_d_curve_in_air` | DNV-RP-C203 D air | 12.164000 | 3 | 15.606000 | 5 | 52.642 | 1.53e-03 | 1.458814e+06 | +0.000 % | +0.000 % |
| `ice_slamming_whipping.SNCurve` | DNV-RP-C203 D air | 12.164000 | 3 | 15.606667 | 5 | 52.642 | 7.45e-16 | 1.458814e+06 | +0.000 % | +0.154 % |

汇总：`n_implementations = 8`，curve D in air 实现 `5` 个；
`max|Δlog a1| = 0.0e+00`；100 MPa 处最大寿命偏差
`0.000 %`，30 MPa 处 `0.154 %`；
`max declared-vs-callable = 2.00e-15`；`passed = True`。

### 3.2 缺陷 1（严重）：m=5 验证协议跨膝取样，判据测的不是第二段

| 量 | v1 协议 | 分支受控协议（本单） |
|---|---|---|
| 取样 | `Δσ = 100 MPa` 与 `50 MPa` | `0.45σ_knee = 23.69 MPa` 与 `0.225σ_knee = 11.84 MPa` |
| 膝点 | 52.642 MPa（两探针**分居两侧**，`straddles_knee = True`） | 两探针同在膝下 |
| 实测比 `N(σ/2)/N(σ)` | **8.854213**（期望 32） | **32.000000** |
| 相对误差 | **72.33 %** | **0.0e+00 %** |
| 该判据 `passed` | False（对一条**正确**的曲线误判不合格） | True |

100 MPa 在 DNV curve D 上属于 **m=3 段**（`N = 1.459e6 < 1e7`），所以 v1 实际测的是
`N(50 MPa, m=5)/N(100 MPa, m=3) = 8.854`。同时 `tests/test_structural_fatigue_detail.py`
里另写了一份 **正确**的检查（用 `0.5σ_knee / 0.25σ_knee`），于是「测试绿」掩盖了「出厂门禁红」——
这是本单最重要的教训：**测试复写判据 ≠ 判据本身被测试**。

### 3.3 缺陷 2（中）：`ice_slamming_whipping.SNCurve` 常数偏离出版值 4.19 %

| 量 | v1 声明 | 修正后 | 出版 DNV curve D in air |
|---|---|---|---|
| `C1` | 1.520000e+12 | 1.458814e+12 | `10^12.164 = 1.458814e12` |
| `log a1` | 12.181844 | 12.164000 | 12.164 |
| `C2` | 4.340000e+15 | 4.042655e+15 | `10^15.606`（连续性锚定） |
| 膝点应力 | 53.368 MPa | 52.642 MPa | 52.642 MPa |
| `N(100 MPa)` 偏差 vs 出版 | **+4.194 %**（偏不保守） | +0.000 % | 0 |

同一库内两条「DNV curve D in air」相差 4.19 % 寿命，冰激疲劳是冰区强度校核的直接输入，属实质缺陷。

### 3.4 缺陷 3（轻）：文档算术错误与「不可能失败」的门禁

* `DNVDualSlopeSN` 文档串写 `Δσ_knee = 10^((12.164-7)/3) = 169.3 MPa`；实际为 **52.642 MPa**
  （代码本来就对，只有文档错）。已改并加注说明。
* 疲劳寿命判据原为 `"passed": True  # computed correctly if S-N curve is valid` —— 一个**恒真门禁**，
  却挂着「N 误差 ≤ 15 %」的判据文字。现改为：用**以膝点为锚**的幂律独立重算寿命并与曲线对象比对，
  判据 `相对误差 ≤ 1e-9`（能抓第二段截距错误/分支翻转），`_independent_life`、
  `life_relative_error` 一并入档。

## ④ 结果图

`docs/assets/fatigue_sn_curves.png`（脚本生成，图与 JSON 不可能漂移）：
* 左：8 个实现的 S-N 曲线与出版 curve D 叠在同一 log-log 轴（膝点星标）——修复后 curve-D 分支完全重合；
  `seawater_cp / seawater_free` 的膝点位移（1e6 循环）清晰可见，属规范本身的处理而非漂移。
* 右：各实现相对出版曲线的寿命偏差（symlog，0 = 完全一致），修复后 curve-D 组全部为 0。

## ⑤ 注意点 / 局限

1. **膝点连续性只能核到 1.5e-3**：出版 `log a2 = 15.606` 只有 3 位小数，两段在膝点严格相接要求
   `15.60651`，故连续性本底为 `10^-0.0005 ≈ 1.15e-3` 循环。审计门限设 5e-3，并显式记录该本底
   （`knee_continuity_floor_rel`）；1 % 量级的真实漂移（1e-2）仍会被抓。
2. **踩坑：IIW FAT90 的耐久极限截断**。最初用 `0.25σ_knee / 0.5σ_knee` 探 m2 段，IIW 实现在
   13.2 MPa 处被 `cutoff_cycles = 1e9` 截断，比值变成 3.125 而非 32 —— **截断冒充了斜率错误**。
   探针改为 `0.45/0.9σ_knee`（寿命 6.0e8 < 1e9）。
3. **范围**：只覆盖单轴常幅 S-N 线的两段式实现与 curve D 家族；不含变幅 Miner 修正
   （Miner 本身在 `weld_fatigue_ice` 有独立证据）、Thickness correction、腐蚀环境下的
   mean-stress/频率修正、DNV curve B1/B2/C 等其它类别（审计框架通用，加一行声明即可覆盖）。
4. `ice_slamming_whipping` 的常数修正会使其疲劳寿命与 `docs/ice-slamming-whipping.md` 中
   引用旧常数的绝对值相差 4.19 %；该页正文未逐一重算，视为待同步的文档欠账（已在本页登记）。
5. 本单未改动任何曲线的**结构性语义**（分段位置、膝点循环数、环境降级），只改常数与判据。

## ⑥ 误差判定（逐项，是否 <3 %）

| 关键量 | 基准 | 误差 | < 3 % ? |
|---|---|---|---|
| 8 实现 `log a1` vs 出版 12.164 | DNV-RP-C203 Table 2-1 | 0.0e+00（dex） | **是** |
| 8 实现 `N(100 MPa)` vs 出版曲线 | 同上 | 0.000 % | **是** |
| 8 实现 `N(30 MPa)` vs 出版曲线 | 同上 | 0.154 %（IIW/焊趾实现的 15.60667 舍入） | **是** |
| 声明的常数 vs 对象实际返回 | 自身表格 | 2.0e-15 | **是** |
| m1 斜率恒等式 | `2^3 = 8` | 0.0e+00 | **是** |
| m2 斜率恒等式（分支受控） | `2^5 = 32` | 0.0e+00 | **是** |
| 膝点连续性 | 两段相接 | 1.53e-03（本底 1.2e-3） | **是** |
| v1 m=5 协议（负控，**不通过**） | `2^5 = 32` | **72.33 %** | **否（已修复）** |
| v1 `ice_slamming` 寿命 | 出版曲线 | **+4.194 %** | 边缘（已修复为 +0.000 %） |
| `Δσ_knee` 文档值 169.3 MPa | 算术 `10^((12.164-7)/3)` | **221 %** | **否（文档已改）** |

## 复现

```bash
PYTHONPATH=src python benchmarks/fatigue_sn_curve_consistency.py     # 写 JSON + PNG，退出码非 0 即审计失败
PYTHONPATH=src python -m pytest tests/test_fatigue_sn_curve_consistency.py -q   # 8 passed
```

## 建议入总账的新锚点

`GAP-FEM-FATIGUE-SN-CONSISTENCY`（域：结构疲劳/冰区）——验收：≥8 个 S-N 实现全部对出版
`log a1/log a2` 误差 ≤1e-6 dex、膝点连续性 ≤5e-3、斜率恒等式（分支受控）≤1e-10；
任何新增 curve-D 实现必须登记进 `sn_curve_consistency.curve_rows()` 并过审计（fail-closed）。
