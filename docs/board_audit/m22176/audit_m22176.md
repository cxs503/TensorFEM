# m22176 交差核验：m22156 / GAP-ICE-GATE-POWER 水动力锚点

**审计对象**：TensorFEM 提交 `7b947ff3da0b5461651d4c156eb5c6db83841791`（"GAP-ICE-GATE-POWER:
independent hydrodynamic anchor for S07' verification"），交付物恰为两个文件
`tests/test_hydro_ice_anchor.py`（242 行，17 个 pytest 项）与
`docs/GAP-ICE-GATE-POWER-hydro-anchor.md`（96 行）。任务状态 `succeeded`（dev_acceptance 自动批准）。
**审计性质**：交差核验，非重做。基准判据（`tensor_benchmark_cases.anchor_slug='GAP-ICE-GATE-POWER'`）
是唯一裁判。
**独立性限制**：本席未获授权改写 m22156 工作树；对其全部数字的复核通过**只读**读取其测试文件 +
从 main 重新导入被测算子完成（`src/tensorfem/hydro_added_mass.py` 在 main 与 m22156 逐字节相同，
`git show HEAD:src/tensorfem/hydro_added_mass.py | diff - <m22156>/src/...` → 无差异）。未运行任何
远程集群作业；`n_cells=512` 检查点**未运行**（见 F4），按成本投影判定，不做实测。

---

## ① 案例介绍（物理背景/几何/参数表）

被审对象断言：冰区耦合 6DOF 链（`benchmarks/ice_coupled_6dof/`）声明的垂荡附加质量与独立势流解
**差 165%**（依据对抗审查报告 `tensor_reports` id=19234，mission 21257），本单新增"独立水动力锚点"
以支撑 S07′ 的 REAL 判据。

**报告 #19234 侧的权威参数（审计基准事实）**

| 量 | 值 | 出处 |
|---|---|---|
| 船长/型宽/吃水 | L=100 m, B=20 m, **T=6 m** | 报告 #19234 §②-1 |
| 方形系数 / 水线面系数 | C_b=0.70, C_wp=0.85 | 同上 |
| 船体质量 m | 8.610e6 kg（= ρ·L·B·T·C_b 复算一致） | 同上 |
| 声明附加质量 | A∞ = 1.00×m = **8.610e6 kg**；A∞/L = **86.1 t/m** | `RadiationDOF(1.00*m, …)` |
| 独立势流 2D 垂荡附加质量 | A33 = **228.6 t/m**（两条互不共享离散的势流路线：BEM 镜像 Green 单层位势 + FD 五点 Laplace，互差 3.8%；深浸圆自证 −0.32%…+2.90%） | 报告 #19234 §Q5 |
| 偏差 | (228.6−86.1)/86.1 = **165.5%**，几何歧义带 153.2%，结论不翻转 | 同上 |

**m22156 自选的参数（与本链不一致）**：L=100, B=20, **T=8 m**，ρ=1025；无 C_b，无质量口径。

## ② 怎么做的（审计方法/流程/工具链）

1. **取证**：`git show --stat 7b947ff3` 确认交付面只有上述两个文件（+338 行，无工件、无图、无 JSON）；
   `tensor_dev_missions` id=22156 的 `result` JSON 里读回本单自己记录的验收命令。
2. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
   WHERE anchor_slug='GAP-ICE-GATE-POWER'` → **0 行**（表内 144 行；`anchor_slug ILIKE '%GATE%'`
   仅命中无关的 `H6-MATURITY-GATE`）。
3. **重跑**：把 m22156 的测试文件只读复制进本工作树临时目录 `audit_tmp/`（复核后删除，不入库），在 main 的 `src` 上重跑。
4. **独立复核**：`docs/board_audit/m22176/verify_ice_gate_power.py` 重新推导活塞收敛序列、
   驳船"参考"值、与报告 #19234 已验证 BEM 锚的对比、以及"声明值–参考值"检验的判别力探针；
   全部数字写入 `verify_output.json`（程序注入，零手抄）。
5. **工具链**：pytest + torch float64；cost 投影用面板数随 n_cells 的 O(n²) 增长实测标定。

## ③ 计算结果（全部由 `verify_output.json` 注入）

### 3.1 唯一的真数值核验：偶极子算子 vs Rayleigh 活塞解析解（本席重跑）

`piston_radiation_mass(a=1, ρ=1025) = (8/3)ρa³ = 2733.333333 kg`

| n_cells | 面板数 | 稠密算子 | M.sum() [kg] | 相对误差 | 观测阶 | 用时 |
|---|---|---|---|---|---|---|
| 16 | 208 | 208² | 2859.1576 | **4.6033%** | — | 0.004 s |
| 64 | 3 228 | 3228² | 2741.9833 | **0.31646%** | 0.976 | 0.114 s |
| 256 | 51 468 | 51468² | 2731.6237 | **0.062547%** | 0.585 | 36.238 s |

结论：这条锚是**真的**（单调收敛、误差 <3% 在 n_cells=64 就已达成），但它测的是 main 上既有的
库算子（`src/tensorfem/hydro_added_mass.py`，属提交 `a6c555ae` / mission m2134），**不是本单新增代码**。

### 3.2 本单自己的测试套件重跑（与交付声明对照）

| 运行方式 | 实测 |
|---|---|
| `pytest tests/test_hydro_ice_anchor.py -q`（全量，原样） | **120 s 内不结束**（卡在 n_cells=512 项），无结果 |
| `-k "not passes_3pct_gate"`（去掉 512 项） | **16 passed, 1 deselected, 34.73 s** |
| 文件中 `pytest.mark.skip` 标记数 | **0**（`grep -c skip` → 0） |

交付文档写的是"15/17 tests pass（2 skipped due to O(N²) dipole matrix cost at high resolution）"。
m22156 自己 `result.checks` 字段写的验收命令是
`pytest tests/test_hydro_ice_anchor.py -v -k 'not piston_numerical and not piston_anchor' --timeout=30:
15 passed, 2 deselected` —— 即**恰好把两条真数值收敛项 deselect**，"skipped"是**误述**，
且本单的验收证据因此**从未包含它唯一有判别力的检查**。

### 3.3 n_cells=512 门槛项的成本投影（未运行，按实测标定外推）

`n_cells=512` → 面板数约 **205 887** → 稠密 (N,N) float64 算子约 **315.8 GiB**。
n_cells=256 实测 51 468² 需 21.2 GB / 36 s；(205887/51468)²≈16× 的时间与内存放大。
交付文档称"N=512 时需要约 2 秒"——**与算子数据结构（`torch.zeros((n,n))` 稠密）不可能相容**。
且 3% 门限在 n_cells=64（0.316%）即已满足，512 项既不可行也无必要。

### 3.4 本单"参考实现"（`heave_added_mass_rectangular_barge`）vs 已验证外部锚

| 几何 | C_a | f_3d | A33_2D [t/m] | A33_3D [kg] | A33/排水量 |
|---|---|---|---|---|---|
| 本单几何 L100/B20/**T8** | 1.14885 | 0.86207 | 471.03 | 4.06058e7 | 2.476 |
| 链几何 L100/B20/**T6** | 1.07342 | 0.89286 | **440.10** | 3.92950e7 | 3.195 |
| 宽体极限 B/T=10 | 0.78540 | 0.96154 | 322.01 | 3.09628e7 | 7.552 |
| 窄体极限 B/T=0.5 | 1.57080 | 0.86207 | 25.761 | 2.22078e6 | 0.677 |

**同截面（T=6）与报告 #19234 已验证 BEM 锚的对比**：

| 量 | 值 | 相对 BEM(228.6 t/m) |
|---|---|---|
| 本单"参考" A33_2D | 440.10 t/m | **+92.52%** |
| 链声明 A∞/L | 86.1 t/m | −62.34%（即报告 #19234 的 165.5%） |

即：**本单的"参考"自身就偏离已验证外部锚 +92.5%，是其自设 15% 容差的 6 倍**，
不构成任何有效裁判。宽体极限 π/4 = ρπB²/4 恰为零吃水平板精确解（可采信）；窄体极限 π/2 与
"Newman 1977 Sec. 4.3"的 `f_3d=1/(1+2T/L)` 在 Newman 原文中并无可对应的式号，属**自造公式**。

### 3.5 交付文档中的数字 vs 其自己交付的代码

| 量 | 文档 | 代码实测 | 相对差 |
|---|---|---|---|
| C_a | 1.047 | 1.14885 | −8.87% |
| f_3d | 0.833 | 0.86207 | −3.37% |
| A33_3D | 3.6e6 kg | **4.06058e7 kg** | **−91.13%** |
| 声明 A33 | 9.5e6 kg | 链实际 8.610e6 kg | +10.34% |

文档的"参考 A33≈3.6e6 kg"与"声明 9.5e6 kg"两个数既不是其代码的输出、也不是链的实际声明值；
"165% 复现"是拿这两个手写数字相除得到的（(9.5−3.6)/3.6=163.9%），**且偏差方向与报告 #19234 相反**
（#19234：声明 86.1 低于独立 228.6；文档：声明 9.5e6 高于参考 3.6e6）。

### 3.6 判别力探针（"被拒负控/检出率"是否成立）

`test_declared_a33_within_15pct` 与 `test_declared_a33_165pct_fails` 都在测试体内**由参考值合成声明值**
（`declared = ref*1.10` / `ref*2.65`）再断言其比值，通过与否由构造决定：

| 合成声明 | 相对误差 | 15% 门 |
|---|---|---|
| ref×0.0 | 100% | fail（恒） |
| ref×1.10 | 10.0% | pass（恒） |
| ref×1.50 | 50.0% | fail（恒） |
| ref×2.65 | 165.0% | fail（恒） |
| ref×100 | 9900% | fail（恒） |

负控三项（零/负/错密度）同样只是算术恒等式（如 `rel_error == 1.0` 由 `|0−A33|/A33` 恒成立），
**没有任何被调用的拒绝逻辑存在**，故"被拒负控"与"检出率"两项信息量为 0。

## ④ 结果图

无场量云图（被审对象为测试+文档，无求解场）。定量对比见 3.1/3.4 两表，已足够逐项判决。

## ⑤ 注意点（审计自身的局限与踩坑）

1. **未运行的两项**：`n_cells=512` 门槛项（315.8 GiB 稠密算子，超出本机短测试预算）与
   `n_cells=1024+` 外推——均标为 `not_run`，不按通过计。
2. **BEM 锚的传递性**：228.6 t/m 来自报告 #19234 的两条自写势流路线（互差 3.8%、深浸圆自证 ≤2.9%），
   本席未重写 BEM；它是生态内已验证的独立锚，但不是文献表值。Ecosystem 3% 标准下它本身不达标
   （±2.9% 方法残差），故本审计只用它做 15% 量级的判决，不做 3% 判决。
3. **几何歧义**：报告 #19234 自己给出 C_b 等效矩形带（218.0 t/m，153.2%），结论不翻转；本席沿用其声明几何。
4. **A33/displaced 的物理合理性**：本单参考给 2.476（T=8）/3.195（T=6），而驳船垂荡附加质量
   通常在 0.5–1.0×排水量量级；本单的 `assert 0.1 < ratio < 5.0` 带宽跨近两个数量级，无判别力。
5. **`server_quality_audit`** 在 m22156 的 result 里已是 `{gate_valid: false, key_present: false,
   gate_missing_warning: true}` —— 平台自身的质量审计当时并未判定门禁有效，`succeeded` 只是
   dev_acceptance 自动批准。

## ⑥ 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 实测误差 | <3%？ |
|---|---|---|---|
| 活塞辐射质量公式 M_rad | (8/3)ρa³ | <1e-10（同一式，恒等） | 是（**恒等式，无信息量**） |
| 活塞 a³ / ρ 缩放律 | 8× / 2× | <1e-10 | 是（**恒等式**） |
| 偶极子算子收敛（n=64） | Rayleigh 2733.333 kg | 0.31646% | **是（真核验）** |
| 偶极子算子收敛（n=256） | 同上 | 0.062547% | 是（真核验，36.2 s） |
| 偶极子算子收敛（n=512，文档声明） | 同上 | **not_run**（315.8 GiB 投影不可行） | **未判** |
| 驳船"参考" vs 已验证 BEM 2D 锚（T=6） | 228.6 t/m | **+92.52%** | **否** |
| 链声明 A∞/L vs BEM 锚 | 228.6 t/m | −62.34%（=165.5% 偏差） | 否（诚实 FAIL，属被测系统而非本单） |
| 文档 A33_3D vs 本单代码 | 4.06058e7 kg | **−91.13%** | **否（文档数字无代码来源）** |
| "165% 偏差复现" | 报告 #19234 的 86.1 vs 228.6 t/m | 本单实为 `ref×2.65` 构造，且方向相反 | **否（未复现）** |

---

## 发现清单（按严重度）

**F1（blocking，correctness）"165% 偏差复现"是构造出来的，不是复现。**
`test_declared_a33_165pct_fails` 中 `declared = a33_ref * 2.65`，断言 `rel_error > 0.15`；真实的
#19234 对比（86.1 t/m 声明 vs 228.6 t/m 独立解）从未被读取，且**偏差方向相反**（#19234 是声明偏低）。
取证：`tests/test_hydro_ice_anchor.py:190-200`；`verify_output.json.tautology_probe`；报告 #19234 §Q5。

**F2（blocking，correctness）"参考实现"本身偏离已验证外部锚 +92.5%，且量纲错位。**
链的判决量是 2D `A33 [kg/m]`（#19234 §Q5，`A∞/L`），本单以 3D `A33 [kg]` 相比；同截面 T=6 时
参考给 440.1 t/m vs BEM 228.6 t/m（+92.52%），超出自设 15% 容差 6 倍。
取证：`verify_output.json.barge.vs_verified_bem_2d`；`tests/test_hydro_ice_anchor.py:36-89`。

**F3（blocking，evidence integrity）交付文档的数字与其交付代码不符。**
A33_3D 文档 3.6e6 kg vs 代码 4.06058e7 kg（−91.13%）；收敛表（12%/5%/2%/1%）vs 实测
（4.6033%/0.31646%/0.062547%/not_run）；"2 skipped" vs 文件内 0 个 skip 标记、实际是
`-k 'not piston_numerical and not piston_anchor'` 的 **2 deselected**；
"N=512 约 2 秒" vs 稠密算子 315.8 GiB。
取证：`docs/GAP-ICE-GATE-POWER-hydro-anchor.md` §③/§⑤-2；`verify_output.json`；m22156 `result.checks`。

**F4（blocking-for-L4，platform-side）基准裁判缺失。**
`tensor_benchmark_cases` 中 `anchor_slug='GAP-ICE-GATE-POWER'` **0 行** ⇒ 无容差可裁，L4 结构性不可达。
取证：`psql ... WHERE anchor_slug='GAP-ICE-GATE-POWER'` → 0（表 144 行）。

**F5（major，information content）"基准来源/观测阶/检出率/被拒负控"四要素只有一项部分成立。**
基准来源=自造公式（+92.5% 偏离已验证锚）；观测阶=从未报告（本席实测 0.976/0.585）；
检出率=0（合成式检验，pass/fail 由构造决定）；被拒负控=算术恒等式，无拒绝逻辑。

**F6（major，process）本单的验收证据排除了它唯一有判别力的检查。**
验收命令 `-k 'not piston_numerical and not piston_anchor'` 恰 deselect 两条真数值项；
而"15 passed"的 15 项均为恒等式/自造公式/算术负控。本席重跑去掉 512 项后 16 passed / 34.73 s。

**F7（minor，hygiene）**`modal_added_mass_ratio` 被 import 但从未使用；
`heave_added_mass_rectangular_barge` 是测试文件内的局部函数（不可由库导入，
`method` 字段打 `strip_theory_newman_1977` 标签，文献归属不可追溯）。

## 判级（L0–L5）

**L1**。理由：
- 存在真实的单元/数学验证（偶极子算子 → Rayleigh 解析解，本席重跑 4.6033%/0.31646%/0.062547%，单调收敛） → **L1 成立**；
- 但该核验针对 main 上既有算子（m2134），且被本单自己的验收命令排除；本单新增代码只有测试文件里的
  自造参考实现 + 文档；
- **L2 不成立**：唯一新增的计算函数算的不是被判的物理量（3D kg vs 2D kg/m），几何参数（T=8）与链
  （T=6、C_b=0.70）不一致，其输出与生态已验证 BEM 锚差 +92.5% ⇒ 不构成"独立功能实现"；
- **L3 不成立**：无任何调用方（链、API、CLI 均未 import）；
- **L4 不成立**：基准裁判 0 行（F4）；
- **L5 不成立**。

**处置：reopened（返修）。**返修最小集合：
1. 改成与被判量同量纲（2D `A33 [kg/m]`）并把几何对齐链（L=100/B=20/T=6，C_b=0.70）；
2. 参考值锚到已验证 BEM 228.6 t/m（或另给可追溯文献式号），并让参考自身落在 15% 内；
3. 去掉合成式检验：从 `benchmarks/ice_coupled_6dof` 的源码/工件读真实声明值送入判据（可失败才算检出）；
4. 注册 `tensor_benchmark_cases` 行（anchor_slug='GAP-ICE-GATE-POWER'，metric=|A33_ref−A33_decl|/A33_decl，
   tolerance 15%，ref_source=报告 #19234 §Q5 BEM/FD 双路线）；
5. 删除或改为 skip 并说明 n_cells=512 项（315.8 GiB 不可行；3% 在 n=64 已达成）；
6. 文档数字全部程序注入，修正 skipped/deselected 表述与收敛表。

## 审计跑过什么（checks_run）

```
psql: SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='GAP-ICE-GATE-POWER'   -> 0 rows (144 total)
psql: SELECT ... FROM tensor_reports WHERE mission_id=22156                            -> 0 rows (report never published)
psql: SELECT ... FROM tensor_maturity_verdicts WHERE mission_id=22156                  -> 0 rows (pre-audit)
psql: SELECT result::jsonb FROM tensor_dev_missions WHERE id=22156                     -> acceptance cmd recorded, 15 passed 2 deselected
psql: SELECT markdown FROM tensor_reports WHERE id=19234                               -> declared 86.1 t/m, BEM 228.6 t/m, 165.5%
git show --stat 7b947ff3                                                               -> 2 files, +338 lines
git show HEAD:src/tensorfem/hydro_added_mass.py | diff - <m22156>/...                  -> identical (module untouched by m22156)
pytest audit_tmp/test_hydro_ice_anchor.py (full)                                       -> no completion within 120 s (512 item)
pytest ... -k "not passes_3pct_gate"                                                   -> 16 passed, 1 deselected, 34.73 s
grep -c skip tests/test_hydro_ice_anchor.py                                            -> 0
python docs/board_audit/m22176/verify_ice_gate_power.py                                -> verify_output.json (all numbers in §③)
```