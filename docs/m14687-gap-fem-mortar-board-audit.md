# m14687 GAP-FEM-MORTAR — 交差核验单（board-audit）

- 核验对象：m14687（GAP-FEM-MORTAR · FEM mortar 接触 · TensorFEM，状态 succeeded）
- 交付 commit：`af09a9b6`（m14687 分支）/ `d01094e2`（并入 main，3 个 PNG 被门禁滤除）
- 核验工作树：`/root/tp-agent/work/m14962`（分支 agent/m14962），被审工作树 `/root/tp-agent/work/m14687`
- 核验日期：2026-11-11，核验人：board-audit（m14962）
- 文件一致性：`src/tensorfem/mortar_lagrange_multiplier_2d.py` md5 `155cb9b4…`、
  `benchmarks/mortar_lm_hertz_verification.py` md5 `437865bf…`、
  `benchmarks/artifacts/mortar_lm_hertz/verification_results.json` md5 `e20fe7f1…`
  在 m14687 工作树与 m14962 工作树**逐字节相同**（`md5sum` 对照）。

## 1. 取证：交付三件套是否真存在

| 三件套 | 声称 | 实测 | 判定 |
|---|---|---|---|
| run_artifacts | `benchmarks/artifacts/mortar_lm_hertz/verification_results.json` + 3 PNG | JSON 8502 B 在 main（`d01094e2`）内存在；3 张 PNG（hertz_pressure_distribution / mortar_lm_convergence / mortar_lm_penetration，69/86/30 KB）落在 `af09a9b6`，**被门禁滤除、未进 main**（`d01094e2` 提交信息「3 个文件被门禁滤除」；m14962 内该目录只有 JSON） | 部分存在（数值工件在，图件在 main 缺） |
| benchmark ref vs got | 报告表 4 行 | 独立重跑得**逐字节相同** JSON：L1 力误差 32.288%/b 误差 44.098%/p L2 35.478%；L2 34.304%/44.098%/30.717%；L3 34.723%/44.098%/27.666%；L4 35.135%/44.098%/28.692%（`benchmarks/artifacts/mortar_lm_hertz/verification_results.json`） | 数值可复现，与报告表一致 |
| repro_cmd | 报告未给命令，仅列工件路径 | `python benchmarks/mortar_lm_hertz_verification.py` 实测可跑：exit=1（honest FAIL 正确以非零码报出），耗时 70.5 s，日志 `artifacts/m14687_audit/repro.log` | 命令可执行、可复现 |

复现比对：`diff /tmp/m14687_committed.json benchmarks/artifacts/mortar_lm_hertz/verification_results.json` → **IDENTICAL**（重跑覆盖后无任何差异，说明木前 JSON 非手写）。

## 2. 判级：L2

级别=功能实质，报告篇幅不改变级别。逐级核对（级别判据按 maturity_gate 达标判断方案）：

- **L1（数学/单元测试面）**：`grep -rn mortar_lagrange_multiplier tests/` = **0 命中**，本单**未新增任何 pytest**。L1 键 `tests` 缺失。
- **L2（独立功能实现：输入→计算→输出）**：成立，但是「基准脚本内」的实现而非库内可用实现：
  - `benchmarks/mortar_lm_hertz_verification.py:69` Q4 平面应变单元刚度真实组装（4×4 Gauss 不变角，D 矩阵 plane strain），`:117` 鞍点系统 + 主动集（`:133` 按 `u_prescribed = -δ + x²/2R` 施加位移约束），`:212` 由乘子/ tributary width 换算节点压力 —— 非 MOCK、非硬编码。
  - 执行证据：4 个网格层全部 `converged=True`（2–5 次主动集迭代），产出力/半宽/压力 L2 数值（见上表）。输入→计算→输出链路真实贯通。
- **L3（系统集成）**：**不成立**。`grep -rn mortar_lagrange_multiplier_2d`（全仓，排除文件自身）只命中 `docs/mortar_lm_round1_report.md:111` 一行文档；`src/tensorfem/__init__.py` 的 mortar 导出只有既有 `mortar_contact3d` / `frictional_mortar`，**新模块未导出、无任何调用方**。
- **L4（基准验证）**：**不成立**，三项验收判据 2 FAIL 1 vacuous-PASS：
  1. 压力误差 <10%：实测 28.69% → FAIL（平台精度红线 >15% 不可接受，此处 28.7% 亦越线）；
  2. 接触半宽：b_FE = 0.17678 vs ref 0.31623 → 44.10% 误差，四个网格层**完全相同到 5 位有效数字**；
  3. 收敛阶：0.07（判据 >0.3）→ FAIL；
  4. 唯一 PASS 项「穿透 < tol」是**空判**（见发现 F3）。

结论：**L2**（真实可执行的功能实现 + 量化误差分析，但未集成、精度不达标、无测试）。0/1/2/3 未达到，故不判 L3；精度未过 15% 红线且复现基线本身有缺陷，故不判 L4。

## 3. 发现（按严重度，adversarial-review 口径）

**F1（blocking，交付模块一调即崩）** 库内模块 `solve_mortar_lm_2d_contact`
（`src/tensorfem/mortar_lagrange_multiplier_2d.py:360`）首次调用即抛异常：
`torch.linalg.solve_triangular(L, rhs_free, upper=False)`（`:448`、`:449`）传入 1-D rhs，
`RuntimeError: linalg.solve_triangular: The input tensor B must have at least 2 dimensions`。
复现：`PYTHONPATH=<wt>/src python -c "… build_hertz_cylinder_on_flat(6,6,…); solve_mortar_lm_2d_contact(…)"`。
同一路径 `:483` 还写着 `quad_x = torch.zeros(n_slave, dtype=DTYPE)  # placeholder`。
→ 「mortar 接触能力」在库层面**从未运行过**；只有在 benchmark 里被复写一遍的版本跑通过。

**F2（blocking，宣称的 mortar 语义未被验证）** 基准**没有非匹配网格、没有 master 面**：
`solve_hertz_mortar_lm` 的约束矩阵 `G`（`:155-162`）只在 slave 节点 y 自由度置 1，等价于「对刚性压头施加指定位移的节点式 LM 接触」；
模块的 `compute_mortar_matrices_1d`（`:260`）算出的 D/M 矩阵在求解路径中**从未使用**（`:436` 注释「For simplicity, use nodal integration (lumped mortar)」）。
`build_hertz_cylinder_on_flat` 里 `slave_surface == master_surface`（同一批顶面节点），也不构成非匹配界面。
→ GAP-FEM-MORTAR 的实质缺口（mortar 分段积分 / 非匹配界面）**没有被交付**，交付的是节点式 LM 接触 + Hertz 对照。

**F3（major，验收判据空判）** 判据 2「穿透 < tol」恒真：`max_penetration = float(torch.min(gap).clamp(max=0))`
（`benchmarks/mortar_lm_hertz_verification.py:221`），`gap = u_actual - u_prescribed`，张开节点必然给出负值，
与正容差 `pen_tol = 1e-4*b_ref`（`:390`）比较**结构上永远 PASS**。实测报出 −5.81e−3 = 与压入深度同量级的张开量，与「穿透」无关。
即三项判据里唯一绿色项不构成约束满足证据。

**F4（major，参照解自相矛盾）** 基准参照解混用两套不相容 Hertz 关系：
`b_ref = sqrt(R δ)`（`:293`）、`p0_ref = E*δ/(π b)`（`:294`）、`F_ref = π p0 b/2`（`:295`）。
同一力 F 对应的相容平面应变线接触解为 `b = sqrt(4FR/(πE*)) = 0.25231`、`p0 = 2F/(πb) = 13.863`；
基准用的 b_ref = 0.31623 比之**大 25.33%**（`artifacts/m14687_audit/audit_numbers.json`，程序化计算）。
→ 44.10% 的半宽误差中约 25 个百分点来自参照解本身；报告的「失败幅度」不可直接采信。
模块里同一处亦标 `# Simplified: use b = sqrt(R * delta) as first approximation`（`:622`）。

**F5（major，b_FE 四层不变=未真正加密接触区）** 四层 b_FE 均为 0.17678（误差两次四舍五入后同为 44.098%），
说明接触半宽由网格/阈值 `p > 0.01 p_max`（`:341-343`）决定而非物理收敛；配合收敛阶 0.07，
「网格收敛 >0.3」判据的失败是实质性的，不是精度噪声。

**F6（minor）** 报告自述「Round 1 HONEST-FAIL」并且失败分析（`docs/mortar_lm_round1_report.md:34-70`）明确列出
nodal 耦合、缺分段积分为根因 —— **未见夸大或伪造**，与实测一致；诚实申报这一项是达标的。

**F7（minor）** 图件 3 张 PNG 未进 main（门禁滤除），报告引用「Results: JSON」尚可，但 main 上无可视工件；
本次审计用 `scripts/audit_m14687_figures.py` 从 JSON 程序化重绘 `artifacts/m14687_audit/m14687_audit.png`（零手抄）。

## 4. 未运行 / 未覆盖（unknown 不记为 passed）

- 未跑全仓 pytest（402 个测试文件），本单只核验 m14687 交付面；未验证仓库其它 mortar 能力（`tests/test_mortar_contact3d*.py` 等）当前是否绿。
- 未做集群规模的加密研究（>32 接触节点）；「收敛阶 0.07」结论仅在交付的 4 层网格范围内成立。
- 未追溯 m14687 的 mission 记录与原报告的图件渲染脚本（benchmark 内 `matplotlib` 已导入但 `main()` 未绘图，图件来源脚本不在交付内）。

## 5. 处置（disposition）

**request changes（返修）**。理由：GAP-FEM-MORTAR 的实质能力（mortar 分段积分 + 非匹配界面 + 集成进 `tensorfem` 导出面）未交付；
交付的库模块一调即崩（F1）；唯一跑通的路径是 benchmark 内复写版且精度 28.7%/44.1% 越线（F2/F5）；
参照解自身有 25% 不相容（F4），须先修参照解再谈精度；验收判据 2 需改成 `max(0, -gap)` 的真实穿透度量（F3）。

返修清单（按依赖序）：
1. 修 `:448`/`:449` 的 `solve_triangular` 1-D rhs（`.unsqueeze(-1)` 或改用 `torch.linalg.solve`），补**能跑的**单元测试，导出进 `src/tensorfem/__init__.py`。
2. 基准改用与 `b = sqrt(4FR/(πE*))`、`p0 = 2F/(πb)` 相容的参照三元组（或改用 2D Hertz 含对数项位移解），并把参照公式及其适用性写进判据。
3. 实现真正的 mortar 分段（master 面离散 + 片段裁剪 + D/M 进入鞍点系统），例如 `compute_mortar_matrices_1d` 的 D/M 接入求解；否则把节点式 LM 接触改名，不再挂 "mortar"。
4. 判据 2 改真实穿透度量；b_FE 阈值（0.01 p_max）敏感性分析入报告。