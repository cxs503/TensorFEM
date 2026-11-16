# CASE-FEM-HEAT-L2 复算闸门：`--check` 从未被跑过，证书与代码之间没有绑定（m22937 / PR#1113 重开）

**锚点** CASE-FEM-HEAT-L2 ｜ **仓库** TensorFEM ｜ **性质** 重开核查（reconciliation）+ 缺陷修复 F1 + 新增复算闸门 ｜
**判据** 收敛率 ≥ 2 阶（manufactured solution）+ 证书可复算（committed artifact == 当前 checkout 的计算结果）

**平台报告** `tensor/reports#20606`（六要素完整版，含场量云图与阶次对比曲线）。

**一句话**：PR#1113（m20143）的 8 个交付文件**早已在 main 上**（经 #963 及 m20372/m20995/m22874 迭代成熟），
但证书与代码之间那条唯一的绑定线——`run.py --check` 的全阶梯复算——**从来没有被执行过**：
它的 `import independent_sparse` 只在 `python benchmarks/verified/.../run.py` 这一种调用方式下成立，
从测试套件里调用立刻 `ModuleNotFoundError`，于是"已合并的证书是否仍由 main 算得出来"无人验证。
本轮修复该导入（`load_independent_solver()`，数字不变），并把复算装进测试套件：
全阶梯 4→64 重算后与 `result.json` / `gates_verified.json` 逐字段比对，**最坏相对漂移 0.0（逐位一致）**，
8/8 闸门通过，独立码交叉 4.42e-11。

---

## 1. 重开核查：PR#1113 的交付内容在 main 上的位置

| PR#1113 声明的文件 | main 上的状态 | 落地提交 |
|---|---|---|
| `benchmarks/verified/fem_heat_conduction_l2/run.py` | 存在（946 行，本轮 +29/−5，修 F1 + 删 1 个未用导入） | 0cf1e2fd → #963 (m20143)，后经 e8bd57c5 (m20995) 换为域积分范数 |
| `benchmarks/verified/fem_heat_conduction_l2/result.json` | 存在（5 层阶梯，8 项 gate 全绿） | 同上 |
| `benchmarks/verified/fem_heat_conduction_l2/gates_verified.json` | 存在（8 gate，`passed=true`） | 同上 |
| `benchmarks/verified/fem_heat_conduction_l2/README.md` | 存在（由 JSON *渲染*，非手抄） | d0918f21 (m22874) |
| `benchmarks/verified/fem_heat_conduction_l2/independent_sparse.py` | 存在（不 import `tensorfem` 的第二份实现） | #963 / m20372 |
| `src/tensorfem/heat_conduction_l2.py` | 存在（333 行，`run_heat_conduction_l2`） | #963 |
| `src/tensorfem/__init__.py`（导出） | 存在（`HeatConductionL2Result`, `run_heat_conduction_l2`） | #963 |
| `tests/test_heat_conduction_l2.py` | 存在（408 行，53 个热传导相关用例） | #963 → m20995 → m22874 |
| `docs/case_fem_heat_l2_report.json` | 存在（258 KB，含内嵌图） | m20143 |

**结论**：重复粘贴 PR#1113 的 diff 是空改动（文件已逐字在 main 上），因此本轮不重贴；
按"重开只做增量"处理，增量必须回答一个此前无人回答的问题——*证书还是不是当前代码的函数？*

## 2. 实验设计（先声明，后看结果）

| 要素 | 内容 |
|---|---|
| 决策 | 本轮 PR 交付什么：**(A)** 原样重贴 PR#1113 文件；**(B)** 只出核查报告；**(C)** 修 F1 并把复算装进套件 |
| 备选与取舍 | (A) 被否：tree 与 main 逐字相同，无实质增量（`git diff` 空）；(B) 被否：报告不是可执行证据，README 与 JSON 的绑定已由 m22874 覆盖、JSON 与代码的绑定仍缺；(C) 采纳 |
| 假设 H1 | 修复 `independent_sparse` 的导入方式后，`recompute_without_writing()`（=`--check` 的只读半边）可以在任意入口执行，且**不改变任何已发布数字** |
| 基线/对照 | 已提交的 `result.json` + `gates_verified.json`（m22874 产物）作为基线；对照组 = 同一份内核在 CLI 模式下的 `--check`（导入原本就成立的那条路径） |
| 结果度量 | `numeric_drift` 的**最坏相对漂移**（无量纲）、漂移字段条数、8 个 gate 的 `passed`、独立码 `worst_norm_relative_difference`、慢道单测墙钟 |
| 预声明阈值 | 漂移 `DRIFT_RTOL = 1e-12`（仓内既有预声明，比 README 打印的 6 位有效数字低 6 个量级；零容差只在一个 BLAS 构建上成立）；gate 必须 8/8 且 `passed=true`；独立码相对差 ≤ 1e-9（沿用 m20995 预声明）；`<3%` 精度判据针对最细网格 L₂ 误差 |
| 变量 | 受控：阶梯 {4,8,16,32,64}²、Q4+2×2 装配、6 点域积分后处理、Dirichlet 全场、`float64`；改变：仅 `independent_sparse` 的导入方式 |
| 混杂因素 | BLAS/线程数改变浮点归约顺序（→ 用 1e-12 而非逐位，且实测逐位一致）；torch/CPU 版本（`environment` 字段列入 `VOLATILE_KEYS`，不参与比对）；机器（证书由 loongarch64 生成，本轮在 x86_64 上复算） |
| 重复与盲法 | 复算 1 次全阶梯即可判定（5 层 × 3 种积分阶、共 15 个范数值全部落在 1e-12 内，且实测 0.0），无需重复；无可盲化对象（数值断言） |
| 停机规则 | 若漂移 > 1e-12 或任一 gate 翻红 → 停止，改为"证书过期"处置（重跑 `run.py` 并连同图/README 一起提交），不调阈值 |

## 3. 发现的缺陷 F1（本轮修复）

**F1（阻断级，可复算性）** `independent_cross_check()` 里 `import independent_sparse` 依赖
"脚本所在目录在 `sys.path` 上"——只有 `python benchmarks/verified/fem_heat_conduction_l2/run.py`
这一种调用方式满足：

```
$ pytest tests/test_heat_l2_reproduction.py -q
ModuleNotFoundError: No module named 'independent_sparse'
  at benchmarks/verified/fem_heat_conduction_l2/run.py:305 in independent_cross_check
```

后果不是"某条用例红"，而是**复算闸门整体不可达**：`recompute_without_writing()` → `independent_cross_check()`
是 `--check` 的必经路径，于是测试套件只能演练 `numeric_drift` 的**合成字典**
（`test_reproduction_detector_flags_drift_and_ignores_volatile_fields`），从未真正重算过阶梯。
这与 m22874 记录的 README 漂移是同一类失败、深一层：README 被绑到 JSON，JSON 没被绑到代码。

**修复**：新增 `load_independent_solver()`，按 `Path(__file__).parent` 用 `importlib.util` 加载，
缓存进 `sys.modules["independent_sparse"]`（`import independent_sparse` 与重复调用共享同一实例）。
CLI 行为不变，数字不变（§5：漂移 0.0）。

## 4. 新增闸门：`tests/test_heat_l2_reproduction.py`

| 用例 | 车道 | 作用 |
|---|---|---|
| `test_committed_certificate_reproduces_from_this_checkout` | slow（登记于 `tests/slow_lane_manifest.txt`） | 全阶梯 4→64 重算，与 `result.json` 逐字段比对 1e-12、与 `gates_verified.json` 比对 verdict，并断言"复现出来的不是一份失败"（`passed=true`、无 `failed_gates`） |
| `test_volatile_keys_do_not_swallow_the_measured_physics` | fast | `VOLATILE_KEYS` 只能是时钟/元数据；把最细层 `l2_relative` 改 1e-6 必须被抓 |
| `test_tampered_norm_is_caught_on_every_level` | fast | 逐层篡改：每一级都在比对范围内（不只是最细层） |
| `test_flipped_gate_verdict_is_caught` | fast | 逐 gate 翻转 verdict（数值不动）必须被抓 |
| `test_relaxed_gate_threshold_is_caught` | fast | 把 `l2_pair_order_ge_1p95` 阈值放宽到 1.80 必须被抓 |
| `test_gate_tolerance_is_predeclared_below_the_printed_precision` | fast | 锁定 `0 < DRIFT_RTOL ≤ 1e-9`，防止用松阈值让已发布数字悄悄漂移 |

最后 5 条是**失败关闭（fail-closed）对照**：它们篡改**真实**已提交 payload 的副本，
所以闸门不可能因为"只对一个恒真比较器做单元测试"而空转。

## 5. 计算结果（程序化，零手抄）

复算环境：x86_64 / Python 3.11.6 / torch 2.7.1 / float64 / `OMP_NUM_THREADS=4`；
基线证书：loongarch64 / torch 2.7.1（`result.json.environment`）。

| 量 | 基线（committed） | 本轮复算 | 相对漂移 |
|---|---|---|---|
| L₂ 4×4 | 4.460090e-02 | 4.460090e-02 | 0.0 |
| L₂ 8×8 | 1.121026e-02 | 1.121026e-02 | 0.0 |
| L₂ 16×16 | 2.806922e-03 | 2.806922e-03 | 0.0 |
| L₂ 32×32 | 7.020126e-04 | 7.020126e-04 | 0.0 |
| L₂ 64×64 | 1.755209e-04 | 1.755209e-04 | 0.0 |
| 观测阶 (8→64) | 1.99225 / 1.99776 / 1.99942 / 1.99985 | 同左 | 0.0 |
| H₁ 阶 | 0.99869 / 0.99972 / 0.99993 / 0.99998 | 同左 | 0.0 |
| 节点 RMS 诊断阶 | 1.81019 / 1.88381 / 1.93509 / 1.96573 | 同左 | 0.0 |
| Roache 幂律残差 | 1.350864e-04 | 同左 | 0.0 |
| 独立码最大相对差 | 4.42e-11 | 4.4215657e-11 | 0.0 |
| gate | 8/8 通过 | 8/8 通过 | verdict 全等 |
| **结果树最坏漂移 / 漂移字段数** | — | **0.0 / 0** | — |
| **gate 清单最坏漂移 / 漂移字段数** | — | **0.0 / 0** | — |

墙钟：全阶梯复算 13.1 s（单次 `pytest` 用例，含 5 层 × 3 种积分阶后处理 + 独立 numpy/scipy 交叉码）；
5 条 fast 对照合计 5.8 s。

## 6. 误差判定（vs 基准，逐项）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 最细网格 L₂ 相对误差 | 解析解 `T=sin(πx)sinh(πy)/sinh(π)` | 1.755e-4（0.0176%） | **是** |
| 观测收敛阶（最细对） | Q4 理论 2.0 | 1.99985 → 相对误差 7.6e-5（0.0076%） | **是** |
| 观测收敛阶（4 个对，最差） | 2.0 | 1.99225 → 0.39% | **是** |
| H₁ 半范阶 | 理论 1.0 | 0.99869 → 0.13% | **是** |
| 独立实现 vs 库 L₂ 范数 | 同一离散格式第二实现 | 4.42e-11 相对 | **是** |
| **本轮的判据：证书 vs 当前代码** | 已提交 `result.json` | 漂移 **0.0**（逐位一致，阈值 1e-12） | **是** |

诚实声明：以上全部是**代码验证（verification）**——与解析制造解、与第二份独立实现对照，
**不是确认（validation）**，没有任何实验数据参与；`<3%` 在这里回答的是"离散误差与实现一致性"，
不是"物理模型对真实换热器的精度"。

## 7. 注意点与局限

* **F1 的教训**：`--check` 这类"存在但够不到"的闸门比没有闸门更糟——它让 `README` 里的
  "reproducible" 看起来像已经被验证过。判据是"最近一次执行在 CI 里跑过"，而不是"文件里有这个模式"。
* **仍未覆盖**：非均匀/扭曲网格、张量热导 `k(x)`、Neumann/Robin 边（`thermal.py` 的
  `convection_edges` 通路没有解析解对照）、三维单元、瞬态项。本锚点只验证 Q4 + Dirichlet + 常系数。
* **重跑会改写三件套**：`run.py` 主模式会重写 `result.json` / `gates_verified.json` / `README.md`
  以及两张 png，其 sha256 记在 `result.json.figures`；只重跑其中一件会造成"图-数不同源"，
  因此本轮**没有**重跑（数字无需更新），只运行只读的 `--check` 路径。
* **本环境的既存红灯（与本轮无关，未修，如实记录）**：`tests/test_slow_lane.py` 有 3 条失败，
  在 `git stash` 掉本轮改动的干净 main 上**同样失败**（3 failed / 14 passed，两次测量一致）：
  ① `test_registry_matches_a_fresh_collection` —— 本机缺可选的同级 TensorLBM checkout，
  `tests/test_fluid_rigid_feedback.py`、`tests/test_passive_rigid_feedback.py` 收集报错（`pyproject.toml` 已声明该 fixture 可选）；
  ② `test_registry_entries_are_canonical_and_sorted` —— 清单在 index 7 处已非字典序（`box_girder` 先于 `boussinesq`）；
  ③ `test_every_registry_entry_names_a_live_test_function` —— `tests/test_pinched_cylinder.py` 的两条登记项
  指向已不存在的函数名。三者属 slow-lane 记账卫生，与热传导无关，留待单独任务。

## 8. 复现

```bash
# 只读复算（证书 vs 当前 checkout），CI 里即 tests/test_heat_l2_reproduction.py 的慢道用例
PYTHONPATH=src python3 benchmarks/verified/fem_heat_conduction_l2/run.py --check

# 慢道用例（全阶梯重算一次）
PYTHONPATH=src python3 -m pytest \
  tests/test_heat_l2_reproduction.py::test_committed_certificate_reproduces_from_this_checkout -q

# 5 条失败关闭对照（fast 车道，秒级）
PYTHONPATH=src python3 -m pytest tests/test_heat_l2_reproduction.py -q -m "not slow"
```