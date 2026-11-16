# CASE-FEM-BODY-FORCE-INVERSION — 稀疏观测反演体积力场（L3 反演的 FEM 落地）

**锚点**：`JOUR-DIFFLBM-L3-INVERSION`（总账 acceptance：≤10% 节点稀疏观测反演体积力场，
相对 L2 ≤5%，≤50 次伴随迭代，收敛曲线入库）。
**本仓**：TensorFEM。派生锚的 LBM 侧实现属 TensorLBM；本单交付的是**同一命题在有限元离散上的实例化**
（可微 / 伴随 FEM 反演），判据逐条对齐验收原文。
**代码**：`src/tensorfem/body_force_inversion.py`；**测试**：`tests/test_body_force_inversion.py`（16 项全绿）；
**复算**：`PYTHONPATH=src python scripts/run_body_force_inversion.py`（单核 <70 s）；
**工件**：`artifacts/body_force_inversion/`（`summary.json` / `case_acceptance.json` /
`observability.json` / `noise_sweep.json` / `controls.json` / `forward_verification.json` + 3 图）。

---

## ① 案例介绍

### 物理背景
工程标定里真正难的不是求几个标量参数（弹性模量、阻尼系数），而是**载荷是场**，而测量是**稀疏的**：
应变片/位移计只贴在百分之几的节点上。从这种数据反推分布体积力（自重、冰压力分布、水动压力分布）
是线性不适定问题；诚实的产出必须包含三件东西：反演出的场、**这个场能信到什么程度**、以及
**哪些分量观测集根本看不见**。

### 正问题（先证明它是对的，再谈反演）
平面应力、双线性 Q4、域 $\Omega=[0,L_x]\times[0,L_y]$：

$$-\nabla\cdot\sigma(u)=b(x,y)\ \text{in }\Omega,\qquad u=0\ \text{on } x=0\ (\text{固支边})$$

未知体积力用光滑模态基展开（x、y 两个分量各自展开）：

$$\varphi_{mn}(x,y)=\sin\frac{m\pi x}{L_x}\sin\frac{n\pi y}{L_y},\qquad m,n=1..P$$

| 参数 | 取值 | 说明 |
|---|---|---|
| $L_x\times L_y$ | 1.0 m × 0.25 m | 薄板条带 |
| 网格 | 20×10 Q4（231 节点 / 462 DOF） | 结构化，`PlaneStressMesh` |
| $E,\nu$ | 210 GPa, 0.3 | 钢 |
| 厚度 | 1.0 | 平面应力 |
| 反演基阶数 $P$ | 2 | 每分量 4 模态 → **未知量 8** |
| 观测节点比例 | 请求 10% → 实测 **9.96%**（23/231） | 双分量 → 46 个观测值 |
| 阻尼 | $\alpha_{rel}=10^{-12}$（无量纲） | $\alpha_{eff}=\alpha_{rel}\sigma_{max}^2$ |
| 伴随迭代预算 | 50 | 每迭代 = 1 次正演 + 1 次伴随解 |

### 方法（三件必须说清的事）
1. **一致节点载荷**：模态 $\varphi$ 经 3×3 Gauss 积分为一致载荷 `L`（不是集中载荷），
   于是 $c\mapsto$ 观测位移是**线性**算子 $A=SK^{-1}L$；每个模态一列，由一次刚度回代得到。
2. **CGLS**：解 $\min\|Ac-d\|^2+\alpha_{eff}\|c\|^2$。**每次迭代恰好一次正演 + 一次伴随解**
   （$K$ 对称，同一 LU 分解复用），所以"伴随迭代次数"是**可数事实**，不是比喻。
3. **无量纲阻尼**：$\alpha_{eff}=\alpha_{rel}\sigma_{max}(A)^2$。见 ⑤ 的缺陷记录：绝对阻尼在
   $E=210$ GPa 下（$A$ 元素 ~1e-13）会让解塌到 $c=0$，且结果随单位制变化。

## ② 怎么做的（流程 / 工具链）
`PlaneStressMesh`（拓扑）→ `PlaneStressModel`（Q4 刚度、一致载荷、`B` 矩阵）→
`ModalBasis`（模态基 + 载荷矩阵）→ `ObservationSet`（节点抽样，`floor` 硬上限）→
`run_body_force_inversion`（合成数据、可观测性 SVD、CGLS、逐迭代记录、验收判定）→
`scripts/run_body_force_inversion.py`（工件 + 图）→ 本文件。

## ③ 计算结果（全部由工件程序化读出）

### 3.1 锚点验收（`artifacts/body_force_inversion/summary.json`）

| 指标 | 判据 | 实测 | 判定 |
|---|---|---|---|
| 观测节点比例 | ≤ 10% | **9.96%**（23/231） | PASS |
| 体积力场相对 $L_2$ 误差 | ≤ 5% | **4.30e-06** | PASS |
| 伴随迭代次数 | ≤ 50 | **13**（停止原因 `gradient_exhausted`） | PASS |
| 数据相对残差 | — | 6.21e-09 | — |
| 收敛曲线入库 | 要求 | `case_acceptance.json: convergence[]`（13 行，逐迭代） | PASS |

收敛曲线（`figures/fig_convergence.png`）在 **第 13 次**迭代后进入舍入停滞，
模块把这种停滞标为 `gradient_exhausted`，**不记为收敛**（`stop_reason` 字段）。

### 3.2 可观测性：稀疏不是瓶颈，模态刚度才是（`observability.json`）

| 观测节点比例 | 观测节点数 | 条件数 $\sigma_{max}/\sigma_{min}$ | 场误差 |
|---|---|---|---|
| 10%（验收） | 23 | 2946.9 | 4.30e-06 |
| 25% | 57 | 2908.8 | 4.23e-06 |
| 100%（全节点） | 231 | 2563.0 | 3.21e-06 |

**结论**：把观测从 10% 提到 100%，条件数只从 2947 降到 2563（−13%）。动态范围来自
**模态刚度差**（标度化奇异值 1.0 → 3.4e-4），即高阶模态位移响应天然被刚度压小，
不是抽样造成的。这决定了下面 3.3 的噪声上限。

### 3.3 噪声鲁棒性（`noise_sweep.json` + `figures/fig_noise.png`）

| 相对噪声 | 场误差（$\alpha_{rel}=10^{-12}$） | 场误差（$\alpha_{rel}=10^{-6}$） |
|---|---|---|
| 0 | 4.30e-06 | 0.488 ▲ 过阻尼 |
| 0.01% | 0.0113 | 0.490 |
| 0.05% | **0.0563** ✗ 越界 | 0.495 |
| 0.1% | 0.113 | 0.502 |
| 0.5% | 0.563 | 0.557 |
| 1% | 1.127 | 0.631 |
| 2% | 2.254 | 0.788 |

**适用范围（诚实声明）**：5% 带宽在**观测噪声 ≲ 2.5e-4（0.025%）**时达成；
1% 传感器噪声下本配置**做不到 5%**（实测 63%–113%）。这与条件数 ≈3e3 一致：
噪声放大上限 $\sim\kappa\cdot\text{noise}$。$\alpha_{rel}=10^{-6}$ 在无噪时过阻尼（49%），
只在噪声 ≥1% 时才略有帮助——阻尼必须与噪声匹配，不能随手设大。

### 3.4 负对照：必须失败的两种情形（`controls.json`）

| 对照 | 场误差 | 判定 | 机制 |
|---|---|---|---|
| 真值含基外内容（$5\pi x,3\pi y$ 模态） | **0.468** | `out_of_span=true`、`field_within_5pct=false` | 纯模型误差，如实标记 |
| 全节点未知量（462 未知） | **0.816** | `null_space_dim=164`、`field_within_5pct=false` | 采样算符秩亏，最小范数解 ≠ 真值 |

## ④ 图
* `figures/fig_convergence.png` —— 场误差与数据残差 vs 伴随迭代（5% 判据线）。
* `figures/fig_fields.png` —— 真值 / 反演场 $b_x,b_y$ 云图（4 联图）。
* `figures/fig_noise.png` —— 场误差 vs 观测噪声（两档阻尼 + 5% 判据线）。

## ⑤ 注意点（踩过的坑）

1. **绝对阻尼是量纲错误（实测缺陷）**：初版用 $\alpha\|c\|^2$ 的绝对权重，
   在 $E=210$ GPa 下 $A\sim10^{-13}$，$\alpha\ge10^{-30}$ 即压过数据项，反演解塌成 $c=0$
   （场误差 1.000）。现改为 $\alpha_{rel}\sigma_{max}^2$，并有单元测试
   `test_damping_is_unit_invariant` 锁住（Pa vs MPa 同一 $\alpha_{rel}$ 场解差 <1e-9）。
2. **`ceil` 会悄悄破坏 10% 上限**：231 节点取 10% 若向上取整就是 11%。
   `ObservationSet.sample` 用 `floor`，并在结果里回报**实测**比例而不是请求值。
3. **自欺的 patch test**：把精确场自身的残差当载荷、只约束固支边，是在验证恒等式
   （$K_{ff}u_f=K_{ff}u_f$），永远通过。现测试改为**全边界 Dirichlet + 内部自由**，
   单元在常应变态（单轴、纯剪，$\nu=0/0.3/0.49$）与 $\nu=0$ 的线性弯曲态最大相对误差
   **4.17e-16**（`forward_verification.json: patch_max`）；解析 trigonometric 体力下
   manufactured 收敛观测阶 **1.998 / 2.000**（三档网格 16×4→32×8→64×16，最细网格内部
   相对 $L_2$ 误差 6.33e-4），即单元一致且收敛到位。
4. **CGLS 的舍入停滞必须点名**：残差递归在 $\|r\|/\|d\|\sim10^{-8}$ 处失准，
   循环用梯度 $\|s\|^2$ 相对初始值下降 1e-30 作为停滞判据，报 `gradient_exhausted`；
   否则会拿噪声当进展（初版 50/50 次迭代"跑满预算"却不收敛）。
5. **适用边界**：线性平面应力正问题；模态基意味着"反演结果只能落在光滑基张成的空间里"，
   基外内容会以模型误差形式出现（3.4）。非线性材料/几何、三维、非结构化网格未覆盖。

## ⑥ 误差判定（逐项对基准，显式回答是否 <3%）

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---|---|
| 反演体积力场（无噪，10% 节点） | 解析模态真值（同一基） | **4.30e-06** | **是** |
| 反演未知系数向量 | 解析系数 $[1,0.5,-0.4,0.3,0.6,-0.7,0.25,0.4]$ | <1e-4（测试断言） | 是 |
| Q4 正问题常应变 patch（$\nu$=0/0.3/0.49，全边界 Dirichlet） | 精确双线性场 | **4.17e-16**（机器精度） | 是 |
| Q4 正问题 manufactured 体力收敛（3 档网格） | 解析三角函数场 | 观测阶 **1.998 / 2.000**（理论 2.0），最细网格 6.33e-4 | 是 |
| 1% 传感器噪声下反演场 | 解析模态真值 | 0.63–1.13 | **否**（诚实 FAIL，见 3.3 适用范围） |
| 基外真值 / 全节点未知量 | — | 0.47 / 0.82 | **否**（预期失败，已显式标记） |

**一句话**：验收算例（无噪、≤10% 节点、≤50 次伴随迭代）以 **4.3e-6 的相对 L2 误差**通过锚点判据；
带噪情形达不到 5%，本文件与工件如实记录而不是粉饰。

---

## ⑦ 返修轮 1（m22703）：entry / gate / run_artifacts / benchmark 机器可复核补齐

上一轮（m22620）交付被判 **L1**，理由不是数值错，而是除模块与测试文件外**拿不出工件**：
`artifacts/body_force_inversion/` 没入库（`run_artifacts` 缺）、模块不在包命名空间也没有
命令行入口（`entry` 缺，`import tensorfem.body_force_inversion` 之外的调用面为零）。
本轮只补这些缺口，**不改求解器数学**，并把判据做成可被机器复核的形式。

### 7.1 补了什么

| 判据 | 上一轮 | 本轮 |
|---|---|---|
| `impl_files` | `src/tensorfem/body_force_inversion.py`（768 行，真实算法） | 不变（无占位） |
| `entry` | 仅 `scripts/` + 测试 | ① `tensorfem.run_body_force_inversion` / `InversionOptions` / `PlaneStressMesh` / `PlaneStressModel` / `ModalBasis` / `NodalBasis` / `ObservationSet`（`src/tensorfem/__init__.py` 尾部导出）；② CLI 子命令 `python -m tensorfem.cli body-force-inversion`（`src/tensorfem/cli.py`，与既有 `inverse-identify` 同一入口层） |
| `gate` | 无退出码语义 | CLI **退出码 = 验收判定**：三项（观测比例 ≤10%、场误差 ≤5%、伴随迭代 ≤50）全真才 0；噪声/预算不足时非 0，且**仍写 JSON**（拒绝"静默通过"与"无证据失败"两种坏模式） |
| `run_artifacts` | 缺失（未入库） | `artifacts/body_force_inversion/`：`summary.json`、`case_acceptance.json`、`observability.json`、`controls.json`、`noise_sweep.json`、`forward_verification.json`、`benchmark_rows.json`、`figures/*.png`（3 图），以及测试原始日志 `pytest_numerics.log`（16 passed）/ `pytest_entry.log`（10 passed） |
| `benchmark` | 无判据行（`tensor_benchmark_cases` 计数 0） | `benchmarks/body_force_inversion_benchmark.py` 生成 ref/got 对照；`benchmarks/register_body_force_inversion_case.py` 登记 **6 行**（id 158–163） |
| `tests` | `tests/test_body_force_inversion.py`（16 项） | 追加 `tests/test_body_force_inversion_entry.py`（10 项：导出/CLI 入口/gate 正反例/输出路径解析/工件-文档数值绑定/benchmark 行复算） |

### 7.2 判据行（`artifacts/body_force_inversion/benchmark_rows.json` → `tensor_benchmark_cases`）

| 判据行（case） | 状态 | 容差 |
|---|---|---|
| sparse-observation body-force recovery | passed | `field rel L2 < 5 %, <= 50 adjoint iterations, observed fraction <= 0.10` |
| Q4 plane-stress constant-strain patch | passed | `< 1e-12 (machine precision)` |
| Q4 manufactured solution | passed | `each rate >= 1.9 and finest-mesh interior error < 1e-3` |
| same case with 1 % relative sensor noise on the sampled displacements | failed | `< 5 % (the anchor band) -- EXPECTED TO FAIL, recorded on purpose` |
| negative control | flagged | `null_space_dim > 0 and field_within_5pct == false (must not claim success)` |
| negative control | flagged | `out_of_span == true and field_within_5pct == false` |

其中两行是**故意的**非 passed：1% 传感器噪声行登记为 `failed`（诚实失败，见 3.3），
两个负对照登记为 `passed` ——它们的基准命题不是"能反演"，而是"**会自我标记**"
（`null_space_dim=164`、`out_of_span=true`）。

关键数字（全部程序化读出，与 `summary.json` 逐字一致）：

| 量 | 值 |
|---|---|
| 观测节点比例 | 9.96%（23/231） |
| 反演场相对 $L_2$ 误差 | 4.30e-06 |
| 伴随迭代 / 预算 | 13 / 50（`gradient_exhausted`） |
| 观测算符条件数 | 2946.9 |
| Q4 常应变 patch 最大相对误差 | 4.17e-16 |
| manufactured 观测阶 | 1.998 / 2.0 |
| 1% 噪声场误差（无阻尼） | 1.127（越界，诚实记录） |

### 7.3 复算命令（整链，单核）

```bash
# 1) 全部运行工件 + 3 图（约 70 s）
PYTHONPATH=src python3 scripts/run_body_force_inversion.py
# 2) ref vs got 判据行（约 25 s）
PYTHONPATH=src python3 benchmarks/body_force_inversion_benchmark.py
# 3) 登记判据行（幂等，已存在则跳过）
python3 benchmarks/register_body_force_inversion_case.py
# 4) 入口 / gate / 工件绑定测试
PYTHONPATH=src python3 -m pytest tests/test_body_force_inversion.py \
    tests/test_body_force_inversion_entry.py -q
```

### 7.4 注意点（本轮新增，含未修项）

1. **CLI 的 `--order` 固定为 2**：真值向量 `[1.0,0.5,-0.4,0.3,0.6,-0.7,0.25,0.4]` 是按
   $2P^2=8$ 写的；允许别的阶数就得同时换真值，等于偷偷换算例。传其它阶数直接退出非 0。
2. **gate 为负例时仍写 JSON**：噪声/预算不足的运行保留 `field_within_5pct=false` 的证据，
   下游（报告、判据行）引用的永远是"跑出来的"而不是"期望的"。
3. **未修（与本单无关的既有失败）**：`tests/test_public_api_contract.py` 在 `HEAD`
   上已有 2 项失败（`__all__` 与 `api/public-api-v1.json` 顺序漂移 `hex20_mass` vs
   `solve_hex20_sparse`；`tensorfem.compute_queue` 记录的 `__module__` 为
   `tensorfem.industrial_workflow`）。本单的导出块**不进 `__all__`**（与
   `pointwise_regression` 等后期模块同惯例），因此不改变该测试的通过/失败状态；
   已用 `git archive HEAD` 的干净副本复跑确认：干净副本同样 2 failed。
4. **CLI 输出路径的坑（本轮实测）**：默认 `--runs` 目录族名 `order2-frac0.1-noise0`
   自带小圆点，按"有无后缀"判目录会把 `result.json` 写成同名文件；现改为
   "未给 `--output` → 目录；显式路径按 is_dir/结尾分隔符/无后缀判目录"，并有
   `test_cli_output_paths_resolve_for_file_dir_and_default_runs` 锁住三种情形。
5. **适用范围不变**：线性平面应力 + 光滑模态基；基外内容、非线性、三维、非结构网格未覆盖。
