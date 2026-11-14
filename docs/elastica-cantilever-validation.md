# 端部集中力悬臂梁大挠度（elastica）验证

> **Round 2（2026-11，锚点 CASE-CANTILEVER-TIP）更正**：本报告的两个结论已被
> [CASE-CANTILEVER-TIP-elastica-mesh-independence.md](CASE-CANTILEVER-TIP-elastica-mesh-independence.md)
> 修正，原文保留不改（历史记录）：
> 1. §4/§5 的「n=8/16 不收敛，指向差分切线条件数恶化」**归因错误**。重放实验
>    （`scripts/elastica_convergence_floor_study.py`）证明 Newton 迭代本身已到不动点，
>    是**绝对残差容差 1e-10 低于双精度舍入地板**（地板随网格/装配规模增长），
>    导致已收敛状态被判为不收敛、载荷步反复折半后归零。修复后 n=2…32 全部收敛，
>    端挠度误差 4.58%→0.0219% 单调下降，门禁 `qualified=True`。
> 2. §1 参数表的两处数字有误：`EI = E b t³/12 = 100`（不是 10），因此
>    `P = α·EI/L² = 1.0·α`（不是 0.1·α）。无量纲 α 与结论不受影响，但表格须以
>    派生量为准；`scripts/run_elastica_cantilever_benchmark.py` 现已改为从几何派生。

## 1. 案例与物理背景

已合并的大转动证据（[large-rotation-shell-validation.md](large-rotation-shell-validation.md)）是**位移控制**的纯弯曲：端部转角被线性指定，参考解为圆弧线。圆弧只是 elastica 的一个特例，该算例从不检验**载荷控制**路径。本报告补的是载荷控制的经典互补算例：悬臂梁自由端受横向集中力 $P$，解为非圆弧的 Kirchhoff/Clebsch elastica。

无量纲载荷 $\alpha=PL^2/(EI)$，参考量取自由端横向挠度 $w/L$ 与轴向回缩 $u/L$。

| 量 | 值 |
|---|---|
| 长度 L | 10.0 |
| 宽度 b | 1.0 |
| 厚度 t | 0.1 |
| 杨氏模量 E | 1.2e6 |
| 泊松比 ν | 0.0 |
| 抗弯刚度 EI = Ebt³/12 | 10.0 |
| 端部载荷 P | 0.1·α（α=1/2/5/10 → P=0.1/0.2/0.5/1.0） |

## 2. 方法

**参考解（独立推导，非图表数字化）**：以自由端为弧长起点、$k^2=P/(EI)$，$\theta''=-k^2\cos\theta$，$\theta(0)=\theta_0$、$\theta'(0)=0$（端部零弯矩）、$\theta(L)=0$（固支）。首次积分 $\theta'^2=2k^2(\sin\theta_0-\sin\theta)$，自由端转角由长度条件确定

$$I_{len}(\theta_0)=\int_0^{\theta_0}\frac{d\theta}{\sqrt{2(\sin\theta_0-\sin\theta)}}=\sqrt{\alpha}$$

端部运动学用同一核乘 $\cos\theta$、$\sin\theta$；积分经 $\theta=\theta_0-t^2$ 消去端点奇异性。两条互不依赖的数值路线：(a) 512 点 Gauss-Legendre 求积 + Brent 求根；(b) 直接对 ODE 系统 $[\theta,\theta',x,z]$ 做 `scipy.solve_bvp`（$k$ 连续化）。

**有限元侧**：共旋二节点 Euler-Bernoulli 梁，每节点 3 自由度 $(u,w,\theta)$，局部能量 $\tfrac12(EA/L_0)\Delta^2+(2EI/L_0)(\alpha_i^2+\alpha_i\alpha_j+\alpha_j^2)$，$\alpha_{i,j}=\theta_{i,j}-\varphi$（部弦转角）。牛顿迭代 + 自适应载荷增量（初增量 0.1，收敛放大 1.5×，失败减半），切线取梯度的中心差分（步长 1e-6）。

## 3. 计算结果

### 3.1 参考解自检（两条独立路线）

| α | w/L（求积） | u/L（求积） | θ₀ [rad] | w/L（BVP） | 两路线最大差 |
|---|---|---|---|---|---|
| 1 | 0.301721 | 0.056433 | 0.461352 | 0.301721 | 8.7e-12 |
| 2 | 0.493457 | 0.160642 | 0.781750 | 0.493457 | 2.3e-12 |
| 5 | 0.713792 | 0.387628 | 1.215368 | 0.713792 | 5.6e-12 |
| 10 | 0.810609 | 0.554996 | 1.430286 | 0.810609 | 2.6e-12 |

小载荷极限：α=1e-3 时 $w/L$ 与 $\alpha/3$ 相对差 <2e-4，$\theta_0$ 与 $\alpha/2$ 相对差 <1e-3。α=1 时 $w/L=0.301721$，与文献表值 0.30172 相差 7.7e-7；α=2/5/10 分别为 0.493457/0.713792/0.810609，与表值最大差 2.5e-6。

### 3.2 有限元 vs 参考解

| α | 单元数 | 收敛 | w（FE/精确） | 挠度相对误差 | u（FE/精确） | 回缩相对误差 |
|---|---|---|---|---|---|---|
| 1 | 2 | yes | 3.03842 / 3.01721 | 0.7030% | 0.54245 / 0.56433 | 3.8777% |
| 1 | 4 | yes | 3.02217 / 3.01721 | 0.1643% | 0.55852 / 0.56433 | 1.0302% |
| 1 | 8 | no | 0.01302 / 3.01721 | not applicable | 0.00001 / 0.56433 | not applicable |
| 1 | 16 | no | -0.00000 / 3.01721 | not applicable | -0.00000 / 0.56433 | not applicable |
| 2 | 2 | yes | 5.02169 / 4.93457 | 1.7655% | 1.57045 / 1.60642 | 2.2389% |
| 2 | 4 | yes | 4.95437 / 4.93457 | 0.4012% | 1.59586 / 1.60642 | 0.6570% |
| 2 | 8 | no | -0.00000 / 4.93457 | not applicable | -0.00000 / 1.60642 | not applicable |
| 2 | 16 | no | -0.00000 / 4.93457 | not applicable | -0.00000 / 1.60642 | not applicable |
| 5 | 2 | yes | 7.38609 / 7.13792 | 3.4769% | 3.89401 / 3.87628 | 0.4573% |
| 5 | 4 | yes | 7.19172 / 7.13792 | 0.7537% | 3.87102 / 3.87628 | 0.1358% |
| 5 | 8 | no | 0.43535 / 7.13792 | not applicable | 0.01134 / 3.87628 | not applicable |
| 5 | 16 | no | -0.00000 / 7.13792 | not applicable | -0.00000 / 3.87628 | not applicable |
| 10 | 2 | yes | 8.47770 / 8.10609 | 4.5844% | 5.67595 / 5.54996 | 2.2702% |
| 10 | 4 | yes | 8.18834 / 8.10609 | 1.0147% | 5.55708 / 5.54996 | 0.1284% |
| 10 | 8 | no | 0.41593 / 8.10609 | not applicable | 0.01035 / 5.54996 | not applicable |
| 10 | 16 | no | -0.00000 / 8.10609 | not applicable | -0.00000 / 5.54996 | not applicable |

图：`artifacts/elastica_cantilever/figures/load_deflection.png`、`artifacts/elastica_cantilever/figures/error_sweep.png`（红色=未收敛）。

## 4. 误差判定（是否 <3%）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 端部挠度 w/L，α=1，4 单元 | elastica 精确解 | 0.1643% | 是 |
| 端部挠度 w/L，α=2，4 单元 | elastica 精确解 | 0.4012% | 是 |
| 端部挠度 w/L，α=5，4 单元 | elastica 精确解 | 0.7537% | 是 |
| 端部挠度 w/L，α=10，4 单元 | elastica 精确解 | 1.0147% | 是 |
| 轴向回缩 u/L，α=10，4 单元 | elastica 精确解 | 0.1284% | 是 |
| 参考解自身（求积 vs BVP） | 两条独立路线 | ≤1.2e-12 | 是 |
| w/L，α=10，2 单元 | elastica 精确解 | 4.5844% | **否**（粗网格，保留未剔除） |
| 细网格 n≥8 全部载荷 | elastica 精确解 | 未收敛，无数值可判 | **无法判定（fail-closed）** |

口径：误差为 $|w_{FE}/w_{exact}-1|$。最细**已收敛**网格（n=4）的最大挠度误差 1.0147%，满足 3% 门限；但门禁 `qualified` 仍为 **False**，因为最细网格（n=8/16）未收敛。

## 5. 注意点与局限

- **细网格不收敛是本次的主要负面结果**：n=8、n=16 在首增量即失败（载荷因子停在 0.00–0.13），8/16 个案例未收敛，全部保留在 JSON 中，未剔除。该现象随网格加密出现，指向差分切线条件数恶化，而非物理极限（悬臂端部死载荷不存在极限点）。本报告**只认证 n≤4 的收敛区**，不外推。
- 有限元侧是**专用梁路径**，不是 Shell4 面元。Shell4 在载荷控制下的试点（端部死载荷条带）单次切线装配约 0.43 s/单元，且未能在预算内稳定收敛，故未纳入；仓库既有 Shell4 大转动证据仍只有位移控制口径。该边界不得外推为「壳单元通过 elastica 验证」。
- 参考解为**不可伸长** Euler-Bernoulli elastica：不覆盖横向剪切（Timoshenko）与轴向可伸长效应，也不覆盖 $\alpha\to\infty$ 的贴卷极限（$\theta_0\to\pi/2$ 处积分仅对数发散，数值上不可达）。
- 文献表值仅用于参考解的粗校验（w/L 四项，最大差 2.5e-6）；参考值由本模块独立推导，未复制任何图表内容。

## 6. 复现

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_elastica_cantilever_benchmark.py \
    --alphas 1,2,5,10 --meshes 2,4 --out artifacts/elastica_cantilever
PYTHONPATH=src python -m pytest tests/test_elastica_cantilever.py -q
```

证据：`artifacts/elastica_cantilever/results.json`（逐案例原始值/参考解/收敛标志）、`artifacts/elastica_cantilever/figures/`。
