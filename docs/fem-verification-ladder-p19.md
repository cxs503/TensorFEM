# P19-N01 — TensorFEM 基础验证阶梯：Euler–Bernoulli 梁 / Kirchhoff 板 / 模态分析

**任务**：TensorFEM 战役节点 P19-N01（mission 5418），在 TensorFEM 上建立三级解析验证阶梯，
用闭式解/级数解证明线性弹性 Kirchhoff 求解器的基本正确性。
**结论**：三级全部 PASS（23 个受判数据点全部低于阈值，收敛阶 2.0–4.0，见 §6）。

| 项 | 内容 |
|---|---|
| L1 梁静态 | 简支梁均布载荷 + 悬臂端部集中力（Hermite 梁单元） |
| L2 板静态 | 四边简支 (SSSS) + 四边固支 (CCCC) 方板均布压力（BFS C1 单元） |
| L3 模态 | 简支梁前 5 阶 + 简支矩形板前 5 阶固有频率 |
| 新库模块 | `src/tensorfem/plate_bfs.py`（BFS 单元 + 一致质量/载荷 + Kirchhoff 解析预言机） |
| 新测试 | `tests/test_plate_bfs.py`（12 项）、`tests/test_p19_ladder_artifacts.py`（10 项） |
| 产物 | `benchmarks/verified/fem_beam_static/`、`fem_plate_static/`、`fem_modal/` 各含 `result*.json` + `gates_verified.json` + 图 |

---

## 1 案例介绍（物理背景与参数）

### 1.1 L1 — Euler–Bernoulli 梁静力

等截面直梁、线弹性、小变形、无横向剪切变形：

    EI w'''' = q,   w(0)=w(L)=0（简支）；EI w'''' = 0, w(0)=w'(0)=0（悬臂）

| 参数 | 简支梁 | 悬臂梁 |
|---|---|---|
| 长度 L | 10 m | 10 m |
| 抗弯刚度 EI | 1000 N·m² | 1000 N·m² |
| 载荷 | 均布 q = 1 N/m | 端部集中 P = 0.1 N |
| 闭式解 | `w(x)=q x(L³-2Lx²+x³)/(24EI)`，`w(L/2)=5qL⁴/(384EI)=0.130208 m` | `w=P L³/(3EI)=3.3333e-2 m`，`θ=P L²/(2EI)=5.0e-3 rad` |
| 对比截面 | x = 0.9, 2.4, 3.9, 5.4, 6.9 m（三档网格下**均非节点**） | 端部 w、θ |

> 截面点特意取在非节点处：Hermite 梁单元的节点位移对 Euler–Bernoulli 分布载荷是**精确**的
> （节点超收敛），若在节点上对比只会验证到舍入误差。取非节点截面才真正检验形函数的插值质量，
> 也才能测出真实的 4 阶收敛率。

### 1.2 L2 — Kirchhoff 薄板静力

方板 a = b = 1 m、t = 10 mm（a/t = 100）、E = 210 GPa、ν = 0.3、均布压力 q = 10 kPa，
抗弯刚度 `D = E t³/(12(1-ν²)) = 19230.77 N·m`。

* **SSSS**：Navier 双正弦级数解

      w(x,y) = 16q/(π⁶D) Σ_{m,n odd} sin(mπx/a) sin(nπy/b) / [m n ((m/a)²+(n/b)²)²]

  按任务要求截断到 **11×11**（m, n ≤ 11）；同时用同一级数的 **301 项**收敛值作为判收敛阶的预言机
  （11 项截断对挠度已达 5e-8，但对弯矩只有 ~1e-3，若用 11 项去判弯矩就只是在测级数本身）。
* **CCCC**：Timoshenko & Woinowsky-Krieger 表 35：`w_c = 0.00126 q a⁴/D`、
  中心 `M_x = M_y = 0.0231 q a²`（三位有效数字，本身带 ~0.4% 舍入）。

### 1.3 L3 — 自由振动

| 参数 | 简支梁 | 简支矩形板 |
|---|---|---|
| 几何 | L = 4 m，b = h = 50 mm（L/h = 80） | a×b = 1.0×0.75 m（a/b = 4/3），t = 10 mm |
| 材料 | E = 200 GPa，ρ = 7850 kg/m³ | E = 210 GPa，ν = 0.3，ρ = 7850 kg/m³ |
| 预言值 | `ω_n=(nπ/L)²√(EI/ρA)`；f₁…f₅ = 7.153, 28.610, 64.373, 114.440, 178.813 Hz | `ω_mn=π²√(D/ρt)[(m/a)²+(n/b)²]`； (1,1) 68.294、(2,1) 142.051、(1,2) 199.418、(3,1) 264.980、(2,2) 273.175 Hz |
| 说明 | 梁无转动惯量（EB 理论） | a/b = 4/3 保证前 5 阶**无简并**，FE 与解析可一一配对 |

---

## 2 怎么做的（方法 / 流程 / 工具链）

```
几何与材料参数
   └─> 离散（N = 10 / 20 / 40，板另加 N = 80）
        └─> 单元矩阵（梁：tensorfem.modal；板：tensorfem.plate_bfs BFS C1）
             └─> 一致载荷/一致质量装配（稀疏）
                  └─> 边界条件（简支 hard-SS：w=0 + 切向斜率=0；固支：w=w,x=w,y=0）
                       └─> 静态：SuperLU / 模态：shift-invert Lanczos（σ=0）
                            └─> 后处理：Hermite 插值恢复 w、w,x、w,y、w,xx、w,yy、w,xy → 弯矩
                                 └─> 与闭式解/级数/表值逐点比较 → gates_verified.json
```

1. **单元实现（新库模块 `tensorfem/plate_bfs.py`）**
   * BFS 双三次 Hermite 矩形（16 自由度/单元：`w, w,x, w,y, w,xy`），位移场 C¹ 协调，
     是 Kirchhoff 板的协调单元。
   * 弯曲刚度：`K_e = ∫ Bᵀ D_b B dA`，`B = [w,xx; w,yy; 2w,xy]`，
     `D_b = D[[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]]`，3×3 Gauss（对二次被积函数精确）。
   * 一致载荷 `f_e = ∫ q N dA`（3×3 Gauss 精确）——**不是**集中质量式节点载荷。
     这一点很重要：仓库里既有的 `benchmarks/verified/fem_plate` 用的是集中载荷，
     同样网格下误差比本模块大两个数量级（N=10：1.45% vs 0.003%）。
   * 一致质量 `M_e = ρt ∫ N_i N_j dA`（4×4 Gauss，对 6 次多项式精确）。
   * 边界：`bfs_boundary_dofs(..., "ssss")` 固定边节点 `w` 与**切向**斜率（Kirchhoff 精确条件，
     同时消去角点扭转自由度，去掉角点上翘）；`"cccc"` 固定 `w` 与两个斜率（角点 `w,xy` 自由，
     Zienkiewicz 经典做法，收敛阶梯证明其正确）。
   * 自校验：本模块的弯曲矩阵与 `tensorfem.plate_buckling_hermite.bfs_plate_element_matrices`
     （独立实现的同族单元）在 `tests/test_plate_bfs.py` 中比对到
     `|ΔK|max / |K|max ≈ 1.6e-16`；刚体运动、常曲率 patch 检验、一致质量/载荷的积分总量
     （ρtA、qA）都被单测钉死。
2. **求解**：静态用 `scipy.sparse.linalg.spsolve`（SuperLU）；板模态用
   `eigsh(K, k, M, sigma=0)`（shift-invert Lanczos），梁模态用
   `tensorfem.modal.solve_modes`（M 的 Cholesky 对称化 + LAPACK `eigh`）。
3. **后处理**：单元内用张量积 Hermite 基恢复 `w` 及其一、二阶导数，弯矩
   `M_x = -D(w,xx + ν w,yy)`。SSSS 情况该恢复与 Navier 级数弯矩吻合到 0.02%（N=40），
   说明恢复公式本身无误。
4. **判据**：所有阈值在驱动器里以常量出现，逐数据点写进 `gates_verified.json`；
   `tests/test_p19_ladder_artifacts.py` 重新读取 JSON 复核“数据点数量 / 阈值 / 是否全部通过”，
   避免“验收后偷偷放宽阈值”。
5. **工具链**：numpy / scipy / torch（仅复用 `tensorfem.modal` 的梁单元矩阵）/ matplotlib；
   `benchmarks/verified/_harness.py` 统一负责 JSON 与 gate schema。

---

## 3 计算结果（数字全部由脚本写入 JSON，零手抄）

### 3.1 L1 梁静态（N = 40，单元数 40，自由度 82）

| x [m] | w_FE [m] | w_exact [m] | 相对误差 |
|---:|---:|---:|---:|
| 0.9 | 3.691983e-02 | 3.691984e-02 | 2.54e-07 |
| 2.4 | 8.986239e-02 | 8.986240e-02 | 1.04e-07 |
| 3.9 | 1.227068e-01 | 1.227068e-01 | 7.64e-08 |
| 5.4 | 1.292094e-01 | 1.292094e-01 | 7.26e-08 |
| 6.9 | 1.081888e-01 | 1.081888e-01 | 8.67e-08 |

| N | h [m] | 最大截面误差 | 位移场 L2 相对误差 |
|---:|---:|---:|---:|
| 10 | 1.00 | 2.67e-05 | 1.78e-05 |
| 20 | 0.50 | 1.81e-06 | 1.11e-06 |
| 40 | 0.25 | 2.54e-07 | 6.97e-08 |

悬臂端部：`w = 3.3333333333548523e-2 m`（精确 3.3333333333333333e-2，误差 6.5e-11），
`θ = 4.9999999996917e-3 rad`（精确 5.0e-3，误差 6.2e-11）。

### 3.2 L2 板静态（D = 19230.77 N·m，q = 10 kPa）

**SSSS**（Navier11 = 2.112399e-3 m，Navier301 = 2.112423e-3 m，表值 α = 0.00406 → 2.1125e-3 m）

| N | h [m] | w_c [mm] | α = w_c D/(q a⁴) | vs Navier11 | vs Navier301 | vs 0.00406 | M_x vs Navier301 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.100 | 2.112460 | 0.0040623 | 2.89e-05 | 1.73e-05 | 5.97e-04 | 3.59e-03 |
| 20 | 0.050 | 2.112426 | 0.0040624 | 1.27e-05 | 1.07e-06 | 5.81e-04 | 8.68e-04 |
| 40 | 0.025 | 2.112424 | 0.0040624 | 1.17e-05 | 6.64e-08 | 5.80e-04 | 2.15e-04 |

中心弯矩（N=40）：`M_x = M_y = 478.76 N`（= 0.047876 q a²），Navier301 为 478.864 N
（0.047886 q a²），表值 0.0479。

**CCCC**（表值 α = 0.00126 → w_c = 0.65800 mm）

| N | w_c [mm] | α | vs 0.00126 | M_x/(q a²) | vs 0.0231 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.660011 | 0.0012693 | 7.34e-03 | 0.02322 | 5.01e-03 |
| 20 | 0.658223 | 0.0012658 | 4.61e-03 | 0.02298 | 5.40e-03 |
| 40 | 0.657999 | 0.0012654 | 4.27e-03 | 0.02292 | 7.71e-03 |
| 80（扩展） | 0.657953 | **0.00126533** | 4.23e-03 | 0.022909 | 8.27e-03 |

N = 80 的 α = 0.00126533 与经典高精度值 0.00126532 吻合到 1e-5，
说明 0.42% 的“误差”**全部来自表值 0.00126 的三位有效数字舍入**，而不是离散误差。

### 3.3 L3 模态

**简支梁**（相对误差）

| 阶 | N=10 | N=20 | N=40 | f_exact [Hz] | 观测阶 |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.75e-06 | 4.23e-07 | 2.72e-08 | 7.153 | 3.98 |
| 2 | 1.07e-04 | 6.75e-06 | 4.23e-07 | 28.610 | 3.99 |
| 3 | 5.34e-04 | 3.40e-05 | 2.14e-06 | 64.373 | 3.98 |
| 4 | 1.66e-03 | 1.07e-04 | 6.75e-06 | 114.440 | 3.97 |
| 5 | 3.95e-03 | 2.60e-04 | 1.64e-05 | 178.813 | 3.95 |

**简支矩形板 a×b = 1.0×0.75 m**（相对误差）

| 模态 (m,n) | f_exact [Hz] | N=10 | N=20 | N=40 | 观测阶 |
|---|---|---:|---:|---:|---:|
| (1,1) | 68.294 | 7.64e-06 | 6.02e-07 | 3.76e-08 | 3.83 |
| (2,1) | 142.051 | 5.33e-05 | 3.37e-06 | 2.11e-07 | 3.99 |
| (1,2) | 199.418 | 2.00e-04 | 1.64e-05 | 1.03e-06 | 3.80 |
| (3,1) | 264.980 | 3.76e-04 | 2.38e-05 | 1.49e-06 | 3.99 |
| (2,2) | 273.175 | 1.22e-04 | 9.63e-06 | 6.02e-07 | 3.83 |

---

## 4 结果图

| 文件 | 内容 |
|---|---|
| `benchmarks/verified/fem_beam_static/deflection_profile.png` | 简支梁三档网格挠曲线 vs 解析解（mm） |
| `benchmarks/verified/fem_beam_static/convergence.png` | 悬臂端部挠度误差（舍入极限 ~1e-11） |
| `benchmarks/verified/fem_plate_static/fields_ssss.png` | SSSS 板 w 云图（N=40）+ 中线和 Navier 曲线重叠 |
| `benchmarks/verified/fem_plate_static/fields_cccc.png` | CCCC 板 w 云图（N=40）+ 中曲线 |
| `benchmarks/verified/fem_plate_static/convergence.png` | 板挠度/弯矩误差 vs h（h²、h⁴ 参考线） |
| `benchmarks/verified/fem_modal/beam_modes.png` | 梁前三阶振型 vs `sin(nπx/L)` |
| `benchmarks/verified/fem_modal/plate_modes.png` | 板 (1,1)(2,1)(1,2) 振型云图 |
| `benchmarks/verified/fem_modal/modal_convergence.png` | 频率误差 vs h（h⁴ 参考线） |

> 图由 matplotlib（Agg）程序化生成，数据来自与 JSON 同一次求解；脚本重跑即可复现。
> 若平台文件门禁滤掉二进制图，`result*.json` 中的数值仍是完整的可审计证据。

---

## 5 注意点（踩过的坑 / 局限 / 适用范围）

1. **BFS 形函数索引坑**：二维张量积的 1D 基函数下标必须带节点偏移
   （`2*di + k%2`、`2*dj + k//2`）。漏掉偏移时刚体运动不被零空间吃掉，
   单元矩阵与参考实现差一个量级，SSSS 挠度偏小 5 个数量级；已在单测里加了
   “刚体运动零能 + 常曲率 patch”双检验。
2. **Jacobian**：在 [0,1]² 上做 Gauss 积分时 `dx dy = lx·ly·ds·dt`（不是 `lx·ly/4`）。
   这两个错误都会静默给出“看起来收敛”的结果，必须靠 patch test 兜住。
3. **一致载荷 vs 集中载荷**：仓库既有 `fem_plate` 用集中载荷，N=10 时 1.45% 误差；
   一致载荷同网格 0.003%。本阶梯判的是求解器能力，因此用一致载荷，
   并在 gates 的 limitations 里注明两者不可直接比较。
4. **预言机截断**：11×11 Navier 对挠度足够（5e-8），对弯矩只有 1e-3；
   若用 11 项去判弯矩，会把级数截断误当成离散误差、并伪造出“误差不下降”的假象。
   本阶梯因此同时记录 11 项（按任务要求）与 301 项（判阶用）两套值。
5. **表值舍入地板**：CCCC 的 `0.00126`、`0.0231` 是三位有效数字，
   分别带来 0.42% 与 0.83% 的地板；因此 CCCC 的**收敛阶用自收敛（Richardson）判定**，
   不用表值判。弯矩对表值的误差非单调（N≈10 时恰好穿过 0.0231），已在限制中写明。
6. **适用域**：Kirchhoff（薄板）与 Euler–Bernoulli（细长梁）理论；a/t = 100、L/h = 80
   的算例处在薄结构适用区间，Mindlin 剪切修正量级 ~1e-4，远低于 1–3% 阈值，
   但本阶梯**没有**验证横向剪切、转动惯量、几何非线性、材料非线性。
7. **只覆盖简支/固支**：自由边、点支承、开孔板、加筋板不在本阶梯范围。
8. **既有仓内遗留问题（非本单造成）**：`tests/test_structural_verification.py` 中
   `test_committed_verified_artifacts_are_consistent[fem_beam/fem_plate/fem_modal]` 因
   `benchmarks/verified/fem_*/convergence.png` 在仓库中缺失而失败（历史上被推送文件门禁滤除，
   见 commit `46c58b9` “4 个文件被门禁滤除”）；`tests/test_public_api_contract.py` 与
   `tests/test_release_quality.py` 的 API 快照测试也已在 main 上漂移失败，与新增模块无关
   （失败信息中不含 `plate_bfs`）。

---

## 6 误差判定（逐项，显式回答“是否 < 3%”）

| # | 数据点 | FE 值 | 基准 | 相对误差 | 阈值 | 是否 < 3% |
|---:|---|---|---|---:|---:|---|
| 1 | L1 截面 x=0.9 m | 3.691983e-02 m | 闭式 3.691984e-02 | 2.5e-07 | 0.5% | 是 |
| 2 | L1 截面 x=2.4 m | 8.986239e-02 m | 闭式 | 1.0e-07 | 0.5% | 是 |
| 3 | L1 截面 x=3.9 m | 1.227068e-01 m | 闭式 | 7.6e-08 | 0.5% | 是 |
| 4 | L1 截面 x=5.4 m | 1.292094e-01 m | 闭式 | 7.3e-08 | 0.5% | 是 |
| 5 | L1 截面 x=6.9 m | 1.081888e-01 m | 闭式 | 8.7e-08 | 0.5% | 是 |
| 6 | L1 悬臂端部 w | 3.333333e-02 m | PL³/(3EI) | 6.5e-11 | 0.5% | 是 |
| 7 | L1 悬臂端部 θ | 4.9999999e-03 rad | PL²/(2EI) | 6.2e-11 | 0.5% | 是 |
| 8 | L2 SSSS w_c | 2.112424e-03 m | Navier 11×11 | 1.17e-05 | 1% | 是 |
| 9 | L2 SSSS w_c | 2.112424e-03 m | 表值 0.00406 q a⁴/D | 5.80e-04 | 2% | 是 |
| 10 | L2 SSSS M_x | 478.76 N | Navier301 478.864 N | 2.15e-04 | 2% | 是 |
| 11 | L2 SSSS M_y | 478.76 N | Navier301 478.864 N | 2.15e-04 | 2% | 是 |
| 12 | L2 CCCC w_c | 6.57999e-04 m | 表值 0.00126 q a⁴/D | 4.27e-03 | 2% | 是 |
| 13 | L2 CCCC M_x | 0.02292 q a² | 表值 0.0231 q a² | 7.71e-03 | 2% | 是 |
| 14 | L3 梁 f₁ | 7.153 Hz | (π/L)²√(EI/ρA)/2π | 2.7e-08 | 3% | 是 |
| 15 | L3 梁 f₂ | 28.610 Hz | 同上 n=2 | 4.2e-07 | 3% | 是 |
| 16 | L3 梁 f₃ | 64.373 Hz | 同上 n=3 | 2.1e-06 | 3% | 是 |
| 17 | L3 梁 f₄ | 114.440 Hz | 同上 n=4 | 6.8e-06 | 3% | 是 |
| 18 | L3 梁 f₅ | 178.816 Hz | 同上 n=5 | 1.6e-05 | 3% | 是 |
| 19 | L3 板 (1,1) | 68.294 Hz | π²√(D/ρt)[(m/a)²+(n/b)²]/2π | 3.8e-08 | 3% | 是 |
| 20 | L3 板 (2,1) | 142.051 Hz | 同上 | 2.1e-07 | 3% | 是 |
| 21 | L3 板 (1,2) | 199.418 Hz | 同上 | 1.0e-06 | 3% | 是 |
| 22 | L3 板 (3,1) | 264.980 Hz | 同上 | 1.5e-06 | 3% | 是 |
| 23 | L3 板 (2,2) | 273.175 Hz | 同上 | 6.0e-07 | 3% | 是 |

**23/23 数据点误差 < 3%（实际最大 7.7e-03 = 0.77%，出现在 CCCC 中心弯矩，主要来自表值舍入）。**

网格收敛（三档 N = 10/20/40）：

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| 梁位移场 L2 收敛阶 | ≥ 2 | 4.00 | PASS |
| 梁截面误差收敛阶 | ≥ 2 | 3.36 | PASS |
| 板 SSSS 挠度收敛阶 | ≥ 1.5 | 4.01 | PASS |
| 板 SSSS 弯矩收敛阶 | ≥ 1.5 | 2.03 | PASS |
| 板 CCCC 自收敛阶 | ≥ 1.5 | 2.99 | PASS |
| 板 CCCC 挠度误差（vs 表值）单调下降 | 严格下降 | 7.34e-3 → 4.61e-3 → 4.27e-3 | PASS |
| 梁前 5 阶频率收敛阶 | ≥ 2 | 3.95–3.99 | PASS |
| 板前 5 阶频率收敛阶 | ≥ 1.5 | 3.80–3.99 | PASS |
| 模态误差单调下降 | 严格下降 | 全部单调 | PASS |

---

## 7 审计链与复现

| 环节 | 命令 / 文件 |
|---|---|
| 库模块 | `src/tensorfem/plate_bfs.py`（BFS 单元 + 一致质量/载荷 + 解析预言机） |
| 单元级测试 | `pytest tests/test_plate_bfs.py -q` → 12 passed（含与 `plate_buckling_hermite` 的独立实现比对） |
| 证据级测试 | `pytest tests/test_p19_ladder_artifacts.py -q` → 10 passed（复核 23 个数据点与阈值） |
| L1 复现 | `PYTHONPATH=src python benchmarks/verified/fem_beam_static/run.py` |
| L2 复现 | `PYTHONPATH=src python benchmarks/verified/fem_plate_static/run.py` |
| L3 复现 | `PYTHONPATH=src python benchmarks/verified/fem_modal/run.py` |
| 证据 schema | `benchmarks/verified/_harness.py` → `tensorfem.verified-gates.v1` |
| L3 遗留锚点 | `benchmarks/verified/fem_modal/result.json`（cantilever，`benchmarks/structural_ladder.py` 生成，**未改动**） |

`gates_verified.json` 里每条 gate 都记录了 quantity / reference（闭式解表达式）/ value / threshold /
passed；`passed: true` 表示该锚点全部判据通过，具体失败项会列在 `failed_gates`。