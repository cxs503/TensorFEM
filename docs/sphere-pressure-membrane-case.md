# CASE-SPHERE-PRESSURE — 均压球壳薄膜应力（TensorFEM · 算例池锚点）

> 算例池单（mission 15405，仓 TensorFEM）。判据：`σ = pR/(2t)`，相对误差 **< 3%**。
> 本文件由 `benchmarks/verified/fem_sphere_pressure_membrane/build_report.py` 从
> `result.json` 程序化生成，**所有数字零手抄**；一键复现：
> `python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py`
> （约 13 s，单进程 float64，未用集群）。

## ① 问题与 ref 出处

**物理背景**：闭合薄球壳受均匀内压 `p`，无自由边 ⇒ 无弯曲边界层，中面膜态是
**精确解**（不是渐近解）。这是薄壳理论里唯一同时具备「精确膜态 + 双曲率 + 压力载荷」
的算例，因此是曲面壳单元的压力工况唯一无争议的应力级锚点；船舶总体性能里对应
耐压球壳/球形舵/球形首部结构的膜态强度估算。

**几何/载荷**：

| 量 | 值 |
|---|---|
| 中面半径 `R` | 10.0 m |
| 壁厚 `t` | 0.04 m |
| 弹性模量 `E` | 6.825e+07 Pa |
| 泊松比 `ν` | 0.3 |
| 内压 `p` | 1.0 Pa |
| 无量纲数 `R/t` | 250 |
| 无量纲数 `t/R` | 0.004 |
| 无量纲数 `p/E` | 1.4652e-08 |
| 无量纲数 `pR/(Et)` | 3.6630e-06 |
| 无量纲数 `σ/p`（ref） | 125 |
| 无量纲数 `Etw/(pR²)`（ref） | 0.35 |
| 弯曲特征长度 `√(Rt)` | 0.6325 m |

**ref 出处（分三层，逐层可复核）**

1. **机器推导（主锚，非手抄）**：`run_verification.py::reference()` 用 sympy 解两个
   **互相独立**的静力平衡：
   - 半球切断体的轴向平衡 `p·πr² = σ·2πrt` ⇒ `σ = pr/(2t)`；
   - 双曲率膜的 Laplace/法向平衡 `N_φ/R_φ + N_θ/R_θ = p`，`R_φ = R_θ = R`、
     `N = σt` ⇒ `2σt/R = p` ⇒ `σ = pR/(2t)`；
   两者在 `r = R` 处相减的符号残差 = `0`（严格零）。
   位移由 Hooke 定律 `ε_θ = (σ_θ - νσ_φ)/E` 得到
   `w = pR²(1-ν)/(2Et)`（同样符号推导）。得
   `σ = 125 Pa`、`w = 1.282051e-05 m`。
2. **文献锚（诚实标注）**：同一闭式即 *Roark's Formulas for Stress and Strain* 薄壁球壳
   内压式（第 7 版 Table 13.1 case 1），与本仓 `benchmarks/lame_cylinder_anchor.py`
   已引的圆柱条目（Table 13.5 case 1）同族。
   ⚠️ **原表原页未能核对**：本沙箱**无外网**（`https` 请求超时），且全盘检索
   （`find /root /opt /usr/share/doc -maxdepth 6 -iregex '.*\(roark|timoshenko|handbook\).*\.\(pdf|txt|md|djvu\)'`）**未找到手册 PDF/文本**，故
   **page-level provenance = WITHHELD**，不作「已核对原页」的声明；数值由第 1、3 层承载。
3. **异理论交叉锚**：精确 Lamé 厚壁球解（3-D 弹性，不用任何壳假设）在壁中面取
   `a = R - t/2`、`b = R + t/2`，当 `t/R → 0` 收敛到 `pR/(2t)`；本算例点
   `t/R = 0.004` 处两者差 +4.013e-01 %（见 ⑤）。
   即：ref 数值同时被「独立静力平衡」与「另一种理论」锁住。

## ② 方法与实现

复用（不另起炉灶）：

* 单元：`tensorfem.axisym_shell.AxisymShell` —— 2 节点轴对称壳环单元，每节点 3 自由度
  `(u, w, β)`，Reissner–Mindlin 退化壳，膜/弯用 2 点 Gauss、横剪用**选择性减缩积分**（SRI，
  防剪切锁死）；关键点是 `meridian_radius = R`，使子午线是**真圆弧**（`R_s = R`），
  单元「看到的」就是它声称的球面。
* 算例驱动：`tensorfem.axisym_shell_benchmarks.sphere_pressure`（极点到赤道的
  1/2 球子午链，`facets` 个单元，`u = β = 0` 于两极/赤道切断面 ⇒ 对称边界）。
  输出逐 Gauss 点的 `N_θ`、`N_φ`，应力 `σ = N/t`；`w` 取极点节点值。
* 对照组（同一物理题，只变一个因素）：
  - **几何对照**：同单元、同网格、`meridian_radius=None`（直弦母线）⇒ 隔离「几何表示」
    这一因素；
  - **单元对照**：已合并的平面面元 `projected_shell4_stiffness`
    （`tensorfem.sphere_membrane_benchmark`，立方球全四边形闭合网格）⇒ 隔离「单元族」这一因素。
* 工具链：Python 3 + PyTorch `2.7.1` float64 + sympy
  `1.14.0` + matplotlib `3.10.8`；稠密约化直接解
  （本算例 ≤ 195 DOF，把线性求解器从精度讨论中剔除）。环境：`Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38`。
* 判据**先于**求解写入代码（`TOL = 0.03`、`FACETS`、`THICKNESS_RATIOS` 模块级常量），
  逐步记录 gate，不因结果调整阈值。

## ③ 参数表（无量纲数完整）

见 ① 表：`R/t = 250`、`t/R = 0.004`、`p/E = 1.4652e-08`、
`pR/(Et) = 3.6630e-06`、`σ/p = 125`（膜态与 `E`、`ν` 无关，
唯一载荷参数是 `p/E`；`ν` 只通过 `(1-ν)` 进位移）、`√(Rt)/R = 0.06325`。

## ④ 对照表（ref vs got，逐项打勾叉）

`σ_ref = 125 Pa = pR/(2t)`，`w_ref = 1.282051e-05 m`；误差为逐 Gauss 点
（或极点点值）相对误差的最大值。判据：**< 3%**。

| facets | 单元数 | DOF | `σ_ref` [Pa] | max `σ_θ` [Pa] | `σ_θ` 误差 | `σ_φ` 误差 | 各向异性 | `w` [m] | `w` 误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 9 | 125 | 125.011119 | +1.237e-02 % | +2.429e-02 % | +2.611e-02 % | 1.282706e-05 | +5.105e-02 % |
| 4 | 4 | 15 | 125 | 125.000827 | +1.128e-03 % | +1.955e-03 % | +2.311e-03 % | 1.282141e-05 | +6.960e-03 % |
| 8 | 8 | 27 | 125 | 125.000046 | +7.309e-05 % | +1.204e-04 % | +1.525e-04 % | 1.282059e-05 | +6.262e-04 % |
| 16 | 16 | 51 | 125 | 125.000001 | +4.603e-06 % | +7.001e-06 % | +9.359e-06 % | 1.282052e-05 | +4.812e-05 % |
| 32 | 32 | 99 | 125 | 125.000000 | +2.828e-07 % | +3.643e-07 % | +5.716e-07 % | 1.282051e-05 | +3.313e-06 % |
| 64 | 64 | 195 | 125 | 125.000000 | +2.534e-08 % | +2.535e-08 % | +3.479e-08 % | 1.282051e-05 | +2.129e-07 % |

工程网格（`facets = 8`，`27` DOF）上
`σ_θ` 误差 +7.309e-05 % → **是否 < 3%：
是 ✓**（余量 4 个量级）。逐 Gauss 点最大误差出现在极点单元，属单元内插值残差。

| 对照 | 现象 | 判读 |
|---|---|---|
| 几何对照（直弦母线，同单元同网格） | 见下表，误差**随加密增大** | ✗ 单靠弦几何无法表示 `R_s = R`，模型退化成锥台链，膜态不存在 ⇒ **几何表示是主因素** |
| 单元对照（平面面元 Shell4，24 面元） | `N_mean = 670.8294 N/m` vs exact 5 N/m ⇒ +1.332e+04 %；`w` 误差 +3.793e+02 %；非径向残差 84.0% | ✗ 复现 `docs/sphere-membrane-validation.md` 记录的 13317%（本单 24 面元 +1.332e+04 %），**单元/几何能力边界**，非求解器 bug |

直弦几何对照（同 `AxisymShell`，`meridian_radius=None`）：

| facets | `σ_θ` 误差 | `w` 误差 |
|---:|---:|---:|
| 2 | +1.102e+02 % | +2.626e+01 % |
| 4 | +1.443e+02 % | +4.945e+01 % |
| 8 | +2.079e+02 % | +1.981e+02 % |
| 16 | +4.411e+02 % | +5.454e+02 % |
| 32 | +9.812e+02 % | +1.220e+03 % |
| 64 | +1.473e+03 % | +2.031e+03 % |

## ⑤ 收敛性 / 参数敏感性（≥2 点，共 4 条轴）

**收敛性（网格加密 6 级）**：`facets = 2, 4, 8, 16, 32, 64`，
`σ_θ` 误差 +1.237e-02 % →
+2.534e-08 %；相邻两级观测阶
`[3.456, 3.948, 3.989, 4.025, 3.48]`（均 ≥ 3.4，高于线性插值的名义二阶：误差由极点单元的圆弧几何/积分
主导，属该核的超收敛，不作一般性阶数声明；最细一级回落）。
**如实标注**：观测阶是对"逐 Gauss 点最大值"这一范数而言，不等于位移场的 $L_2$ 阶。
位移误差同级下降（+5.105e-02 % →
+2.129e-07 %）。对比之下直弦对照**发散**
（+1.102e+02 % →
+1.473e+03 %），
所以「加密能救」只对几何正确的路径成立。

**敏感性 A（壁厚 `t/R`：ref 自身适用域）**——这是本单最重要的负面发现：

| `t/R` | `σ_ref = pR/2t` [Pa] | Lamé 中面 `σ_θ` [Pa] | ref vs Lamé | FEM vs ref | `w` 误差 |
|---:|---:|---:|---:|---:|---:|
| 1/1000 (0.001) | 500 | 499.5 | +1.001e-01 % | +4.585e-06 % | +5.043e-05 % |
| 1/500 (0.002) | 250 | 249.5 | +2.003e-01 % | +4.557e-06 % | +4.963e-05 % |
| 1/250 (0.004) | 125 | 124.5 | +4.013e-01 % | +4.603e-06 % | +4.812e-05 % |
| 1/100 (0.01) | 50 | 49.5008 | +1.008e+00 % | +4.577e-06 % | +4.451e-05 % |
| 1/50 (0.02) | 25 | 24.5017 | +2.034e+00 % | +4.525e-06 % | +4.055e-05 % |
| 1/20 (0.05) | 10 | 9.50417 | +5.217e+00 % | +5.990e-06 % | +3.431e-05 % |
| 1/10 (0.1) | 5 | 4.50836 | +1.091e+01 % | +7.406e-06 % | +2.929e-05 % |
| 1/5 (0.2) | 2.5 | 2.01686 | +2.396e+01 % | +8.941e-06 % | +2.433e-05 % |

`|σ_thin - σ_Lamé|/σ_Lamé` 的 log-log 斜率 = **1.0276**，即偏差
≈ `t/R` 一阶。**机理是机器推导出来的**（同 `reference()`）：Lamé 中面 hoop 应力按
`ε = t/R` 展开为

```
eps*p/12 - p/2 + p/(2*eps)
```

首项 `p/(2ε)` 正是薄壳律 `pR/(2t)`，常数项 `-p/2` 给出相对偏差
`(p/2)/(p/(2ε)) = ε = t/R` —— 薄壳给的是壁厚平均应力，Lamé 给的是中面应力，
两者差恒为 O(t/R)，与实测斜率 1.0276 一致。
因此 **ref 自身在 `t/R ≲ 0.03` 内才满足 3%**；本算例点 `t/R = 0.004` 处为
+4.013e-01 %，安全。FEM vs ref 在整个扫描带上都 ≤
8.94e-06%（同一壳理论内部一致）——
**注意：这只证明实现与理论一致，不证明理论相对三维弹性正确**，后者由 ref vs Lamé 那一列承载。

> 预登记预测 **G8b 已被证伪**：「ref 在 `t/R ≤ 0.05` 内保持 3%」不成立（实测带止于
> `t/R ≈ 0.02`）。该条留在 `gates_verified.json` 的
> `falsified_predictions` 里，不删除、不改判据。

**敏感性 B（泊松比 `ν`，8 单元）**：

| `ν` | `σ_θ/σ_ref` | `σ_θ` 误差 | `w/w_ref` | `w` 误差 |
|---:|---:|---:|---:|---:|
| 0.0 | 1.000000204 | +6.912e-05 % | 1.000003 | +3.482e-04 % |
| 0.2 | 1.000000300 | +7.086e-05 % | 1.000005 | +5.123e-04 % |
| 0.3 | 1.000000365 | +7.309e-05 % | 1.000006 | +6.262e-04 % |
| 0.4 | 1.000000447 | +7.798e-05 % | 1.000008 | +7.742e-04 % |
| 0.49 | 1.000000543 | +8.572e-05 % | 1.000010 | +9.517e-04 % |

`σ` 对 `ν` 不变（膜态断言，误差全在 8.57e-07 量级），
`w` 严格按 `(1-ν)` 缩放 ⇒ 位移比值恒 ≈ 1。

**敏感性 C（载荷 `p`，无量纲 `p/E`）**：

| `p` [Pa] | `p/E` | `σ/p` | `σ(p)/σ(p=1)` | 误差 | `w/p` [m/Pa] |
|---:|---:|---:|---:|---:|---:|
| 1e+00 | 1.4652e-08 | 125.000046 | 1.000000000 | +7.309e-05 % | 1.282059e-05 |
| 1e+03 | 1.4652e-05 | 125.000046 | 1000.000000000 | +7.309e-05 % | 1.282059e-05 |
| 1e+06 | 1.4652e-02 | 125.000046 | 1000000.000000000 | +7.309e-05 % | 1.282059e-05 |

`σ/p`、`w/p` 在 6 个量级载荷跨度上逐位相同 ⇒ 线性、无载荷相关误差（薄壳小变形假设的内部自洽检查）。

## ⑥ 失败模式与工件

**已知失败模式（都被本单固化为对照，不再重复踩）**

1. **平面面元 + 曲面 + 压力**：内压在一个平面面元内部没有面内分量，只能靠折角以弯曲平衡；
   薄壳弯曲长度 `√(Rt) = 0.632 m` ⇒ 需 ~`3142` 量级面元才有希望，
   实测 24 面元 +1.332e+04 %。**该路径不可用于压力球壳**。
2. **直弦母线**：`R_s = ∞` 丢掉子午曲率项，加密发散（
   +2.079e+02 % @8 单元 →
   +1.473e+03 % @64 单元）。凡「曲面壳 + 内压」
   必须传 `meridian_radius`。
3. **把 shell-vs-shell 的一致性当验证**：FEM 与 `pR/2t` 吻合到 1e-7 只说明单元实现了同一壳理论；
   超出 `t/R ≳ 0.03` 后 ref 与三维弹性分道扬镳（⑤-A）。本单用 Lamé 列显式分开这两件事。
4. **极点奇异/对称约束**：极点节点被 `u = β = 0` 固定，`w` 取极点值；`σ` 的逐点最大值受极点单元
   内插值影响，因此判据用「逐 Gauss 点最大相对误差」而非单点抽样。

**工件（真实文件）**

* `benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py` — 驱动器（ref 推导 + 收敛 + 3 条敏感性 + 2 个对照 + 3 图 + gates）
* `benchmarks/verified/fem_sphere_pressure_membrane/result.json` — 全部数字（含 env、provenance、repro_cmd）
* `benchmarks/verified/fem_sphere_pressure_membrane/gates_verified.json` — 11 条 gate 判定 + 被证伪预测
* `benchmarks/verified/fem_sphere_pressure_membrane/figures/{convergence.png,field_meridian.png,sensitivity.png}`
* `benchmarks/verified/fem_sphere_pressure_membrane/build_report.py` — 本报告生成器（零手抄）
* `tests/test_sphere_pressure_membrane_case.py` — 锁定本算例的 pytest 门

**Gate 汇总（程序判读）**

| gate | 值 | 判据 | 结果 |
|---|---|---|---|
| `G10_flat_facet_shell4_control` | 1.3317e+02 | > 3% (recorded 13317% at 24 facets) | ✓ |
| `G1_sigma_theta_vs_pr_2t` | 7.3091e-07 | rel err < 3% at facets=8 | ✓ |
| `G2_sigma_phi_vs_pr_2t` | 1.2039e-06 | rel err < 3% at facets=8 | ✓ |
| `G3_isotropy` | 1.5248e-06 | < 3% at facets=8 | ✓ |
| `G4_radial_displacement` | 6.2617e-06 | < 3% at facets=8 | ✓ |
| `G5_monotone_refinement` | [1.1280e-05, 7.3091e-07, 2.8282e-09] | e(4) > e(8) > e(32) | ✓ |
| `G6_observed_order` | [3.4560e+00, 3.9480e+00, 3.9890e+00, 4.0250e+00, 3.4800e+00] | min observed order >= 1.5 | ✓ |
| `G7_poisson_invariance` | 8.5723e-07 | < 3% for every nu in 0..0.49 | ✓ |
| `G8a_reference_valid_at_case_point` | 4.0134e-03 | /sigma_thin - sigma_Lame//sigma_Lame < 3% at t/R of the case | ✓ |
| `G8b_reference_band_reaches_t_over_R_0.05` | [1.0000e-03, 2.0000e-02] | predicted: band reaches t/R = 0.05 | ✗ |
| `G9_chord_control_fails` | 2.0787e+00 | > 3% (control must fail: geometry representation matters) | ✓ |

* gating 全通过：**True**；含预测的整体判定：
  False；被证伪预测：
  ['G8b_reference_band_reaches_t_over_R_0.05']。

## 生产模块、入口与集成调用者（m15703 返修轮补齐）

本算例的逻辑已从一次性脚本迁入**生产模块** `src/tensorfem/sphere_pressure_case.py`
（唯一真源；复用既有 `AxisymShell` + `axisym_shell_benchmarks.sphere_pressure`，未新增单元）：

| 层 | 位置 | 说明 |
|---|---|---|
| 实现 | `src/tensorfem/sphere_pressure_case.py` | `membrane_reference` / `axisym_case` / `quick_check` / `convergence` / `thickness_sensitivity` / `poisson_sensitivity` / `pressure_scaling` / `gates` / `figures` / `run_case` / `cli` |
| 入口 | `tensorfem-sphere-pressure`（`pyproject.toml [project.scripts]` → `sphere_pressure_case:cli`） | `--out DIR`（默认 `artifacts/CASE-SPHERE-PRESSURE`）、`--no-figures`、`--quiet`；`python3 -m tensorfem.sphere_pressure_case` 等价 |
| 集成调用者 | `src/tensorfem/benchmark_registry.py::run_registered_benchmarks()` | 注册 3 条 `_evidence`：`shell.sphere.pressure_sigma_theta/-_sigma_phi/-_radial_displacement`，随 `tensorfem verify` 一起跑 |
| | `src/tensorfem/cli.py` capabilities 清单 | 声明该算例入口 |
| | `benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py` | 薄壳 runner（只做路径转发 + 复用导出，不再持有逻辑） |
| 测试 | `tests/test_sphere_pressure_case_module.py` | 引用推导=0 残差、生产网格 <3%、chord 对照必须失败、注册/入口/runner 接线、CLI 端到端写工件 |
| 工件 | `artifacts/CASE-SPHERE-PRESSURE/`（result.json、gates_verified.json、figures/*.png、run.log） | 由 CLI 真实运行产生（含环境、repro_cmd、逐门判据） |

复现（一键）：

```
tensorfem-sphere-pressure --out artifacts/CASE-SPHERE-PRESSURE
# 或（未安装时）
PYTHONPATH=src python3 -m tensorfem.sphere_pressure_case --out artifacts/CASE-SPHERE-PRESSURE
```

**生态摩擦（如实记）**

* `knowledge_base` / `negative_knowledge_ledger` / `oss_solutions` 三张表在任务书里要求先查，
  但本会话的 DB 角色对它们 **permission denied**（`psql` 默认 root 角色）；改用
  `psql -U postgres` 才读到（knowledge_base 有 TensorFEM 能力条目 12+ 条，无球壳压力条目；
  oss_solutions 同域仅 TensorLBM 球体条目）。建议平台把只读查询暴露成 API，别让每个工人各自试权限。
* 无外网、盘上无手册 ⇒ 「引用原表原页」在本沙箱**结构上不可达**。本单按证据纪律处理为
  机器推导 + 异理论交叉 + 文献锚标注 WITHHELD；建议平台给算例池预置一份合法可引的手册摘录
  或公认 ref 数值库，否则每个算例都要重复这一轮免责声明。
* 2 节点壳环单元的正确用法（必须传 `meridian_radius`）只在源码 docstring 里写明，没有
  「误用会发散」的显式门；本单的 chord 对照可考虑提升为库级回归测试。

## 复现

```
python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py   # 13 s
python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py
pytest tests/test_sphere_pressure_membrane_case.py -q
```

环境：Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38，Python 3.11.6，torch
2.7.1，sympy 1.14.0，numpy 2.4.6，
dtype float64，单进程，未用集群（运行 13 s < 10 min 阈值）。

## 参考

* Roark's *Formulas for Stress and Strain*, 7th ed., Table 13.1 case 1
  （薄壁球壳内压；**原页未核对，见 ①ref 出处第 2 层**）。
* Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., McGraw-Hill, 1959
  （回转壳膜理论；本仓已在多处引用该版本）。
* 仓内证据：`docs/sphere-membrane-validation.md`（平面面元 13317% 失败记录）、
  `docs/benchmark-qualification-status.md`、`benchmarks/verified/fem_axisym_shell/`。
