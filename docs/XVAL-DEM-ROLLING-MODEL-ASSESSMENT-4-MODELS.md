# XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS — 四种滚动阻力模型的可机器裁剪解析基准

锚点：`XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS`（literature/Ai-2011 rolling-model-assessment-4-models，
doi:10.1016/j.powtec.2010.11.003）。验收线：对标误差 <15%（owner 精度标准），本文按更严的 3% 自检。

代码：`src/tensorfem/rolling_resistance_models.py`（基准面） ·
`examples/rolling_model_assessment.py`（评估harness） ·
`tests/test_rolling_resistance_4models_xval.py`（22 项机器门） ·
产物：`artifacts/rolling_model_assessment.json`、`artifacts/rolling_model_assessment.png`

---

## ① 案例介绍（物理背景 / 几何 / 参数）

单颗粒刚性滚子（球，`k=I/(mR²)=2/5`）在倾角 θ 的平面上**无滑滚动**，接触点受滚动阻力矩 τ。
Ai et al. (2011) 的核心问题是：DEM 里四种滚动阻力模型族在**同一 μ_r** 下并不等价，
宏观行为差异全部来自"有没有静态支路"。本单把这四种族放在同一个单颗粒基准上，逐条对表解析解。

| 参数 | 值 |
|---|---|
| 滚动阻力系数 μ_r | 0.30 |
| 滚子半径 R | 0.01 m |
| 质量 m | 1.0 kg |
| 转动惯量因子 k | 0.4（球） |
| 重力 g | 9.81 m/s² |
| 临界角 θ_c = atan(μ_r) | 16.699244° |
| 超临界工况 θ_super | 20° (tan=0.364 > μ_r) |
| 亚临界工况 θ_sub | 5° (tan=0.0875 < μ_r) |
| 滚动弹簧屈服应变 φ_y | 1e-3 rad（k_r = M_cap/φ_y = 27.66 N·m/rad） |
| 阻尼比 ζ（EPSD2 滚动阻尼器） | 1.0（临界）；纯族对照取 0 |

四族共同上限：`M_cap = μ_r F_n R`（θ=20° 时 0.0276552 N·m）。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **运动方程**：无滑滚动的接触点平衡 `m(1+k) dv/dt = m g sinθ + τ/R`，`ω = v/R`，
   滚动位移 `θ_rolled = x/R`。四族只差在 τ 的构成：
   - `cdt`（恒向力矩）：`τ = −M_cap·sign(ω)`，无静态支路，方向由上一个非零转速样本保持；
   - `viscous`（黏性）：`τ = −(M_cap/ω_ref)·ω`，严格耗散但在 ω=0 时力矩为 0；
   - `hysteretic`（双刚度滞回弹簧）：卸载支路刚度 `k_r/(1−h)`，`|φ| ≤ φ_y` 弹性截断——静态支路来自截断；
   - `epsd`（弹塑性弹簧+阻尼器，Iwashita–Oda / Jiang）：加载纯弹性，`|φ|` 越限即塑性流动，附 `c_r = 2ζ√(k_r I)` 阻尼器。
2. **积分器**：记忆无关族用精确步（`cdt` 常加速度；`viscous` 线性 ODE 的指数解），弹簧族用半隐式步。
   故线性解析基准被复现到 ~1e-13，测试里的 1e-9/1e-6 容差是真门限而非摆设。
3. **对照组**：每个判据都写成闭式（`saturated_acceleration` / `cdt_stopping_distance` /
   `viscous_terminal_velocity` / `viscous_time_constant` / `static_windup_angle` /
   `damped_terminal_velocity` / `damped_time_constant`），模拟量与之比对，误差表由 harness 程序化生成（零手抄）。
4. **工具链**：`PYTHONPATH=src python examples/rolling_model_assessment.py --out artifacts`
   （28 s，单核）→ JSON + PNG；`pytest tests/test_rolling_resistance_4models_xval.py`（25.6 s，22 passed）；
   `ruff check` 全绿。

## ③ 计算结果（数字表，程序化注入）

### 3.1 解析基准对表（本单的"成绩单"）

| 判据 | 族 | 模拟值 | 解析参考 | 相对误差 |
|---|---|---|---|---|
| S1 平停距 `(1+k)v0²/(2μ_r g)`，v0=1 m/s | cdt | 0.237852534 m | 0.237852531 m | 1.34e-8 |
| S1 平停耗功 vs 动能 0.7 J | cdt | 0.7000155 J | 0.7 J | 2.21e-5 |
| S1 平停距 vs cdt 基准 | hysteretic / epsd | 0.2378478 / 0.2378440 m | 0.2378525 m | 1.98e-5 / 3.60e-5 |
| S1 平停末速（指数衰减闭式） | viscous | 2.25189e-46 m/s | 2.25189e-46 m/s | 1.09e-13 |
| S2 静态保持位移 `R·φ_y/2` | epsd | 5.000000e-6 m | 5.000000e-6 m | 2.34e-14 |
| S2 静态保持位移 | hysteretic | 5.209372e-6 m | 5.000000e-6 m | 4.19e-2（族差异，见⑤） |
| S2 蠕变速度 `Rω_ref·τ_app/M_cap` | viscous | 4.9999999e-3 m/s | 5.0e-3 m/s | ~4e-14 |
| S3 饱和加速度 `g(sinθ−μ_r cosθ)/(1+k)` | cdt | 0.4212158736083 m/s² | 0.4212158736085 m/s² | 3.80e-13 |
| S3 同上 | hysteretic / epsd | 0.4212158736083 m/s² | 0.4212158736085 m/s² | 3.80e-13 |
| S3 尾窗平均力矩 vs −M_cap | 三族 | −0.0276551538297293 N·m | −0.0276551538297293 N·m | 3.76e-16 |
| S3b 阻尼族终端速度 `(mgH R−M_cap)R/c_r` | hysteretic / epsd | 8.86512236190348e-4 m/s | 8.86512236190337e-4 m/s | 1.22e-14 |
| S4 亚临界弹性缠绕位移 `Rφ_y tanθ/μ_r` | epsd | 2.9162888e-6 m | 2.9162888e-6 m | 1.36e-11 |
| S4 同上 | hysteretic | 3.0379204e-6 m | 2.9162888e-6 m | 4.17e-2（族差异） |
| S4 亚临界末速（指数闭式） | viscous | 2.916288784e-3 m/s | 2.916288784e-3 m/s | 1.78e-14 |
| S5 临界角（阻尼族漂移二分） | hysteretic / epsd | 16.703001° | 16.699244° | 2.25e-4（0.0038°） |
| S6 `h=0` 退化为 epsd | hysteretic | max\|Δv\| = 0.0 | 0 | 0 |

### 3.2 族间行为矩阵（这是 Ai-2011 评估的实质结论）

| 族 | 静态支路（0.5 M_cap 恒定力矩，1 s） | 饱和稳态 | 亚临界 5° | 单位位移耗散 |
|---|---|---|---|---|
| cdt | **无**：极限环抖动 rms 速度 1.287e-5 m/s（O(dt)） | a=0.4212 m/s² | 不静止，max\|v\|=5.41e-5 m/s | μ_r mg |
| viscous | **无**：持续蠕变 4.976e-3 m（1 s） | v→v∞=0.012132 m/s，a→8.2e-8 | 永不停，末速 2.916e-3 m/s | 速度相关 |
| hysteretic | 有：位移 5.209e-6 m（+4.19% vs epsd） | a=0.4212 m/s² | 缠绕 3.038e-6 m 后静止 | μ_r mg + h 滞回 |
| epsd | 有：位移 5.000e-6 m（精确） | a=0.4212 m/s² | 缠绕 2.916e-6 m 后静止 | μ_r mg + 塑性 |

S2b（cdt 抖动分辨率依赖）：dt=1e-5 → rms 1.28722998e-5 m/s；dt=1e-6 → 1.28729220e-6 m/s；
比值 **9.9995**（位移比值 10.000）。Δt 减 10 倍抖动减 10 倍 ⇒ 静态接触状态**不收敛**，是离散伪像。

S7（亚屈服驱动的滞回环耗散，ζ=0，驱动幅值 0.4 M_cap、20 Hz、5 周期）：
h=0 → 4.52e-8 J（保守）；h=0.1 → 6.77e-6 J；h=0.25 → 1.53e-5 J；h=0.5 → 3.33e-5 J（随 h 单调）。

## ④ 结果图

`artifacts/rolling_model_assessment.png`（matplotlib，`make_figure()` 程序化生成）：
- 左：S4 亚临界倾角 5° 的滚动位移轨迹 —— cdt/viscous 持续漂移，hysteretic/epsd 缠绕到 `Rφ_eq` 后静止；
- 中：S2 静态保持位移柱状图（对数轴）—— 四族跨 3 个数量级（5e-6 ~ 5e-3 m）；
- 右：S7 亚屈服滞回环（τ vs θ_rolled，h=0/0.25/0.5）—— h=0 退化为单线（保守），h 越大环面积越大。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **阻尼器会吃掉滚动阻力判据**：EPSD2 的滚动阻尼器 `c_r=2ζ√(k_r I)` 在 `|ω| ≫ ω_ref` 时力矩
   远大于 `M_cap`。同一 μ_r=0.3、v0=1 m/s 的平停距，ζ=0 时 0.23785 m，ζ=1 时 **0.0019776 m**
   （−99.17%）。所以"弹簧族 = cdt"只在 ζ=0 的纯族对照下成立；带阻尼器的标定必须把 ω_ref、ζ 一起报。
2. **νiscous 族的 ω_ref 是隐藏参数**：ω_ref=1 rad/s 时平停距 0.004757 m（比饱和族短 50 倍）。
   离开 ω_ref 谈 viscous 的 μ_r 没有意义。
3. **cdt 在静平衡处方向盲**：无转速历史时力矩方向只能由外载力矩推定，这正是它产生伪蠕变/极限环的根源；
   本实现如实复现（并在 `simulate` 里显式记录该选择），不掩盖。
4. **hysteretic 族 +4.17%（S2/S4）不是积分误差**：同一积分器下 epsd 误差 1e-11，故 4.17% 是双刚度卸载支路
   在初始缠绕段的真实刚度效应。它<15% 验收线，但不<3%，本文如实单列，不并入"对表误差"。
5. **Ai-2011 原文公式未逐位复现**（手头无原式）：本文复现的是**族级本构**与解析推论，并对
   `hysteretic` 的刚度调度做了显式声明；与论文图件逐点比对的定量一致性 **WITHHELD**（仓内无数字化数据集，
   零手抄纪律下不做图读数比对）。
6. **生产实现归属**：DEM 生产法在 `tensordem/rolling_resistance.py`（cdt / viscous / epsd 三族，569 行）；
   本模块是 TensorFEM 侧的**参考级验证面**（与 `broken_ice_floe_field`、`cattaneo_hertz` 同一定位），
   并补上生产侧缺失的第四族 `hysteretic`。建议后续派单在 TensorDEM 侧补齐 `hysteretic` 并直接用本基准对表。
7. 单颗粒、无滑、单自由度（无自旋/钻磨摩擦、无滑移耦合）。多颗粒堆的宏观量（休止角、麦粒堆）
   不在本单范围，需 DEM 装配级验证。

## ⑥ 误差判定（逐项，显式回答"是否 <3%"）

| 关键量 | 基准类型 | 误差 | <3%？ |
|---|---|---|---|
| cdt 平停距 | 闭式解析 | 1.34e-8 | **是** |
| cdt 平停耗功 = 动能 | 闭式解析 | 2.21e-5 | **是** |
| viscous 平停末速 / 终端速度 / 亚临界末速 | 闭式解析 | ≤1.09e-13 | **是** |
| 三族饱和加速度 / 平均力矩 | 闭式解析 | ≤3.80e-13 | **是** |
| 阻尼族终端速度 | 闭式解析 | 1.22e-14 | **是** |
| epsd 静态保持位移 / 亚临界缠绕位移 | 闭式解析 | ≤1.36e-11 | **是** |
| 临界角 θ_c = atan(μ_r) | 二分模拟 vs 解析 | 0.0038° / 2.25e-4 | **是** |
| `h=0` 退化到 epsd | 族间一致 | 0 | **是** |
| hysteretic 静态保持/缠绕位移（族差异，非误差） | vs epsd 解析 | 4.17% | 否（单列，族本构所致） |
| Ai-2011 图件定量一致性 | 文献图读数 | — | **WITHHELD**（无数字化数据） |

结论：所有"模拟 vs 解析闭式"的判据误差均 **≪3%**（最大 2.25e-4，即 0.0225%），远优于 15% 验收线；
唯一的 >3% 项是两个弹簧族之间的**真实本构差异**（4.17%），已单列声明，不计入对表误差；
与论文图件的定量一致性如实标注 WITHHELD。

## 复现

```bash
PYTHONPATH=src python examples/rolling_model_assessment.py --out artifacts   # 28 s, 单核
PYTHONPATH=src python -m pytest tests/test_rolling_resistance_4models_xval.py -q   # 22 passed, 26 s
ruff check src/tensorfem/rolling_resistance_models.py tests/test_rolling_resistance_4models_xval.py examples/rolling_model_assessment.py
```

参考：Ai, Chen, Rotter, Ooi (2011) Powder Technology 206:269-282；
Iwashita & Oda (1998) J. Eng. Mech. 124:285-292；Jiang, Yu & Harris (2005) Comput. Geotech. 32:340-357。