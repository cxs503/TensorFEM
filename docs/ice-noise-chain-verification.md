# 冰载荷 → 湿船体梁 → 水下噪声链：验证报告（ice_noise_chain）

本报告对已合并但**无任何测试**的 `src/tensorfem/ice_noise_chain.py`（808 行，
campaign node IN13-W2-N06）做独立验证。模块 docstring 声明"每一级都由独立解析锚
钉住、任何一级回归都会被测试捕获"——合并时该声明没有任何测试支撑。本次补齐
`tests/test_ice_noise_chain.py`（25 passed + 1 strict xfail），并在验证过程中发现并
修复 4 个必然崩溃的缺陷。

## 1. 案例介绍（物理背景与参数）

链条三阶段：

| 阶段 | 物理 | 输出 |
|---|---|---|
| S1 | DEM 冰接触载荷（半正弦脉冲序列）→ FFT 幅值谱/PSD | 载荷谱 |
| S2 | 湿船体梁模态叠加（2-D 势流切片附连水 + 模态阻尼）| 中跨位移/加速度时程 |
| S3 | 水下辐射噪声（障板矩形板 Rayleigh/BEM 远场，1 m 参考） | 声压级 dB re 1 µPa |

统一参数（SI）：

| 量 | 值 |
|---|---|
| 船体梁长 L / 单元数 / EI / 结构线密度 ρa | 20 m / 10–40 / 1.0e10 N·m² / 5000 kg/m |
| 梁截面（圆形）半径 | 0.5 m |
| 附连水（π ρ r²，ρ=1025 kg/m³） | 805.033 kg/m |
| 阻尼比 ζ（时程比对） | 0.05 |
| 辐射板 a × b / 振动模态 | 1.0 m × 0.6 m / (1,1) |
| 水 ρ / c | 1025 kg/m³ / 1500 m/s |
| 冰冲击（链驱动）| (0.05 s, 0.02 s, 5e5 N) + (0.25 s, 0.03 s, 2e5 N) |
| 采样 dt / t_end（链） | 2e-3 s / 0.4 s |

## 2. 怎么做的（方法/流程）

1. **缺口扫描**：`src/tensorfem/*.py` 与 `tests/test_*.py` 名称匹配 + 全文引用检索，
   发现 `ice_noise_chain`（808 行、被 campaign 判定"stage-verified"）在 `tests/` 与
   `docs/` 中**零引用**。以此作为验证欠账靶子（而非新功能）。
2. **先跑再判**：直接调用三个 stage 的入口函数，暴露 4 个异常（见 §5）。
3. **最小修复**（只改该模块 5 处，不改语义）：
   - `baffled_monopole_pressure`：`math.exp(complex)` → `cmath.exp`；
   - `girder_modal_superposition` / `girder_newmark_reference`：`solve_modes` 接收的是
     **自由度缩减后**的 K/M，返回模态按缩减编号索引，原代码却用全局 DOF 号索引
     （`phi[free, :]` → IndexError）；改为 `phi_f = eig.modes` 并把模态散射回全局
     `phi[free, :] = phi_f`；
   - 模态坐标/时程缓冲 dtype 由默认 float32 改为与梁矩阵一致的 float64（原代码必抛
     `RuntimeError: expected scalar type Double but found Float`）；
   - Newmark 参考解：`K_eff = M/(βΔt²) + γ/(βΔt)C + K` 是位移形式的算子，原代码直接
     用它解**加速度**形式，响应被缩小 `1/(βΔt²)`=4e6 倍；补上 `/ (βΔt²)` 重标定。
4. **锚点测试**（`tests/test_ice_noise_chain.py`）：解析/闭式/异构数值方法三路交叉，
   `pytest` 26 s 内跑完。
5. **程序化出数**：`scripts/verify_ice_noise_chain.py` 重算全部锚点，写
   `artifacts/ice_noise_chain/numbers.json` 与三张图，并自动生成
   [`ice-noise-chain-anchors.md`](ice-noise-chain-anchors.md)（报告中的数字全部由脚本
   注入，无手抄）。

## 3. 计算结果（数字表）

完整表见 [`ice-noise-chain-anchors.md`](ice-noise-chain-anchors.md)（脚本生成）。摘要：

| 锚点 | 量 | 计算 | 参照 | 相对误差 |
|---|---|---:|---:|---:|
| A1 | 基频对齐三音调幅值（20.5/37/91 Hz） | 120.0000/45.0000/18.0000 N | 120/45/18 N | +0.000% |
| A1 | 非整 bin 音调 213.37 Hz | 17.2296 N | 18.0 N | −4.280% |
| A2 | 湿梁一阶 f1，10 单元 | 5.154190479 Hz | 5.154155712 Hz | +6.75e−4% |
| A2 | 同上，20 单元 | 5.154157890 Hz | 5.154155712 Hz | +4.23e−5% |
| A2 | 同上，40 单元 | 5.154155844 Hz | 5.154155712 Hz | +2.56e−6% |
| A2 | 阶跃载荷峰值 / 2×静挠度（模态叠加） | 0.9977 | 1.0 | +0.228% |
| A2 | 同上（Newmark 参考） | 0.9978 | 1.0 | +0.218% |
| A2 | 中跨 \|u\| RMS 模态 / Newmark（0.7 f1，ζ=0.05） | 2.0792e−4 m | 2.4340e−4 m | **−14.58%** |
| A2 | 中跨 \|a\| RMS 模态 / Newmark | 1.0734e−1 m/s² | 1.8096e−1 m/s² | **−40.68%** |
| A3 | σ(5 Hz), ka_eff=0.016 | 1.376053e−5 | 1.376065e−5 | −0.0009% |
| A3 | σ(20 Hz), ka_eff=0.065 | 2.201373e−4 | 2.201705e−4 | −0.0151% |
| A3 | σ(80 Hz), ka_eff=0.260 | 3.514247e−3 | 3.522728e−3 | −0.2407% |
| A3 | σ(200 Hz), ka_eff=0.649 | 2.168825e−2 | 2.201705e−2 | −1.4934% |
| A3 | 远场 vs 单极子，r=1 m | — | — | −1.3862 dB |
| A3 | 远场 vs 单极子，r=10 m | — | — | −0.0175 dB |
| A3 | 远场 vs 单极子，r=50 m | — | — | −0.0007 dB |
| 链 | 1 m 处声压级 | 168.325 dB re 1 µPa | — | 报告值 |
| 链 | 载荷放大 s=0.5/2/4 的级差 vs 20log10 s | — | — | 0.0000 dB |

参照式：f1=(π/L)²√(EI/(ρa+m_a))/2π；静挠度 PL³/(48EI)；
σ = k²Q²/(π a b V0²)，Q=4V0ab/π²；p = j ρ ω Q e^{−jkr}/(2πr)。

## 4. 结果图

| 图 | 内容 |
|---|---|
| [`artifacts/ice_noise_chain/girder_modal_anchor.png`](../artifacts/ice_noise_chain/girder_modal_anchor.png) | 左：中跨加速度时程（模态叠加 vs Newmark 参考，0.7 f1）；右：f1 相对解析解的误差 vs 网格（10/20/40 单元，单调收敛） |
| [`artifacts/ice_noise_chain/radiation_sigma_anchor.png`](../artifacts/ice_noise_chain/radiation_sigma_anchor.png) | 左：σ(Rayleigh 积分) 与单极子极限 k²Q²/(πabV0²) 的 log-log 对照；右：σ 相对误差 vs 频率（ka_eff<0.4 时 <0.1%） |
| [`artifacts/ice_noise_chain/chain_noise_levels.png`](../artifacts/ice_noise_chain/chain_noise_levels.png) | 左：S2 输出的中跨速度谱；右：链末端声压级 vs 冰载荷放大系数（与解析 20log10 s 重合） |

## 5. 注意点（坑与局限）

1. **模块合并时完全未被执行**：4 个缺陷中任意一个都会让入口函数直接抛异常
   （`TypeError`/`IndexError`/`RuntimeError`），即 `run_ice_noise_chain` 与
   `girder_*` 自合并起从未跑通。教训：docstring 里的"stage-verified"声明若无测试即
   为无证据声明。
2. **A2 的 <3% 加速度级一致度未达标（honest FAIL）**：0.7 f1 驱动、4 模态、
   ζ=0.05 下加速度 RMS 差 −40.68%，位移 −14.58%。原因不是积分器（阶跃载荷锚点两者
   差 0.01%，无阻尼时首峰一致到 1e−4），而是 `girder_newmark_reference` 的阻尼矩阵
   `C = Φ diag(2ζω) Φᵀ` **只覆盖被保留的模态**：截断外的模态在参考解里无阻尼、持续
   振铃，而模态叠加把这些模态整个丢弃 → 两者不是同一物理系统。要修需把阻尼改为
   Rayleigh 形式或对全自由度基构造，属于模块自身设计修改，本单未做；该 FAIL 已用
   `@pytest.mark.xfail(strict=True)` 锁定，一旦以后达标会立刻报警。
3. **A1 的 <1% 只对整 bin 音调成立**：非整 bin 音调（213.37 Hz，bin 间距 0.5 Hz）
   回收误差 −4.28%（Hanning 泄漏 + 未做幅值校正）。链条驱动用整 bin 无碍，但用于
   实测 DEM 数据时需加插值校正或更长窗。
4. **A3 的远场公式有几何近场项**：r=1 m 与板尺寸（1.0×0.6 m）同量级，单极子极限本身
   是 ka→0 **且** r→∞ 的极限，r=1 m 处恒定 −1.39 dB 偏差来自 1/R 在板面积分上的
   平均（几何因子），随 r 收敛（10 m −0.018 dB，50 m −0.0007 dB）；r=0.5 m 已越出
   远场声明（−3.75 dB）。报告与测试都按"r≥1 m 且 <3 dB"设门。
5. **适用范围**：小应变线弹性梁 + 势流附连水 + 障板 Rayleigh 辐射；不覆盖冰-结构
   接触非线性、水弹性耦合、非障板（自由场）辐射、结构阻尼非模态分布。
6. `single_mode_steady_state` 要求 `f0_over_m ≥ 0`，而模态振型符号任意（本算例
   φ_mid=−0.00415），直接传 `F0·φ_mid` 会抛 ValueError——用 `abs()` 或约定振型符号。

## 6. 误差判定（是否 <3%）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---:|---|
| 湿梁 f1（10/20/40 单元） | Euler–Bernoulli 闭式 | 6.75e−4% / 4.23e−5% / 2.56e−6% | **是** |
| 阶跃载荷首峰 / 2×静挠度 | PL³/48EI | 0.228% / 0.218%（两法） | **是** |
| 单模态稳态幅值 | 标量阻尼振子 RK4 数值解 | <1%（测试门，3 组频率） | **是** |
| σ（ka_eff ≤ 0.4） | 单极子极限 k²Q²/(πabV0²) | ≤0.095%（5–50 Hz） | **是** |
| σ（ka_eff ≈ 0.65, 200 Hz） | 同上 | 1.49% | **是**（但已偏离 ka≪1 前提） |
| 远场声压（r ≥ 10 m） | 障板单极子 | 0.018 dB（0.2%） | **是** |
| 远场声压（r = 1 m） | 同上 | 1.39 dB（14.9%） | 不适用（近场几何项，非算法误差） |
| 模态叠加 vs Newmark 加速度 RMS | 直接积分参考 | **40.68%** | **否**（honest FAIL，见 §5.2） |
| 链输出对载荷的线性度 | 20log10 s | 0.0000 dB | **是** |

结论：本模块**三个阶段各自的解析锚点全部通过且远优于 3%**（频率 6.8e−4% 量级、
σ ≤0.1%、线性度机器精度）；**模块 docstring 自述的"模态叠加 vs 直接 Newmark 加速度
<3%"未通过**，原因是参考解的模态阻尼截断而非积分误差，已作为严格 xfail 与本文
§5.2 如实登记，不得据此宣称链条的时程一致性已认证。