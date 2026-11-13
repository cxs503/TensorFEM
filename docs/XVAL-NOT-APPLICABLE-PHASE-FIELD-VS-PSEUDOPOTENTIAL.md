# XVAL-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT — 相场 AT2 与伪势 LBM 不可横向比较的
机器裁决，及相场一侧的解析替代验证

锚点：`XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT`
（外部做法：`AminZar69/CudaPhaseField` 的 not-applicable-phase-field-not-pseudopotential 声明）。
验收线：对标误差 <15%（owner 精度标准）；本文对替代验证路径自设 3% 更严门槛。
代码：`src/tensorfem/xval_applicability.py`（判据机 + 解析锚点） ·
`examples/xval_not_applicable_phase_field.py`（证据 harness） ·
`tests/test_xval_applicability_phase_field.py`（23 项机器门） ·
`scripts/gen_xval_not_applicable_doc.py`（本文数字由该脚本从 JSON 程序化注入，零手抄）。
产物：`artifacts/xval_not_applicable_phase_field.json`、`artifacts/xval_not_applicable_phase_field.png`。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

**被筛查的配对**：本仓已实现的 AT2 变分相场损伤（`tensorfem.phase_field_fracture`）
vs 单组分多相伪势 LBM（Shan–Chen 类，外部 `CudaPhaseField` 仓的背景方法）。
外部仓给出的结论是"相场不是伪势"——本单的任务不是照抄这句话，而是把它变成
**可审计的机器裁决**：给出五条阻断判据、逐条证据，以及"既然不能横向比较，
那这一侧该拿什么来验证"的替代路径。

**相场侧的物理背景**：正规化变分断裂泛函（Miehe AT2）

```
E(u, d) = ∫ g(d) ψ⁺(ε) + ψ⁻(ε) dV + ∫ Gc [ d²/(2l) + (l/2)|∇d|² ] dV ,  g(d) = (1-d)² + k
```

不可逆性由历史场 `H(x,t)=max_{s≤t} ψ⁺(ε(x,s))` 承担。关键结构事实：
**界面宽度 l 是泛函里的显式自由参数**，而不是某种耦合常数的涌现结果。

**验证算例（替代路径 A1/A2）**：一维、无应变的 AT2 平衡。取 `ψ⁺=0`（历史场恒零）后
强形式退化为 `d = l² d''`，边界条件 `d(0)=1`、`d'(L)=0` 的精确解为
`d(x)=cosh((L-x)/l)/cosh(L/l)`（`L ≫ l` 时即常见的 `exp(-x/l)`）。几何取
`[0,L]×[0,h]` 一条 Q4 带（厚度方向一个单元，`h=L/n`），全部位移自由度钉零
（应变能→0，故历史场与驱动力恒为零），仅 `x=0` 处钉 `d=1`。

| 参数 | 值 |
|---|---|
| 正则化长度 l | 0.05 m |
| 计算域长 L | 1 m |
| L/l | 20（远场 `d(L)=1/cosh(L/l)` 已到 4.12e-09，`d'(L)=0` 条件充分近似） |
| 网格（基准算例） | n = 160，h = 0.00625 m，h/l = 0.125 |
| 杨氏模量 / 泊松比（位移钉零，仅用于装配刚度） | E = 1.0e3，ν = 0.3 |
| 断裂能 Gc | 1.0（归一化） |
| 均匀历史锚点 H | 0.5, 2, 10 |

**验证算例（替代路径 A3）**：均匀历史下的损伤饱和。空间常数驱动力 `H` 使 Euler–Lagrange
方程塌缩为 `Gc d/l = 2(1-d)H`，解为代数式 `d = 2Hl/(Gc + 2Hl)`。该算例走的是
**生产求解器的历史场分支**（`solver.history`），不是纯 Laplace 项。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **把"能不能横向"拆成五条可判定判据**（`screen_applicability`），每条都从两侧的
   `MethodDescriptor`（PDE 类 / 状态变量 / 界面宽度控制 / 闭式基准 / 本仓实现模块）算出，
   并附一句可复核的理由字符串：
   - X1 控制方程类相同？——变分能量极小 vs 离散速度格气动力学输运没有共同离散极限；
   - X2 状态变量语义相同？——非守恒序参量（含不可逆下限）无法映射到守恒质量密度；
   - X3 界面宽度控制同类？——一侧是显式参数，一侧是耦合常数/EOS 的涌现输出；比剖面
     等于拿"旋钮"和"后果"相比；
   - X4 有共享的闭式**场级**基准？——Maxwell 构造是体相热力学约束，不算场级参考；
   - X5 两侧在本仓都有实现？——一侧根本没有实现时，"横向"只是纸面声明；
     X5 由独立实现的**代码清单扫描**（`platform_implementation_inventory`）取证，
     不看描述符自述。
2. **判据机的自检**：`screen_applicability` 必须能把"同类孪生对"判成 APPLICABLE——
   否则一个恒返回 NOT_APPLICABLE 的空壳也能"通过"。测试里放了这个正控制，
   以及"清单被污染（伪造 `pseudopotential_hits`）时孪生对必须被踢回 NOT_APPLICABLE"的负控制。
3. **替代验证路径（这一侧本来就能证）**：
   - 用生产 AT2 求解器（`PhaseFieldSolver.newton_step`）解上述一维问题。由于位移全钉、
     历史场为零，损伤子问题是**线性**的，一次 Newton 即到精确离散解——所以误差里没有
     迭代残差混入（实测残差 1.183e-19，迭代 1 次）；
   - 与闭式解 `cosh((L-x)/l)/cosh(L/l)` 比 L2 相对误差与节点最大相对误差；
   - h 加密阶梯（n = 5, 10, 20, 40, 80, 160）
     检验二阶收敛，并把**渐进区外**的格子原样记录（不隐藏，见 ⑤）；
   - 均匀历史三档（H = 0.5, 2, 10）
     对表代数解；
   - 界面宽度标度：`l → 2l → 4l` 同时按比例放大域，`x(d=1/e)/l` 必须不变——这是 X3
     的**定量**侧面：宽度确实是自由参数决定的。
4. **工具链**：
   `PYTHONPATH=src python examples/xval_not_applicable_phase_field.py --out artifacts`（约 3 s，单核）
   → JSON + PNG；`pytest tests/test_xval_applicability_phase_field.py`（23 passed，约 12 s）；
   `ruff check --no-cache` 全绿；`python scripts/gen_xval_not_applicable_doc.py` 生成本文。

## ③ 计算结果（数字表，程序化注入）

### 3.1 裁决表（本单的主结论）

**配对**：`phase_field_at2` vs `pseudopotential_lbm` → 结论 **`NOT_APPLICABLE`**，
阻断判据：`X1, X2, X3, X4, X5`（5/5 条全部未通过）。

| 判据 | 检查的问题 | 意图 | 结论 | 证据 |
|---|---|---|---|---|
| X1 | Do both candidates discretise the same governing PDE class? | 控制方程类：同一离散极限 → 才有可解释的场量差 | FAIL（阻断） | `pde_class phase_field_at2='variational_diffuse_interface' vs pseudopotential_lbm='kinetic_bgk_lattice' -- a variational energy minimisation and a discrete-velocity kinetic transport equation have no shared discrete limit, so a field-by-field difference has no interpretation` |
| X2 | Do both candidates carry the same state-variable semantics? | 状态变量语义：同一物理量 → 才能逐点相减 | FAIL（阻断） | `state_variable phase_field_at2='non_conserved_order_parameter' vs pseudopotential_lbm='density' -- a non-conserved order parameter (source-type evolution, irreversibility floor) cannot be mapped onto a conserved mass density; there is no common field to subtract` |
| X3 | Is the diffuse-interface width fixed by the same kind of control? | 界面宽度控制方式：同一类旋钮 → 剖面比较才有意义 | FAIL（阻断） | `interface_width_control phase_field_at2='free_parameter' vs pseudopotential_lbm='emergent' -- one width is an explicit functional parameter, the other is an output of the coupling constant / EOS; comparing profiles would compare a knob with a consequence` |
| X4 | Is there a shared closed-form field-level reference to measure error against? | 共享闭式场级基准：有共同参考 → 误差数字才有分母 | FAIL（阻断） | `shared=none; phase_field_at2 field refs=2, pseudopotential_lbm field refs=0 -- without a common analytic reference any 'error' number would measure the difference between two models, not the error of either` |
| X5 | Are both candidates implemented in this platform (so the pair can be run)? | 本仓双实现：两侧可执行 → 横向才不是纸面声明 | FAIL（阻断） | `implemented=['phase_field_at2'] missing=['pseudopotential_lbm']; inventory modules_scanned=436 pseudopotential_hits=[] -- one side of the pair has no implementation here; the cross-validation cannot be executed, only claimed` |

清单扫描证据：`modules_scanned = 436`，
`pseudopotential_hits = []`（即本仓 436 个模块里
一个伪势多相实现都没有）。

替代路径（裁决要求的下一步，机器生成）：

- phase_field_at2: verify against its own closed-form references (1D AT2 equilibrium profile d(x)=cosh((L-x)/l)/cosh(L/l); 1D homogeneous damage d=2Hl/(Gc+2Hl) under uniform history H; AT2 critical stress sigma_c=sqrt(9EGc/(16l)) (uniaxial, plane strain))
- pseudopotential_lbm: absent from this repository -- no verdict may be claimed here; re-open the XVAL only after an implementation lands
- record the model-level distinction (different PDE class / state variable) as the deliverable instead of an error table

### 3.2 替代路径 A：一维 AT2 剖面对解析解

| 判据 | 数值结果 | 参考值/解析式 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| A1 1-D AT2 剖面 L2 相对误差（n=160, h/l=0.125） | 4.597430e-04 | 解析剖面 `cosh((L-x)/l)/cosh(L/l)`（误差即该值的 L2 范数比） | 4.597e-04 | 是 |
| A2 1-D AT2 剖面节点最大相对误差 | 1.295906e-02 | 同上 | 1.296e-02 | 是 |
| A3 均匀历史饱和 d=2Hl/(Gc+2Hl)，H=0.5 | 0.047619047619 | 0.047619047619 | 0.000e+00 | 是 |
| A3 均匀历史饱和 d=2Hl/(Gc+2Hl)，H=2 | 0.166666666667 | 0.166666666667 | 1.665e-16 | 是 |
| A3 均匀历史饱和 d=2Hl/(Gc+2Hl)，H=10 | 0.5 | 0.5 | 0.000e+00 | 是 |
| A4 收敛阶（L2 vs h 最小二乘斜率） | 1.9644 | 2（Q4 双线性，L2 范数） | 1.779e-02 | 是 |

### 3.3 h 加密阶梯（误差与收敛阶的原始数据）

| 单元数 n | h/l | L2 相对误差 | 节点最大相对误差 | Newton 迭代 | 收敛 |
|---|---|---|---|---|---|
| 5 | 4 | 3.5918e-02 | 2.9092e+03 | 8 | 否 |
| 10 | 2 | 9.0147e-02 | 9.9823e-01 | 1 | 是 |
| 20 | 1 | 2.7242e-02 | 6.0912e-01 | 1 | 是 |
| 40 | 0.5 | 7.2139e-03 | 1.9295e-01 | 1 | 是 |
| 80 | 0.25 | 1.8317e-03 | 5.1100e-02 | 1 | 是 |
| 160 | 0.125 | 4.5974e-04 | 1.2959e-02 | 1 | 是 |

误差判定门槛所需的分辨率（程序化从阶梯里读出）：**h/l ≤ 0.125
（n ≥ 160）时 L2 与节点最大相对误差同时 < 3%**。

### 3.4 界面宽度标度（判据 X3 的定量侧面）

| l [m] | l/L | x(d=1/e)/l（数值） | 解析 `exp(-x/l)` 期望 |
|---|---|---|---|
| 0.05 | 0.050 | 0.999388 | 1 |
| 0.1 | 0.050 | 0.999388 | 1 |
| 0.2 | 0.050 | 0.999388 | 1 |

三档最大/最小宽度比 = 1.000000
（不变性门槛 1.10）——宽度完全由自由参数 l 决定，与 l 的绝对大小无关。

## ④ 结果图

`artifacts/xval_not_applicable_phase_field.png`（matplotlib，1×3 面板，由 harness 直接生成）：

- 左：一维 AT2 损伤剖面（对数纵轴），TensorFEM 数值点 vs 解析曲线 `cosh((L-x)/l)/cosh(L/l)`，
  角标给出 L2 与节点最大相对误差；
- 中：h 加密阶梯双对数图（L2 / 节点最大误差 vs h/l），叠画斜率 2 参考线与 3% 门槛线，
  `h/l=1` 处画竖线标出渐进区边界；
- 右：`x(d=1/e)/l` 随 l 的变化（l = 0.05, 0.1, 0.2），
  与解析期望 1 的水平线重合，说明宽度是被自由参数设定的量。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **"NOT_APPLICABLE" 不是"已验证"**。本单证到的是：(a) 这一对不该做场级数值横向；
   (b) 相场一侧有自洽的闭式验证。伪势 LBM 一侧的判据 X1/X2/X3 基于该方法的**定义与文献**，
   本仓没有实现，因此**没有**任何本仓算出来的伪势数据——不得把它读成"伪势被证伪/被验证"。
2. **判据机必须能被证伪**。恒返回 NOT_APPLICABLE 的空壳是这类工作最典型的造假形态；
   因此测试里放了同类孪生对的正控制（必须 APPLICABLE）与清单污染负控制（必须被踢回）。
3. **渐进区外不许混进收敛阶**。`h/l ≥ 1` 时离散问题已不在渐进区：`h/l = 4` 这一档
   Newton 甚至不收敛（见 3.3 第一行，节点误差量级 1e3）。所以拟合阶只用 `h/l ≤ 1` 的档位，
   并把这些粗档**原样列表**而不是删掉——删掉就等于粉饰。
4. **损伤子问题线性的前提**：位移全钉 ⇒ `ψ⁺ ≡ 0` ⇒ 驱动力项为零，方程对 d 是线性的。
   这是为了让"误差里只有离散误差"这一条成立；一旦放开位移，误差会掺入 Newton 停止准则，
   本单的 <3% 结论就不再干净。
5. **适用范围**：本裁决只禁止**场级**数值比较（逐点损伤 vs 逐点密度）。宏观量级比较
   （表面张力随相互作用强度的趋势、体相 Maxwell 构造的一致性）在伪势实现落地后仍可做，
   届时应新开 XVAL 条目，而非复用本条的结论。
6. **环境坑（如实记录）**：本单执行期间宿主盘一度 **100% 满（可用 28 KB）**，`ruff`
   因无法建临时文件直接报 `os error 28`。清理工作树内 `__pycache__/.pytest_cache/.ruff_cache`
   后恢复（腾出 2.3 GB）。凡在此后跑绿的测试都已重跑确认，但读者须知当时的环境异常。
7. **零手抄溯源**：③ 的全部数字由 `scripts/gen_xval_not_applicable_doc.py` 从
   `artifacts/xval_not_applicable_phase_field.json` 渲染，JSON 由 harness 直接写出，
   无人工转录环节。

## ⑥ 误差判定（逐项对基准 + 显式回答"是否 <3%"）

| # | 量 | 我方 | 基准 | 误差 | <3% ? |
|---|---|---|---|---|---|
| E1 | 1-D AT2 剖面 L2 相对误差（h/l=0.125） | 4.597430e-04 | 解析 `cosh` 剖面 | 4.597e-04 | **是** |
| E2 | 1-D AT2 剖面节点最大相对误差 | 1.295906e-02 | 解析 `cosh` 剖面 | 1.296e-02 | **是** |
| E3 | 收敛阶 | 1.9644 | 2 | 1.779e-02 | **是** |
| E4 | 均匀历史饱和 d（三档最大相对误差） | 1.665e-16 | `2Hl/(Gc+2Hl)` | 同左 | **是** |
| E5 | 宽度比 `x(d=1/e)/l`（三档） | 0.999388, 0.999388, 0.999388 | 1（解析 `exp(-x/l)`） | 最大 6.121e-04 | **是** |
| E6 | 判据结论 | NOT_APPLICABLE（阻断 5 条） | 期望 NOT_APPLICABLE | — | **是** |

**逐项回答**：E1 4.597e-04 < 3% → 是；E2 1.296e-02 < 3% → 是；
E3 |1.9644 − 2|/2 = 1.779e-02 < 3% → 是；
E4 最大相对误差 1.665e-16 < 3% → 是；
E5 宽度比偏离 1 最大 6.121e-04 < 3% → 是。
**E6（伪势一侧的数值误差）：无基准、无实现，故不存在可报的误差数字——本单不声称其精度，
只给出"不可横向 + 其一侧已自证"的结构性结论。** 全部判定字段：`{"at2_homogeneous_ok": true, "at2_profile_l2_ok": true, "at2_profile_nodal_ok": true, "l2_order_ok": true, "rel_tol": 0.03, "verdict_is_not_applicable": true, "width_invariant_under_l_scaling": true}`。
