# 交差审计 m15883 — CASE-HOLLOW-SPHERE-3D（m15862, TensorFEM）

**结论先行**：m15862 是**真实现**（非 MOCK/STUB，独立复跑逐位复现），验证等级 **L2 确认**；
`passed:true` 的 1.71% 不能作为算例池 <5% 的达标证据 —— 按工件自己声明的参考泛函
（`reference_functional = lame_layer_average`，即"体积加权层积分"）重算，
选中阶梯 hex20 的 `radial_stress_inner` 误差为 **8.03% > 5%（不达标）**，而基线 hex8 为 4.10%（达标）。
即**基线与选型的通过/失败排序在两种泛函下反转**。L3 缺（未进 benchmark_registry），L4 不予确认。

## ① 案例介绍（物理背景/几何/参数表）

厚壁球壳（a=8 m、b=10 m）受内压 p=1 MPa，外表面自由，属经典三维弹性静力问题；
参考解为 Lamé 闭式解（Lamé 1852；Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed. 1970, §5.6）：

| 量 | 值 |
|---|---|
| 内半径 a / 外半径 b | 8.0 m / 10.0 m |
| 壁厚 t、径比 b/a | 2.0 m、1.25 |
| E / ν / p | 2.1e11 Pa / 0.3 / 1e6 Pa |
| A = p a³/(b³−a³) | 1 049 180.33 Pa |
| B = A b³ | 1.0491803e9 Pa·m³ |
| σ_r(a)、σ_r(b) | −1.000000e6 Pa（=−p）、0 Pa |
| u_r(a)、u_r(b) | 6.672912e−5 m、5.245902e−5 m |
| 精确壳体体积 / 应变能 | 2044.130 m³ / 26833.37 J |

以上 A/B/边界值/位移由本审计的独立脚本重新推导（`docs/audit_m15883/analyze_m15862.py`），
与工件 ref 值逐位一致。

## ② 怎么做的（方法/流程/关键设置/工具链）

- 被审对象：m15862（已 succeeded），commit `e92f7e43`，工作树 `/root/tp-agent/work/m15862`。
- 取证：读 `artifacts/case_hollow_sphere_3d/case_hollow_sphere_3d.json`（187 kB，含两族阶梯全部 mesh/metrics/fields）、
  3 张 png、`src/tensorfem/hollow_sphere_3d.py`（668 行）、`benchmarks/case_hollow_sphere_3d.py`、
  `tests/test_hollow_sphere_3d.py`；裁判取 `tensor_benchmark_cases id=25`（tolerance `<5%`）。
- 三件套核验：**工件**✅（json+3 png 存在且非空，03:35:00 生成）；**ref vs got 数值**✅（表内 7 项逐项有 ref/got/error/pass）；
  **repro_cmd 可执行**✅（`PYTHONPATH=$PWD/src python3 benchmarks/case_hollow_sphere_3d.py --quick`，
  m15883 独立重跑 exit 0，hex8 28.2996%/15.3843%、hex20 2.8848%/1.7096307692412875%，
  与工件逐位相同，末行 `CASE VERDICT: PASS`）。
- 独立复算：重推 Lamé 解与"体积加权层积分"的闭式 <σ_r>=A−3B·ln(ro/ri)/(ro³−ri³) 等，
  并用数值积分交叉验证；再按**工件声明的泛函**与**工件实评分泛函**两读法逐项算误差。
- 单元测试：`pytest tests/test_hollow_sphere_3d.py -q` → **13 passed in 329.93 s**（含扰动场必须被拒的负对照）。
- 网格侧证：重建 `_mesh(6,3,2)`，得节点半径 8.0/8.333/8.667/9.0/9.333/9.667/10.0，
  末环单元形心半径 9.549–9.628 ⇒ 单元径向跨度是 [9.333,10]，而非工件声明的 [9.667,10]。

## ③ 计算结果（数字表，零手抄：全部由脚本从工件读出并注入）

两读法逐项（tolerance 5%；`scored` = 工件实际评分用的"单元形心处采样参考"，`declared` = 工件声明并写进 `layer_integrals` 的精确体积加权层积分）：

| 判据项 | hex20 got | scored ref | scored 误差 | declared ref | declared 误差 |
|---|---|---|---|---|---|
| u_r(inner) | 6.6737e−5 | 6.67291e−5 | 0.012% | 6.67291e−5 | 0.012% ✅ |
| u_r(outer) | 5.24064e−5 | 5.24590e−5 | 0.100% | 5.24590e−5 | 0.100% ✅ |
| σ_r inner | −825 133 Pa | −811 263 | 1.710% ✅ | −763 804 | **8.029% ❌** |
| σ_θ inner | 1 951 680 | 1 979 400 | 1.400% ✅ | 1 955 668 | 0.204% ✅ |
| σ_vM inner | 2 776 820 | 2 790 670 | 0.496% ✅ | 2 719 493 | 2.109% ✅ |
| σ_θ outer | 1 625 400 | 1 644 480 | 1.160% ✅ | 1 629 930 | 0.278% ✅ |
| σ_r outer（以 p 归一） | −151 670 | −141 419 | 1.03%·p ✅ | −112 322 | **3.94%·p** |

基线 hex8（层界 [8,9]、[9,10]）：σ_r inner got −632 184 / declared ref −659 239 = **4.104% ✅**（scored ref −747 124 = 15.384% ❌）；
σ_θ inner 0.129% ✅、σ_vM inner 1.133% ✅、σ_θ outer 0.292% ✅、σ_r outer 1.78%·p。
⇒ 按 declared 泛函，被判"失败"的 hex8 反而在 5% 线内，被判"通过"的 hex20 反而超线。

全局一致性证据（真实求解的旁证）：FE 应变能 26 822.05 J vs Lamé 26 833.37 J（0.042%）；
自由残差 8.1e−12、压力载荷合力 6.4e−17、hex20 网格体积误差 2.83e−4。

## ④ 结果图

- `fig_fields_vs_lame.png`：hex8/hex20 形心处 σ_r、σ_θ、σ_vM 与 Lamé 场对比（对数式内壁陡梯度区可见 hex8 欠分辨）。
- `fig_metric_two_readings.png`：同一 got 值在两种参考泛函下的误差柱状对比 + 5% 判据线（scored 与 declared 排序反转一目了然）。
- `fig_convergence.png`：交付的两级阶梯最差误差（hex8 28.3%→15.4%，hex20 2.88%→1.71%），显示降低但仅 2 级。

## ⑤ 注意点（坑/局限/适用范围）

1. **声明泛函 ≠ 实评泛函**（F1，高）：工件 `reference_functional` 写的是 `lame_layer_average`（精确体积加权层积分），
   但表内 `ref` 实际是"在 FE 单元等参中心处采样精确场再加权"。二者在 σ_r inner 上相差 7.5%（hex20）/13.3%（hex8）。
   读者若照工件自己的声明核验，会得到 8.03%（不达标）而非 1.71%。
2. **层界取成半层**（F2，高）：`probe_metrics` 用 `torch.unique(node radii)` 定层界，HEX20 的边中点节点半径
   （8.333/9.667）被当成层界，于是"精确层积分"在 [8,8.333]、[9.667,10] 这半个单元层上积分，
   与 FE 侧按整层加权平均的对象不一致 —— 这恰好抹掉了内层 σ_r 的 8% 误差。
3. **探针位置向内偏**（F3，中）：σ 在 HEX20 等参中心（ξ=η=ζ=0）取值，曲壳单元的该点比体积形心更靠内
   （末环 9.549–9.628 vs 形心 ~9.68 m），对 σ_r∝1/r³ 会引入 2–3% 的位置偏差，`got` 本身也不是真正的体积平均。
4. **阶梯太短**（F5，低）：每族仅 2 级，hex20 末对观测阶 1.29（二阶单元期望≈2）；(8,4) 级代码中有（`LONG_HEX20_LADDER`）但未跑，GCI 带未量化。
5. 本审计未跑：loong64 集群复跑（本机 x86_64）、Gauss 点应力恢复重算、跨仓平台消费（无调用方）。
6. 复跑副作用：`--quick` 会重写被审工作树的 artifacts（数值逐位不变，仅 `generated` 时间戳/文件 mtime 变），已在 `repro_cmd` 注明。

## ⑥ 误差判定（逐项 + 显式回答"是否<3%"）

| 关键量 | 误差 | <3%？ |
|---|---|---|
| u_r(a) / u_r(b) | 0.012% / 0.100% | ✅ 是 |
| σ_θ inner / σ_vM inner / σ_θ outer（declared 泛函） | 0.20% / 2.11% / 0.28% | ✅ 是（σ_vM 2.11%） |
| **σ_r inner（declared 泛函，算例池判据）** | **8.03%** | ❌ 否 |
| σ_r inner（工件实评 scored 泛函） | 1.71% | ✅ 是 |
| σ_r outer（近零量，以 p 归一，declared） | 3.94%·p | ❌ 否（作为相对量无意义，故以 p 归一） |
| 应变能（全局标量） | 0.042% | ✅ 是 |
| 压力载荷合力 / 自由残差 | 6.4e−17 / 8.1e−12 | ✅ 是（机器精度） |

**是否 <3%？** 位移与环向/等效应力：是（≤2.11%）；**径向应力（算例池主判据项）按工件声明的体积加权层积分：否，8.03%，
连 5% 池线都未通过**。因此本审计不确认"benchmark 达标"，按 L2 结账，并给出返修建议。

## 判级与返修

- **L1** ✅（闭式解 + Clapeyron/强式平衡/形函数/载荷自平衡检验）
- **L2** ✅ 确认（真代码 input→assembly→约束 CG→应力恢复→误差表→工件；独立复跑逐位复现；13 测试通过）
- **L3** ❌（只有 `tensorfem hollow-sphere` CLI 与包导出；`src/tensorfem/benchmark_registry.py` 31 条 `_evidence` 中无本算例，平台面看不到）
- **L4** ❌（ref-vs-got 表、误差分析、repro 都在，但达标结论建立在与自身声明不符的泛函上；按声明泛函 8.03% 超线）
- 返修：①层界改用角节点半径/单元层厚；②FE 层平均改单元内 Gauss 积分；③按精确层积分重评并补内层径向加密（如 (8,4) 级）；
  ④注册进 `benchmark_registry.py`（补 L3）。

**工件**：`docs/audit_m15883/{verdict_m15862.json, analysis_m15862.json, analyze_m15862.py, fig_*.png}`；
verdict 落表 `tensor_maturity_verdicts id=393`（repo=TensorFEM, scope=`m15862 CASE-HOLLOW-SPHERE-3D`, mission_id=15883, level=2, verdict_by=board-audit）。