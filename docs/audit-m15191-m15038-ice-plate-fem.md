# 交差核验 m15038 `H3-ICE-PLATE-FEM`（冰板弯曲首裂 FEM，TensorFEM）

**核验对象**：mission m15038（succeeded，commit `19a51812`，分支 `agent/m15038`），工作树
`/root/tp-agent/work/m15038`；核验者在 `agent/m15191` 独立复核，未改动被审交付物。
**性质**：结榜交差核验，非重做。**基准判据**：`tensor_benchmark_cases` id=11，
`anchor_slug=H3-ICE-PLATE-FEM`，metric「挠度/弯矩/破坏位置 vs 级数解」，tolerance `<5%`。

## ① 案例介绍（物理背景/几何/参数）

浮冰盖上的圆形均布载荷（冰上作业、冰区结构接触）→ 首裂（弯曲拉伸破坏）极限载荷与破坏位置。
无限大冰板 + Winkler 水弹性地基（k=ρ_w g），板受半径 a 的均布压力 p，求主弯曲力矩达到弯曲
强度时的载荷 P_f = σ_f h²/(6·m̂)，以及临界纤维（下表面）与临界半径。

| 参数 | 值 | 出处 |
|---|---|---|
| E | 7.0e9 Pa | `evidence.json::sheet.young` |
| ν | 0.30 | 同上 |
| h | 0.30 m | 同上 |
| k | 10055.25 N/m³ | 同上（1025×9.81） |
| σ_f | 0.60 MPa | 同上 |
| D=E h³/12(1-ν²) | 1.730769e+07 N·m | 审计复算一致 |
| l=(D/k)^¼ | 6.4411 m | 同上 |

## ② 怎么做的（方法/流程/工具链）

1. **被审交付物的实质**：m15038 只做**公共 API 暴露**——`src/tensorfem/__init__.py:259` 导入并在
   `__all__` 加 11 个符号、`src/tensorfem/public_api_manifest.json` 11 条 manifest、
   `api/public-api-v1.json` 11 条契约；**实现体 724 行来自 m14883 的 merge #652**（本单未改）。
   因此本审计把「实现」与「暴露」分开记账。
2. **复核面①：单测**——`tests/test_ice_plate_flexural_limit.py` 22 passed / 43.5 s（本单实跑）。
3. **复核面②：基准复现**——`python benchmarks/ice_plate_flexural_limit.py` 实跑 100.7 s，
   `script_sha256=d9f61087…` 与交付工件一致，全部物理数值逐位复现（仅 `seconds`/`runtime_seconds`
   计时项不同），最大 |误差 vs Kelvin| 0.339%。
4. **复核面③：独立解法补测**——用 **Hankel 变换**独立解同一 PDE（不调用任何 TensorFEM 数值代码、
   不用 kei 级数，只用 scipy Bessel + 分段自适应求积）：
   w(0)=(pa/kl)∫J₁(ρu)/(1+u⁴)du，M_r(r)=pal∫u²J₁(ρu)[J₀(σu)-(1-ν)J₁(σu)/(σu)]/(1+u⁴)du。
   机具自校：M_r(0) 与仓库 Kelvin 锚一致到 **-0.0149%**。
   脚本 `artifacts/audit_m15191/independent_deflection_and_failure_location.py`。
5. 工具链：python 3.11.6 / numpy 2.4.6 / scipy 1.16.0 / loongarch64，单核。

## ③ 计算结果（数字表，程序化注入）

**首裂载荷 vs 解析板理论**（`evidence.json::limit_load`，基准 <5%）：

| a/l | FEM P_f (kN) | Kelvin 解析 (kN) | err vs Kelvin (%) | err vs Nevel (%) |
|---|---|---|---|---|
| 0.10 | 29.900 | 29.799 | +0.339 | +0.305 |
| 0.20 | 39.076 | 39.025 | +0.130 | -0.047 |
| 0.30 | 47.616 | 47.576 | +0.085 | -0.394 |
| 0.50 | 65.299 | 65.274 | +0.039 | -1.745 |
| 1.00 | 123.495 | 123.447 | +0.039 | -12.572 |
| 2.00 | 395.790 | 395.791 | -0.000 | -135.143 |

最大 |err vs Kelvin| = **0.339%**；
网格收敛段中心弯矩最大误差 **0.936%**（40→160 单元）。

**本审计补的挠度肢**（FE 中心挠度 vs 独立 Hankel）：

| a/l | w_FEM (m) | w_Hankel (m) | 相对误差 (%) |
|---|---|---|---|
| 0.1 | 3.885870e-07 | 3.882609e-07 | +0.084 |
| 0.3 | 3.389053e-06 | 3.385338e-06 | +0.110 |
| 1.0 | 3.038008e-05 | 3.037181e-05 | +0.027 |
| 3.0 | 1.058561e-04 | 1.058580e-04 | -0.002 |

**本审计补的破坏位置肢**（FE 临界半径 vs 独立 Hankel M_r 极大点，r/a）：

| a/l | FE 临界 r/a | Hankel argmax r/a | \|差值\| | 临界纤维 |
|---|---|---|---|---|
| 0.3 | 0.0112 | 0.0200 | 0.0088 | bottom |
| 1.0 | 0.0034 | 0.0200 | 0.0166 | bottom |
| 3.0 | 0.4361 | 0.4283 | 0.0078 | bottom |

## ④ 结果图

- `artifacts/audit_m15191/audit_closure_figure.png`：A=挠度误差 vs a/l（5% 容差带）；
  B=a/l=3 的独立 M_r(r)/p 曲线 + FE 临界半径标线（破坏位置肢）。
- `benchmarks/artifacts/ice_plate_flexural_limit/limit_load_vs_radius.png`（交付工件，本单复现）：
  首裂载荷-载荷半径曲线 vs 两条解析参考。

## ⑤ 注意点（坑与局限）

1. **实现 ≠ 暴露**：m15038 的 3 个改动文件全部是 API 面；把 L2 判给「本单新增能力」会失真，
   真实增量是「能力被公共 API 承认并可被发现」。
2. **交付 evidence 的 benchmark 只覆盖两肢**：挠度无对照，破坏位置仅网格收敛自证（内部一致而非
   外部对照）。本审计用独立 Hankel 补齐（0.11% / ≤0.0078a），功能本身无缺陷。
3. **`PASS/FAIL` 字样无阈值比较**：`benchmarks/ice_plate_flexural_limit.py:356` 计算 `worst` 后仅打印，
   恒 `return 0`；容差口径三处不一：evidence `acceptance`=10%、基准表 5%、单测断言 3%。实测 0.339%
   远严于三者，但口径应统一（低危）。
4. **工件溯源**：`evidence.json` 记 `dirty=true` 与非合并分支 commit，未绑合并后 commit（与 m14883
   审计同一发现，仍未闭合）。
5. **m15038 无自身报告**：`tensor_reports` 无 mission_id=15038 行（`report_owed=false`），六段报告只有
   m14883 的 report#14005；交差时必须按「单号-报告号」对应阅读。
6. **Mindlin 剪切**：h/l=0.047，FE 挠度与 Kirchhoff-Hankel 差 0.11% 量级，剪切影响可忽略——正是该
   对照能用 5% 判据的前提。
7. **m15019 独立 Hankel（弯矩 0.0002%）与本单独立 Hankel（挠度/位置）互不依赖**，两次第三方路径
   都指向同一结论：0.34% 是真验证，非同源互证。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | got | 误差 | <3%? | <5%(注册容差)? |
|---|---|---|---|---|---|
| 首裂载荷 a/l=0.1 | 29.799 kN (Kelvin) | 29.900 kN | +0.339% | 是 | 是 |
| 首裂载荷 a/l=1.0 | 123.447 kN | 123.495 kN | +0.039% | 是 | 是 |
| 首裂载荷 a/l=2.0 | 395.791 kN | 395.790 kN | -0.000% | 是 | 是 |
| 中心弯矩（最粗网格 40 单元） | 2.2191 (Kelvin) | 2.1983 | −0.936% | 是 | 是 |
| 中心挠度（本审计，a/l=0.3 最差） | Hankel | FE | +0.110% | 是 | 是 |
| 破坏位置 a/l=3 | Hankel argmax | FE 临界点 | Δr/a=0.0078（0.78% of a） | 是 | 是 |

**结论：注册基准（<5%）全部满足；关键量均在 3% 以内。** m15038 = **L4**
（L5 不成立：无实验/外场锚）。

## 附：判级与交差结论

- **verdict**：`('TensorFEM', 'm15038 H3-ICE-PLATE-FEM', 4, board-audit)`，evidence 键齐
  （impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor）。
- **交差结论**：功能实质 L4 成立 → 可消账；三项低危改进项（容差口径统一、benchmark 加机器阈值
  比较、工件绑合并 commit）留给后续维护单，不构成本单不合格。
