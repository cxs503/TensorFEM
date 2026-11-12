# 弹性地基板（Winkler / Pasternak 两参数）进入可执行门禁：3 条注册证据 + 4 级收敛阶梯

**模块** `src/tensorfem/foundation_plate_registry.py` · **接入点**
`src/tensorfem/benchmark_registry.run_registered_benchmarks()`（追加 3 条，
门禁总数 69 → 72）· **测试** `tests/test_foundation_plate_registry.py`（7 项全绿）·
**数据/图** `artifacts/foundation_registry/foundation_convergence.{json,png}`

## 1. 案例介绍

船体板格下方的弹性地基（冰-结构接触中的冰盖/海床、双层底压载舱的弹性支撑）在
TensorFEM 里由 Pasternak（Vlasov）两参数模型描述：地基反力

    p = k w - G ∇²w ,      k = 1e+08 N/m³ (Winkler 弹簧),  G = 5e+04 N/m (剪切层)

算例为四边软简支（w=0，转角自由）方板，均布压力 q：
a=b=1.0 m，t=0.01 m，E=2.10e+11 Pa，ν=0.3，q=1e+04 Pa。
待验量＝板中心挠度。

## 2. 怎么做的

1. **发现缺口**：`run_registered_benchmarks()` 的 69 条检查里**没有一条带地基**——
   地基板的证据此前只存在于 `benchmarks/verified/fem_kerr_plate/result.json`
   这类**静态工件**里。静态工件不会被门禁重跑，因此它无法 FAIL。
2. **补可执行面**：新增 `plate_foundation_evidence()`，用 Q4 Reissner-Mindlin 单元
   （2×2 弯曲 + MITC 横向剪切）经 `assemble_plate_foundation_system` 直接装配求解，
   与**同一连续问题的精确 Navier 级数**
   `plate_foundation_interface.fsdt_navier_ss_deflection`（地基符号 k + Gλ）逐点比较。
3. **三条检查**：`plate.foundation.winkler_center_w`、`plate.foundation.pasternak_center_w`
   （容差 3%）与 `plate.foundation.pasternak_zero_reduces_to_winkler`
   （G=0 归约必须是**精确等式**，容差 1e-12——它不是精度判据，防止剪切层装配缺陷躲在 3% 带里）。
4. **物理不变量测试**：剪切层只能**变刚**（变形减小），方向错了就是符号反了；
   该断言正是本轮抓到的一个写法错误（见 §5）。

## 3. 计算结果（程序化生成，零手抄）

注册网格 n=24（1875 自由度，稠密求解）:

| 检查 | FE 读数 (m) | 精确解 (m) | 相对误差 | 门限 | 判定 |
|---|---:|---:|---:|---:|:--:|
| Winkler（G=0） | 1.195303e-04 | 1.190091e-04 | **0.438%** | 3% | PASS |
| Pasternak（G=5e+04） | 1.187165e-04 | 1.182078e-04 | **0.430%** | 3% | PASS |
| G→0 归约 | 1.244606e-04 | 1.244606e-04 | 0.0 | 1e-12 | PASS |

收敛阶梯（见上图）：h 从 1/8 到 1/24，误差单调下降到 ~0.4%，两条曲线近乎重合，
说明剪切层项没有破坏弯曲精度；误差随 h 二次下降与 Q4 Mindlin + MITC 的期望阶一致。

## 4. 结果图

`artifacts/foundation_registry/foundation_convergence.png`：双对数下 Winkler/Pasternak
两族的中心挠度相对误差 vs 单元尺寸，含 3% 门限线。

## 5. 注意点（坑与适用边界）

- **方向写错一次**：初版测试断言「加剪切层挠度变大」（softens），实测 1.1872e-04 < 1.1953e-04 当场 FAIL。
  物理上 p = k w − G∇²w 中 G>0 只会**提高**地基刚度，故改为断言挠度减小。
  这条错断言正是新增物理不变量检查的价值所在——纯数值比对（都在 3% 内）不会暴露它。
- **G→0 归约检查是「同路两次装配」**：它检验装配路径的自洽，不检验精度；写成 1e-12 容差
  而非 3%，避免又一个「宽容差掩盖缺陷」的口子。
- **适用范围**：薄到中厚板（t/a=0.01 在此范围内），k/G 取正；负刚度（脱空）不在此面，
  脱空由 `foundation_plate_tensionless` 的单向路径覆盖。
- **未做**：Kerr 三参数（上/下弹簧 + 剪切层串联）尚未进注册门禁，仅由
  `kerr_equivalent_pasternak` 的归约测试覆盖；其精确解需要对弹簧串联做模态级重排，
  留作后继。

## 6. 误差判定

| 关键量 | vs 基准 | 是否 <3% |
|---|---|---:|
| Winkler 板中心挠度 | 精确 Reissner-Mindlin Navier 级数 | 0.44% → **是** |
| Pasternak 板中心挠度 | 同上（含剪切层符号 k+Gλ） | 0.43% → **是** |
| G→0 归约 | Winkler 模型恒等 | 0（精确） → **是** |

基准为同一单元族的封闭解析解（非文献近似值），故容差按平台 3% 标准执行，实测远优于该线。
