# CASE GAP-PP-3 · 板格低温钢弹塑性 + 永久挠度 — 3 网格极限载荷证据（含审计发现）

锚点：`GAP-PP-3-ICE-STRUCT-PLASTIC-SET`（板格低温钢弹塑性 + 永久挠度，大斑载荷下的真实失效模式）
本文件记录**本轮实测**的数值证据、审计发现与判定。仓：TensorFEM，分支 `agent/m22559`。

## 1. 案例（程序化注入，非手抄）

| 项 | 值 |
|---|---|
| 板 | a=2.0 m, b=1.0 m, t=0.020 m（无加筋板格，aspect=2.0, t/b=0.02） |
| 材料 | DH36，T = −20 °C（DH36 设计温度；DBTT = −65 °C，故塑性可用） |
| 流变应力 | σ_y(−20 °C) = 374.2 MPa（线性拟合 slope 0.48 MPa/K，`low_temperature_flow_stress`） |
| 硬化 | 等向硬化 H = 1000 MPa |
| 载荷 | 中心斑压 0.5 m × 0.5 m（斑边与单元边重合，`patch_plate_mesh` 强制校验） |
| 边界 | 简支 w=0 + `inplane_restrained=True`（板格嵌于骨架，面内全部嵌固） |
| 解析屈服压力 | p_y = 0.7125 MPa（`analytic_first_yield_pressure`，Navier 双正弦，ν=0.3） |
| 上限解（屈服线/斑） | p_UB = 2.0528 MPa（`yield_line_patch_upper_bound`） |
| 单元 | 分层 Mindlin Shell4（`layered_shell4_plasticity`），layers = 5 / 9 |

## 2. 本轮实测（8×4，DH36@−20 °C）

load ladder `[0.32, 0.64, 0.86, 1.07, 1.28] MPa`，卸至 0 MPa（3 步）。

| 量 | 值 | 基准 | 误差 |
|---|---|---|---|
| 同网格弹性偏置 `elastic_bias` | **−15.57 %** | — | — |
| FE 首次屈服 | 0.8439 MPa | p_y = 0.7125 MPa | **+18.44 %** |
| 斑中心挠度 @1.28 MPa | 29.74 mm（弹性的 1.12 倍） | — | — |
| 路径实测极限（切线交点） | 1.0615 MPa | p_UB = 2.0528 MPa | **−48.3 %** |
| 永久挠度（卸载 3 步，`permanent_set_valid=True`） | **2.839 mm** | 峰值 29.74 mm | 峰值之 9.5 % |
| 层模型截面能力比 | 0.9600（5 层）/ 0.98765（9 层）/ 0.99174（11 层） | 1.0 | 层离散缺口 (n²−1)/n² |

layers=9、ladder `[0.64, 1.07, 1.50, 1.92] MPa`：路径点 13.3 / 22.7 / 43.1 mm
（α=0.0035, |M|/m_p=0.953），下一步 1.924 MPa 收敛到 |M|/m_p = 1.057 → 被截面门拒绝。

## 3. 审计发现（本轮）

### N1（阻塞级，假通过）同网格「偏置归一」屈服误差是恒等式，不是证据
`plate_capacity_summary` 报的 `first_yield_error_bias_normalised` 两侧都由**同一个量**
构成：路径首次屈服因子是该 FE 弹性面应力达到 σ_y 时的插值载荷，而
`analytic_first_yield_same_mesh = σ_y / (FE 应力/Pa)`。比值恒为 1。实测残差 **6.1e−14**。
旧测试 `test_first_yield_bias_normalised_under_5pct` 断言 `|误差| < 0.05`，因此对任何网格、
任何单元、任何物理都必然通过 —— 这是「执行了代码而非断言不变量」的典型。
**处置**：测试改为钉住该恒等式（`|residual| < 1e−9`）并显式说明它**不构成** <5 % 证据；
验收由原始误差 + 网格收敛承担。

### N2（阻塞级，选错网格）「finest mesh」用字符串排序
`Test3GridConvergenceFromCluster` 取 `sorted(results["meshes"])[-1]` 作最细网格：
`"12x6" < "16x8" < "8x4"` → 被当作最细的是 **8×4（最粗）**。极限载荷 10 % 判据一直在
最粗网格上被检查。**处置**：按单元数排序（`_by_element_count`）。

### N3（阻塞级，能力未接通）`plate_capacity_summary` 报的「collapse」是加载终点
该函数不转发 `geometric` / `inplane_restrained` / `max_deflection_factor`，默认配置下
分层板在弯矩饱和后横向切线仍正定，载荷控制永远不失稳，于是 `collapse_factor` = ramp 最后
一个规定因子（`collapse_is_limit_point=False`，实测如此）。以它为「塑性极限载荷」与上限解
比较是无意义的。**处置**：新增三个关键字透传 + 在 `fe` 里给出
`limit_measurements` / `measured_limit_load` / `measured_limit_error_vs_upper`
（`limit_load_from_path` 的双切线/2 倍挠度/刚度三类定义 + 可接受性门）。

### N4（阻塞级，路径被截断）截面门拒绝 → 整条路径截断，极限点永远搜不到
旧 `advance()` 一旦 `_step_rejection` 返回原因就 `ramp_truncated` 并结束 ramp：实测在
1.50 MPa（|M|/m_p = 0.964）拒绝后立刻终止，而**物理极限点还在前面**；layers=5/9 都是这样，
所以 `measured_limit_load` 从未被真正测到过。拒绝的是「步长太大」而不是「载荷水平」。
**处置**：拒绝按与 Newton 失败相同的方式**二分**，只有增量小于 `bisection_tolerance` 才截断。

### N5（非阻塞，两门容差不一致）运行门 1 % vs 提取门 0.1 %
`_step_rejection` 容差 `1e-3`（可配置前）与 `path_admissibility` 默认 `tolerance=1e-3`
对同一状态判定不一致的根源在 `section_moment_capacity` 给的是**名义**层截面能力
（0.96 m_p），而收敛态可以合法到达 0.964–0.98 m_p（层积分比名义值略高）。实测：1.496 MPa
的点被运行门接受、被提取门判为 inadmissible，`limit_load` 被整体扣留。**未修**：
应让截面能力由模型自身的分层求积给出（或给门一个由此导出的容差），留作后续。

## 4. 判定（honest）

| 验收判据 | 结果 | 证据 |
|---|---|---|
| 弹→塑过渡点 vs 解析屈服载荷 <5 % | **未达（8×4 实测 +18.44 %）** | 同网格偏置 −15.57 %，误差来自单元弹性应力偏低；网格极限待 cj4232 |
| 塑性极限载荷 vs Ilyushin/上限解 <10 % | **未达（−48.3 %）** | 实测极限 1.0615 MPa vs UB 2.0528 MPa；FE 落在塑性上下界之间（能量允许），但 10 % 一致性不成立 |
| 3 档网格收敛 | **未完成（台账 not_run）** | `scripts/gap_pp3_3grid_study.py` 已递交 cj4232（8×4/12×6/16×8，layers=9），排队中 |
| 永久挠度后处理 | **达成** | 2.839 mm（卸载 3 步收敛，`permanent_set_valid=True`），9.5 % 峰值，随网格记录 |
| pytest ≥10 绿 | 见 §5 | 新增 5 绿 + 重型不变量测试按环境变量 gate |

**为什么 <10 % 这条判据本身可疑**：屈服线解对「斑」载荷给的是很宽的括号（长条下限 ≈ 8m_p/b²
量级 vs 斑机制上限），且分层板 n 层的截面能力比 (n²−1)/n² 使机制在 n 有限时无法完全成形。
因此更诚实的表述是：FE 极限应落在塑性上下界内并以网格收敛为稳定性证据，逐项 10 % 对齐单一
上限解在这个问题上不是一个可满足的判据。后续建议把该判据改写为
「p_lower ≤ p_FE ≤ p_UB 且相邻网格差 <10 %」。

**层数敏感性**：n=5 → 截面能力 0.96 m_p，路径在 0.98 m_p 处被门拒绝；n=9 → 0.9877 m_p，
路径可推到 |M|/m_p = 0.953 且挠度已明显非线性（43.1 mm vs 弹性 31.8 mm）。因此层数与
截面能力必须与「极限载荷」一起报告，否则数值只反映层离散。

## 5. 复现

```bash
# 单网格（本机可跑，约 45 s）
python - <<'EOF'
import sys; sys.path.insert(0,'src')
from tensorfem.plate_patch_plastic_limit import plate_capacity_summary
s = plate_capacity_summary(a=2.0,b=1.0,t=0.020,young=206e9,poisson=0.3,grade="DH36",
    temperature_c=-20.0,patch=(0.5,0.5),hardening=1000e6,
    load_factors=[0.45*712514,0.90*712514,1.50*712514,2.10*712514],
    nx=8,ny=4,layers=9,unload_steps=3,geometric=True,inplane_restrained=True)
print(s["fe"]["first_yield_error"], s["fe"]["measured_limit_error_vs_upper"],
      s["fe_permanent_deflection"])
EOF

# 3 网格矩阵（重算，递交调度）
python scripts/gap_pp3_3grid_study.py   # -> outputs/gap_pp3_3grid_results.json

# 不变量测试（重路径需显式开启）
TF_RUN_HEAVY_PLASTIC_PATH=1 python -m pytest tests/test_plate_patch_plastic_path_bisection.py -q
python -m pytest tests/test_plate_patch_plastic_path_bisection.py -q      # 5 passed, 5 skipped
```

## 6. 未运行 / 未知（不得当作通过）

* 3 网格矩阵数值（cj4232 排队中）：**not_run**。
* layers 网格（5/9/11）× 网格 3 档的全矩阵：**not_run**。
* 卸载滞回 / 大变形参照（永久挠度的独立基准）：**not_run**，本文件只给出自洽性与
  「永久挠度 < 峰值」不变量。
* N5 的截面能力容差修正：**未做**。