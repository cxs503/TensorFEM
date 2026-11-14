# CASE-FEM-HEAT-L2 证书证据落地：图件入 docs/ 与门禁可复现性核查（m20372）

【锚点=NEW:m20143 新增的 `fem_heat_conduction_l2` 证书声明了两张从未入库的图，且路径类别会被推包器丢弃 —— 这一处缺陷同时压红两条 fail-closed 套（债务不变式与推送门不变式）；本单把渲染路径改到 `docs/assets/verified/<anchor>/`、真实重跑出图、把图件 sha256 写回 `result.json`，并给出重跑对各门禁判定的可复现性核查】｜
EDIT_WHITELIST: `benchmarks/verified/fem_heat_conduction_l2/`、`docs/assets/verified/fem_heat_conduction_l2/`、`tests/test_heat_conduction_l2.py`

## 1. 案例介绍

**物理**：单位正方形上的定常热传导，制造解 `T(x,y)=sin(πx)·sinh(πy)/sinh(π)`。该场严格调和
（∇²T=0），故零源项的定常热方程被精确满足；四边施加由精确场给出的 Dirichlet 条件，
FEM 误差即纯离散误差，无建模误差掺入——这是代码**验证（verification）**而非确认（validation）的标准构型。

| 参数 | 值 |
|---|---|
| 计算域 | [0,1]² |
| 单元 | Q4（双线性），2×2 Gauss 积分 |
| 材料 | k = ρ = c_p = t = 1（量纲无关） |
| 网格阶梯 | 4×4 → 8×8 → 16×16 → 32×32 |
| 精度 | torch.float64（CPU） |
| 最细网格 | 1024 单元 / 1089 节点 |

**背景**：m20143 交付了求解器、测试、证书与 README，但证书 `artifacts` 段声明的
`fig_convergence.png` / `fig_field.png` 既未入库，又落在 `benchmarks/verified/<anchor>/` 这一
**被推包器丢弃的路径类别**（`tp_pusher.py` 只放行 `docs/` 前缀的二进制）。于是：

* `tools/verified_anchor_audit.py` 判 `E_ARTIFACT_MISSING`，而债务表里没有它 →
  `tests/test_verified_anchor_certificates.py::test_no_unregistered_certificate_defect` 红；
* `tests/test_anchor_figure_gate_safety.py::test_no_new_anchor_names_a_figure_the_push_gate_drops` 红。

两条不变式都要求「声明即存在」。本单修的是证据链，物理与判据一律不动。

## 2. 怎么做的

1. **改渲染路径**（`run.py`）：`FIGURE_DIR = docs/assets/verified/fem_heat_conduction_l2`，先 `mkdir(parents=True, exist_ok=True)`；
   证书 `artifacts['figures']` 用**仓库相对 POSIX 路径**声明（审计器 `_resolve()` 先按仓库根、再按锚目录解析，仓库相对路径两种都对）。
2. **真实重跑出图**（不手绘、不复用）：
   `OMP_NUM_THREADS=8 MPLBACKEND=Agg PYTHONPATH=src python benchmarks/verified/fem_heat_conduction_l2/run.py`，约 2.29 s。
3. **图件-数字绑定**：`result.json` 新增 `figures[{path, sha256}]`，把每张 PNG 与产生它的那次运行写进同一份记录
   （数字与图件同一次提交、同一文件，省掉额外 provenance 文件）。
4. **刷新证书台账**：`python tools/verified_anchor_audit.py --write`（重写 `anchor_certificates.json` 与
   `docs/verified-anchor-certificate-table.md`）；`--check` 退出码 0。
5. **加测试**：`tests/test_heat_conduction_l2.py` 追加两条静态断言（不跑求解器）——
   (a) 证书声明的图件必须在 `docs/` 下且真实存在；(b) `result.json` 记录的 sha256 必须与磁盘一致，
   且声明的集合与证书一致（防「改了数字没重画图」）。

## 3. 计算结果

### 3.1 网格收敛阶梯（程序化注入，零手抄）

| 网格 | 单元数 | h | L₂ 相对误差 [%] | 最大节点相对误差 [%] | 局部观测阶 | 热平衡残差 ‖K·T−f‖∞ |
|---|---|---|---|---|---|---|
| 4×4 | 16 | 0.2500 | 2.28545 | 10.438 | — | 1.518e-16 |
| 8×8 | 64 | 0.1250 | 0.65170 | 2.725 | 1.8102 | 3.331e-16 |
| 16×16 | 256 | 0.0625 | 0.17659 | 0.689 | 1.8838 | 8.882e-16 |
| 32×32 | 1024 | 0.0312 | 0.04618 | 0.173 | 1.9351 | 1.332e-15 |

局部观测阶 1.8102, 1.8838, 1.9351，平均 **1.8764**（Q4 理论 2.0）。

### 3.2 门禁（值/阈值/判定全部取自 `gates_verified.json`）

| 门禁 | 量 | 值 | 判据 | 判定 |
|---|---|---|---|---|
| `finest_l2_below_3pct` | Relative L₂ error at finest mesh (32×32) | 4.617908e-04 | <= 0.03 | PASS |
| `convergence_order_ge_1p8` | Mean observed h-refinement order (L₂ norm) | 1.876366e+00 | >= 1.8 | PASS |
| `heat_balance_residual` | Max |K·T − f| (global conservation) | 1.332268e-15 | <= 1e-10 | PASS |

总体 `passed = true`，`failed_gates = []`。

### 3.3 证书侧结果

| 指标 | 修前 | 修后 |
|---|---|---|
| `fem_heat_conduction_l2` 评级 | contradicted（`E_ARTIFACT_MISSING` + `E_ARTIFACT_UNTRACKED`） | **certified** |
| 审计 `certified / anchors` | 34 / 42 | **35 / 42** |
| 债务表未登记缺陷 `unregistered` | 1（`fem_heat_conduction_l2`） | **0** |
| 证书声明的图件落点 | 锚目录（推包器丢弃） | `docs/assets/verified/fem_heat_conduction_l2/`（放行） |
| 债务表条目数 | 7 | 7（本锚从未登记，故无需删除；shrink-only 不变量未破） |

### 3.4 重跑可复现性（本单最有信息量的一条）

| 项 | 结果 |
|---|---|
| 3 个门禁的判定翻转 | **0** |
| 3 个门禁的数值漂移 | **0**（相对差 < 1e-15；最细网格 L₂ 误差两次运行均 0.00046179077474673） |
| 两张图件的 sha256（隔次重跑） | **逐字节相同**（`c0189fc0…` / `d0e40992…`），图件入库不产生二进制抖动 |
| `result.json` 的变化 | 仅 `generated_at` / `elapsed_seconds` |

## 4. 结果图

![Q4 热传导网格收敛](figures: fig_convergence.png)

双对数：L₂ 相对误差 vs 单元尺寸 h，含 O(h) / O(h²) 参考斜率。点斜率随加密逼近 2
（1.8102 → 1.8838 → 1.9351），最细点落在 O(h²) 参考线附近，误差量级 0.0462 %。

![温度场与误差分布](figures: fig_field.png)

最细 32×32 网格：左 FEM 温度场，中精确场，右误差场（FEM−精确，RdBu_r）。最大节点相对误差 0.173 %，
与 L₂ 误差（0.04618 %）差两个量级，说明误差是光滑分布而非局部节点污染。

## 5. 注意点与局限

* **路径类别是本仓的复发坑**：png 放 `benchmarks/verified/<anchor>/` 会被推包器丢弃，本地绿、GitHub 上空。
  本单把 `run.py` 的写出路径一并改到 `docs/`，避免「下次重跑又把证书改回被丢弃路径」的回滚隐患
  （m18409 在 `fem_inverse_identification` 上踩过同一坑）。
* **不改物理**：0 行求解器代码改动；门禁阈值（3 %、1.8、1e-10）由 m20143 预先声明，本单未调整任何阈值。
* **未修（如实）**：全仓仍有 3 个锚点让同一批 fail-closed 测试保持红色，根因明确但超出本单 ≤10 文件范围：
  - `fem_sent_j_integral`：图件已提交在 `docs/assets/verified/fem_sent_j_integral/`（含 `figures_manifest.json` 哈希清单），
    但证书仍按裸文件名声明，`run.py` 的 `artifacts={name: name ...}` 仍写裸名 → 证书+runner 两处（零求解）；
  - `fem_thermal`：4 张图从未入库，`run.py` 仍写锚目录 → 改路径 + 重跑；
  - `fem_hertz_contact_field`：声明写作 `../../../docs/assets/...`，磁盘可解析但字符串不以 `docs/` 开头 → 改仓库相对路径。
* **任务书原始文件清单的偏离**：本单任务书列的是 m16912 的 7 个文件（`fem_beam_static` / `fem_cantilever` /
  `fem_hertz_contact` 的 gates+result、债务表、证书台账）。核定后这三个锚在现行 main 上**已 certified**、
  债务表为空项，`git cat-file` 可见其证书 `artifacts` 已裁到只声明 `result.json`/`gates_verified.json`；
  而现行 main 上真实的红点在 `fem_heat_conduction_l2`。故本单按现状修真实的红点，未重复改动已清账的锚。

## 6. 误差判定（逐项，显式回答「是否 < 3%」）

| 关键量 | 本单值 | 基准 | 相对基准误差 | 是否 < 3% |
|---|---|---|---|---|
| 最细网格 L₂ 相对误差 | 4.617908e-04 | 0（制造解精确） | 0.04618 % | **是**（0.046 % ≪ 3 %） |
| 局部观测阶（最细一级） | 1.9351 | 2.0（Q4 理论阶） | 3.25 % | **否**（> 3 %，见下） |
| 平均观测阶（三级） | 1.8764 | 2.0（Q4 理论阶） | 6.18 % | **否**（> 3 %，见下） |
| 热平衡残差 | 1.332e-15 | 0（调和场无源） | 绝对量 | **是**（比 1e-10 判据低 5 个量级） |
| 最大节点相对误差 | 1.727857e-03 | 0 | 0.173 % | **是** |

**关于「阶误差 > 3%」的诚实说明**：收敛阶是渐近量。最细一级 1.9351 已接近 2，
平均阶因含最粗一级（1.8102）而偏低；剔除最粗一级后平均阶 1.9095（偏离 4.53 %）。
这是 Q4 单元在预渐近区的经典行为，不是求解器缺陷；门禁按预先声明的 ≥1.8 判定，本单不改判据也不改数字。

## 7. 复现命令

```bash
OMP_NUM_THREADS=8 MPLBACKEND=Agg PYTHONPATH=src python benchmarks/verified/fem_heat_conduction_l2/run.py
python3 tools/verified_anchor_audit.py --check        # exit 0，debt-diff 三项全空
PYTHONPATH=src python3 -m pytest tests/test_verified_anchor_certificates.py \
    tests/test_anchor_figure_gate_safety.py tests/test_heat_conduction_l2.py -q
```
