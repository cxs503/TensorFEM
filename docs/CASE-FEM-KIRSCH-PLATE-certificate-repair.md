# CASE-FEM-KIRSCH-PLATE — 锚点证书修复：Kirsch 孔应力集中对解析解复核 + 证据链入库

【锚点=NEW:算例池 Kirsch 孔经典基准的锚点证书欠账（E_ARTIFACT_MISSING）——证书引用 4 张图从未入库，门禁判据在仓内无工件可核】

本单是**验证欠账**类：`benchmarks/verified/fem_kirsch_plate/` 的 `gates_verified.json`
声明了 4 张 PNG 工件，但合并入 main 的只有 `run.py`/`result.json`/`gates_verified.json`，
4 张图全部缺失（审计缺陷 `E_ARTIFACT_MISSING`，登记于
`benchmarks/verified/anchor_certificate_debt.json`）。任何人按证书去取工件都会 404，
14 条门禁判据（K_t 误差 <5%、Richardson 外推 <1%、Q8 交叉 <5%、ν/E 不变性 <1%、
Saint-Venant <1%、有限宽趋势 <3%、载荷平衡 <1e-10 等）在仓内**无对应工件可核**。
本单重跑驱动、把工件补齐入库、让该锚点从债务登记表中删除，并顺带修复债务登记表的一处
**陈旧**（`fem_shell_modal` 的图同样从未入库，此前未被登记——本单如实登记，不粉饰）。

## ① 案例介绍

无限大平板中心圆孔在远场单向拉应力 T 下的应力集中，FEM 最经典基准之一。
Kirsch (1898) 给出精确解，孔边环向应力 σ_tt(a,θ)=T(1−2cos2θ)，垂直载荷轴两极处

    K_t = σ_tt(a,π/2)/T = 3

K_t 与 E、ν、绝对尺度无关（本身即一条敏感性判据）。参考：Kirsch (1898) VDI 42:797；
Timoshenko & Goodier, *Theory of Elasticity* 3rd ed. Art. 39 (a)–(c)。**零手抄**：
K_t=3 由闭式解在驱动内求值，且 oracle 自检（孔边无牵引、远场单向）先于任何 FE 运行。

| 参数 | 值 |
|---|---|
| 模型 1 | 有限宽四分之一板 0≤x≤L, 0≤y≤W/2，Q4 平面应力，星爆网格向孔分级（grading=6） |
| 模型 2（交叉） | Q8 serendipity 极坐标环带，r=R_out 上给定精确 Kirsch 位移 |
| d/W 阶梯 | 0.02 / 0.05 / 0.10（0.02 最接近无限板极限） |
| 网格阶梯 | n_boundary = 12 / 24 / 48（r=2） |
| 判据（预声明） | 极限误差 <5%、外推 <1%、Q8 交叉 <5%、收敛阶 ≥1.0/≥1.5、载荷平衡 <1e-10 |

## ② 怎么做的

1. 重跑 `benchmarks/verified/fem_kirsch_plate/run.py`（OMP_NUM_THREADS=8，墙钟 55 s，单进程）：
   9 个四分之一板 + 3 个 Q8 环带 + ν/E 不变性 4 例 + Saint-Venant 板长研究 + 有限宽趋势 5 档。
2. 驱动重新生成 `result.json` + `gates_verified.json`（schema `tensorfem.verified-gates.v1`）
   + 4 张 PNG，全部落锚点目录。
3. `git add` 工件后 `tools/verified_anchor_audit.py --write` 刷新债务登记表：
   `fem_kirsch_plate` 从债务表删除（已认证），`fem_shell_modal` 如实登记（预存欠账）。
4. `tests/test_verified_anchor_certificates.py` 16 条全绿（fail-closed 契约：
   新债务必须登记、已修复锚点必须从表删除、认证锚点必须带已跟踪证据）。

## ③ 计算结果（全部程序化注入自 `result.json`/`gates_verified.json`）

**主判据（d/W=0.02，n=48 最细）**：K_t = 3.001408，对解析 3 的误差 **+0.047 %**。

| d/W | n | K_t(FE) | 对 K_t=3 误差 | 墙钟 |
|---:|---:|---:|---:|---:|
| 0.02 | 12 | 2.879925 | +4.002 % | 0.1 |
| 0.02 | 24 | 2.977466 | +0.751 % | 0.2 |
| 0.02 | 48 | 3.001408 | +0.047 % | 5.7 |
| 0.05 | 12 | 2.861285 | +4.624 % | 0.0 |
| 0.05 | 24 | 2.970483 | +0.984 % | 0.2 |
| 0.05 | 48 | 3.001349 | +0.045 % | 3.0 |
| 0.10 | 12 | 2.874571 | +4.181 % | 0.0 |
| 0.10 | 24 | 2.991240 | +0.292 % | 0.1 |
| 0.10 | 48 | 3.025609 | +0.854 % | 2.8 |

**Q8 独立单元族交叉校核**（不同单元/拓扑/边界形式）：

| 网格 | K_t | 对解析误差 |
|---|---:|---:|
| 24x6 | 3.019739 | +0.658 % |
| 48x12 | 3.005729 | +0.191 % |
| 96x24 | 3.001476 | +0.049 % |

**14 条门禁（值/阈值均取自工件，可离线复算）**：

| 门禁 | 值 | 判据 | 判定 |
|---|---:|---|---|
| `oracle_selfcheck` | 0 | <= 1e-12 | PASS |
| `oracle_kt_value` | 3 | >= 3 | PASS |
| `kirsch_limit` | 0.0004692 | <= 0.05 | PASS |
| `kirsch_extrapolated` | 0.00306528 | <= 0.01 | PASS |
| `mesh_convergence_order` | 1.76325 | >= 1 | PASS |
| `mesh_error_monotone` | 0.0117227 | <= 0.5 | PASS |
| `q8_crosscheck` | 0.000492089 | <= 0.05 | PASS |
| `q8_convergence_order` | 1.71999 | >= 1.5 | PASS |
| `nu_E_invariance` | 0.00705612 | <= 0.01 | PASS |
| `saint_venant_length` | 0.00219973 | <= 0.01 | PASS |
| `finite_width_trend` | 0.00984781 | <= 0.03 | PASS |
| `finite_width_monotone` | 1 | >= 1 | PASS |
| `hoop_profile_shape` | 0.00317641 | <= 0.05 | PASS |
| `load_balance` | 1.01099e-12 | <= 1e-10 | PASS |

## ④ 结果图

- `kirsch_field.png`：场量云图（von Mises，孔边集中）。
- `kirsch_convergence.png`：K_t 收敛阶梯 + Richardson 外推 vs 解析 3。
- `kirsch_hoop_profile.png`：孔边环向应力 vs 解析 1−2cos2θ。
- `kirsch_width_trend.png`：有限宽趋势 vs Howland/Heywood 拟合。

## ⑤ 注意点 / 局限

- K_t=3 是**无限板极限**；有限宽板推到 d/W=0.02 并做 Richardson 外推，d/W 趋势对
  Howland/Heywood 拟合分级，不声称=3。
- Q8 环带在 r=2a 给定精确位移，只认证离散化误差，看不到远场/有限宽模型误差。
- 平面应力、线弹性、各向同性、单孔、无塑性无接触。
- 板长研究必须分辨率匹配（n~L）：固定射线数时星爆网格在顶点处变粗，K_t 会漂移出
  百分位级的假"载荷引入效应"——result.json 同时给出匹配/不匹配两张表。
- 债务登记表净变化：kirsch 出表（修复）、shell_modal 入表（预存欠账，非本单新增债务）；
  表内计数 26→26，但集合与活审计一致（`--check` 通过）。

## ⑥ 误差判定（逐项 vs 基准，显式回答 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---:|---|
| K_t（d/W=0.02, n=48） | 解析 3 | +0.047 % | **是** |
| K_t（Richardson 外推, d/W=0.02） | 解析 3 | +0.307 % | **是** |
| K_t（Q8 环带最细 96x24） | 解析 3 | +0.049 % | **是** |
| 孔边环向应力剖面形状 | 1−2cos2θ | +0.318 % | **是** |
| ν/E 不变性散布 | 自洽 | +0.706 % | **是** |
| Saint-Venant 板长带 | 自洽 | +0.220 % | **是** |
| 有限宽趋势 | Howland/Heywood | +0.985 % | **是** |
| 载荷平衡残差 | 0 | 1.01e-12 (<1e-10) | **是** |

**总判：PASS（14/14 门禁，failed_gates=[]）。** 关键量对解析基准误差最大 0.854%（d/W=0.10,
n=48 的有限宽模型误差，已单独归因），主判据 0.047%——**全部 <3%**。

## 复现

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py
python3 tools/verified_anchor_audit.py --check
PYTHONPATH=src python -m pytest tests/test_verified_anchor_certificates.py -q
```
