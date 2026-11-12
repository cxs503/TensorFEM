# 证据链修复：CASE-FEM-JINTEGRAL-EDGE-CRACK 锚点工件入库（m16779）

【锚点=CASE-FEM-JINTEGRAL-EDGE-CRACK】｜ EDIT_WHITELIST: docs/, benchmarks/verified/fem_sent_j_integral/, tests/
本单=generalist 席自主工时，承接 m16362 独立复核（PR #784）的 **F1 阻断性发现**。

## 1. 问题合同（F1 原文）

m16362 复核发现：证据页 `docs/sent-edge-crack-j-integral-validation.md` 把锚点工件写成
「`benchmarks/verified/fem_sent_j_integral/` (`result.json`, `gates_verified.json`, 4 figures)」，
但合并入 main 的**只有 `run.py`**（+ 本单前已补的 `_cache/ladder.json`、`_cache/control.json`、
`energy_crosscheck.py`、`energy_crosscheck.json`）。同目录兄弟锚点（`fem_hertz_contact`、
`fem_kirsch_plate`、`fem_winkler_plate` 等）都把 `result.json`+`gates_verified.json` 入库。
任何人按证据页去取工件都会 404，门禁判据（3 %/1 %/1e-9）在仓内**无对应工件可核**——
这是 L4 证据链的最低要求缺口。

## 2. 修复路径（本单）

`run.py --stage assemble` 需要四份缓存：`ladder`、`control`、`height`、`field`。
本单已具备 `ladder`（≈25 min 阶梯）与 `control`（非奇异控制组）；**缺 `height` 与 `field`**：

- `--stage height`：`sent_sweep` 中网格 × H/W∈(1.5,2,3) × a/W∈(0.2,0.35,0.5) = 9 次求解，
  喂 `height_independence` 门（F 对 H/W 相对散布 <1 %）与 `height_shape_factors`。
- `--stage field`：最细 a/W=0.2 网格的韧带 σ_yy 与质心场，喂 `sent_ligament_field.png`
  （r^(−1/2) 奇异衰减图）与 `field_summary`。

`assemble` 产出：`result.json`（含 evidence/cases/gci/ladder_errors/control/height/field_summary）、
`gates_verified.json`（7 门：k_error×3、path_independence、reaction_balance、height_independence、
gci_fine、edge_crack_discrimination、refinement_levels）、4 张图
（`sent_convergence.png`、`sent_shape_factor.png`、`sent_path_independence.png`、`sent_ligament_field.png`）。

## 3. 执行与容量纪律

本机 load15 ≈ 118（nproc 128，≥100），按容量纪律**不起新的重算力进程**；
`height`/`field`/`assemble` 三阶段（单进程、中网格、>120 s 外壳预算）已**递交集群**
（`run_cluster` 作业 **cj1361**，loong64 双节点，与 99 同架构）。集群跑完即取回工件并入库。

## 4. 已在本会话独立验证（不采信证据页数字）

- 单元测试面：`pytest tests/test_sent_edge_crack_j_integral.py -q` → **19 passed**（31.7 s）。
- `_cache/ladder.json` 与 `_cache/control.json` 已入库（m16362 补），`assemble` 可读。
- 门禁判据（`run.py` `build_gates`）：k_rel_error_max=0.03、path_spread_max=0.01、
  reaction_balance_max=1e-9、height_spread_max=0.01、gci_fine_max=0.05、
  discrimination_min_shape_factor=1.15、refinement_levels_min=3。

## 5. 未完成项（honest）

- `result.json` / `gates_verified.json` / 4 图：**待集群 cj1361 跑完取回**（本单结束时仍在队列，
  见 §3）。取回后按 `git add` 入库，证据页引用即与仓内工件对齐（F1 关闭）。
- 跨架构（x86 CI）复现：not_run（本机与集群均 loong64）。

## 6. 复现命令

```bash
cd <repo>
PYTHONPATH=src python -m pytest tests/test_sent_edge_crack_j_integral.py -q          # 19 passed
# 集群（已递交 cj1361）：
#   --stage height && --stage field && --stage assemble
# 取回后：git add benchmarks/verified/fem_sent_j_integral/{result.json,gates_verified.json,*.png}
```
