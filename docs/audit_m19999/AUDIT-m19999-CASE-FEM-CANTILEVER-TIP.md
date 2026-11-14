# 交差审计 · m19999 · CASE-FEM-CANTILEVER-TIP（TensorFEM）

- 被审对象：mission m19999，分支 `agent/m19999`，提交 `5504271e`（工作树 `/root/tp-agent/work/m19999`，HEAD 即该提交，`git status` 干净）
- 审计工作树：`/root/tp-agent/work/m20061`（分支 `agent/m20061`）
- verdict 行：`tensor_maturity_verdicts`（repo=TensorFEM, scope=`m19999 CASE-FEM-CANTILEVER-TIP`, verdict_by=`board-audit`）
- 判级：**L3（系统集成）**；L4 扣留（见 §5 F1，登记缺位），L5 拒绝（教材级单锚点验证，非科研工程成果）

## 0. 独立性说明

本审计不采信被审报告的自述数字，凡数字均取自下列之一：

1. 被审工件本体（`docs/audit_m19999/audited_case/`，与被审工作树文件 **byte-identical**，sha256 见 `audited_case/SHA256SUMS.txt`）；
2. 本审计自己的工作树里用 **main 的 `src/tensorfem`** 重跑被审脚本副本（`artifacts/m19999_audit/repro/`，sha256 与被审文件逐一相同）；
3. 本审计自己从第一性原理重推的解析量（`artifacts/m19999_audit/recheck_analytic.py`，**不 import** 被审脚本、不 import tensorfem）；
4. 平台库真实行（`tensorbench_results` / `tensor_case_pool` / `tensor_benchmark_cases` / `tensor_reports`，psql 直查）。

局限：被审报告里 `--write-docs`（文档页 + 三张 PNG 的重新生成）与**全量** `run_study`（含 32x2x2 稠密档与 64x4x4 档，本机单次 >120 s）未在本机完成；全量 workload 已递交集群 cj2457，审计结论不以该作业为条件（见 §4 not_run）。

## 1. 三件套核验（工件 / 基准数值 / 复现命令）

| 项 | 判据 | 实测 | 结论 |
|---|---|---|---|
| run_artifacts 文件 | 工件真实存在且已入库 | `benchmarks/case_fem_cantilever_tip_evidence.json`（400 行，sha256 `68816c13…`）、`docs/assets/case-fem-cantilever-tip/{convergence,levels,field_uz}.png`（66 688 / 51 650 / 30 007 B，sha256 见清单）——均由提交 `5504271e` 引入，`git ls-files` 追踪 | ✅ 存在 |
| 证据文档自洽 | `verdict` 段算术自洽 | 工件 `got=9.964045336499231e-03`、`ref=9.999999999999998e-03`、`relative_error=-0.3595466%`、`tolerance_rel=0.02`、`passed=true`；审计重算 `(got-ref)/ref = -0.359547%`，`|·|<0.02` → True | ✅ 自洽 |
| 基准数值（裁判表） | `tensor_benchmark_cases WHERE anchor_slug='CASE-FEM-CANTILEVER-TIP'` | **0 行**（全表 121 行；TensorFEM 仅 7 行：clamped-plate / J-integral / cohesive3d / elastoplastic / winkler / ice-plate / TPB1976，无极值梁端挠度条目） | ❌ **无登记裁判**（F1） |
| 基准数值（替代裁判） | 平台机器判定行 | `tensorbench_results WHERE mission_id=19999` → 2 行：`tip_deflection_solid_reference_m` ref=1.000000000e-02 got=9.964047018716e-03 rel=+0.359530% tol=0.02 **passed=t**（job 2435, run `CASE-FEM-CANTILEVER-TIP-20261114T081205Z`）；`tip_deflection_beam_eb_m` rel=+1.3e-12 tol=0.02 passed=t | ✅ 机器判定 PASS（但不在任务指定裁判表内） |
| 容差出处 | 预声明容差可追溯 | `tensor_case_pool.case_code='FEM-CANTILEVER-TIP'`：`tol_policy='<2%'`，`ref_source='解析 PL³/3EI'`，status `dispatched`；被审脚本模块常量 `TOLERANCE=0.02`（`benchmarks/case_fem_cantilever_tip.py:113`） | ✅ 与 case-pool 一致 |
| repro_cmd 可执行 | 命令真能跑且能复现 | (a) `PYTHONPATH=src python3 <copy>/benchmarks/case_fem_cantilever_tip.py --quick` → 与工件 13 位有效数字逐位一致（8x1x1 `9.748606755302e-03`、16x2x2 `9.913672415396e-03`）；(b) 参考网格单档复算 `solid_level(32,4)` → `9.964045336499231e-03`，与工件 **相对差 0.0**（`artifacts/m19999_audit/verify_reference_mesh.json`）；(c) `pytest -q <copy>/tests/test_case_fem_cantilever_tip.py` → **9 passed in 16.83 s** | ✅ 可执行、可复现（全量 `--json` 全档见 §4） |
| 报告六段 | `tensor_reports.id=18246`（mission 19999） | ①案例介绍 ②方法与实现 ③计算结果（程序化注入）④结果图 ⑤收敛性/敏感性 ⑥注意点/失败模式 ⑦误差判定 —— 六要素齐 + 显式「是否 <2%：是」；**但** `figures` 三条只有 `name`/`caption`，**无 `png_base64`**（同期其它报告如 id=17486 有 224 768 B base64） | ⚠️ 六段齐、图未内嵌（F2） |

## 2. 基准裁判逐项核对（唯一裁判视角）

任务指定的裁判表 `tensor_benchmark_cases` 对 `anchor_slug='CASE-FEM-CANTILEVER-TIP'` **无行**，故不存在平台登记的 `metric`/`tolerance` 可对照。作为替代，本审计用两路独立证据判定数值真伪与容差合规：

1. **解析锚自推**（`recheck_analytic.py`，零依赖被审代码）：I=bh³/12=8.333333333333335e-10 m⁴、EI=166.6667 N·m²、δ_EB=PL³/3EI=**9.999999999999998e-03 m**、
   κAG=6.410256e+06 N、剪切项 PL/(κAG)=**7.800000e-07 m**（占弯曲 0.0078%）、δ_T=**1.000078000000000e-02 m**；
   另用 20 001 点复合 Simpson 对弯矩场/剪力场独立积分，与闭式解相对差 **+3.47e-16**（弯曲）、**-7.87e-15**（剪切）。
   被审工件 `reference` 段与之一致到浮点末位（rel=0.0）。
2. **平台 ingest 行算术**：`rel_error=0.003595298128370936` 与审计重算 `(9.964047018716e-03-1e-2)/1e-2=+0.359530%` 一致；`0.0035953 < 0.02` → PASS 判定成立。

误差判定（关键量 vs 解析基准）：

| 关键量 | 基准 | got | 误差 | <2%（case-pool） | <3%（生态线） |
|---|---:|---:|---:|:--:|:--:|
| HEX20 参考网格 32x4x4 端挠度（判据行） | 1.000000e-02 m | 9.964045e-03 m | **-0.3595%** | 是 | 是 |
| HEX20 最粗 8x1x1 | 1.000000e-02 m | 9.748607e-03 m | -2.5139% | 否（工件亦标 ❌，诚实） | 是 |
| EB 梁 FE n=16 | 1.000000e-02 m | 1.0000000000013e-02 m | +1.30e-10% | 是 | 是 |
| Timoshenko 梁 FE n=8 | 1.000078e-02 m | 1.0000779999997e-02 m | -3.2e-14% | 是 | 是 |

## 3. 实现实质审查（是否真算，而非硬编码）

- `benchmarks/case_fem_cantilever_tip.py:203 beam_eb_level / :233 beam_timoshenko_level / :258 solid_level` 均调用 `tensorfem.frame2d`/`hex20` 求解器（`:108-110` import），端挠度取自解向量 `result.displacement[...]`（`:138, :150`），**无任何硬编码结果**；
- `:171 castigliano_integral` 独立重导解析锚（Simpson 20001 点），`:320 run_study` 组装 11 档阶梯 + 细长比扫描 + 载荷线性 + 6 项 `checks`；
- 反证：本审计在**另一个工作树的 main 求解器**上重跑副本，参考网格得 `9.964045336499231e-03`（与工件相对差 0.0）——若结果为硬编码，跨工作树不会逐位一致且与 32x2x2→64x4x4 单调收敛序列自洽（-0.4015% → -0.3595% → -0.1810%）。
- 集成面：workload 经 TensorBench 管道（`tensorbench_results.spec_sha256/result_sha256/stdout_sha256` + `cluster_job_id=2435`）落在平台库；`grep -rn case_fem_cantilever_tip` 除 `tests/test_case_fem_cantilever_tip.py`（importlib 按路径加载）与 docs 外无其它调用方。

## 4. 严重度排序发现

**F1（blocking for L4，medium-high）— 登记裁判缺位。** `tensor_benchmark_cases` 无 `CASE-FEM-CANTILEVER-TIP` 行（0/121）。任务指定裁判是唯一裁判，因此本算例的 PASS 属"工件自证 + 自 ingest"，不是"平台登记基准判定"。复现：
`psql -c "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FEM-CANTILEVER-TIP'"` → `(0 rows)`。
返修：登记 `(anchor_slug='CASE-FEM-CANTILEVER-TIP', case_name='Cantilever tip deflection, EB beam, L/h=100', ref_source='PL^3/(3EI) (Timoshenko 1955 ch.4)', metric='tip_deflection_solid_reference_m', tolerance='<2%', repo='TensorFEM')`。

**F2（medium，经对照实验改判为平台侧）— 报告图未内嵌。** `tensor_reports.id=18246` 六段齐全（含显式「是否<2%：是」），但 `figures` 三条只有 name/caption、无 `png_base64`。**对照实验**：本审计把自己 99 700 B + 74 684 B 的 base64 随 `POST /tensor/reports` 递交（id=18284），落库后 `figures->0 ? 'png_base64'` 仍为 **false**；而旧报告 id=17486 有 224 768 B。⇒ 当前 ingest 丢弃图载荷，**非 m19999 交付缺陷**，返修责任转平台服务。

**F3（low-medium）— 证据 provenance 不自洽。** 工件 `implementation.revision = 9ad824f7…`（= 提交 `5504271e` 的父提交），该 revision **不含** `benchmarks/case_fem_cantilever_tip.py`（该文件由 `5504271e` 引入）；平台行 `implementation_revision='unknown'`。即"从记录 revision 复现"不成立。返修：工件写 `5504271e`，ingest 时带真实 revision。

**F4（low）— 跨运行末位漂移未声明。** 工件 `got=9.964045336499231e-03` vs 平台 ingest 行 `9.964047018716e-03`，相对差 1.7e-9（两者均 <2% 通过）。可能为 SuperLU 线程/排序差异；报告未提"逐位不可复现、容差内可复现"。本审计在 main 上复算得与**工件**逐位一致，说明漂移来自 cj2435 那次运行环境。

**F5（low，历史遗留，不属 m19999 交付）— 陈旧工件指针。** `benchmarks/cantilever_tip_artifacts/results.json`（m15330/m17033 时期）`artifacts.convergence_png/comparison_png` 指向 `benchmarks/cantilever_tip_artifacts/{convergence,comparison}.png`，该两文件在库中**不存在**（`git ls-files` 无、磁盘无）；其 `repro_cmd.sh` 调的是旧脚本 `benchmarks/cantilever_tip_deflection.py` 而非本锚点脚本。建议后续清理，避免被当作 m19999 的 run_artifacts。

**非缺陷观察：** `agent/m19999` 提交 `5504271e` 尚未并入 main（`git log --all` 中仅 `agent/m19999` 包含之），符合"验收后平台自动推送"的流程，不计为缺陷。

## 5. 判级依据

- **L2（通过）**：真代码 input→compute→output；`--quick`/参考网格跨工作树逐位复现；9/9 pytest 绿。
- **L3（通过）**：接入平台 TensorBench 管道（workload_id + spec/result/stdout sha256 + cluster job 2435 → `tensorbench_results` 2 行机器判定），并复用 main 的 `frame2d`/`hex20` 求解器；测试与文档成套入库。
- **L4（扣留）**：算例完整性、误差分析、可复现性在实质上已达 L4 水准（11 档 h 收敛 + 细长比扫描 + 载荷线性 + 0.3595% 判据行 + 逐位可复现），但裁判登记缺位（F1）+ provenance 不自洽（F3）使"平台侧可复现、可判定"这一 L4 要件不成立。与先例 m19997（0 referee rows → L4 refused）保持一致。
- **L5（拒绝）**：单锚点教材级解析基准验证，非科研工程成果。

处置（disposition）：**request changes（小返修，reopened）**——F1 登记裁判行、F2 重发内嵌图报告、F3 更正 revision；F4/F5 记入建议。返修后按同一判据可直接确认 L4。

## 6. 复现与工件清单（本工作树）

```
docs/audit_m19999/AUDIT-m19999-CASE-FEM-CANTILEVER-TIP.md   # 本文
docs/audit_m19999/audited_case/                             # 被审工件副本（byte-identical, SHA256SUMS.txt）
docs/audit_m19999/make_figures.py, figs/fig1_convergence.png, figs/fig2_got_vs_ref.png
artifacts/m19999_audit/recheck_analytic.py / .json          # 解析锚独立重推
artifacts/m19999_audit/verify_reference_mesh.py / .json     # 参考网格 + 1-D 档独立复算
artifacts/m19999_audit/repro/{benchmarks,tests}/            # 被审脚本/测试副本（sha256 与被审一致）
artifacts/m19999_audit/rerun*.json|log                      # 全量 workload 重跑（cj2457）
```

审计侧执行过的命令：

```bash
python3 artifacts/m19999_audit/recheck_analytic.py                                    # 解析锚自推
OMP_NUM_THREADS=4 PYTHONPATH=$PWD/src python3 artifacts/m19999_audit/verify_reference_mesh.py
PYTHONPATH=$PWD/src python3 -m pytest -q artifacts/m19999_audit/repro/tests/test_case_fem_cantilever_tip.py
python3 docs/audit_m19999/make_figures.py
```