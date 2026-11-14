# 交差审计：m19614 · GAP-PP-1-ICE-STRUCT-CONTACT-PATCH（TensorFEM）

审计方：board-audit（本单 m19696）　审计方式：独立取证 + 复算，不重做实现
被审对象：m19614（`selfstart-tp-impl-5-1113-2257-rework1`，status=succeeded，finished 2026-11-14 01:03:15Z）
基准裁判：`tensor_benchmark_cases`（postgres `tensorplatform`），锚点 GAP-PP-1-ICE-STRUCT-CONTACT-PATCH

---

## 0. 结论摘要（先行）

| 项 | 裁定 |
|---|---|
| 三件套（run_artifacts / benchmark ref-vs-got / repro_cmd） | **全部不可取证**（0/3） |
| 基准裁判 | **锚点无基准行**：`SELECT ... WHERE anchor_slug='GAP-PP-1-ICE-STRUCT-CONTACT-PATCH'` → `(0 rows)` |
| 判级 | **L0**（m19614 自身功能实质 = 0；报告文本不改变级别） |
| 差异说明 | 该锚点的**能力实体**在仓内存在，但归属 m18657/m18930 分支（`agent/m18930`），非 m19614；且未合入 main ⇒ 不构成 m19614 的 L2/L3 |
| 处置 | **reopened**（返修：把实现/测试/工件落到可达 ref 并补基准行，否则该锚点不得再申报 ≥L2） |

---

## 1. 取证范围与独立性

* 独立读取源：`tensor_dev_missions`（19614、19482）、`tensor_reports`、`tensor_maturity_verdicts`、
  `tensor_benchmark_cases`（postgres）；git 对象库与 200 个 `agent/*` 分支引用；文件系统全盘检索。
* 独立复算：`agent/m18930` 分支的 `src/tensorfem/contact_patch_load.py` +
  `tests/test_contact_patch_load.py`（另一 mission 的同一锚点实现）—— 作为**旁证实现**复跑，
  用以区分「数字造假」与「链路断裂」。
* 独立性限制：m19614 的工作树 `/root/tp-agent/work/m19614` 已被平台回收，其原始运行日志不可得；
  `origin` 远端可达性在本沙箱内不可验证（`git ls-remote origin` 对本仓已存在分支 `agent/m18930`
  同样返回空且 exit=1，属网络受限 ⇒ **远端未推送结论记为 unknown，不作为判据**）。
  本地判据（对象库 + 本地 200 分支 + 本地镜像 `/root/tensor-ecosystem/TensorFEM`）自洽且互相印证。

## 2. 逐项核验（三件套）

### 2.1 run_artifacts —— 缺失（SEV-1）

m19614 result 自述 `on_disk_changes` 含：

```
artifacts/contact_patch_load/{study.json,stress_fields.json,peak_stress.png,pressure_fields.png,stress_contour.png}
```

实测：

* `git log --all -- artifacts/contact_patch_load/stress_fields.json benchmarks/run_contact_patch_study.py
  docs/CASE-FEM-CONTACT-PATCH-LOAD-COUPLING.md` → **空**（无任何 ref 含这些路径）；
* 全盘 `find / -name stress_fields.json -o -name peak_stress.png -o -name stress_contour.png` →
  **无** m19614 版本（仅命中无关的 `work/artifacts/stiffened_panel_ice_ultimate/stress_contour.png`）；
* `main` 无 `src/tensorfem/contact_patch_load.py`、无 `docs/CASE-FEM-CONTACT-PATCH-LOAD-COUPLING.md`。

⇒ 报告与 result 引用的工件**在仓内不存在**，无法核 ref-vs-got。

### 2.2 承诺的 commit / 分支 —— 不可达（SEV-1）

result headline 称「已 commit 826e860a（分支 agent/m19614）」。实测：

* `git cat-file -t 826e860a` → `fatal: Not a valid object name`；
* `git branch -a | grep 19614` → 无；本地镜像 `/root/tensor-ecosystem/TensorFEM` 同样无该分支；
* `git log --all --oneline --grep="19482\|19614"` → 空。

⇒ **commit 与分支均不存在**，m19614 的交付物没有任何可复算载体。这与返修单的初衷（补齐
m19482 被 gate 判 L1 的「impl_files/entry/tests 工件缺失」）**恰好同病**：返修轮 1 仍未把工件
落到可达 ref。

### 2.3 repro_cmd —— 不可执行（SEV-1）

result 自述复现命令：

```
PYTHONPATH=src python benchmarks/run_contact_patch_study.py
PYTHONPATH=src python -m tensorfem.cli contact-patch-load --contact-case hertz --mesh 24x21
PYTHONPATH=src pytest tests/test_contact_patch_load.py tests/test_ice_belt_collapse.py -q
```

在 `main`（本工作树基线 `9ad824f7`）逐条实测：

* `benchmarks/run_contact_patch_study.py` → 文件不存在（`ls` 失败）；
* `tensorfem.cli` 的 `contact-patch-load` 子命令 → `main` 的 `cli.py` 无此入口；
* `tests/test_contact_patch_load.py` → 不存在（**只有** `tests/test_ice_belt_collapse.py` 存在）。

⇒ 三条 repro_cmd 在基线树上**均不可执行**；自述「42 passed」无法复核
（`test_ice_belt_collapse.py` 存在但缺另一半，不能凑出 42）。

### 2.4 报告纪律 —— 未发布（SEV-2）

`SELECT id,title FROM tensor_reports WHERE mission_id=19614` → **0 行**。
m19614 仅在 result 里声称写了 `/tmp/report_m19614.json`（未 POST）；该文件亦已不存在
（`ls /tmp/report_m19614.json` 失败）。同锚点上一轮 m19482 的 report（id=17792）是存在的，
返修轮反而把报告发布这一步丢了。

## 3. 基准裁判（唯一裁判）

```
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='GAP-PP-1-ICE-STRUCT-CONTACT-PATCH';
 -> (0 rows)
```

* 全表 118 行 / 83 个锚点，**无** GAP-PP-1-ICE-STRUCT-CONTACT-PATCH。
* 最近邻：`H3-ICE-PLATE-FEM`（Hetényi 冰板弯曲，挠度/弯矩/破坏位置 vs 级数解，`<5%`，status=`pending`）
  —— 这是**母锚点**判据，不覆盖「非均匀斑 vs 均匀斑峰值应力」这一新量。
* 结论：**该锚点在基准裁判中无判据行**，因此按判据体系**任何 ≥L4 的申报都不可能成立**；
  m19482 报告里「+8.0%~+27.0%」这一核心量自认「无解析基准」，只用守恒 + 网格无关性替代。
  这一点原报告是诚实的，但其数值因此不具备 L4 的「基准验证」资格。

## 4. 旁证复算：能力实体存在，但不属于 m19614

`agent/m18930`（m18657 + m18930 两单）含完整实现：

* `src/tensorfem/contact_patch_load.py`（999 行，真实代码：`patch_from_contact`
  = `C^T(w p)` 一致投影 → `PlateModel` MITC4/Mindlin → `principal_moments` → σ=6|M|/t²；
  另含 `navier_series_reference` 任意载荷级数解与 `solve_contact_plate_coupling` 双向定点）；
* `tests/test_contact_patch_load.py`（360 行，14 个测试）——**本次实跑：14 passed in 88.81s**；
* `artifacts/contact_patch_load/{study.json,concentration_sweep.json,*.png}`。

三点审计读数（对 m19482/m19614 的数值主张做独立挑战）：

1. **同总力非均匀斑的放大方向成立，量级更小**：`concentration_sweep.json`（同总力、同足迹）
   12×24 / 24×48 网格：centre-peak 放大 1.0088/1.0090（amp0.5）、1.0151/1.0155（amp1.0）、
   1.0247/1.0241（amp2.0）；one-sided ramp 1.0899/1.0955（amp0.5）、1.1798/1.1910（amp1.0）。
   ⇒ 方向与 m19482「非均匀抬高峰值」一致，但**幅度对构型高度敏感**（+0.9% ~ +19%），
   因此 m19482 的「+8%~+27%」**不能当作可迁移常数修正因子**（该报告 §⑤ 亦如此自述）。
2. **绝对峰值应力网格敏感**：同表 `mesh_stability.absolute_sigma_peak_uniform` coarse/fine = **0.874**
   （≈12.6% 差异），而放大比 coarse/fine ≈ 1.0002（稳定）。⇒ 以绝对 MPa 作结论（67.60 MPa 等）
   在未做细化研究前不达 L4 强度；以**比值**作结论是稳的。
3. **口径不可跨实现直接比数**：m18930 的算例（clamped bay、4 MPa 级压强）与 m19482 的 JOUR 算例
   （四边简支 0.8×0.7×0.022 m、a=0.2 m、均值 0.3 MPa）不同，故**不能**用 1.025 去否证 +10.96%；
   只能证明「方向一致、幅度构型相关」。此为本次审计对 m19482 数字的唯一可辩护判断。

## 5. 判级推导（L0–L5）

| 级别 | 判据 | m19614 |
|---|---|---|
| L1 | 数学/单元验证 + tests 工件在仓 | ✗（tests 缺失） |
| L2 | 独立功能实现（真代码 输入→计算→输出）在仓 + 测试 | ✗（impl 缺失） |
| L3 | 接入主求解器/平台（有 caller） | ✗（main 无 import，无 CLI 入口） |
| L4 | 完整算例 + 误差分析 + 可复现 + 基准判据 | ✗（无基准行；repro 不可执行） |
| L5 | 科研工程成果 | ✗ |
| **L0** | 文档/文本成果 | ✓（仅 mission result 文本；连 doc 文件都未落仓） |

**裁定：L0。** 差异说明：锚点能力实体（真代码 + 14 项通过的测试）存在于 `agent/m18930`，
按功能实质可评 L2、且缺 caller 故不达 L3；但这属 m18657/m18930 的交付，**不是 m19614 的交付**。
仓内既有 `tensor_maturity_verdicts` id=794（scope `m19482 GAP-PP-1-...`, level 2, mission_id=19562）
指向同一条能力血缘，本审计不撤销该行，只声明 **m19614 自身不得据其继承级别**。

## 6. 发现清单（按严重度）

* **SEV-1 · blocking** 承诺 commit `826e860a`/分支 `agent/m19614` 不可达（对象库 + 200 本地分支 + 镜像三处皆无）。
* **SEV-1 · blocking** 五个 run_artifacts + 实现 + 测试 + 文档路径在任何 ref 均不存在；repro_cmd 三条全部不可执行。
* **SEV-1 · blocking** 基准裁判无该锚点判据行 ⇒ 涉该锚点的 ⩾L4 申报结构性不可能。
* **SEV-2** 报告未发布（`tensor_reports` 无 mission_id=19614 行），违反报告纪律。
* **SEV-2** 以 `fail_stage=budget_exhausted+forced_finish`（tool_calls=80）收尾，仍以「42 passed /
  67.60 MPa」等具体数字申报成功，属**未经复核即结构化自证**。
* **SEV-3 · non-blocking** 绝对峰值应力网格敏感（旁证 12.6%），报告以绝对 MPa 呈现放大结论易被误读为收敛值。

## 7. 未执行的检查（记 unknown，不记 pass）

* 远端 GitHub 是否曾有 `agent/m19614`（网络受限，`ls-remote` 对已知存在分支同样失败）——**unknown**。
* m19614 当时的原始 pytest/benchmark 输出与 `/tmp/report_m19614.json`——**not_run**（文件已被回收）。
* JOUR 算例上 67.60 MPa / +10.96% 的直接复算——**not_run**（实现不可达；已用旁证实现证明「链路断」而非「算错」）。
* 该锚点的外部实验/文献锚（ISO 19906 冰压斑放大系数）对标——**not_run**（基准裁判无行，无容差可依）。

## 8. 处置（disposition）

**reopened（返修）**。返修最小闭环：

1. 把 `contact_patch_load.py` + `tests/test_contact_patch_load.py` + 说明书合入一个**可达 ref**
   （最省力路径：直接采纳 `agent/m18930` 的实现，14 项测试已由本审计实跑通过）；
2. 由平台侧在 `tensor_benchmark_cases` 补一行锚点判据（建议：`navier_series` 均匀斑中心弯矩
   `M_max/(q a²)=0.0479`，tol `<3%`；以及「同总力 Hertz vs uniform 放大比」的 `ref=独立实现`、
   tol `<5%`），否则该锚点永不可能达 L4；
3. 报告按六要素发布（m19482 的 17792 可直接作为数值底稿，需补「比值而非绝对值」的网格细化表）。