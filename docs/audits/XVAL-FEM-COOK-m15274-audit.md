# XVAL-FEM-COOK 交差核验审计（被审单 m15274 / 审计单 m15312）

- 被审对象：`m15274` 「XVAL-FEM-COOK · Cook 翘曲膜三方横向（我方 vs 开源）」succeeded
- 被审工作树：`/root/tp-agent/work/m15274`，分支 `agent/m15274`，HEAD `1e701d1f`（≠ origin/main）
- 审计工作树：`/root/tp-agent/work/m15312`，HEAD `6567b330`（= `/root/tensor-ecosystem/TensorFEM` 检出）
- 裁判来源：`tensor_case_pool` id=50（`COOK-MEMBRANE`，`tol_policy='<10%'`，`linked_anchor='XVAL-FEM-COOK'`，status=pending）；
  `tensor_benchmark_cases` 对 `anchor_slug='XVAL-FEM-COOK'` **0 行**（见 D2）
- 审计类型：交差核验（非重做）。审计者未修改被审工作树任何文件。

---

## 0 结论（disposition）

**approve with notes — 级别 L4（基准验证），无阻断性物理错误；4 项必改（D1/D2/D4/D6），3 项登记（D3/D5/D7）。**

一句话：三件套**真实存在且可复现**（审计者在当前 main 上逐位复跑，90 个物理数字相对差 0.0）；主判据
（`u_y(48,52)` vs 锚点 23.96，n=16 误差 **2.2103%**）**经第三方独立内核复核成立**；但报告 §4 的
「平衡 ΣR_y 三方通过（偏差<1e-6）」是**与自表数字自相矛盾的假陈述**，其根因是 skfem 两方的
`reaction_y` 度量只累加了顶点自由度（漏掉 P2 单元的中边自由度）。该列已被审计者用全约束自由度重算为
**−1 ± 7e-13**，即结论（平衡成立）为真而印出的数字为假。

---

## 1 三件套取证（逐项，含 sha256）

| 交付物 | 事实 | 判定 |
|---|---|---|
| 实现 `benchmarks/cook_membrane_xval.py` | 517 行，sha256 `85362c8c…c8c41c`；真实 3 内核（TensorFEM 生产 Q4 / skfem ElementQuad2 / skfem ElementTriP2），无 placeholder | REAL |
| 文档 `docs/cook-membrane-xval.md` | 65 行，sha256 `3831c786…5f6159`，六段齐（问题/方法/参数/对照/收敛/失败模式） | REAL（含 1 处假陈述，见 D1） |
| run_artifacts `artifacts/cook_xval/metrics.json` | 12163 B，sha256 `612d6934…5cd557b`，合法 JSON，含 3 方 × n=8,16 + 探针 + env | REAL（有重复键，见 D5） |
| run_artifacts `artifacts/cook_xval/xval_convergence.png` | 56678 B，sha256 `5bfc3de0…2a2091`，PNG 896×588 RGBA，可解码 | REAL |
| repro_cmd 可执行性 | `PYTHONPATH=src python benchmarks/cook_membrane_xval.py --levels 8,16 --parties tensorfem_q4,skfem_quad2,skfem_tri2 --out …` 在审计树 `artifacts/audits/m15312-cook-xval/repro_copy/` 实跑 **rc=0** | REAL |
| 提交状态 | `git status` 显示上述 4 个交付物**全部未跟踪**（`??`），仅存在于工作树；`agent/m15274` 未推 origin | 见 D6 |

### 1.1 复现比对（审计者实跑 vs 归档）

`repro_copy/repro_out/metrics.json` 对比 `work/m15274/artifacts/cook_xval/metrics.json`：
97 个可比节点中 **90 个数值节点相对差 0.0（逐位一致）**，仅 6 个 `wall_s` 计时与 1 个 `parties` 列表不同。
复跑使用**当前 main 的 `src/`**（`m15312` HEAD 6567b330），即生产 Q4 路径在 main 上稳定。

---

## 2 基准裁判（machine referee）

| 判据源 | 内容 | 判定 |
|---|---|---|
| `tensor_benchmark_cases WHERE anchor_slug='XVAL-FEM-COOK'` | **0 行** — 总账判据空转 | 缺口 D2 |
| `tensor_case_pool` id=50（`COOK-MEMBRANE`，`linked_anchor='XVAL-FEM-COOK'`） | `tol_policy='<10%'` | 事实判据 |
| 报告自声明判据 | 挠度 <3%（严于总账 <10%） | 保守 |

逐项（审计者复算，数字取自 `artifacts/audits/m15312-cook-xval/corrected_table.json`）：

| 参与方 | n | ref | got `u_y(48,52)` | 误差 | <3% | <10%（pool） |
|---|---:|---:|---:|---:|:--:|:--:|
| tensorfem_q4 | 8 | 23.96 | 22.079183 | 7.8498% | ✗ | ✓ |
| **tensorfem_q4** | **16** | 23.96 | **23.430411** | **2.2103%** | **✓** | **✓** |
| skfem_quad2 | 8 | 23.96 | 23.925405 | 0.1444% | ✓ | ✓ |
| skfem_quad2 | 16 | 23.96 | 23.949409 | 0.0442% | ✓ | ✓ |
| skfem_tri2 | 8 | 23.96 | 23.860706 | 0.4144% | ✓ | ✓ |
| skfem_tri2 | 16 | 23.96 | 23.927125 | 0.1372% | ✓ | ✓ |

**锚点身份（审计者独立裁定，被审单未做）**：锚点 23.96 究竟属于加载边中点 (48,52) 还是角点 (48,60)？
审计者用第四套独立实现（自写双线性映射 + skfem P2，n=8…64）裁定：

| 探针点 | Quad2 n=64 | TriP2 n=64 | 对 23.96 误差 | 结论 |
|---|---:|---:|---:|---|
| 中点 (48,52) | 23.965040 | 23.961786 | 0.021% / 0.007% | **锚点=中点，映射正确** |
| 角点 (48,60) | 25.163954 | 25.152089 | 5.02% / 4.98% | 锚点不是角点 |

Richardson 外推：中点 23.9665 / 23.9650；角点 25.1730 / 25.1660。⇒ 被审单把 got 与 23.96 在中点比较是
**正确的**（若误比角点，误差会变成 5% 量级）。

### 2.1 被审单「踩坑」结论的独立复核

报告 §2 声称：直接用 skfem `lame_parameters(E,ν)`（三维常数）会静默变成平面应变，n=8 给出 21.45（低 10.5%）。
审计者独立复算：

```
plane_stress_lambda_star   uy(48,52)=23.925405  err_vs_23.96=0.144%
3d_lame_plane_strain       uy(48,52)=21.452602  err_vs_23.96=10.465%
```

数值与报告一致（21.45 / 10.5%）⇒ 该「规格传递」发现是**真的、可复现的**，是本单最有价值的产出。

---

## 3 发现（按严重度排序）

### D1 [严重·必修] 平衡列度量错误 + 报告假陈述（与自表数字自相矛盾）

- 事实：报告 §4 印出 `Q2 -0.423109；P2 -0.569671`，紧接着断言「三方均为**通过**（相对 −1 偏差 <1e-6）」
  —— 0.42 与 0.57 相对 −1 的偏差是 **58% / 43%**，不可能 <1e-6。
- 根因（代码级）：`benchmarks/cook_membrane_xval.py:356`
  `residual[basis.nodal_dofs[1, left_nodes]].sum()` 只累加**顶点**自由度；`ElementQuad2/ElementTriP2`
  左边界还有中边自由度承载反力，故部分和偏小。审计者用 `basis.get_dofs(...).all()` 复算：
  - quad2 n=16：全约束 −1.0000000000007236（误差 7.2e-13）；仅顶点 −0.390049881（= 被审单 n=16 值）
  - tri2 n=16：全约束 −1.0000000000001759；仅顶点 −0.467722395（= 被审单 n=16 值）
  - n=8：仅顶点 −0.423109（quad）/ −0.569671（tri）——与被审单印出值**逐位一致**，机制确认。
- 另一层：Q4 的 `reaction_y = -1.0000000000000313` 并非独立校核，而是恒等式（自由自由度残差为 0
  ⇒ 约束自由度残差和 ≡ −Σf ≡ −1）。被审单把它当「平衡校核通过」属**判据自证**。
- 影响面：仅 §4 的一列与该句判词；**主判据与所有挠度/能量数字不受影响**。审计者已将真值登记入
  `corrected_table.json`（`reaction_y_auditor`）。
- 修法（一行）：`benchmarks/cook_membrane_xval.py:356` →
  `reaction_y = float(residual[constrained_dofs].sum())`，`constrained_dofs = basis.get_dofs(lambda x: np.abs(x[0]) < 1e-9).all()`；
  Q4 侧应改报「与独立施加的外载总量比对」或删除该列。

### D2 [中·必修] 总账判据空转：`tensor_benchmark_cases` 无 XVAL-FEM-COOK 行

`SELECT count(*) … WHERE anchor_slug='XVAL-FEM-COOK'` = 0（全表 14 行）。任务定义的「唯一裁判」为空，
本审计改用 `tensor_case_pool` id=50（tol `<10%`）作事实判据。建议登记一行
`('XVAL-FEM-COOK','Cook 翘曲膜（加载边中点挠度）','Simo & Rifai (1990); 收敛值 23.965（审计者 P2 n=64）','u_y(48,52)','<3%','TensorFEM')`。

### D3 [中·登记] 「三方」的独立性弱于字面

skfem Quad2 与 TriP2 是**同一库的两种单元**，不是两个独立代码库。⇒ 两方一致（0.14% / 0.41%）
证明的是**规格传递与装配正确**，不能作为「跨独立实现」证据。被审单声明的 `tensorfem_q8`、`fealpy` 两方
未跑通（**已如实声明**，不扣分），故本单实际的「我方 vs 外部」只有 1 个外部库。

### D4 [中·必修] 目标③未交付：`oss_solutions` 无 XVAL-FEM-COOK 沉淀行

m15274 goal 明确要求「方法选型+迁移方案落 `oss_solutions`（`feature_slug='XVAL-FEM-COOK'`）」。
`SELECT count(*) FROM oss_solutions WHERE feature_slug ILIKE '%COOK%'` = **0**（全表 233 个 slug 无 COOK）。

### D5 [低·登记] `metrics.json` 自描述不一致

`tensorfem_q4` 下 `"8"` 键出现两次（重复值相同，末值胜出，解析无歧义）；`parties` 列表含 `tensorfem_q8`
但 `results` 无其条目。复跑产物（审计者）只有 3 方 ⇒ 归档文件是多次运行合并的残留。

### D6 [低·必修] 交付物未提交，工件易失

`git status --short` 在 `work/m15274`：`?? benchmarks/cook_membrane_xval.py`、`?? docs/cook-membrane-xval.md`、
`?? artifacts/cook_xval/`、`?? scratch/…` 全部未跟踪；`agent/m15274` 未推 origin。本审计的全部指针
目前只在本地工作树可解析。

### D7 [低·登记] 分辨率不足以给收敛阶

仅 n=8/16 两点，无阶估计（被审单已如实说明）。审计者把外部两支补到 n=64（`independent_check.json`）：
quad2 误差 0.1444%→0.0442%→0.0032%→0.0210%，tri2 0.4144%→0.1372%→0.0326%→0.0075%（n=8/16/32/64），
n=8→16 段 quad2 拟合阶约 1.70、tri2 约 1.59（两点拟合，仅作趋势，不作判据）。

---

## 4 判级（L0–L5）

| 级 | 判据（maturity_gate SCHEME） | 本单是否满足 | 证据 |
|---|---|---|---|
| L1 | 测试 | 部分：本单**未加测试**；被验证路径有既有测试 `tests/test_continuum_benchmarks.py:33,38` | — |
| L2 | impl_files + entry + run_artifacts | ✓ | 517 行真实现；`main()` CLI；metrics.json/png |
| L3 | + integration_callers | ✓（**限定**：xval 脚本自身 0 调用方；被验证的生产路径有主求解器调用方 `additional_benchmark_fields.py:88`、`benchmark_registry.py:34`） | grep 复核 |
| L4 | + benchmark(ref vs got) + repro_cmd | ✓ | 2.2103% vs 23.96；审计者逐位复跑 rc=0 |
| L5 | + external_anchor | ✗ 未主张 | 文献仅仓内二手引用（Simo & Rifai 1990），未取原文数字化；无外网 |

**裁定：L4。** 理由=功能实质：完整 BVP（几何/材料/载荷/边界逐项独立复算）+ 与发表锚点的误差分析
+ 可复现（逐位）+ 第三方内核横向。级别不因文档篇幅或 vM 峰值未达标而改变（被审单已把 vM 登记为
honest-部分达标，这是正确做法）。

**级别敏感性（如实声明）**：若仲裁认为「结果表出现一列错误数字 + 一句假判词」足以否定 L4 的证据资格，
则应降 L3（L3 全部要件亦为真）。审计者不取此口径，因：①主判据与全部挠度/能量数字经逐位复现与第四套
独立内核实算成立；②D1 的真值已由审计者重算登记（−1±7e-13），账上不留假数；③错误列不参与达标判定。

---

## 5 审计者做了什么 / 没做什么

**做了**（全部可复跑）：

1. 三件套存在性与 sha256；`metrics.json` 合法性；PNG 解码。
2. 逐位复跑 repro_cmd（当前 main 的 `src/`）：`repro_copy/`，90 个物理数值相对差 0.0。
3. 自写第四套独立实现（`independent_check.py`，不 import 被审脚本）：双线性映射 + skfem P2，
   n=8/16/32/64 × {quad,tri}，判定锚点所属探针点；计算全约束反力与仅顶点反力（机制归因）。
4. 复核「三维 Lamé→平面应变」踩坑数值（21.452602 / 10.465%）。
5. 判据表：`tensor_benchmark_cases`（0 行）/ `tensor_case_pool` id=50 / `oss_solutions`（0 行）。
6. 生成审计图 2 张 + 校正表（`gen_figures.py` → `figs/`、`corrected_table.json`）。

**没做 / 局限**：

- 无外网：Simo & Rifai (1990) 原文图表未核（锚点值由收敛性裁定，非原文核对）。
- 未重做被审单：未跑 `tensorfem_q8` / `fealpy` 两方（被审单已声明未通）。
- 未改动被审工作树；D1 的修法以文字+行号给出，未代改（避免与被审单交付物混淆）。
- 未评 vM 峰值三方门限（被审单已声明不作判据，审计者同意：加载角点奇异，峰值随网格单调上升）。
- 级差敏感性未消解（见 §4），若第二位审计者判 L2 则触发 disputed。

---

## 6 复跑命令

```bash
# 1) 三件套逐位复现（审计者已跑，rc=0）
cd /root/tp-agent/work/m15312/artifacts/audits/m15312-cook-xval/repro_copy
PYTHONPATH=/root/tp-agent/work/m15312/src:/root/tp-agent/work/m15274/scratch/pylibs \
python3 cook_membrane_xval.py --levels 8,16 \
  --parties tensorfem_q4,skfem_quad2,skfem_tri2 --out repro_out

# 2) 审计者独立内核 + 锚点身份裁定 + 反力机制归因
cd /root/tp-agent/work/m15312/artifacts/audits/m15312-cook-xval
PYTHONPATH=/root/tp-agent/work/m15274/scratch/pylibs python3 independent_check.py 8,16,32,64 > independent_check.json

# 3) 校正表 + 图
PYTHONPATH=/root/tp-agent/work/m15274/scratch/pylibs python3 gen_figures.py
```

审计环境：python 3.11.6 / numpy 2.4.6 / scikit-fem 12.0.2（工作树外 `scratch/pylibs`）/ loongarch64。