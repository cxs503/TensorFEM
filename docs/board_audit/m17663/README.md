# m17663 · 交差审计：m17604（XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE · TensorFEM）

**结论：claim = L2，实测支持 L2；L3/L4 均不成立。处置 = request changes（原单不消账到 L4，但功能实质已到独立实现级）。**

| 项 | 值 |
|---|---|
| 被审单 / 树 | m17604 · `/root/tp-agent/work/m17604`（只读；执行全在 `.scratch/m17604_repro` 副本） |
| 被审分支 | `agent/m17604` == `HEAD` == `a86365e7`，**工作树 7 项改动 0 提交**（`in_main=false`） |
| 上游审计 | verdict #582（m17488，scope `m17412 …`，level 2 → gate L1 → reopened 出本单） |
| 基准裁判 | `tensor_benchmark_cases WHERE anchor_slug='XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE'` → **0 rows** |
| 本单 verdict | `tensor_maturity_verdicts`（repo=TensorFEM, scope=`m17604 XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE`, mission_id=17663, **level=2**, verdict_by=board-audit） |
| 工件 | `verdict_evidence.json`、`reproduced_numbers.json`、`fig_xval_audit.png`、`make_audit_figure.py` |

---

## ① 案例介绍（物理背景 / 几何 / 参数）

被审对象不是一个求解器，而是**一个「方法选型记录 vs 独立实现」的交叉验证 harness**：

- 物理算例 `CASE-HOLLOW-SPHERE-3D`：内压厚壁球壳（Lamé 问题）。几何 a=8 m, b=10 m（壁比 b/a=1.25）；
  材料 E=210 GPa, ν=0.3（λ=121.15 GPa, μ=80.77 GPa）；载荷 p=1 MPa 内压；外表面 σ_r=0。
- 解析锚点（Timoshenko & Goodier 厚壁球解）：C₁=p·a³/(b³−a³)=**1.04918 MPa**、C₂=p·a³b³/(2(b³−a³))=**524.59 MPa·m³**；
  σ_r(r)=C₁−2C₂/r³，σ_t(r)=C₁+C₂/r³ ⇒ σ_r(8)=−1.000 MPa、σ_t(8)=2.0738 MPa、σ_r(10)=0、σ_t(10)=1.5738 MPa、
  von Mises(8)=2.7156 MPa。
- 被验证的记录（`oss_solutions` id=400，repo=TensorFEM，tag=`method-selection+first-evidence`）声明 6 条：
  C1 HEX20(n=4) 全部探针 <3%；C2 HEX8 同网格 worst 15.4%；C3 HEX8 需 n≥12 才逼近 5%；
  C4 体积误差 7.1%→0.14%；C5 闭合面合力/矩 <1e-16；C6 HEX8 n=8 PCG 67 迭代。
- 本单交付（m17604）要证明的东西：用**独立成熟 OSS 实现 scikit-fem 12.0.2（TET4/TET10 icosphere 壳）**
  复算同一算例，并把「记录里的数」逐条复现 / 证伪（M1=形心半径采样，M2=精确层积分，两种泛函并列）。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **只读取证**：`git status/ls-files/log`、`git ls-tree main` → 交付是否入库；`tensor_dev_missions.result`（m17604 原文）。
2. **干净副本执行**：`cp -r src benchmarks tests scripts` → `/root/tp-agent/work/m17663/.scratch/m17604_repro`；
   被审树零写入。因仓内 `pyproject.toml` 的 `pythonpath=["src"]` 会污染 rootdir，测试用
   `--rootdir=$PWD -o pythonpath=$PWD/src` 强制指向副本（否则 editable 安装的 m17663/src 抢先命中）。
3. **测试复核**：`pytest tests/test_xval_method_selection.py -q` → **17 passed / 108.43 s**（被审单自报 17 passed/68.39 s，量级一致）。
4. **入口复核**：`xval.run(quick=True)` 实跑 → 崩；benchmark `--quick` 同崩；`make_figures()` 单独喂真实行 → 崩（见 ⑤）。
5. **基准裁判**：SQL 直查 `tensor_benchmark_cases` / `tensor_reports`（不引用被审报告任何自述数字）。
6. **数值复现（本审计自己的数）**：按交付的同一 ladder 逐档调用 `arm_tensorfem` / `arm_skfem`，
   用交付的 `reproduce_claims()` / `decision_table()` 判，共 **6 档**：HEX8 n=4/6、HEX20 n=4、TET4 n=3、
   TET10 n=3（投影/非投影）。原始数字落 `reproduced_numbers.json`（含每探针 ref/got 对照）。
7. **出图**：`make_audit_figure.py` 从 `reproduced_numbers.json` + 行数据程序化生成 `fig_xval_audit.png`（零手抄）。

未用集群（`cj1642` 递交时排队 362 位，收工时仍 queued）：HEX8 n=8/n=12、HEX20 n=6 三档本审计未复测。

## ③ 计算结果（数字表）

**A. 逐档 worst 探针误差（本审计复算；M1/M2 = 两种泛函）**

| 臂 | 单元 | n | dofs | M1 worst | M2 worst | 体积误差 | 与 m17604 自报 |
|---|---|---|---|---|---|---|---|
| TensorFEM | HEX8 | 4 | 882 | **28.182%** | 6.460% | 7.116% | 一致（28.182） |
| TensorFEM | HEX8 | 6 | 1962 | **15.350%** | 4.104% | 3.284% | 一致（15.350） |
| TensorFEM | HEX20 | 4 | 3198 | **2.875%** | **18.186%** | **0.1359%** | 一致（2.875/18.186） |
| scikit-fem | TET4 | 3 | 828 | **24.620%** | 10.599% | — | 一致（24.620） |
| scikit-fem | TET10 投影 | 3 | 6150 | **10.237%** | 9.163% | — | 一致（10.237） |
| scikit-fem | TET10 不投影 | 3 | 6150 | **19.130%** | 6.883% | — | 未自报（本审计补测） |

**B. 记录 6 条 claim 的复现判定（本审计复算）**

| claim | 判定规则（预注册） | recorded | got | reproduced |
|---|---|---|---|---|
| C1 HEX20 n=4 全探针 <3% | 阈值 | 3% | 2.875% (M1) / **18.186% (M2)** | M1 True / **M2 False** |
| C2a HEX8 同网格 worst 15.4% | 数值 ≤15% | 15.35% | **28.182%**（n=4 同网格） | **False** |
| C2b 15.35% 属 n=6 | 数值 ≤15% | 15.35% | 15.350% | True |
| C3 HEX8 n≥12 逼近 5% | 阈值 | 5% | 15.350%（本审计只到 n=6） | False（未复测中段；m17488 已测 n=12=5.685%） |
| C4a 体积误差 HEX8 | 数值 ≤15% | 7.1% | 7.116% | True |
| C4b 体积误差 HEX20 | 数值 ≤15% | 0.14% | 0.1359% | True |
| C5 闭合面力/矩 <1e-16 | 严格 < | 1e-16 | **0.0**（HEX20 n=4）/ 6.87e-17（HEX8 n=4） | 规则上 False（见 ⑤ m1） |

**C. 预注册决策表（交付 `decision_table` 原样）**：
`quadratic_under_3pct_m1=True`、`quadratic_under_3pct_m2=False`、`ranking_m2_reproduced=False`、
`functional_dependence_gap=0.1531`（HEX20 n=4 的 M2−M1）、**`selection_robust=False`**。

⇒ 选型记录「二次单元全部探针 <3%」**只在 M1 泛函下成立**；换成记录自己声明的
`reference_functional=lame_layer_average`（M2）反而 **18.19%**。跨实现稳健性同样不成立
（skfem TET10 投影 10.24% > 3%；TET4 24.62%）。

## ④ 结果图

`fig_xval_audit.png`（三面板，程序化生成）：
左 = M1 口径 worst error vs dofs（四臂 + 3%/5% 判据线）；中 = M2 口径同轴；
右 = 场量校核：HEX20 n=4 单元 von Mises 值沿半径 vs Lamé 闭式曲线（σ_r、von Mises）。
被审单自带的 `fig_xval_convergence.png` / `fig_xval_probe_split.png` 存在，但因 `evidence.json` 从未落盘，
**无法由数字追溯**，本审计不采信其数值，只采信自己复算的数。

## ⑤ 注意点（缺陷 / 局限 / 适用范围）

**被审单确实修好了 m17488 的两条 blocking**：F1（`dev` 被 float 重绑定 → claim 表 TypeError）已修，
`reproduce_claims()` 正常产出 7 条判定；F2（skfem `arm_skfem` TET10 IndexError）已修，TET10 投影/非投影两档都出数。

**但引入两条新的 blocking（本审计各复现 ≥2 次）**：

| id | 级 | 缺陷 | 证据 |
|---|---|---|---|
| B1 | blocking | `make_figures()` 读 `row["fields_radial_stress"]`，而 `arm_tensorfem` 只存 `fields_radius / fields_von_mises / fields_centres` ⇒ **KeyError 死在图 3**；而 `evidence.json` 写盘在 `make_figures` **之后**（`benchmarks/xval_method_selection_first_evidence.py:256-266`）⇒ **交付的 benchmark 入口永远产不出 evidence.json** | 15 s 内复现：`FIGURES FAILED: KeyError 'fields_radial_stress'` |
| B2 | blocking | `run(quick=True)` 在 `observed_order` 除零：quick skfem ladder 有两条 order=2、n=3（投影/非投影）行，`log(n_hi/n_lo)=log(1)=0` ⇒ CLI `--quick` 与 benchmark `--quick` 双崩 | 两次复现 `ZeroDivisionError` @ `xval_method_selection.py:674` |

**其余（非 blocking）**：
- M1（major）run_artifacts 只有 2 张 PNG，无 evidence.json/run.log/ref-vs-got 表 —— 被审单自述「预算截断」实为 **B1 的 KeyError**，不是预算。
- M2（major）0 文件入库：`agent/m17604 == HEAD == a86365e7`，`git log HEAD..agent/m17604` 空；
  且 main 已由 m17412（merge #876）放入**同路径但不同实现**的 `benchmarks/xval_method_selection_first_evidence.py`（32956 B vs 交付 12058 B）⇒ 若直接套用是覆盖冲突，需先合并语义。
- M3（major）锚点 **0 行**注册（`tensor_benchmark_cases`）；`scripts/register_xval_method_selection_cases.py` 需要 evidence.json → 死循环，未跑。
- M4（major）`tensor_reports WHERE mission_id=17604` **0 行**（report_owed=true）。
- m1（minor）C5 判据用严格 `<1e-16`，而 HEX20 实测闭合力恰为 **0.0** ⇒ 规则把「完美闭合」判成 not reproduced（应为 `<=`）。
- m2（minor）M2 泛函实现相对 m17488 审计改变（形心带 → 层索引切片），HEX20 n=4 的 M2 由 11.989% 变 18.186%，交付未把「定义变更」写明。
- 适用范围：本结论全部 loongarch64 本地实跑，未做 x86 CI；HEX8 n=8/n=12、HEX20 n=6 未复测（集群排队），C3/C6 的再判定沿用 m17488 的实测值。

## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

| 关键量 | 基准（ref） | 本审计 got | 相对误差 | <3%? |
|---|---|---|---|---|
| HEX20 n=4 最差探针（M1） | Lamé@⟨r⟩ | — | **2.875%** | **是**（阈值判定 2.875<3，余量 0.125 pct-pt） |
| HEX20 n=4 最差探针（M2，记录声明的泛函） | 精确层积分 | — | **18.186%** | **否** |
| HEX8 n=4 最差探针 | Lamé@⟨r⟩ | — | **28.182%** | **否** |
| HEX8 n=6 最差探针 | Lamé@⟨r⟩ | — | **15.350%** | **否** |
| TET10 n=3（投影） | Lamé@⟨r⟩ | — | **10.237%** | **否** |
| TET4 n=3 | Lamé@⟨r⟩ | — | **24.620%** | **否** |
| 网格体积误差 HEX20 n=4 | 精确壳体积 4π(b³−a³)/3 | — | **0.1359%**（记录 0.14%，偏差 2.9%） | **是** |
| 网格体积误差 HEX8 n=4 | 同上 | — | **7.116%**（记录 7.1%，偏差 0.22%） | 否（记录本就承认 7.1%） |

**回答**：只有「HEX20 n=4 在 M1 泛函下」与「体积误差」两项满足 <3%；**记录的核心主张放在其自己声明的
M2 泛函下不满足 <3%**（18.19%），因此「方法选型 = 全部探针 <3%」必须重述为「M1 口径下 <3%」。
此外锚点 **0 条**机器基准行，`CASE-HOLLOW-SPHERE-3D` 仍是 `in_progress`（tolerance <5%）——
所以本单的「基准误差」只能由本审计按交付预注册规则复算给出，**不是**平台 benchmark 裁判的判定。

## 判级（L0–L5，只认功能实质）

| 级 | 判 | 依据 |
|---|---|---|
| L1 | 有 | 17 个测试（含两条负对照）本审计复跑全绿 108.43 s |
| **L2 独立功能实现** | **是** | `arm_tensorfem` / `arm_skfem` 真代码：几何+材料+压力 → HEX8/HEX20/TET4/TET10 装配 + 载荷 + PCG 求解 → 探针/能量/体积误差；本审计复算 6 档，与自报逐位一致 |
| L3 系统集成 | 否 | 生态仓 grep `xval_method_selection` = 0；交付 0 提交（不在 main）⇒ 无集成面 |
| L4 基准验证 | 否 | 锚点 0 注册行；交付 `repro_cmd` 在 `--quick` 崩、full 模式死在图 3 前 ⇒ 端到端不可复现；无 evidence.json |
| L5 科研工程 | 否 | 外部数值锚点仍缺（`oss_inventory` 无 3D 结构 FEM 可比；skfem 是库不是参考算例）；且核心结论在 M2 泛函下被自己推翻 |

## 处置与返修最小集（给下一轮）

1. **B1 一行修**：`arm_tensorfem` 增存 `fields_radial_stress`（或图 3 改读 `metric_m2` 的层平均）→ evidence.json 才可能落盘。
2. **B2 一行修**：`observed_order` 对 `len(ladder)<2 or hi["n"]==lo["n"] → return {}`，并把投影/非投影行去重（按 `n, curved_projection` 分组）。
3. 跑 full 一次，落 `evidence.json` + `run.log` + 4 张图，再跑 `register_xval_method_selection_cases.py` 注册 C1/C2a/C2b/C4a/C4b（C3/C6 补 n=12/n=8），`CASE-HOLLOW-SPHERE-3D` 一并推动 passed。
4. 入主干前先解与 m17412 同路径文件的语义冲突；把 M2 定义变更写进交付说明。
5. C5 判据改 `<=`；C1 结论重述为「M1 泛函下 <3%，M2 下 18.19%」。
## 附：平台门禁观察（非本单交付，供 gate 硬化）

本审计在落表前用 `maturity_gate.judge()` 做了三次 dry-run：claim=L2 → `(2, [])`；claim=L3 → `(3, [])`；
claim=L4 → `(4, [])`。后两者**不是**本单真实结论，而是门禁的字符串检查太宽松：
`integration_callers` 只要是非空串就 `grep -rl -F "<串>"`，本审计写的 `"NONE: grep … -> 0 hits"` 因仓内
存在含 `NONE` 的 .py 文件即被"复核通过"；`benchmark` 只要含数字、`repro_cmd` 只要长度 >5 即过。
故本单**只 claim L2**（不利用该漏洞），并建议 gate 硬化：`integration_callers` 改「grep 到 `import`/调用行」、
`repro_cmd` 改实际执行（rc=0 才过）。
