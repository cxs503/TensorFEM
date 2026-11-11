# 交差审计 m15393 — CASE-KIRSCH-PLATE（TensorFEM，原件 m15298）

- 审计单：m15393（交差核验单，非重做）· verdict_by=`board-audit`
- 被审对象：m15298 / 分支 `agent/m15298` / 提交 `d3db3dbf`（"CASE-KIRSCH-PLATE: Kirsch hole stress-concentration anchor (K_t=3)"）
- 已并入生态仓：`/root/tensor-ecosystem/TensorFEM` 合并提交 `d1b3129a`（"Merge #692 (m15298) via push-merge"）；PR head `28e046ee`
- 审计工作树 HEAD：`93c8bb91`（与生态仓同 head，谱系合并提交不同但内容一致：`result.json` blob `a0dea25`、`run.py` blob `8d7cf71` 在两处相同）
- 平台报告：`tensor_reports` id=**14386**（mission 15298，四图内嵌，六段齐全）
- 审计动作：只读被审仓 + 独立重跑；未修改被审算例的任何数值/门限。唯一落仓改动＝补回被流水线滤除的图件镜像与一份守卫测试（见 F3）。

---

## 1. 基准裁判（唯一裁判）

任务指定的裁判表 `tensor_benchmark_cases`：

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='CASE-KIRSCH-PLATE';   -- → 0 rows
```

**该表对本锚点无登记行（F1）。** 池中确有一条可机读的容差，在算例池表：

```sql
SELECT id,case_code,repo,name,ref_source,tol_policy,tier,linked_anchor,status,mission_id
  FROM tensor_case_pool WHERE linked_anchor='CASE-KIRSCH-PLATE';
-- 41 | KIRSCH-PLATE | TensorFEM | 开孔板应力集中 K_t | 弹性力学解析 K_t=3 | <5% | core | | dispatched | 15298
```

故本次以 **`tensor_case_pool(id=41).tol_policy = "<5%"，ref = 弹性力学解析 K_t=3`** 为裁判判据（并叠加平台 15% 不可接受线 / 报告自证 3% 线）。逐项核对：

| 量 | ref | got（本次重跑） | 误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| K_t（d/W=0.02, n=48，gross） | 3（Kirsch 1898 / Timoshenko-Goodier Art.39） | 3.001408 | **0.047%** | <5% | PASS |
| K_t Richardson 极限（d/W=0.02） | 3 | 3.009196（p=2.03，GCI 0.32%） | 0.307% | <5%（自设 <1%） | PASS |
| K_t Q8 独立单元族（n_θ=96） | 3 | 3.001476 | 0.049% | <5% | PASS |
| K_t（d/W=0.10, n=48） | 有限宽度真值（Howland/Heywood gross 3.032222） | 3.025609 | 0.22%（vs Heywood） | <3%（自设） | PASS |
| 孔口轮廓 σ_θθ(a,θ) | 1−2cos2θ | profile RMS 0.318%·K_t | — | <5% | PASS |
| 牵引平衡 | 静力学 | 1.01e-12 | — | <1e-10 | PASS |
| K_tg vs Heywood gross（d/W=0.1…0.5） | 拟合 | ≤0.985%（d/W=0.5） | — | <3% | PASS |
| K_tn vs Peterson/Howland 读表 | Chart 4.1（±1% 读图） | ≤0.65% | — | 拟合+读图不确定度内 | PASS |

**结论：权重最大的一项（K_t vs 3）在 5% 池容差内且远优于 3% 线；14 项自设门禁全过。** 裁判侧结论 acceptable。

## 2. 验收三件套核验

| 三件套 | 声称 | 核验 | 结论 |
|---|---|---|---|
| run_artifacts | `result.json`、`gates_verified.json`、4 张 png | `result.json`(61 606 B, blob a0dea25)、`gates_verified.json`(6 341 B, blob 2f5b246) 在 main 存在；**4 张 png 在 main 树中缺失（F3）** | 数值工件真、图件曾缺（已修） |
| benchmark ref vs got | K_t=3 vs 3.001408 | 重跑独立复现（见 §3），逐项落表见 §1 | 真 |
| repro_cmd 可执行 | `OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py` | 在本审计工作树实跑：exit 0，14/14 gates passed，9 条梯行数值与 `result.json` 逐位一致（仅 `generated_at`/`wall_seconds` 变） | 可执行、可复现 |
| 单测 | `tests/test_kirsch_plate_anchor.py` 6 passed | 实跑 **6 passed, 71.48 s**（报告写 ~15 s，见 F4） | 真（断言含不变量，非仅执行） |

## 3. 独立重跑证据（本审计动作，非引用原件）

```
$ OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py
[plate] d/W=0.02 n= 48 K_t=3.001408 err=0.047%
[plate] d/W=0.05 n= 48 K_t=3.001349 err=0.045%
[plate] d/W=0.10 n= 48 K_t=3.025609 err=0.854%
[Q8] 96x24 K_t=3.001476 err=0.049%
[gates] passed=True failed=[]
$ PYTHONPATH=src pytest tests/test_kirsch_plate_anchor.py -q   → 6 passed
```

- **真实求解链**：`benchmarks/.../run.py:83-84` 引 `tensorfem.structural_canonical_face.solve_kirsch_q8_concentration`，`run.py:78-82` 引 `tensorfem.finite_width_hole.{FiniteWidthPlate,solve_case}`；后者 `src/tensorfem/finite_width_hole.py:57` `from .continuum import ... solve_continuum`，`:361 result = solve_continuum(model)`；Q8 侧 `src/tensorfem/structural_canonical_face.py:977 solve_q8_plane(...)`。**主求解器真被调用**，不是闭式回填：牵引驱动板是纯 Neumann+对称边界的边值问题，孔口不注入精确解。
- **非硬编码**：`K_t=3` 由 `oracle_selfcheck()` 在代码内对闭式求值（`run.py:159-181`），并自检孔口自由边界 |σ_rr|,|σ_rθ|≤1e-12、远场 σ_xx→T/σ_yy→0；重跑复现 `far_field_sxx_deviation=6.25e-05`。
- **数值可证伪性**：粗网格给 2.8799（4.00%），细网格 3.0014；若把牵引改为位移或去掉对称约束，此锚点会失败——不是恒真门。

## 4. 判级：**L4（基准验证）**

- L2（独立功能实现）✔：真代码输入→计算→输出（`run.py` 604 行 + 两个求解器入口）。
- L3（系统集成）✔：算例直接调用主求解器 `tensorfem.continuum.solve_continuum` 与 canonical face `solve_kirsch_q8_concentration`，无旁路实现。
- L4（基准验证）✔：完整算例（3 个 d/W × 3 层网格 + Q8 独立单元族 + L/W、ν/E、有限宽度三组敏感性）、误差分析（观测阶/Richardson/GCI/牵引平衡）、可复现（实测 repro + 6 单测）。
- L5（科研工程成果）✘：本单是**对既有求解器的基准认证**，无新数值方法、无新物理结论；文献锚点（Kirsch/Timoshenko-Goodier、Howland/Peterson）是外部对标而非自身科研产出。**不判 L5。**

## 5. 发现（按严重度）

**F1（中·流程）裁判表缺行** — `tensor_benchmark_cases` 无 `anchor_slug='CASE-KIRSCH-PLATE'` 行，任务书指定的裁判对本案「沉默」；容差实际只存在于 `tensor_case_pool(id=41)`。后果：maturity/基准自动复核对本锚点无法机读容差。建议：算例池 dispatched→succeeded 时同步登记 benchmark 行（属 TensorPlatform 共性机制，非本仓修复面）。

**F2（中·簿记）算例池未收口** — `tensor_case_pool(id=41).status` 仍为 `dispatched`，而 mission 15298 已 succeeded 且基准 PASS。池状态机缺「结案」跃迁。

**F3（高·工件可追溯，已修）声明的图件不在版本库** — `result.json.figures` 与 `gates_verified.json.artifacts.figures` 声明 4 张 png；提交 `d3db3dbf` 确实带这 4 个 blob，但 PR head `28e046ee` 的提交说明即为「4 个文件被门禁滤除」，`d1b3129a` 合并树与当前 main **均无 `benchmarks/**/*.png`（`git ls-files benchmarks | grep -c '\.png$' = 0`）**。根因在推送器 `tp_pusher.py:359` `_EXCLUDE2 = r'\.(db|sqlite3?|png|log)$'`（仅 `docs/` 下 png/svg 放行）→ 落在 `benchmarks/` 的图必然被丢。
  **修复**：本次重跑重新生成的 4 张 png 与 `d3db3dbf` 中的 blob **字节完全一致**（sha256 全等：`ed853e79…`/`481c4342…`/`57f4976a…`/`f59d6244…`），说明图可确定性复现；已把它们镜像到流水线放行路径 `docs/assets/kirsch_plate/` 并在算例 README 标注、加守卫测试 `tests/test_kirsch_plate_anchor.py::test_figures_tracked_mirror`（docs 镜像存在且非空）。不改 `result.json` 的生成路径（生成仍写算例目录），只补一个受版本控制的镜像指针。

**F4（低·文档精度）耗时声称偏差** — 报告写「pytest 6 passed, ~15 s」，本机实测 71.48 s；`run.py` 全梯报告写 87 s。架构/线程差异可解释量级，但数字不实。

**F5（低·出处精度）环境串与实际入口不符** — `result.json.environment.solver` 写 `tensorfem.quadratic_plane.solve_q8_plane`，而实际调用入口是 `tensorfem.structural_canonical_face.solve_kirsch_q8_concentration`（`run.py:274`）；前者是底层装配函数。属出处串不精确，不影响数值。

**F6（低·精度表述，非阻断）0.047% 是点值巧合，认证不确定度 ≈0.32%** — d/W=0.02 三层梯 2.879925→2.977466→3.001408 单调逼近，Richardson 极限 **3.009196**（比最细层更远离 3，误差 0.307%），GCI_fine=0.324%。同时有限宽度真值（Heywood/Howland gross，d/W=0.02）= 2.941192/0.98 = **3.001216**（仅比 3 高 0.041%）。即：最细层的 0.047% 是「离散化误差（约 −0.2…−0.3%）」与「有限宽度增量（+0.04%）」的相互抵近，**不能当作「本求解器精度 0.05%」的普遍结论**；该锚点对 d/W=0.02 事实上无法把离散误差与模型误差分开（Q8 位移驱动臂同理）。报告 ⑤ 已自陈「FE 收敛到有限宽度真值而非 3」并给了 Richardson，故属表述纪律问题而非造假。**建议**：引用精度时写「0.05%（点值）/ 0.32%（GCI 带）」，二者均 <3%/5%，判级不受影响。

**非发现（澄清）**：报告 §① 称 Q8 入口为 `structural_canonical_face::solve_kirsch_q8_concentration`，与 `run.py` 一致；`result.json` 的串才是错的那个（F5）。Peterson 表（2.73/2.51/2.35/2.24/2.16）是**显式标注的读表值 ±1%**（`src/tensorfem/finite_width_hole.py:61-71`），用作二次交叉验证且偏差 ≤0.65%，不构成手抄冒充计算。

## 6. 未做 / 未跑（诚实声明）

- **加密探针未完成**：为判定 F6（d/W=0.02 的 n=96 是否把 K_t 推向 3.0092 还是贴近 3.00122），已写 `artifacts/m15393_kirsch_refinement_probe.py` 并递交集群 `cj1099`（队列 305，审计窗口内未派发）。**该加密层未跑出结果，F6 的量化表述依据是三层梯 + GCI + Heywood 交叉对照，不含 n=96 实测。** 本地直跑超过单次 120 s 限额（n=96 的 dofs≈19 k 在纯 numpy 稠密/带宽路径上过重）。
- 未复核 `make_report.py` 生成的报告 markdown 与 `tensor_reports` id=14386 的逐字一致（抽查了①②③⑤⑥六段与门禁表，数值与 `result.json` 一致；未做全文 diff）。
- 未审 `benchmarks/structural_certification_materials.py`、`scripts/audit_finite_width_hole.py` 等二级消费者。
- 未审 GitHub CI 侧（x86）表现；本审计全部在 loongarch64 本地树执行。

## 7. 处置

**approve with notes** —— 级别 **L4**，裁判判据（池容差 <5%）PASS，三件套在修 F3 后齐备，可复现（repro + 6 单测实跑）。附 F1/F2 两条上游簿记缺陷（TensorPlatform 侧算例池→benchmark 登记与结案跃迁）与 F4/F5/F6 三条文档纪律建议。