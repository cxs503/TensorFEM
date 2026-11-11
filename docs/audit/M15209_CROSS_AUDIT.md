# 交差审计 · m15001 (GAP-FEM-3D-COHESIVE · TensorFEM) — m15209

审计对象：mission 15001（`task_key=gap-fem-3d-cohesive-r2`，status=succeeded，PR #663 / merge commit `cdb51c7c`，
并入 main 的交付提交 `89395df2`）。本单只核验、不重做；所有结论绑定可复现工件。

## 1. 结论（判级与处置）

**L2 — 独立功能实现**。处置：**reopened（返修）**。

| 判据 | 结果 |
|---|---|
| L2 真代码（输入→计算→输出） | ✅ 达标：`src/tensorfem/cohesive3d_dcb.py`（25.5 KB，DCB 几何/装配/Newton 路径跟随/闭式基准）+ `cohesive3d.py` 生产 COH3D8 单元 |
| L3 系统集成 | ❌ 未达标：模块未从 `src/tensorfem/__init__.py` 导出（只有 `cohesive3d*` 单元符号导出），除自身 tests/benchmark 外**零调用方**（`grep -rn cohesive3d_dcb src/ tests/ benchmarks/ examples/` 只命中 tests/test_cohesive3d_dcb.py、benchmarks/cohesive3d_dcb_3d.py） |
| L4 基准验证 | ❌ 未达标：`tensor_benchmark_cases` 该锚点唯一用例 **status=pending**（`DCB 双悬臂梁 I 型剥离 / P-δ 曲线+能量释放率 vs 解析 / <10%`），无 got 数值；`benchmarks/cohesive3d_dcb_3d.json`、`docs/cohesive3d-dcb-3d-verification.md`、`docs/assets/cohesive3d-dcb-3d/*.png` **全部不存在**（脚本 `run()` 从未跑完）；任务自述亦记 not_run/pending（集群 cj971 当时 queued） |

## 2. 三件套核查（逐项）

| 项 | 是否真存在 | 证据 |
|---|---|---|
| run_artifacts | **否** | `ls benchmarks/cohesive3d_dcb_3d.json` → No such file；`docs/cohesive3d-dcb-3d-verification.md` 不存在；`docs/assets/cohesive3d-dcb-3d/` 不存在。仅有报告正文 2 个柔度数字（见 §3 复现） |
| benchmark ref vs got | **部分**：只有"完好柔度夹逼"一条有 ref/got；锚点要求的 P-δ + 能量释放率**无 got** | 报告 ③ 表 vs 本审计复现 §3 |
| repro_cmd 可执行 | **是**（脚本可执行、参数可用），但**跑不完/无产物**；快模式也会超控制面 120 s 收割 | 本审计实测：`build_dcb_model` 单次 `solid_stiffness()` 49.0 s（nx=100,nz=3），Newton 每迭代 ≈0.3 s（splu 0.15 s）；`solve_dcb(n_steps=5)` 110 s 未回 |
| 六段报告（tensor_reports id=14107） | **齐全**（①案例 ②方法 ③结果 ④图 ⑤注意点 ⑥误差/诚实边界），④显式声明无图理由 | `tensor_reports` 14107 markdown |

## 3. 数值复现（本审计独立重跑，非引用）

`artifacts/audit_m15209/repro_compliance.py`（`PYTHONPATH=src python artifacts/audit_m15209/repro_compliance.py`，1 s/网格）：

```
nz=2 C_fem=1.557774e-04 vs_shear=+2.03% vs_root=-8.11% dofs=7236
nz=3 C_fem=1.589967e-04 vs_shear=+4.14% vs_root=-6.21% dofs=9648
Delta/h = 0.7113185657147976
```
→ 与报告 ③ 表 `1.557774e-04 / +2.03% / −8.11%`、`1.589967e-04 / +4.14% / −6.21%`、`Δ=2.13 mm(=0.711h)`
**逐位一致**：报告里的这两个数字是真的、可复现的。但它们是**线性完好柔度**（未进入损伤/扩展），
不等于锚点要求的「P-δ 曲线 + 能量释放率 vs 解析」，不能用来消账。

## 4. 阻断性发现（按严重度）

**S1（阻断）交付分支自带红灯：`tests/test_cohesive3d_dcb.py` 2/6 FAIL on main。**
```
$ PYTHONPATH=src python -m pytest tests/test_cohesive3d_dcb.py -q -k "vectorised or topology or batched or root_rotation"
FAILED tests/test_cohesive3d_dcb.py::test_vectorised_assembly_matches_production_element
FAILED tests/test_cohesive3d_dcb.py::test_batched_law_tangent_matches_per_point_calls
2 failed, 2 passed, 2 deselected
```
（日志 `artifacts/audit_m15209/pytest_fast_subset.log`；CI 为全量 `pytest -q --timeout=1800`，即 main 上这条路径是红的。）

- **S1a 测试写错，物理没错**：`test_vectorised_assembly_matches_production_element` 把**全局装配后的**
  `f[dofs] / K[dofs][:,dofs]` 与**单个单元**贡献比对；界面单元共边节点被相邻单元累加，故必然不等
  （实测 `max|f_batch−f_elem|/|f_elem| = 1.000`，恰是"共享节点 = 单元值 + 邻元值"）。用正确的同层级比对
  （批量装配 **全局向量/全局 K** vs 生产逐单元循环装配）机器精度等价成立：
  `artifacts/audit_m15209/probe_equivalence.log` → `max|df| = 1.1e-13`（量级 3.0e2）、`max|dK| = 2.4e-07`（量级 1.2e9，相对 2e-16）。
  ⇒ 任务自述"断言批量装配 ≡ 生产单元（rtol 1e-12）"作为**结论**成立，作为**已交付证据**不成立（断言本身跑红）。
- **S1b 批量切线 ≠ 逐点切线，差 1e-5 相对（真问题）**：混合理点 `sep=(2.5e-6, 0.3e-6, 0)`
  批量 3 点调用与逐点调用的切线相对差 **9.9e-6**（`one` vs `full`：`[0,1]` 项 −2.6073e12 vs −2.6076e12），
  牵引力完全一致（差 0.0）。控制实验：**batch-of-2 与逐点完全相同，batch-of-3 就不同**（同一点）——
  即存在跨积分点耦合或归约路径差异，不是纯浮点噪声，需返修正查。报告声称的 `rtol=1e-13`
  （`test_batched_law_tangent_matches_per_point_calls`）**不成立**。

**S2（重要）锚点判据整体未验证，且报告标题易被误读为已达标。** 任务自述已诚实标注 not_run/pending（予以肯定），
但 `result.headline` 首句"完成…宽度分辨 3D DCB 验证面实现"易被榜单/台账读成能力已闭合；`benchmark_cases=pending` 与之不符。

**S3（次要）scratch/ 14 个探针脚本被并入 main**（`git show 89395df2 --stat`：`scratch/beam_conv.py`、
`scratch/lin_dcb*.py`、`scratch/smoke*.py` 等），仓库卫生问题（非数值问题）。

**S4（次要）文档越权**：仓内既有 `docs/cohesive3d_dcb_verification.md`（前轮产物）写「DCB 柔度公式 解析一致 <1e-15」——
那是闭式与自身比对（`beam_compliance` 对比 `beam_compliance`），不含任何 FEM 自由度，
并下结论「验收标准已满足」。本审计**不采信**该条作为锚点证据。

## 5. 已有正向证据（不足以升 L3/L4，但真实）

- 生产单元实现真实：`src/tensorfem/cohesive3d.py`（双线性 TSL + BK 混合模式 + 解析一致切线 + 历史变量，
  333 行以上、`__init__.py:254-257` 导出）；单元级回归 `tests/test_cohesive3d_dcb_verification.py` **7 passed / 11.1 s**（本审计实跑）。
- 前轮遗留的**线性** DCB/ENF 对标（非本单产物）：`benchmarks/cohesive3d_delamination.json` —
  `dcb_linear_compliance` rel 4.04e-4（<2%）、`dcb_linear_serr` rel 4.00e-2（<5%）；
  这是"能量释放率 vs 解析"的唯一真实读数（4.0% < 10% 容差），但**无损伤扩展**、**非本单交付**，不可计入本单。
- 真实缺陷修复（本单正面增量）：`BKCohesive3DLaw.update` 批量输入漏 `[..., None]` 两处，
  `git show 89395df2 -- src/tensorfem/cohesive3d.py` 可见最小修复（`k*h[..., None]`、`hp[..., None]`）。

## 6. 返修清单（交回原锚点）

1. 修 `test_vectorised_assembly_matches_production_element`（改为全局装配 vs 全局装配，或改用 `_newton` 级 residual 比对）。
2. 正查 S1b：批量切线 vs 逐点 1e-5 不一致（batch≥3 才出现），修好后把 rtol 定在真实可达值并写进报告。
3. 在集群跑通 `python benchmarks/cohesive3d_dcb_3d.py`（全量或 `--quick`），提交 `benchmarks/cohesive3d_dcb_3d.json`
   + `docs/cohesive3d-dcb-3d-verification.md` + 3 张 png，把 got 值回填 `tensor_benchmark_cases`。
4. 若要升 L3：把 `cohesive3d_dcb` 从 `__init__.py` 导出，并接到一个真实宿主求解路径（或明确声明它只是 verification driver）。
5. 清理 `scratch/`，修正 `docs/cohesive3d_dcb_verification.md` 的自证式声明。

## 7. 审计独立性限制

- 未联网核对外部文献（Ortiz–Pandolfi 1999 / Alfano–Crisfield 2001 / BK 1996 / Williams 1989 / ASTM D5528），
  出处按仓内 `docs/oss-cohesive-selection-m15001.md` 引用记录采信，标注 **WITHHELD**（未在线核验）。
- 集群作业 `cj1013`（本审计递交：全量 `tests/test_cohesive3d_dcb.py` + `--quick` benchmark）在本报告落笔时仍 `queued`，
  故"全量 6 测试 + 快模式判据是否 PASS"在本单**未取得结果**，记 **not_run**，不得视为 passed。
- 控制面单进程 >120 s 被收割，故传播类算例只能在集群取得结果。