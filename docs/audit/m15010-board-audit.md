# m15010 交差审计（GAP-FEM-ELASTOPLASTIC · TensorFEM 平面应力 J2 弹塑性膜）

* 审计单：m15185（board-audit） · 对象：m15010（succeeded，PR #669 merged）
* 审计人：board-audit · 基准裁判：`tensor_benchmark_cases WHERE anchor_slug='GAP-FEM-ELASTOPLASTIC'`（id=4）
* 独立复核环境：审计工作树 `/root/tp-agent/work/m15185`（TensorFEM main 线，head = Merge #681）
* 结论：**L2（独立功能实现）**，`pending → gate 机器复核`；L3/L4 均不成立，理由见 §3。

## 1. 基准裁判（唯一裁判，先立后判）

| 字段 | 值 |
|---|---|
| case_name | Cook 翘曲膜 + 单轴拉伸循环 |
| metric | 位移解/屈服点 vs 参照 |
| tolerance | <10% |
| ref_source | ABAQUS Verification Manual 算例 + 实验应力应变曲线 |
| status | pending |

## 2. 三件套逐项验收（不是看报告，是看工件）

| 验收项 | 判定 | 证据指针 |
|---|---|---|
| run_artifacts 真存在 | ✅ 真（7 文件，非空，sha256 已登记） | `artifacts/audit_m15185/MANIFEST.json`；原件 `/root/tp-agent/work/m15010/artifacts/j2_plastic_membrane/{report.json,fig1..5.png,tests_pytest.log}`；分支 commit `eedefded` |
| 工件是否进主库 | ❌ **未进 main** | `git show --stat 93f77a1e`（Merge #669）只带 `scratch/build_report_m15010.py`；`git ls-tree -r origin/main \| grep j2_plastic` 只有 `benchmarks/j2_plastic_membrane/{README.md,run.py}` → 7 个工件被合并门禁滤除，原件只剩工作树 |
| benchmark ref vs got | ⚠️ 部分 | 位移解：Cook 弹性 n=32 **23.81763 vs 文献 23.96 = 0.594% < 10% ✅**（DOI 10.1002/nme.1620290802）；弹塑性侧 **C10 WITHHELD**（无 ABAQUS/实验表），"屈服点 vs 参照"无外部参照 → case 按自声明 ref_source **未满足** |
| repro_cmd 可执行 | ✅ 真（审计方独立实跑） | `python3 benchmarks/j2_plastic_membrane/run.py --quick` → exit 0、47 s、C1–C5 与已发布 report.json **逐位相同**（`artifacts/audit_m15185/repro_quick_report.json`） |
| tests | ✅ 真（审计方独立实跑） | `pytest tests/test_j2_plane_stress.py -q` → **10 passed in 86.87 s**（交付方日志 33.18 s，同数同级，机器差异） |
| 报告六要素 | ✅ 齐 | tensor_reports id=14155：①案例/②方法/③数字表/④5 图/⑤注意点/⑥误差判定逐项 + 显式回答 <3% |
| 文档零手抄 | ✅ 仓库侧成立 | `docs/j2-plastic-membrane.md` 的 C4/C7 行与 report.json 逐字段一致（C4 got 5.93e-11/thr 1e-10、C7 got 40） |

## 3. 判级（L0–L5）：**L2**

* **L2 ✅ 独立功能实现**：`src/tensorfem/j2_plane_stress.py`（751 行）真代码——平面应力算子（eps_zz 标量 Newton + 3D 切线 Schur 缩聚）、Q4 膜、Newton 驱动（力控/位移控）、每 Gauss 点状态提交与失败增量自二分；入口 `benchmarks/j2_plastic_membrane/run.py`；工件齐；测试断言不变量（σ_zz=0、屈服面一致性、能量闭合、卸载斜率=E）。
* **L3 ❌ 系统集成**：`grep -rn j2_plane_stress src/`（排除自身）= **0 hit**；`src/tensorfem/__init__.py` 无该模块导出（只导出 `j2_elastoplasticity`）；`benchmark_registry.py` 未登记该模块（只登记 `plasticity`/`tet4_j2`）。→ 未接入主求解器/平台。
* **L4 ❌ 基准验证**：case 要求 ABAQUS+实验参照 <10%，弹塑性侧 C10 诚实 WITHHELD；"单轴拉伸**循环**"只做到部分卸载（tests/test_j2_plane_stress.py:149-190 弹性卸载 α 冻结、斜率=E），无反向屈服-再加载全程。
* **L5 ❌**：无外部工程锚（无 ABAQUS/实验，仅弹性文献锚）。

## 4. 只增不改的发现（按严重度）

1. **[中] 工件未进 main（可复现性承重）**：`eedefded` 的 7 个工件在 Merge #669 被滤除；main 上只剩 `run.py`。缓解=run.py 在 main 且审计方实跑复现 C1–C5 逐位相同，故能力未丢；但"工件入库"只成立于分支层，第三方 clone main 拿不到证据文件。建议：工件指纹（sha256）+关键数值以文本入 main（本审计已落 `MANIFEST.json`）。
2. **[低] 平台报告与 JSON 两格漂移**：tensor_reports 14155 §3.1 表 C4 写 `thr 1e-7`、C7 `got —`，而 report.json 为 `1e-10`、`40`；判定结论不变（均 PASS），但"零手抄"在报告文本侧不严格成立。仓库 `docs/` 侧无漂移。
3. **[低] C9 表头误导**：§3.5 列"上界 2/√3"=1.1547，而 H=3000 两行峰值比 1.2479/1.2412 越界——非缺陷：C9 实际判据是硬化放大界 `2/√3(σy+Hα_max)tw`（run.py:325-344，worst 利用率 0.8795 ≤ 1）；表头未写"硬化放大"易被误读为越界。
4. **[低] 文档自述与实现不符**：模块 docstring 称"3D 径向返回保存在 `tensorfem.plasticity` 中（薄封装）"，实际 `_return_3d`（j2_plane_stress.py:184）是本模块内的批量重实现；C4 正是拿它与 `plasticity.update_j2` 互验（2.27e-13）——两实现并存且互验，比 docstring 声称的更稳，但措辞应改。

## 5. 本次未做（诚实声明）

* 全量 lane（412 s 级、n=32×40 增量）审计方未在本机跑完（预算内未完成）；已递交集群作业 **cj1002**（`python3 benchmarks/j2_plastic_membrane/run.py`，stage_dir=m15185）排队中，`--quick` 已证 C1–C5 逐位复现。
* 未核 ABAQUS 数据集（离线不可达，与交付方同一限制）→ C10 只能维持 WITHHELD，未试图替代。

## 6. 判决

**L2 confirmed 候选**：impl/entry/artifacts/tests 四件真、repro 可执行；L3 无调用方、L4 无外部弹塑性锚 → 不允许上跳。证据键与指针见 `artifacts/audit_m15185/verdict.json`。