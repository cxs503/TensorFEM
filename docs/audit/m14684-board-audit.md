# m14684 交差审计（board-audit）：GAP-FEM-ELASTOPLASTIC · TensorFEM

审计单 m14959｜对象 m14684（status=succeeded, PR #640 merged, commit `c3883c74`）｜审计席位 board-audit
审计对象代码 = `c3883c74` 的 11 文件（3180 插入行）；本审计在独立工作树 `/root/tp-agent/work/m14959` 复核，
先用 `git hash-object` 证明五个受审文件与 `c3883c74` 逐字节一致（SAME×4，见下），再做重跑。

## 0. 取证结论（三件套核验）

| 三件套 | 是否真存在 | 证据（真实指针） |
|---|---|---|
| run_artifacts 真实运行工件 | ✅ 真 | `artifacts/j2_plastic_membrane/report.json`（1502 行，含 criteria/代价/曲线数组）+ `fig1_uniaxial.png`…`fig5_cook_elastic_convergence.png`（5 张，38–77 KB） |
| benchmark ref vs got + 误差 | ✅ 真（除 C10） | `report.json:criteria[*]` 逐条含 `got/reference/error/threshold/passed/anchor`；C5 文献 23.96 vs 得 23.81763 = 0.594 %；C10 = WITHHELD（无 ref 无 got） |
| repro_cmd 可执行 | ✅ 真 | 本审计实跑：`python3 benchmarks/j2_plastic_membrane/run.py --quick` → 32.13 s、exit 0、C1–C9 全 True、C10=None；`python3 -m pytest tests/test_j2_plane_stress.py -q` → **10 passed in 40.23 s** |

文件同一性（`git hash-object` vs `git rev-parse c3883c74:<path>`）：

```
src/tensorfem/j2_plane_stress.py            47a7695c…  SAME
tests/test_j2_plane_stress.py               564a38ed…  SAME
benchmarks/j2_plastic_membrane/run.py       4bda5e7b…  SAME
docs/j2-plastic-membrane.md                 25a79991…  SAME
```

## 1. 实现实质（不是壳）

`src/tensorfem/j2_plane_stress.py`（752 行）实现的是**平面应力** J2 弹塑性，并明确把它建在仓库既有
3-D 核之上而不是另起一套返回映射：

* 面外应变 `eps_zz` 作为标量未知量，用标量 Newton 强制 `sigma_zz = 0`（3-D 核是平面应变构造，膜/壳材料点必须补这一半）；
* 平面应力一致切线 = 3-D 算法切线对 `zz` 行/列的 Schur 缩聚（隐函数定理结果），不是数值差分；
* 塑性功增量取硬化积分 `∫(σ_y+Hα)dα` 的闭式，使 `W_ext = U + W_p` 可被审计而不是被假设；
* FE 面：2×2 高斯平面应力 Q4 膜 + Newton–Raphson 驱动（力控/位移控）+ 逐高斯点状态提交（失败步可回滚重试）+ 软化输入 fail-closed 拒绝。

与仓库既有件的差异（能力缺口判定依据）：`src/tensorfem/plasticity.py:105 condense_plane_stress` 只缩聚**切线**，
被 `layered_shell4_plasticity.py:210` 用于壳的切线层面，并未提供平面应力**返回映射**（σ_zz=0 的一致性求解）。
故本模块填补的是真实缺口，不是重复造轮子。

## 2. 机器判据复核（本审计重跑，非引用）

| id | 判据 | got | ref | 误差 | 阈值 | 通过 |
|---|---|---|---|---|---|---|
| C1 | 单轴闭式（含塑性斜率 EH/(E+H)） | 307.075 | 307.0754717 | 1.93e-14 | 1e-10 | ✅ |
| C2 | σ_zz = 0（平面应力约束） | 5.93e-11 | 0 | 2.37e-13 | 1e-9 | ✅ |
| C3 | 塑性点落在屈服面 q=σ_y+Hα | 3.72e-11 | 0 | 1.49e-13 | 1e-8 | ✅ |
| C4 | batched 核 vs `plasticity.update_j2` | 5.93e-11 | 0 | 2.27e-13 | 1e-10 | ✅ |
| C5 | Cook 膜弹性挠度 vs 文献 23.96 | 23.81763 | 23.96 | **0.594 %** | 1 % | ✅ |
| C6 | 全局能量闭合 W_ext=U+W_p | 81.9671 | 81.97785 | 1.31e-4 | 1e-3 | ✅ |
| C7 | 增量塑性功 vs 闭式 σ_y a+H a²/2 | 40 | 0 | 2.11e-16 | 1e-12 | ✅ |
| C8 | Cook 弹塑性网格无关（n=24 vs 32） | 0.136673 | 0.135925 | 0.547 % | 5 % | ✅ |
| C9 | 缺口试件 ≤ 平面应力极限分析上界 2/√3(σ_y+Hα_max)tw | 比值 max 0.88（≤1） | [1.0,1.1547] | — | ≤1+1e-6 | ✅ |
| C10 | Cook/缺口 vs ABAQUS 文献 <10 % | — | ABAQUS 数据集 | — | 0.10 | **WITHHELD** |

复现保真度：本审计 `--quick` 重跑得到的 Cook 弹性行（n=16 → 23.43041、n=32 → 23.81763）与入库
`report.json` 的对应行**逐位一致**（同一代码路径确定性复现），C1–C9 判决与入库一致。
入库 `report.json` 的 `meta.quick=false` 且含 quick lane 不可能产生的 n=24 行/三档硬化 6 行 → 入库工件确为完整 lane；
而 m14684 的 `result.checks` 却写「入库工件为 --quick lane」——**自述与工件不一致（文书级瑕疵，非数据造假）**：
本审计另于 2026-11-11 13:13 在独立工作树启动**完整 lane 端到端重跑**（不覆写入库工件，输出隔离到
`scratch/audit_m14684/full_rerun/`），完成耗时 **405.29 s**（自述 395.75 s，差 2.4 %，与「~7 min」自述一致）。
重跑结果与入库 `report.json` **逐位一致**：Cook 弹性三档 23.43041126007023 / 23.713596320933593 / 23.817633955706043（n=16/24/32）、
Cook 弹塑性三档挠度与 W_ext（0.133985916 / 0.135925087 / 0.136672629；80.077166 / 81.429104 / 81.967103）与能量闭合 1.28e-4 / 1.27e-4 / 1.31e-4、
DENT 六行峰值载荷全 6 位一致（18718.19 / 18618.51 / 16866.19 / 16800.52 / 15855.12 / 15790.12）、材料点最大相对误差 1.9315e-14、
判据 C1–C9 = True / C10 = null 全部相同。**即：入库工件可由 `repro_cmd` 一键端到端复现，无手抄、无篡改。**

## 3. 判级 L0–L5

**判定：L2**（独立功能实现：真代码 输入→计算→输出）。

* L0/L1 显然超过：不只文档，10 条不变量测试（材料点闭式、平面应力约束、屈服一致性、batched 核与 oracle 核一致、
  单元等应变 patch、能量闭合、单调耗散、缺口上界）本审计实跑 10 passed。
* L2 ✅：自带 Q4 膜装配 + Newton 驱动 + 状态提交，端到端跑出载荷-位移/能量/耗散曲线，可复现。
* L3 ❌（系统集成）：`grep -rn j2_plane_stress src/` = **0 hit**；`src/tensorfem/__init__.py` 无该模块导出（顶层不可 `from tensorfem import J2PlaneStressMaterial`）；
  `src/tensorfem/benchmark_registry.py` 无该算例登记。即：既有求解器/仓库基准台账都不消费它。
* L4 ❌（基准验证）：**弹塑性本身没有外部基准**。唯一外部锚 C5/Cook 23.96 是**弹性**锚（验证的是单元与装配，不是塑性路径）；
  弹塑性部分只有自洽证据（网格收敛 0.55 %、能量闭合 1.3e-4、极限分析上界），
  自洽不构成"对标实验/参考值"。任务验收线「Cook 膜/缺口试件 vs ABAQUS 文献基准 <10 %」= C10 WITHHELD，**未达成**。
  按平台精度纪律（对标误差 >15 % 不可接受），无外部锚即不能声称 L4。
* L5 ❌：无科研工程级成果（无新方法、无外部验证闭环）。

## 4. 审计发现（按严重度）

| 级别 | 发现 | 证据 | 影响 |
|---|---|---|---|
| 阻断（对锚点而言） | **任务验收判据未达成即被 auto-approve 结单**：C10（ABAQUS <10 %）WITHHELD，但 m14684 status=succeeded、`decision_note`=`[auto-approve] TP_AUTO_APPROVE_DEV_ACCEPTANCE` | `report.json:criteria[9].passed=null`；m14684.result.decision_note | GAP-FEM-ELASTOPLASTIC 不能就此消账；应 reopen 进第 2 轮 |
| 中 | **无集成面**：模块未导出、无 caller、未登记 benchmark_registry | 见 §3 L3 三条 grep | 平台/求解器无法消费，能力停在"库里有文件" |
| 中 | **自述与工件不一致**：mission result 称工件为 quick lane，实际工件为完整 lane | §2 保真度段 | 审计若只读自述会误判证据强度 |
| 低 | C9 行的 `error` 字段承载的是载荷/上界**比值**（0.88），与其它行"相对误差"语义不同 | `run.py:338-350` | 报表可读性/自动聚合歧义，非正确性缺陷 |
| 低 | C4 自称"independent code path"，实为同一数学的镜像实现（batched 版） | `run.py` C4 行 + 模块 `_tangent_3d` | "独立"是弱化的（可检出复制粘贴级错误，不能检出共同误解） |

无造假迹象：所有入库数字都能由重跑复现（重叠行逐位一致），WITHHELD 是如实申报，`failure_reason` 落到 `report.json` 并指明第 2 轮输入。

## 5. 返修建议（构成 reopen 的完成条件）

1. **外部锚**：取一条公开可引的弹塑性载荷-位移表（如 ABAQUS 验证手册 Cook 膜/带孔板算例，或文献中给出材料卡 + 峰值载荷的表），
   做 ref vs got 的逐点误差表并进 `criteria`，目标 <10 %（平台线 15 %）。
2. **集成**（升 L3）：`tensorfem/__init__.py` 导出材料/求解入口；在 `benchmark_registry.py` 登记该算例（`run_registered_benchmarks()` 能跑到）；
   并把 `layered_shell4_plasticity` 的平面应力切线路径改为复用本模块的 Schur 缩聚（消一份重复实现）。
3. **文书**：`result.checks` 与入库工件 lane 一致；C9/C4 的表述改为比值/镜像实现的准确措辞。
4. 完整 lane 端到端重跑一次并记录 runtime（本审计在本机 >12 min 未完成，与自述 7 min 有差，需在干净机器上确认）。

## 6. 命令与工件

```bash
# 文件同一性
git hash-object src/tensorfem/j2_plane_stress.py            # vs git rev-parse c3883c74:...
# 测试
python3 -m pytest tests/test_j2_plane_stress.py -q          # 10 passed in 40.23s
# 基准（本审计重跑，输出隔离到审计目录，不污染入库工件）
python3 benchmarks/j2_plastic_membrane/run.py --quick --out artifacts/audit_m14684/quick_rerun
python3 benchmarks/j2_plastic_membrane/run.py --out scratch/audit_m14684/full_rerun   # 405.29 s，与入库工件逐位一致
```

* 审计重跑工件：`artifacts/audit_m14684/quick_rerun/report.json`、`artifacts/audit_m14684/quick_rerun.log`、`artifacts/audit_m14684/pytest.log`；完整 lane 重跑 `scratch/audit_m14684/full_rerun/report.json`
* 判决载荷：`artifacts/audit_m14684/verdict.json`（已 INSERT 入 `tensor_maturity_verdicts`）
* 受审工件：`artifacts/j2_plastic_membrane/report.json` @ `c3883c74`