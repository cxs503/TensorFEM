# 交差核验单：m14685 · GAP-FEM-PARALLEL（TensorFEM 并行）

审计单 m14964｜verdict_by = `board-audit`｜审计对象：mission 14685（succeeded，PR #634 / commit
`21b0c55e` "m14685: matured mission deliverables"，merge `8299c5b1` "Merge #634 (m14685) via
merge-master"）｜核验日期 2026-11-11｜审计工作树 `/root/tp-agent/work/m14964`（TensorFEM main + 本次
审计产物）。

本单**不重做** m14685，只做四件事：①取证三件套是否真存在；②按 L0–L5 判级；③落表
`tensor_maturity_verdicts`；④留可机核的 evidence 指针。

---

## 1 交付物清单（实际读到的原件）

| 类 | 指针 | 状态 |
|---|---|---|
| 源码 | `src/tensorfem/parallel/comm.py`（393 行，`Comm`/`SerialComm`/`ProcessComm`/`MPIComm`）、`partition.py`（204 行，RCB + 图增长/KL）、`poisson.py`（325 行，`DistPoisson` DMPlex 式 ghosted 向量 + Jacobi-PCG）、`scaling.py`（197 行，预注册测量协议） | 真代码，非占位 |
| 入口 | `benchmarks/parallel_scaling.py`（CLI: `--mesh/--iters/--repeats/--ranks/--weak-base`，自带 PNG/JSON 输出） | 存在且实测可跑 |
| 测试 | `tests/test_parallel_fem.py`（14 项：单元矩阵 vs 6×6 Gauss、RCB 均衡、collectives、ghost 值、解一致性、二阶收敛、gate 计算、工件存在性） | 1 failed / 12 passed / 1 **skipped** |
| 运行工件 | `/root/tp-agent/work/m14685/artifacts/parallel_scaling/{strong,strong_800,strong_1280,weak,scaling}.json` + 3 PNG | 存在且非空（唯一副本，未随 PR #634 合入主仓） |
| 文档 | `docs/parallel-domain-decomposition.md`（协议/结果/踩坑/文献锚） | 存在 |
| 报告 | `tensor_reports#13841`（六要素齐全，含 honest FAIL 声明与 failure_reason） | 存在 |
| 三件套 | run_artifacts ✅（绝对路径可访问）／benchmark 数值对照 ✅（单元矩阵 5.6e-16、L2 阶 2.00、分区分不变性、E(4) 判据）／repro_cmd ✅（**本次独立复跑通过**，见 §3） | 通过 |

---

## 2 判级：**L2（独立功能实现）**

按判据逐档核（口径与 `maturity_gate.SCHEME` 一致）：

| 档 | 判据 | 结论 |
|---|---|---|
| L1 数学/单元验证 | 测试存在且钉住不变量 | ✅ 单元刚度阵 vs 独立 6×6 Gauss 5.6e-16；L2 收敛阶 2.00（本次复算 nx=64 收敛解 L2=9.2509e-06，与文档 9.25e-6 一致） |
| L2 独立功能实现 | 输入→计算→输出真跑通 | ✅ 分布式 Q1 Poisson 求解：1/2/4/8 rank 的 L2 误差 16 位一致（2.0729e-02 全同），说明 halo/ghost 计划正确 |
| L3 系统集成 | 主求解器/平台侧调用方 | ❌ `grep -rn "tensorfem.parallel" src/tensorfem` **除 `src/tensorfem/parallel/` 自身外零命中**；无 `assembly/solver/analysis_workflow` 调用方，平台侧零调用。PR #634 只是把包放进主仓，不是接入 |
| L4 基准验证 | L3 全 + ref vs got + repro | ❌（被 L3 卡住；且基准本身不稳定，见 D5） |
| L5 科研工程 | L4 + external_anchor | ❌ |

**判级理由（功能实质，不是文档厚度）**：这是一个**自洽、可运行、数值正确**的独立并行内核，但它只解一个
自造算例（单位方形 Poisson + 制造解），没有被任何主求解器/平台路径调用；报告的六要素完整、honest FAIL
如实登记，但"百页报告不改变级别"——实质就是 L2。

---

## 3 独立复跑（repro_cmd 可执行性核验）

在审计工作树（TensorFEM main 检出）用文档给的命令原样跑：

```
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 benchmarks/parallel_scaling.py --mesh 640 --iters 20 --repeats 2 --weak-base 160
   → 48 s，产出 artifacts/parallel_scaling/{strong,weak,scaling}.json + 3 PNG
PYTHONPATH=src python3 -m pytest tests/test_parallel_fem.py -q
   → 1 failed, 12 passed, 1 skipped（6.4 s）
```

复跑工件副本：`artifacts/audit_m14964/repro_strong_640/`、`artifacts/audit_m14964/repro_strong_1280/`
（含 `run.log`）。数值对照表：`artifacts/audit_m14964/scaling_compare.txt`。

| 系列 | mesh | iters/reps | T(1) [s] | E(2) | **E(4)** | E(8) | gate E(4)≥0.60 |
|---|---|---|---|---|---|---|---|
| 原件 strong.json | 640² | 20/2 | 1.1148 | 0.679 | **0.534** | 0.521 | FAIL |
| 本次复跑 640² | 640² | 20/2 | 0.8505 | 0.687 | **0.748** | 0.443 | **PASS** |
| 原件 strong_800.json | 800² | 10/1 | 0.8868 | 0.442 | **0.576** | 0.523 | FAIL |
| 原件 strong_1280.json | 1280² | 10/2 | 2.9485 | 1.022 | **0.805** | 0.432 | PASS |
| 本次复跑 1280² | 1280² | 10/2 | 2.0026 | 0.778 | **0.678** | 0.601 | **PASS** |

→ **同一台机、同一份代码、同一条命令，E(4) 在 0.534↔0.748 之间翻面，gate 判定随之翻面。**
m14685 报的 honest FAIL 不是造假（它如实写了自己测到的 0.534/0.576，并主动标注"本机共享控制面抖动
±25%"），但**它得出的结论"未达标"同样不可复现**：正确说法是"该机上的 E(4) 测量不可判定
（0.53–0.75），判据既未证明通过也未证明不通过"。

---

## 4 发现（按严重度）

**D1（高，数值报告缺陷）`DistPoisson.cg()` 返回的 `residual` 恒等于 ‖b‖₂，不是迭代残差。**
`poisson.py` 里 `nxt = r0` 之后再未更新，`return {"residual": sqrt(nxt)}`。实测 nx=64：预算
1/2/5/10/25/50/93 次迭代，`residual` 全部 = 1.087482e-02 = ‖b‖₂，`residual_rel` 恒 1.000。
后果：①工件与报告里的"残差 1.09e-3"没有收敛含义；②报告 4 项数值判据中的**"1/2/4/8 rank 残差逐位一致
≤1e-12"是空转判据**（该字段在迭代前就定死了），真正有效的分区分不变性证据是 L2 误差一致（那一项是真
的，本次复算一致到 16 位）。复现图 `artifacts/audit_m14964/residual_field_bug.png`。

**D2（高，功能缺陷）`element_adjacency()` 的面邻接用了对角点对。**
`partition.py` 里取的边是 `(0,1),(1,3),(3,2),(2,0)`，其中 `(1,3)`、`(2,0)` 是四边形单元的**对角**，
真正的边 `(1,2),(3,0)` 没取。后果：结构化 16×16 网格单元度数 min/max = 1/2（正确应为 2/4）；
`graph_partition` 只沿 y 方向连通 → 4 分区每个分区在真邻接图下**碎成 4 块**（实测
components=[4,4,4,4]，单块 x 跨度 13/16 列，蛇形交错）。这正是 `tests/test_parallel_fem.py::
test_graph_partition_on_structured_grid_is_valid_and_balanced` 失败的原因（m14685 已如实登记为已知缺陷）。
连带：工件里 `cut_fraction` 对 p=8 少算 25%（0.002347 vs 真值 0.003130；p=4 恰好数值相同，属巧合）。

**D3（中，证据放置）运行工件未随 PR #634 合入主仓，守护测试因此永久空转。**
merge `8299c5b1` 的 stat 里没有 `artifacts/parallel_scaling/*`；`git ls-files | grep parallel` 在 main
上也没有它们。于是 `test_committed_scaling_artifacts_exist` 走 `pytest.skip("scaling artifacts not
generated in this checkout")` → 交付仓里**没有任何测试保护这份证据**；工件只剩 agent 工作树一份。

**D4（高，安全）`.tok` 被 m14685 提交并随 #634 合入 main。**
`git show 8299c5b1:.tok | wc -c` = 209 字节，`eyJhbGciOiJI…` = JWT 访问令牌明文入库（本次审计只核长度
与前缀，未在报告中回显内容）。违反"禁止任何秘密入库"纪律，需吊销+清史。

**D5（高，可复现性）gate 指标不可复现（见 §3 表）。** 报告自述 ±25% 抖动，实测 640² 的 T(1)
1.1148→0.8505（-24%）、T(4) 0.5218→0.2844（-45%），共享控制面上用 min-of-2 的墙钟做 0.60 判据，
测量分辨率低于判据裕度。**判据与测量装置不匹配**：要么提 repeats/取 min-of-N 更大 N，要么按每 rank
DOF 归一后在大规模上判（1280² 复跑 E(4)=0.678 三次里两次过线），要么改判"固定规模"为"每 rank ≥200k
DOF 时的效率"。

**D6（中，目标未达成）没有真 MPI 执行证据。** 工件 `backend = {"mpi4py": false, "process": true}`：
本机无 mpi4py，`ProcessComm`（管道）跑的；`MPIComm`(mpi4py) 分支**从未被执行**。集群递交 cj942
在单结束前未取回，**无任何集群工件**；本次审计递交的 cj966 仍在队列（queued ahead 189）。任务书要求的
"n315/n330 各 128 核集群实测"= 无证据。

**D7（中，集成缺位）零调用方**（§2 L3 行）：并行内核与主求解器 `assembly/solver/analysis_workflow`
无接口，平台侧无调用，故对本平台"船舶总体性能"能力面净增为 0，只有库内可复用资产价值。

**D8（低，一致性）报告 evidence 写"13 passed, 1 failed"，交付仓实跑 12 passed/1 failed/1 skipped**
（差额就是 D3 那个 skip）。

**D9（低，卫生）`scripts/_patch_pool.py`（会改写 `src/tensorfem/parallel/comm.py` 源码的补丁脚本）
与 `_probe_halo.py`/`kb_probe.py` 已合入 main。** 交付仓里留"改自己源码"的脚本不设防。

---

## 5 处置（disposition）

* **请求修改（request changes）**，但**不重开原单终态账**：`level = 2` 落表，gate 将按判据自动裁定；
  若 gate 判 confirmed 则消账，D1/D2/D6 作为返修/后续节点输入而非推翻数值正确性。
* 无阻塞级"造假"：m14685 的数值层（单元矩阵、二阶收敛、分区分不变性、honest FAIL）**经独立复算为真**；
  问题集中在**上报字段语义、邻接实现、证据放置、测量装置、安全卫生**。
* 明确未做/未验证项：集群侧 MPI 实测（cj966 排队中）、`MPIComm` 分支执行、弱扩展的稳定复现
  （本次复跑 E(8)=0.398 vs 原件 0.270，同向 FAIL）、大规模（>1.6M DOF）gate 稳定性。
* 独立性限制：本审计在同一台共享控制面主机上复跑计时，无法排除与原件不同的整机负载；因此
  "E(4) 不可复现"这一结论本身也只在本机噪声条件下成立。

## 6 证据指针

* 原件工件（绝对路径，gate `files_exist` 可核）：`/root/tp-agent/work/m14685/artifacts/parallel_scaling/
  {strong,strong_800,strong_1280,weak,scaling}.json`
* 本次复跑工件：`artifacts/audit_m14964/repro_strong_640/`、`artifacts/audit_m14964/repro_strong_1280/`
* 图：`artifacts/audit_m14964/efficiency_orig_vs_repro.png`、`artifacts/audit_m14964/residual_field_bug.png`
* 生成脚本（零手抄，全部从 JSON 读）：`scripts/audit_m14964_figures.py`
* 复跑命令：见 §3