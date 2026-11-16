# AUDIT-RESTAT v3 · 成果落地存活层（manifest survival）

生成：2026-11-16T04:52:31+00:00 ｜ 快照 cutoff：`2026-11-16 05:00:00+00` ｜ 复跑：`python3 tools/audit_restat/restat_v3_survival.py --snapshot '2026-11-16 05:00:00+00'`

> 本文件补的是 AUDIT-RESTAT 三层既有交付（11-11 首批 / m22264 校准 / m22304 v2 快照锁定）共同的空白：**三层都没有看仓库本身**。所有「交付了什么」的判断都来自 `mission.result` 的自述元数据（`changed_files`/`on_disk_changes`/`pr_gate`）。本层把自述
> 清单逐条搬到各仓 HEAD 核对，得到「宣称 → 落地」矩阵。它不评价正确性（tests/gate 的事），只回答：报过账的文件，现在还在仓库里吗？

## 0. 结论摘要

| 项 | 值 | 含义 |
|---|---:|---|
| succeeded 任务（快照内） | 14,855 | 名义分母 |
| 其中宣称了文件清单 | 14,783（99.5%） | 清单覆盖率 |
| 宣称文件条目 | 254,378 | — |
| 严格存活（路径命中 HEAD） | 109,142（42.9%） | 文件级落地率 |
| 仅同名存在（弱证据） | 24,072 | 上界 52.4% |
| 清单内 worktree 绝对路径 | 48,165（18.9%） | ⇒ 从未进版本控制 |
| 0 存活任务（P0） | 3,409（23.1%） | 硬性落地亏损 |
| 宣称/落地 膨胀因子（任务级） | 1.3 | 宣称单 ÷ 至少 1 条目落地单 |
| 宣称/源码本体落地 膨胀因子 | 1.52 | 更严口径 |
| 宣称/落地 膨胀因子（条目级） | 2.331 | 文件级 |
| 争夺路径占存活条目 | 51.2% | 存活率的假阳性上界 |
| gate≥L2 判定却 0 存活 | 63 单 | 门自身的落地亏损 |
| merged∧0 存活 抽样历史探针 | 路径级：曾入主干 78.6% / 仅分支 0.0% / 无史 21.4% | 见 §6.1 |

## 1. 六类成果统计表（快照内，程序化生成）

### 1.1 任务状态

| 状态 | 单数 | 说明 |
|---|---:|---|
| revoked | 588 | 撤销 |
| failed | 5,129 | 诚实失败 |
| running | 28 | 快照时在跑 |
| awaiting_pr_merge | 260 | 灰区：分支已推未合并 |
| queued | 2 | 排队 |
| stopped | 188 | 中止 |
| succeeded | 14,855 | 已交付（本文件核落地） |

### 1.2 能力台账

| 台账状态 | 条数 |
|---|---:|
| in_progress | 70 |
| verified | 41 |
| done | 280 |
| open | 172 |
| planned | 12 |
| confirmed | 1 |
| dropped | 1 |
| **合计** | **577** |

## 2. 落地档分布与膨胀量化

| 落地档 | 定义 | 单数 | 占宣称清单单 |
|---|---|---:|---:|
| 无清单 | 无任何可解析文件清单 | 72 | 0.5% |
| P0 全丢 | 有清单但 0 条路径命中 HEAD | 3,409 | 23.1% |
| P1 仅文档 | 仅文档/报告类存活 | 1,651 | 11.2% |
| P1 外围实现 | 外围实现（bench/tools/tests）存活、无源码本体 | 2,085 | 14.1% |
| P2 源码本体 | 源码本体存活（src/ 或包内代码） | 2,354 | 15.9% |
| P2 源码+测试 | 源码本体 + 测试都存活 | 5,284 | 35.7% |
| 仓名未映射 | repo 名对不上磁盘目录（不猜） | 0 | 0.0% |
| 无 HEAD | 仓目录无 HEAD（空仓/损坏） | 0 | 0.0% |

任务级漏斗（逐段保留率）：

| 段 | 单数 | 保留率 |
|---|---:|---:|
| succeeded | 14,855 | — |
| 宣称了文件清单 | 14,783 | 99.5% |
| 至少 1 个条目在 HEAD | 11,374 | 76.9% |
| 至少 1 个实现类条目在 HEAD | 9,723 | 85.5% |
| 至少 1 个源码本体条目在 HEAD | 7,638 | 78.6% |

## 3. 文件类别 × 存活（「一次性脚本冒充实现」现在有落地证据）

| 类别 | 宣称 | 严格存活 | 存活率 |
|---|---:|---:|---:|
| docs | 67,412 | 31,641 | 46.9% |
| artifacts | 58,370 | 4,062 | 7.0% |
| src | 47,394 | 28,015 | 59.1% |
| tests | 23,007 | 14,492 | 63.0% |
| bench | 19,542 | 12,024 | 61.5% |
| tools | 14,798 | 9,156 | 61.9% |
| root_script | 11,212 | 4,929 | 44.0% |
| root_doc | 5,170 | 3,087 | 59.7% |
| root_other | 4,167 | 750 | 18.0% |
| other | 2,289 | 455 | 19.9% |
| scratch | 988 | 531 | 53.7% |
| unparsed | 29 | 0 | 0.0% |

### 3.1 读法（三条可直接引用的对比）

1. **源码 vs 根目录脚本**：src（47394 条，59.1%） vs root_script（11212 条，44.0%）。同一批 agent 写的根目录一次性脚本，落地率比源码本体低 15.1 个百分点——这正是 m22264 校准单用「脚本冒充实现」扣分的经验判据在仓库层的独立佐证（该单估计污染面 0.75%，本层给出的是更宽的「写了但没进库」面）。

2. **证据工件基本不入库**：artifacts（58370 条，7.0%）——`artifacts/outputs/runs` 下的证据文件绝大多数没有进版本控制（多数仓 `.gitignore` 掉 `artifacts/`），所以「报告里有数」与「仓库可复现」是两件事；引用任何历史精度数字时，先确认其证据文件是否在库。

3. **记账说合并、仓库说没有**：`pr_gate=merged` 的 7,063 单里，P0（0 条存活）881 单占 12.5%。这批单最值得逐个复核——它们既拿得到「已合并」的名分，也拿不出任何仓库内实体。


## 4. 分仓落地率

| 仓 | succeeded 单 | 有清单 | 条目宣称 | 严格存活 | 条目存活率 | P0 全丢单 |
|---|---:|---:|---:|---:|---:|---:|
| TensorPlatform | 2,626 | 2,556 | 33,741 | 12,006 | 35.6% | 1,258 |
| ship-performance-platform | 2,207 | 2,207 | 44,391 | 13,889 | 31.3% | 227 |
| TensorLBM | 1,989 | 1,989 | 28,999 | 14,023 | 48.4% | 433 |
| TensorDEM | 1,160 | 1,160 | 20,358 | 6,296 | 30.9% | 424 |
| TensorFEM | 1,024 | 1,024 | 18,205 | 11,657 | 64.0% | 67 |
| TensorBEM | 997 | 997 | 15,375 | 9,904 | 64.4% | 59 |
| TensorSPH | 897 | 896 | 13,692 | 8,568 | 62.6% | 45 |
| TensorFVM | 853 | 853 | 16,880 | 8,977 | 53.2% | 238 |
| TensorPapers | 797 | 797 | 24,254 | 3,511 | 14.5% | 330 |
| TensorCAD | 723 | 723 | 11,391 | 4,077 | 35.8% | 186 |
| TensorDG | 671 | 671 | 8,889 | 5,709 | 64.2% | 47 |
| TensorKnowledge | 514 | 514 | 9,235 | 5,486 | 59.4% | 64 |
| TensorPhysics | 350 | 350 | 7,723 | 4,544 | 58.8% | 19 |
| TensorIce | 25 | 25 | 632 | 254 | 40.2% | 12 |
| TensorBench | 15 | 14 | 383 | 175 | 45.7% | 0 |
| TensorData | 7 | 7 | 230 | 66 | 28.7% | 0 |

## 5. pr_gate × 落地档（记账层 vs 仓库层）

| pr_gate | 单数 | 有清单 | 条目存活率 | 无清单 | P0 全丢 | P1 仅文档 | P1 外围实现 | P2 源码本体 | P2 源码+测试 | 仓名未映射 | 无 HEAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 7,655 | 7,583 | 30.9% | 72 | 2520 | 641 | 935 | 1754 | 1733 | 0 | 0 |
| merged | 7,063 | 7,063 | 52.5% | 0 | 881 | 999 | 1121 | 588 | 3474 | 0 | 0 |
| open | 137 | 137 | 74.1% | 0 | 8 | 11 | 29 | 12 | 77 | 0 | 0 |

## 6. 判定层 × 落地档交叉表

行=机器门 `gate_level`（无判定=ungated），列=落地档。

| gate 档 | 无清单 | P0 全丢 | P1 仅文档 | P1 外围实现 | P2 源码本体 | P2 源码+测试 | 仓名未映射 | 无 HEAD | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L5 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| L4 | 0 | 8 | 19 | 10 | 12 | 5 | 0 | 0 | 54 |
| L3 | 0 | 12 | 8 | 7 | 12 | 3 | 0 | 0 | 42 |
| L2 | 0 | 43 | 52 | 16 | 40 | 12 | 0 | 0 | 163 |
| L1 | 0 | 67 | 110 | 30 | 67 | 14 | 0 | 0 | 288 |
| L0 | 0 | 3 | 9 | 0 | 3 | 4 | 0 | 0 | 19 |
| ungated | 0 | 2 | 2 | 1 | 0 | 0 | 0 | 0 | 5 |

gate≥L2 却 0 存活的样本（最多 12 条，全量见 JSON `gate_l2plus_zero_persistence`）：

| mission | gate_level | claim_level | 落地档 | 仓 | 清单条目 |
|---:|---:|---:|---|---|---:|
| 17588 | 2 | 2 | P0_gone | TensorDEM | 11 |
| 15900 | 2 | 2 | P0_gone | TensorLBM | 14 |
| 17491 | 3 | 3 | P0_gone | TensorDEM | 6 |
| 15017 | 3 | 3 | P0_gone | TensorDEM | 11 |
| 15187 | 3 | 3 | P0_gone | TensorDEM | 25 |
| 15184 | 2 | 2 | P0_gone | TensorPapers | 23 |
| 15708 | 2 | 2 | P0_gone | TensorDEM | 83 |
| 15189 | 2 | 2 | P0_gone | TensorDEM | 22 |
| 15226 | 2 | 2 | P0_gone | TensorFEM | 15 |
| 15709 | 2 | 2 | P0_gone | TensorDEM | 11 |
| 15218 | 4 | 4 | P0_gone | TensorLBM | 19 |
| 16226 | 4 | 4 | P0_gone | TensorDEM | 20 |

## 6.1 历史探针：merged 却 0 存活的单，路径到底进过主干吗？

抽 50 单（按 mission_id 等距；条件 pr_gate=merged 且 P0 全丢，共 730 条宣称路径），对每条路径跑两次 `git log --name-only --diff-filter=AM -- <路径>`：一次 `--all`（任何分支）、一次 `HEAD`（主干可达历史）。路径级归因：

| 路径判定 | 含义 | 路径数 | 占比 |
|---|---|---:|---:|
| landed_then_removed | 主干历史里有 ⇒ 入库后被删/替换 | 574 | 78.6% |
| branch_only | 只在 agent 分支历史 ⇒ 从未进主干 | 0 | 0.0% |
| never_existed | 任何历史都没有 ⇒ 清单夸大/路径写错 | 156 | 21.4% |

任务级：**2 单（4.0%）在主干历史里毫无踪迹**（其全部宣称路径都只活在分支或根本不存在）。

| mission | 仓 | 宣称路径 | 任务判定 | 主干命中 | 仅分支 | 无史 |
|---:|---|---:|---|---:|---:|---:|
| 4316 | TensorBEM | 1 | never_existed | 0 | 0 | 1 |
| 8752 | TensorPapers | 10 | trunk_evidence | 10 | 0 | 0 |
| 10119 | TensorDEM | 21 | trunk_evidence | 9 | 0 | 12 |
| 11157 | TensorDEM | 32 | trunk_evidence | 22 | 0 | 10 |
| 13998 | TensorDEM | 8 | trunk_evidence | 3 | 0 | 5 |
| 14681 | TensorDEM | 19 | trunk_evidence | 13 | 0 | 6 |
| 14951 | TensorDEM | 12 | trunk_evidence | 8 | 0 | 4 |
| 15192 | TensorDEM | 13 | trunk_evidence | 13 | 0 | 0 |
| 15418 | TensorDEM | 21 | trunk_evidence | 17 | 0 | 4 |
| 16016 | TensorDEM | 1 | trunk_evidence | 1 | 0 | 0 |

## 7. 未达验收待完成名单（供 owner 裁定重开）

台账未完成项 **254** 条；按状态：open=172，in_progress=70，planned=12

| 域 | 未完成条数 |
|---|---:|
| TensorLBM | 101 |
| TensorKnowledge | 14 |
| TensorIce | 11 |
| TensorCAD | 11 |
| TensorSPH | 9 |
| platform | 8 |
| Platform | 8 |
| TensorPlatform | 7 |
| 可微LBM/反演 | 7 |
| TensorDEM | 5 |
| ship | 5 |
| TensorBEM | 5 |
| knowledge | 4 |
| 冰区航行 | 4 |
| TensorDG | 4 |
| ship-performance-platform | 4 |
| quality | 3 |
| LBM | 3 |
| TensorFVM | 3 |
| 冰区航行/结构 | 3 |
| TensorAppFactory | 2 |
| knowledge-infrastructure | 2 |
| tensorfvm | 2 |
| TensorPapers | 2 |
| TensorBench | 2 |
| TensorData | 2 |
| TensorPhysics | 2 |
| lbm | 1 |
| bem | 1 |
| cad | 1 |
| dg | 1 |
| fem | 1 |
| fvm | 1 |
| sph | 1 |
| platform-mechanism | 1 |
| ship-platform | 1 |
| cross-repo | 1 |
| FEM | 1 |
| FVM | 1 |
| ice | 1 |
| DEM | 1 |
| 冰区航行/平台可判定性 | 1 |
| Papers | 1 |
| bench | 1 |
| TensorFEM | 1 |
| cavitation | 1 |
| Verif | 1 |
| 冰区结构/冰载荷 | 1 |

灰区：`awaiting_pr_merge` **260** 单（分支已推未合并，既不进 done 也不进 failed）。

<details><summary>未完成条目明细（前 60 条；全量见 JSON `ledger_unfinished`）</summary>

| 状态 | 域 | slug | 验收线（截断） |
|---|---|---|---|
| in_progress | quality | `AUDIT-ASSET-12REPO` | 13 仓全扫：分类清单（算法/API/测试/文档/占位/重复/未接入）+每核心功能六问+verdicts 落表 |
| in_progress | TensorKnowledge | `AUDIT-DK` | ①盘点近万报告的存量与结构（SQL实数）②四契约Schema/评审状态机代码真实性③抽取1-2个真实样本走Raw→Pending→Governed全链留证据④可复用性实测：新任务引用Governed KO是否产生可测收益（对照不用）；禁只写 |
| in_progress | TensorAppFactory | `AUDIT-P0LINE` | 审计报告五维PASS/FAIL/NOT-TESTED逐项带证据：①Manifest校验器/项目生成器真跑 ②生成项目可独立构建出APK+SHA256（SDK=/opt/android-sdk）③CI工作流与本地构建一致性 ④ProductF |
| in_progress | TensorPlatform | `AUDIT-PLAT` | 按issue#60验收步序出PASS/FAIL/NOT-TESTED报告：①部署身份（容器SHA vs main）②worker自主领单无外部Claude会话驱动③重启后任务/checkpoint/记忆恢复④失败重试终态；本单Agent即跑 |
| in_progress | quality | `AUDIT-RESTAT` | 六类成果统计表+未达验收待完成名单（供 owner 裁定重开）+成果膨胀量化 |
| in_progress | quality | `AUDIT-RUNTEST-6` | 6 仓各真跑核心测试+代表算例+负向+收敛/守恒；重计算走 cluster_queue；实测 vs 文档声称对照表 |
| in_progress | TensorDEM | `AUDIT-SCI` | 两仓逐条对账：README每条能力声称→定位代码文件+测试+基准数值证据；DEM重点=冰脊触底/管道地基/船体冰磨损是否真有实现与算例；SPH重点=3D算子验证真伪+自由液面/表面张力/多相未完成项如实标；输出声称-证据对照表，虚标逐条列 |
| in_progress | ship | `CAP-LEDGER-EVIDENCE` | 【基准牵引/台账判据集｜ref: 21 项能力台账逐项判据（总账 acceptance 原文）｜判据: 有判据+工件+独立复核项数/21 =21/21】21 项台账全部有判据+工件+独立复核（现仅 3 有证据） |
| in_progress | lbm | `FEAT-LBM-VERIFY-BENCH` | Young-Laplace ≥3σ 误差<5% 或 honest-FAIL+根因；spurious current 基线数字表（网格×σ）；pytest 固化；消费链映射表≥5 条 oss_solutions 条目 id→工装代码 |
| in_progress | platform | `FEAT-MATURITY-L0L5` | tensor_maturity_verdicts 表+finish/打勾路径挂成熟度证据门（分级上线：v1 周报披露→v2 tb-eval-loop 后硬门） |
| in_progress | bem | `FEAT-OSS-INVENTORY-BEM` | 普查≥20 仓 registry JSON+tier 分级+GAP-BEM-* 锚点提议≥6；深盘波按 LBM 协议接力 |
| in_progress | cad | `FEAT-OSS-INVENTORY-CAD` | 普查≥20 仓 registry JSON+tier 分级+GAP-CAD-* 锚点提议≥6；深盘波按 LBM 协议接力 |
| in_progress | dg | `FEAT-OSS-INVENTORY-DG` | 普查≥20 仓 registry JSON+tier 分级+GAP-DG-* 锚点提议≥4；波2 深盘按 LBM 协议接力 |
| in_progress | fem | `FEAT-OSS-INVENTORY-FEM` | 普查≥20 仓 registry JSON+tier 分级+GAP-FEM-* 锚点提议≥6；深盘波按 LBM 协议接力 |
| in_progress | fvm | `FEAT-OSS-INVENTORY-FVM` | 普查≥20 仓 registry JSON+tier 分级+GAP-FVM-* 锚点提议≥6；深盘波按 LBM 协议接力 |
| in_progress | LBM | `FEAT-OSS-INVENTORY-LBM` | ≥24 个 GitHub LBM 项目逐一代码级盘点（T1 多相 6/T2 可微 GPU 8/T3 HPC 10+发现波，总帽 30）；每项目 oss_inventory 落档；按功能点归集 oss_solutions ≥40 条（每条带文 |
| in_progress | sph | `FEAT-OSS-INVENTORY-SPH` | 普查≥20 仓 registry JSON+tier 分级+GAP-SPH-* 锚点提议≥4；波2 深盘按 LBM 协议接力 |
| in_progress | platform-mechanism | `FEAT-OSS-PLAYBOOK` | docs/oss-to-tk-playbook.md：五步各带机器可校验判据+已知坑+六域推广 checklist+度量指标 |
| in_progress | ship-platform | `FEAT-SHIP-DEPLOY-ALIGN` | 部署同步线常设（pull→迁移→前端重建→后端重启→健康门→回滚）+三车道对齐（8001/8004 死道）+契约漂移门进 deploy gate |
| in_progress | cross-repo | `FEAT-SIM-DATAHOUSE` | 【①compute 类成功单场数据自动捕获覆盖≥80%（worktree 清理前抢收 .npz/.npy/.vtk/.csv+参数+几何）②≥60 事件带场演化动画 mp4 入库可查③多模态数据集 v1：≥60 事件×五模态对齐（场快照序列 |
| in_progress | platform | `FEAT-TENSORBENCH` | 六物理锚 spec 化+RE-Bench 式任务集+结果回写评价回路三件全通 |
| in_progress | knowledge | `FEAT-TK-ASSET-OPS` | KO 版本链与生命周期登记+主页知识专栏三 tab 真数据+版本化数据集下载服务(sha256+增量+published-only 硬门) |
| in_progress | knowledge | `FEAT-TK-FEEDBACK` | task-kit 派单装配+AnchorCard 进任务契约与 finish 门容差对照+run_ke_query 主动检索工具+预注册 A/B 证知识增益（判伪即负知识回写） |
| in_progress | knowledge | `FEAT-TK-GOV-INFRA` | VISION 正典化+sediment_line 常设沉淀守护+训练语料/评测集导出三件套落地；P0(存储统一/KE API/旗舰闭环)为地基 |
| in_progress | knowledge-infrastructure | `FEAT-TK-KNOWLEDGE-ENGINE` | ①选型评估给决策矩阵+三路建议且实测 demo（m8618）；②四契约映射规格对试点包零损耗或显式披露（m8619）；③原型对试点包查询全可复现且 xref 一致性断言过（m8620） |
| in_progress | knowledge | `FEAT-TK-ONTOLOGY` | 本体 schema v1（六形态 JSON Schema+关系词表）+LBM 方法轴全量重组+船海领域轴试点，task-kit 契约成文 |
| in_progress | knowledge-infrastructure | `FEAT-TK-OSS-LANDING` | A1 ≥20 多相/MCMP claims+A2 ≥15 可微/框架 claims：schema 全过+全部 oss-cache 实读定位逐字引文+自带校验器全绿+无自批+每条映射回 GAP 锚点 |
| in_progress | tensorfvm | `FEAT-TURB-SST-GATE` | ①NASA TMR 零压力梯度平板：Cf 对标 TMR 验收目标容差内+对数区 u+ 律+首格 y+ 敏感性披露+三级网格收敛②槽道 Re_tau=180 对数律+摩擦速度 vs DNS 容差内③每案 benchmark_cases 绑定  |
| in_progress | tensorfvm | `FEAT-UNSTR3D-MESH` | 3D cell-centered 非结构 FVM：①Gmsh ASCII v2 tet 导入带物理边界组②最小二乘梯度+扩散非正交修正+迎风对流③tet 网格 Poisson 解析解收敛阶≥2（两级加密实测）④45°对流有界性容差内⑤pol |
| in_progress | TensorLBM | `GAP-DIFFLBM-ADJOINT-VERDICT` | 同一目标（升力）同一参数（盘心 y_c）下：cos(展开式AD, 冻结场代理) 若 <0.99，代理不得进优化路径，且 docstring/返回值显式标注 shape_derivative_only=True。实测 cos=-1.000、幅 |
| in_progress | TensorLBM | `GAP-DIFFLBM-CAV-RES` | 给出 max-point 误差 vs n 的双对数曲线与观测阶 p≥1.8；max-point <3% 须在 n≥97 达成。现状：n=65 LBM max-point 5.29%（诚实 FAIL，L2 0.61% 达标）、FD n=65  |
| in_progress | TensorLBM | `GAP-DIFFLBM-PAPER-EVIDENCE` | 解析 2 + 文献 3 + 交叉求解器 1 共 ≥6 基准，逐项列误差并显式回答是否 <3%；现 1 个解析基准（Poiseuille ≤2.36e-3，观测阶 2.21），文献/交叉求解器基准缺 |
| in_progress | TensorPapers | `GAP-DIFFLBM-XSOLVER-ORDER` | 两族无关离散（BGK-LBM 与涡量-流函数 FD）同网格中心线差 max-point <3% 且该差随 h 的观测阶 p≥1.8；须覆盖 n≥{65,97,129,161} 四档且每档两族均已判据收敛（LBM /Δu/<1e-8·U_li |
| in_progress | FEM | `GAP-FEM-3D-COHESIVE` | 【基准牵引/DCB 双悬臂梁 I 型剥离｜ref: ASTM D5528 + 梁理论 P-δ 解析（LEFM）｜判据: P-δ 曲线+能量释放率 vs 解析 <10%】3D cohesive zone 单元+验证算例 |
| in_progress | FVM | `GAP-FVM-TWOPHASE` | 【基准牵引/Zalesak 旋转缺口盘 + Hysing 上升气泡｜ref: Zalesak 1979；Hysing et al. 2009 benchmark 数据集｜判据: 界面形状误差/末速 vs 基准数据 形状 qualitativ |
| in_progress | TensorIce | `GAP-ICE-FLOE-VALID` | 【基准牵引/多面体堆体休止角｜ref: atan(mu) 单调关系 + 力链尾部指数 ｜判据: 11/11 判据全 PASS（现状 9/11，case3 NOT VALIDATED）；dt 细化 5 倍使 omega*dt ≤0.025，能 |
| in_progress | ice | `GAP-ICE-GATE-POWER` | 【判据信息量闸/JOUR-ICE-STRUCT REAL 面｜ref: report 18644 S02/S07/S08=RESTATE｜判据: 每条 REAL 判据须给 (a) 基准来源 (b) 门限对应观测阶或检出率 (c) 至少一个被 |
| in_progress | TensorIce | `GAP-ICE-LIT-TRANSCRIPT` | 【基准牵引/Lindqvist 1989 POAC'89 三分解｜ref: 原文系数表｜判据: ≥4 个 (公式,quote,page/table) 三元组逐字转录，M1 由 WITHHELD 转 PASS；系数等级不得在未达成时升级为 R |
| in_progress | TensorIce | `GAP-ICE-PA-IDENT` | 【基准牵引/p-A 指数分离｜ref: ISO 19906 ex=-0.28 + 本链内生性地板 /ex/<0.126 ｜判据: 处理臂（速率相关挤压）恢复的 ex 与 -0.28 偏差 ≤15%，且高出负控地板 3σ（bootstrap  |
| in_progress | TensorIce | `GAP-ICE-TAIL-NEFF` | 【基准牵引/设计分位统计不确定度｜ref: Weibull delta 法观测信息阵 + 移动块 bootstrap + copula MC 标定 ｜判据: (1) 每工况设计分位 p99 的 95% CI 半宽 ≤10%（现状 base  |
| in_progress | LBM | `GAP-LBM-MCMP-POISEUILLE` | 【基准牵引/MCMP 分层 Poiseuille 解析解｜ref: Huang-Krafcyk-Luo 2011 MCMP 解析｜判据: 界面位置/流量/spurious current vs 解析 <5%/<10%/<0.005U】①稳态 |
| in_progress | LBM | `GAP-LBM-SPHERE` | 【基准牵引/球绕流阻力关联｜ref: Clift-Grace-Weber 球阻力标准关联（Schiller-Naumann 0.1<Re<1000）｜判据: C_D vs 关联值 <8%】曲面边界力法+球体绕流验证 |
| in_progress | DEM | `H1-BROKEN-ICE-DEM` | 【基准牵引/浮冰浮态+航道清空｜ref: 阿基米德浮态解析（密度比 0.9 干舷/吃水）+冰区航道清空实测统计｜判据: 干舷 vs 解析；清空率统计 干舷<5%；清空率如实报】浮沉/接触/航道清空（摩擦触达需声明） |
| in_progress | Platform | `H6-MATURITY-GATE` | 【基准牵引/gate 覆盖率计量｜ref: KPI-GATE-100 计量口径（产出线×门禁矩阵可失败性）｜判据: 门禁覆盖率=可失败门/产出线 =100%】G1/H6 验收门报告 |
| in_progress | 可微LBM/反演 | `JOUR-DIFFLBM-ARTIFACT-INREPO` | 对 cited 工件路径集合执行 git ls-files 命中率=100%，且 CI 增加一条 MISSING 即红的检查 |
| in_progress | 可微LBM/反演 | `JOUR-DIFFLBM-EVID-TRACE` | 重写 GAP-DIFFLBM-PAPER-EVIDENCE 报告后，papers/tools/jd_audit.py 复跑得 n_claims_without_artifact=0、n_withheld_undefined_metric=0 |
| in_progress | 可微LBM/反演 | `JOUR-DIFFLBM-L3-INVERSION` | L3 反演：≤10% 节点稀疏观测反演体积力场，相对 L2 ≤5%，≤50 次伴随迭代，收敛曲线入库 |
| in_progress | 可微LBM/反演 | `JOUR-DIFFLBM-REF-VERIFY` | 立项引用逐条 DOI 核证：题名/作者/年/DOI 一致；不一致者改引或降级为“未核证” |
| in_progress | 冰区航行 | `JOUR-ICE-STRUCT-BENCH-DATA` | 【论文牵引/JOUR-ICE-STRUCT 基准数据集】why_blocked: 现网 12 项判定中 1 项 WITHHELD（平冰总阻力无参考值）、1 项 NOT_SUPPORTED（碎冰 DEM-FEM 耦合），根因是缺一套可复现的基 |
| in_progress | 冰区航行 | `JOUR-ICE-STRUCT-CALIB-PROV` | 【论文牵引/JOUR-ICE-STRUCT 标定系数溯源】why_blocked: 现网 total = raw × calibration_factor，cf 船内恒 0.753028、跨船离散 34.5%（0.670077-0.9011 |
| in_progress | 冰区航行/平台可判定性 | `JOUR-ICE-STRUCT-GUARD` | 【论文牵引/JOUR-ICE-STRUCT 越域护栏】why_blocked: 冰域越域输入 7/12 静默外推（h=50 m→2450 MN 级、σ_c=100 MPa→143.9 MN），护栏只在 schema 层；且 GET /ice |
| in_progress | 冰区航行 | `JOUR-ICE-STRUCT-REF-LADDER` | 论文中每个精度主张必须并列三个数：参考带（最优已发表公式 22.70% MAPE vs 10 工况实测）、实现债（仓内 Lindqvist 与已发表列差 38.43%）、数值/采样条（dt 8.51%、种子 14.72%）；新增任何外部参考 |
| in_progress | ship | `KPI-BENCH-10` | ≥10 个公开对标算例通过 |
| in_progress | ship | `KPI-CYCLE-50-30` | 相对基线的设计周期与试验成本降幅 |
| in_progress | Platform | `KPI-GATE-100` | 全部产出线接质量门禁 |
| in_progress | Platform | `KPI-QC-RECHECK` | 已合并功能 100% 有独立参考复核且通过 |
| in_progress | Platform | `KPI-REPRO-95` | 第三方复现成功率 ≥95% |
| in_progress | ship | `KPI-RES-5-10` | 阻力预报误差达标实测 |
| in_progress | Papers | `KPI-STD-PATENT-PAPER` | 四类知识产权产出达标 |
| in_progress | Platform | `KPI-SUCCESS-90` | 任务成功率 ≥90% 且失败自动归因 |

</details>

## 8. 注意点与局限（honest）

1. **清单本身是自述**：`changed_files`/`on_disk_changes` 由 agent 填写，可能漏写或夸大；本层只核「写进去的」，不核「没写的」。故落地率是**下界口径的宣称精度**，不是绝对产出。
2. **存活 ≠ 正确**：路径命中只说明该文件存在于 HEAD，不说明内容来自该单、也不说明测试通过。内容级归属需要 `git log -S`/blame，本层未做（明示范围）。
3. **同名假阳性**：争夺路径（被 ≥2 单宣称的相对路径）占存活条目 51.2%——这是存活率的假阳性上界；`basename_only` 弱证据已单列、未混入严格存活。
4. **HEAD 时点**：各仓 HEAD sha/日期逐仓记录于 JSON `repo_heads`；仓镜像由 `mirror_ff.sh` 每 15 分钟追平，快照后合并的交付不计入存活（保守方向）。
5. **仓名映射**：仅收录确有目录对应的别名（如 `ship-platform`→`ship-performance-platform`）；对不上者单列 0 单，不猜。
6. **绝对路径条目 18.9%**：这类条目在语义上「从未入库」，但其 worktree 相对候选仍被严格比对过一遍（避免把「后来以相对路径合并」误判为丢失）。

## 9. 复算与已执行验证

```bash
python3 tools/audit_restat/restat_v3_survival.py --snapshot '2026-11-16 05:00:00+00'
python3 -m pytest tests/test_restat_survival.py -q
```

| 检查 | 结果 |
|---|---|
| 口径纯函数单测（清单解析/路径归一/分类/落地档/膨胀/漏斗/仓名映射） | 见 pytest 日志 |
| 同快照两次运行 JSON 一致性 | 见 §下方运行时输出（保留 generated_at 外逐字节一致） |
| 仓 HEAD 身份记录 | 逐仓 sha + 日期见 JSON `repo_heads` |

## 10. 图

- `fig_restat_v3_funnel.png`
- `fig_restat_v3_kind_survival.png`
- `fig_restat_v3_repo_mix.png`
- `fig_restat_v3_gate_heatmap.png`
