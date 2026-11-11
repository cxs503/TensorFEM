# 交差核验审计：m15144 REPRO-FEM-HILLERBORG1976（TensorFEM）

**审计席位** board-audit｜**审计单** m15268｜**被审单** m15144（status=succeeded，claimed_by=tp-qw38-5）
**被审分支** `agent/m15144` head `fb4a40158e417e5596fdefccec4cfdbaa56c343a`
**审计工作树** `/root/tp-agent/work/m15268`（main head `1e701d1f`）｜**被审工作树** `/root/tp-agent/work/m15144`
**复跑指令** `python3 scripts/audit_m15144_m15268.py`（产出 `artifacts/m15268_audit/verify_output.json` + 2 图）
**本文件所有数字**均由该脚本现查现算（文件 sha256 / git 祖先关系 / SQL 行 / 独立复跑产物），无手抄。

---

## 0. 裁定（disposition）

| 项 | 结论 |
|---|---|
| 功能等级 | **L1（数学/单元级验证脚本 + 可核工件）**，claim 上限即 L1；L2/L3/L4 均不达 |
| 判据裁定 | 论文三条判据：**C1 NOT_RUN（a/W 序列未构型）/ C2 FAIL（G_F 反算未跑通，能量口径误差 408%）/ C3 WITHHELD（记录值 +43% 与复跑 +2.9% 相互矛盾）** |
| 基准裁判 | `tensor_benchmark_cases` 中 **anchor_slug='REPRO-FEM-HILLERBORG1976' 注册行数 = 0** → 无机器裁判基准，容差核验无从对表 |
| 三件套 | run_artifacts **存在且可核**（分支内 sha256 固定）；ref vs got **存在**；repro_cmd **存在但不可完成**（rc=1） |
| 报告六段 | 文档形态 **齐 7 段**（分支 `docs/repro-hillerborg1976.md`，127 行）；但 `tensor_reports` **无该单行（0 行）** → 六段报告未入库 |
| 处置 | **reopened（返修）**：核心缺陷是「工件不可由 repro_cmd 再生」而非「数值不达标」——被审单的 honest-FAIL 结论方向正确，但其 FAIL 的具体数字本身不可信 |

---

## 1. 独立性声明与检查范围

- 独立性：本单不复用被审单任何结论；所有 got 值从 `git show fb4a4015:...json` 读、所有 ref 值从闭式公式与被审脚本原式重算、所有「是否入库」从 git 祖先关系与 `ls-remote` 判定、所有账目从 `psql -U postgres -d tensorplatform` 现查。
- 局限（已知未做）：①未取得论文原文 PDF 逐图数字化（本单不掌握 Cem. Concr. Res. 6:773-781 的图表数据），因此「论文 ref」一律限定为**被审脚本自带的 σ_MOR 闭式**；②未改被审单代码（交差核验非重做）；③未在被审工作树内复跑（避免污染其工件），复跑在 `artifacts/m15268_audit/repro_copy/` 独立副本内进行。

---

## 2. 三件套逐项取证

| 件 | 指针 | 核验结果 |
|---|---|---|
| run_artifacts | `scratch/repro_hillerborg1976/coarse_16x16.json` (1737 B, sha256 `9b32e1e01b13…`)、`coarse_16x16.npz` (6770 B, `e2c98135a81b…`)、`fine_32x32.json` (2330 B, `8ac3a3b903ef…`)、`fine_32x32.npz` (8682 B, `846e47d58ce8…`)、`brittle_16x16.npz` (3744 B, `3dcbde8ca9cc…`) | ✅ 全部在 `fb4a4015` 中，工作树与提交逐字节一致（`worktree_matches_branch_commit=True`）；⚠ `brittle_16x16.npz` 为**孤儿工件**：提交的脚本无任何路径产出该 tag（`grep brittle scratch/m15144_repro.py` 仅命中注释），无同名 json |
| ref vs got | 被审 `coarse_16x16.json`: `analytic_load_n=2666.6666666666674`（=2bh²f_t/3L, `src/tensorfem/cohesive_bending.py:1093-1103`）vs `load_n=2797.875566433862` → **+4.92%**；`fine_32x32.json`: `load_n=4004.263531629244` → **+50.16%** | ✅ 数值真在工件里，可复算；⚠ ref 是**梁理论无缺口极限**（模块 docstring 自述 "analytical anchor for a brittle cohesive law"），**不是论文 1976 的 a/W 序列数值** |
| repro_cmd | 文档 `repro_cmd: python3 scratch/m15144_repro.py` | ❌ **rc=1**：`AttributeError: module 'numpy' has no attribute 'trapz'`（本环境 numpy 2.4.6；`np.trapz` 在 numpy 2.0 已移除），崩在 `m15144_repro.py:176` 的 G_F 反算步 → 判据 C2 的产出从未打印 |

**入库判定（L2 硬门槛）**：`git merge-base --is-ancestor fb4a4015 HEAD` → **False**；`git cat-file -e HEAD:<每个交付文件>` → **8/8 ABSENT**；`git ls-remote origin | grep m15144` → **空**。即：交付物**只在本地分支**，main 与 GitHub 均无。加分项：`scratch/m15144_precheck.py` 连分支都未提交（`?? ` 未跟踪）。

---

## 3. 基准裁判面（唯一裁判）

```sql
SELECT id,anchor_slug,case_name,metric,tolerance,status
FROM tensor_benchmark_cases WHERE anchor_slug='REPRO-FEM-HILLERBORG1976';   -- 0 rows
SELECT count(*) FROM tensorbench_results WHERE mission_id=15144;            -- 0
SELECT count(*) FROM tensor_reports    WHERE mission_id=15144;             -- 0
```

- 库内 14 个 anchor（`H2-FRAZIL-RHEOLOGY / GAP-FEM-3D-COHESIVE / H3-ICE-PLATE-FEM / GAP-DEM-CLUMP / XVAL-FEM-TPB1976 / H6-MATURITY-GATE / GAP-FEM-WINKLER / H1-BROKEN-ICE-DEM / GAP-LBM-MCMP-POISEUILLE / GAP-FVM-TWOPHASE / CAP-LEDGER-EVIDENCE / GAP-DEM-FRICTION / GAP-FEM-ELASTOPLASTIC / GAP-LBM-SPHERE`）中**没有 REPRO-FEM-HILLERBORG1976**，同域最近的是 `id=14 XVAL-FEM-TPB1976`（Hillerborg 1976 SENB，status=pending，另一单 m15165）。
- 结论：本题**无注册基准行 ⇒ 无容差可对表**，判据只能按 mission.goal 原文容差（<10%）人工裁定；这也解释了为何被审单能在「三条判据全不达标」的情况下以 succeeded 结单——验收三件套只查「有没有」，不查「对不对」。

---

## 4. 判据逐项裁定（判据 = mission 15144 goal 原文容差）

| # | 判据（原文） | 容差 | 记录值 | 复跑值 | 裁定 |
|---|---|---|---|---|---|
| C1 | 三点弯 P_cr 闭式解（a/W 序列） | <10% | 无缺口 coarse +4.92% | 同值（逐位一致） | **NOT_RUN**：`build_three_point_bend` 无初始缺口参数，a/W 序列几何无法构型；只跑了无缺口 σ_MOR 极限。**不能**把 +4.92% 记作论文复现 |
| C2 | CMOD 断裂能 G_F 反算 | <10% | `energy_relative_error_pct(fine)=408.42%`；coarse `gf_area_j=0`（裂缝未扩展，反算退化） | 崩溃，无产出 | **FAIL** |
| C3 | 网格敏感性（两尺寸 P_cr 漂移） | <10% | **+43.12%**（2797.9 → 4004.3 N，76 步） | **+2.92%**（2797.9 → 2879.6 N，52 步，3 次独立复跑全同） | **WITHHELD**：两次结论相反（FAIL vs PASS），说明提交的细网格工件不是提交脚本在其当前状态的产物 |

---

## 5. 严重度排序的发现（adversarial-review 口径）

**B1（阻断，可复现性）— 提交的 `fine_32x32` 工件不可由 `repro_cmd` 再生。**
本单在独立副本（同样 `src/`、同样脚本、同样环境）复跑：`coarse` 与提交工件**逐位一致**（2797.875566433862 N / 53 步 / 耗散 0.0115739 J，sha256 相同），证明环境与脚本是确定性的；但 `fine` 给出 **52 步 / 2879.6078 N / 耗散 0.014464 J**，提交工件是 **76 步 / 4004.2635 N / 0.31776 J**（相差 39%）。再做 3 次独立复跑，三次全部 `52 / 2879.6078`（逐位一致）。⇒ **不存在数值非确定性**，而是**细网格工件出自另一版本脚本/另一组参数**（artefact–script mismatch）。被审单的 honest-FAIL 叙事（"32×32 比 16×16 高 43%，不收敛"）建立在不可再生的数字上。

**B2（阻断，执行面）— `repro_cmd` 在当前生态环境 rc=1。**
`np.trapz` 在 numpy 2.0 被删除（环境 numpy 2.4.6），脚本第 176 行崩溃；C2 判据的 G_F 反算从未产出任何数字，而文档 ④ 表格里却呈现了「stop=max_steps / err」等完整行，读者会误以为反算已跑。修复面极小（`np.trapezoid`），但此缺陷同时说明**该单交付时未做端到端重跑**。

**B3（阻断，基准面）— 无注册基准行，验收三件套只看有无、不看对错。**
`tensor_benchmark_cases` 0 行 → 三条判据无从机器裁定，succeeded 结单只代表「工件文件存在」。这是平台侧的机制缺口（mission→benchmark 注册未强制），不是被审单个人问题。

**B4（高，报告纪律）— 六段报告未入库。**
`tensor_reports WHERE mission_id=15144` = 0 行。文档本体在分支上质量尚可（7 段结构、参数表、ref vs got 表、失败模式、7 条生态摩擦），但**平台无报告 ⇒ 守望线视角该单欠报告**。注意被审工作树根目录的 `report.json` 是**别的任务的陈旧文件**（title=IN71-N09 welding、mission_id=9270），不可当作本单报告。

**B5（中，物理论断）— 材料参数替换改变了问题所处的物理区间。**
论文混凝土 `G_F=100 N/m`（l_ch/d=1.875），被审单因数值困难改为 `G_F=10 N/m`（l_ch/d=0.1875），文档注释称"qualitative behavior 相同"。但 Hillerborg 1976 的立论点正是**软化/过程区尺寸效应**随 l_ch/d 的迁移；把 l_ch/d 改小 10 倍等于绕开论文的物理靶心，即便数值收敛也不能算复现。这是「can't do it ⇒ change the problem」的典型风险，必须在报告里显式标为**范围降级**而非等价替换。

**B6（中，内部一致性）— 能量口径残差 −127.7%(coarse) / −42.3%(记录 fine) / −163.2%(复跑 fine)。**
平衡残差被脚本按耗散量归一（`100*residual/diss`），分母是 0.0116~0.0145 J 的小量，导致百分比放大；但无论归一口径如何，`external_work ≠ strain_energy + dissipated` 量级达 40–160%，说明**能量记账本身不自洽**，`converged=True` 不能当成解可信的凭据。

**B7（轻，交付卫生）— 孤儿工件 `brittle_16x16.npz`、未提交 `scratch/m15144_precheck.py`、交付全在 `scratch/`（非包内入口）。**

---

## 6. 等级裁定（L0-L5 机械判据口径）

| 档 | 判据 | 本单实测 |
|---|---|---|
| L0 文档 | 有文档/图 | ✅（127 行 7 段） |
| L1 数学/单元验证 | 有验证脚本，无独立功能实现 | ✅ **达标**：`scratch/m15144_repro.py` 187 行，跑既有 `cohesive_bending` 模块做闭式对照 + 网格敏感性，工件可核 |
| L2 独立功能 | 实现文件 + 工程入口 + 可运行工件（**入库**） | ❌ 三道全断：无 `src/` 实现增量；入口是 `scratch/` 脚本；**未入库**（main 8/8 ABSENT、origin 无分支）；且 repro_cmd rc=1 |
| L3 集成 | 被集成面调用 | ❌ `integration_callers=0`（api/backend/cli/tests 均无引用） |
| L4 基准 | L2 + 对标文献 + run 工件 | ❌ 无注册基准行；C1 NOT_RUN / C2 FAIL / C3 WITHHELD |
| L5 | 新方法 | ❌ 不适用 |

**verdict = L1**（已落表 `tensor_maturity_verdicts`，`verdict_by='board-audit'`）。被审单 headline 自称 "attempted reproduction … honest FAIL"，未虚标 L2+，此点予以肯定；**扣级原因是「可复现性与入库性不达 L2 门槛」，不是「结论是 FAIL」**——honest FAIL 在 S-N 轨是正收益，本单的 FAIL 结论方向（solver 在 l_ch/d≈0.19 下不收敛）与复跑现象方向一致（均 `stop_reason=max_steps` 且裂缝仅扩展 1 层单元），故 B5/B6/B1 的返修不否定其叙事，只否定其数字。

---

## 7. 未做的检查 / 判为 not_run 的项

- 未取得论文原文图表并逐点数字化 → 论文 ref 用不上（C1 的 "a/W 序列" 无法对表，只能报 NOT_RUN）。
- 未评估 `cohesive_bending` 模块本身的正确性（它不在本单交付面；仓内 `tests/test_cohesive_bending.py` 属 main 既有测试）。
- 未在 loong64 集群复核（时长 <1 min/例，无需集群，非未做而是不必做）。

## 8. 返修建议（按性价比排序）

1. `np.trapz → np.trapezoid`（或 `scipy.integrate.trapezoid`）并**端到端重跑**，把 rc=0 与三条判据输出写进工件（B2）。
2. 重跑并**只提交由当前脚本产出的工件**；若记录数字来自旧参数集，须在工件里落 `script_sha256 + 参数集`（B1，结构性防再造）。
3. 给 `build_three_point_bend` 加初始缺口（预损伤底边 a/W 段），否则 C1 永远 NOT_RUN（B5 的共同根因）。
4. 恢复 `G_F=100 N/m`（l_ch/d=1.875）并改用粘性正则/损伤型弧长控制替代 CMOD 牛顿路径跟随——这才是论文靶心区间（B5）。
5. 注册基准行（建议挂 `XVAL-FEM-TPB1976` 家族，另开 REPRO anchor 行），让 gate 能机器裁容差（B3）。
6. 把文档 ⑦ 段发到 `tensor_reports`（六段报告入库），并删掉工作树里 mission_id=9270 的陈旧 `report.json` 避免误挂（B4）。