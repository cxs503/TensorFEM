# 交差审计：m17412 · XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE（TensorFEM）

审计单：m17488（board-audit）｜被审单：m17412（status=succeeded，
`result.fail_stage=budget_exhausted+forced_finish`，自报 honest-partial）
｜裁决：**L2**（verdict 行 `tensor_maturity_verdicts.id=581`，gate_state=pending）
｜结论：**request changes（reopened）** —— 臂 A 真数值可复现，但 XVAL 的第二独立臂、
claim 判定表、run_artifacts、入库四项均不成立。

机器证据：`docs/board_audit/m17488/reports/audit_xval_msfe.json`（本单全部数字的唯一来源）
实跑日志：`docs/board_audit/m17488/reports/xval_full_run.log`
图：`docs/board_audit/m17488/fig_xval_ladder.png`

## 0. 独立性与范围

* 被审树 `/root/tp-agent/work/m17412`（branch `agent/m17412`，HEAD `a1572ffb`）**只读**；
  一切执行在 `shutil.copytree` 出的 `/tmp/aud17488_t` 副本里做，被审树零字节写入。
* 基准裁判只用 SQL（`su postgres -c psql -d tensorplatform`），不引用被审报告的任何自述数字。
* ref 侧（Lamé 闭式解）由本审计脚本**自写公式**（Timoshenko & Goodier §5.6）第三方重算，
  与被审 `lame_at` 的最大相对差 **1.33e-15**（r=8/9/9.5/10 四点）——ref 侧无造假。
* 未做的检查见 §6；不做 GitHub CI（x86）复跑，本单全部 loongarch64 本地实跑。

## 1. 基准裁判（唯一裁判）：`tensor_benchmark_cases`

```sql
SELECT case_name,metric,tolerance,status FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE';   -- 0 rows
```

| 查询 | 结果 |
|---|---|
| 锚点注册行数 | **0** —— 没有机器裁判可判「got 是否在容差内」 |
| 下游算例 `CASE-HOLLOW-SPHERE-3D` | tolerance `<5%`，status **in_progress**（未 passed） |

⇒ 按「基准判据是唯一裁判」的口径，本锚点**不存在通过的基准**；任何 L4 主张都无据。
容差内判定只能用记录自身声明的 predeclared 规则（15% 复现容差 / <3% 阈值 / <5% 案例线），
本单按此口径逐项复核（§3）。

## 2. 三件套取证（任务书要求的「真存在」）

| 项 | 实况 | 判定 |
|---|---|---|
| impl 文件 | `benchmarks/xval_method_selection_first_evidence.py` 32956B sha256=`60dd869ce5d7520b…`；`…record.json` 2219B sha256=`9b2f2572768dd139…` | **存在**，但 `in_main=false`（5/5 文件未合主干） |
| run_artifacts | `artifacts/xval_method_selection_first_evidence/run.log` **472B / 11 行**，末行停在 `[arm A] TensorFEM order=2 n=6 ...`（无结果行）；`…_quick/` **空目录**；**无 evidence.json、无图** | **不完整**（未达「可复现产物」） |
| benchmark ref vs got | 锚点 0 注册用例；被审树内 `evidence.json` 从未生成 ⇒ **无落盘 got 值可比** | **缺失** |
| repro_cmd | `PYTHONPATH=$PWD/src python3 -m pytest tests/test_xval_method_selection_first_evidence.py -q` → **4 failed / 4 passed (6.6s)**；`python3 benchmarks/xval…py` 端到端 → **rc=1**，在 `arm_skfem` 第 375 行 IndexError 崩溃，**从未跑到 claims 段** | **不可跑通** |
| tests 文件 | 6160B sha256=`859fb490cb04d756…`，8 个用例，协议断言写得实（不是只执行代码） | 存在但**红** |
| 平台集成 | `grep -r xval_method_selection src/` 命中 0；无调用方；`in_main=false` | **无 L3 面** |
| 外部锚点 | Lamé 闭式解（教科书），记录自述 `oss_inventory` 无 3D FEM 可横向对照 | 有解析锚点、**无外部数值锚点** |

## 3. 声明复核（被审 harness 的 6 条 claim，全部由本单复算）

被审 harness 自定「数值 claim 复现判据 ≤15%」。复算在 `/tmp` 副本内跑臂 A 六档 + 臂 B 三档，
臂 A 六档 M1/M2 与 mission 日志**逐位相同**（最大差 0.00000）。

| claim | 记录原文 | 本单复算 | 判 |
|---|---|---|---|
| C1 | HEX20(n=4,192 单元) **全部探针 <3%** | M1=**2.875%**（成立）/ M2=**11.989%**（不成立，最差量 radial_stress_inner） | **不可判**：结论取决度量泛函，记录未声明泛函 ⇒ 必须重述为「M1<3%」 |
| C2 | HEX8 **同网格** worst 15.4% | 同网格 n=4(192 单元) M1=**28.182%**；15.35% 实为 **n=6(432 单元)**（复算 15.350%，与记录一致） | **记录自相矛盾**（method 字段 vs evidence 字段），且被审 harness 照抄该错措辞 |
| C3 | HEX8 需 n≥12 才逼近 5% | 阶梯 28.18→15.35→9.93→5.685%（单调下降 True）；n=12 仍 **>5% 案例线** | 措辞「逼近」成立，「n≥12 达标」**不成立** |
| C4 | 体积误差 7.1% → 0.14% | HEX8 n=4=**7.116%**、HEX20 n=4=**0.136%** | **成立** |
| C5 | 闭合面合力/合力矩 <1e-16 | HEX8 n=8 = **7.83e-16 / 1.58e-16**；HEX20 n=4 = 2.20e-16 / 5.70e-17 | **不严格成立**（合力超 8 倍，机器精度量级，结论不变） |
| C6 | HEX8 n=8 PCG 仅 67 迭代 | **67** | **成立（逐位）** |

## 4. 发现（按严重度）

| id | 级 | 发现 | 证据 |
|---|---|---|---|
| F1 | blocking | `reproduce_claims` 内 `dev` 被同名 float 重绑定（第 476 行），第 492/493 行调用 `dev(...)` → `TypeError: 'float' object is not callable`：**claim 判定表在交付态永远产不出来** | `pytest::test_claim_reproduction_rules` STDOUT 堆栈 |
| F2 | blocking | 臂 B（独立 OSS scikit-fem）`arm_skfem:375` IndexError（掩码 3610 vs 轴 486）：**TET10 投影档从未产出数字**，而选型结论的跨实现稳健性正依赖该档；`decision_table` 因而无从判定 | `reports/xval_full_run.log`、`pytest::…[1]/[2]` |
| F3 | major | run_artifacts 残缺：run.log 11 行截断、quick 目录空、无 evidence.json/图 | §2 |
| F4 | major | 0/5 交付文件入主干（`in_main=false`），mission 记录 `server_quality_audit.gate_valid=false, key_present=false` | `A_files[*].in_main` |
| F5 | major | 锚点 0 注册基准行；下游 CASE-HOLLOW-SPHERE-3D 仍 in_progress | §1 |
| F7 | major | **OSS 记录本身**内部矛盾：`method` 称「HEX8 同网格 worst 15.4%」，`evidence` 称「HEX8 n=6/8=15.35%/9.93%」；真值同网格 28.18%、n=6 15.35%。被审 harness 照抄 ⇒ 必然输出「C2 not reproduced」，把记录的正确数误判为错 | `F_claims.C2…` + `oss_solutions` 行原文 |
| F8 | minor | C5 `<1e-16` 在 HEX8 n=8 实测合力 7.83e-16（超 8 倍） | `F_claims.C5…` |
| F6 | minor | 交付树根 `skfem-0.0.0-py3-none-any.whl` 1129B **空壳包**（`__init__.py` 0 字节），非真实依赖；实际解析到 site-packages skfem 12.0.2 | wheel namelist + import 路径 |
| F9 | minor | `test_band_average_is_not_centroid_sampling` 断言 M1/M2 差 >3%，实测 **1.816%** ⇒ 该测试假设本身被数据否定（不是代码 bug，是测试写死了未验证的期望） | pytest 输出 |

## 5. 判级（L0–L5，只认功能实质）

| 级 | 判 | 依据 |
|---|---|---|
| L0 文档 | 否 | 非纯文档 |
| L1 数学/单元 | 部分 | `lame_band_average` 与数值积分一致性测试通过；但 4/8 红，F9 的期望被否 |
| **L2 独立功能实现** | **是** | `arm_tensorfem:118-157` 真代码：几何/材料/压力 → HEX8/HEX20 装配 + gauge 消去 + 对角预处理 PCG → 探针误差/能量/体积误差；本单复算 **6 配置**，M1/M2 与 mission 日志逐位一致，PCG 迭代 67 逐位复现 |
| L3 系统集成 | 否 | `in_main=false`、`src` 调用方 0 处 |
| L4 基准验证 | 否 | 锚点 0 注册基准 + 端到端 rc=1 + claim 表 TypeError + 工件残缺（无 evidence.json/图） |
| L5 科研工程 | 否 | 无外部数值锚点（记录自述无 3D FEM 可比） |

百页报告不改变级别：被审单的**功能实质**是「一个能跑出真数值的臂 + 三条未跑通/未判定的支柱」，
故 L2；补上 F1/F2 并把锚点注册进 `tensor_benchmark_cases` 后才够 L4。

## 6. 未做的检查（如实标记 unknown / not_run，不是 passed）

* 未跑 GitHub CI（x86）——本单全部 loongarch64 本地实跑，架构差异未判。
* 未做敏感性/收敛阶估计（被审 harness 也无；C3 仅阶梯数值，无观测收敛阶）。
* 未审 `scripts/report_xval_method_selection_first_evidence.py`（17357B）产出的三张图是否正确
  —— 该脚本依赖的 `evidence.json` 从未生成，无从执行。
* 未审 `scratch/xval_proto.py`（临时原型，未入库）。
* cluster 作业 cj1552（同一复跑，双节点）递交后排队未派发；本地复跑已足够，结论不依赖它。

## 7. 被夸大/需重述的主张

1. 「HEX20 全部探针 <3%」→ 应为「**M1（形心半径采样）口径下 <3%**；M2（精确层积分）口径 11.99%」。
2. 「HEX8 同网格 worst 15.4%」→ 应为「HEX8 n=6 worst 15.35%；**同网格(n=4) 为 28.18%**」。
   换成正确数字后，二次单元优于线性的论证**更强**（不是更弱）。
3. 「交叉对标（XVAL）已完成」→ 独立臂只在 TET4 档出数（n=4/6/8：15.96%/8.93%/6.29% M1），
   TET10 档崩、claim 表崩 ⇒ 「跨实现稳健」**未验证**，只能称「首个证据（first-evidence）开工」。

## 8. 处置（disposition）

**request changes / reopened**。返修最小集（按代价排序）：

1. F1：把 `dev` 改名（`dev_rel`），claim 表跑通——**1 行改动**，产出的判定表是 L4 的前提。
2. F2：修 `arm_skfem:375` 的掩码（`sel` 长度与 `un[:, sel]` 轴不一致，投影后节点集变了），
   补齐 TET10 档；或如实降级声明为「仅线性臂对标」。
3. F3：跑通后落 `evidence.json` + 图，run.log 完整。
4. F5：把 C1/C2/C3/C4/C6 按**声明泛函**注册进 `tensor_benchmark_cases`（anchor_slug=本锚点），
   让 gate 能机器复核。
5. F7：修 `oss_solutions` 记录的 C2 措辞（n=4 28.18% / n=6 15.35%），避免错误再被下游照抄。
6. 修好后整树入主干（现 0/5 入库），L3/L4 才可能记。