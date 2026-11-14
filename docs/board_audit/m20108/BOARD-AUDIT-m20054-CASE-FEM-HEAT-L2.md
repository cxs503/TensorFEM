# BOARD AUDIT — m20054 · CASE-FEM-HEAT-L2 (TensorFEM, 稳态热传导 L2 收敛率 ≥2 阶)

审计单：m20108（交差核验单，非重做） · verdict_by = `board-audit`
被审对象：mission m20054，commit `8867c2a4`（worktree `/root/tp-agent/work/m20054`，分支基 `d111cb7b`）
审计者产物位置：`docs/board_audit/m20108/`（本仓 m20108 工作树，仅新增审计文件）

| 项 | 值 |
|---|---|
| 审计环境 | python 3.11.6 · torch 2.7.1 · numpy 2.4.6 · CPU（OMP 4 线程） |
| 复现时刻 | 2026-11-14，`/tmp/audit20054_wt`（m20054 工作树的字节拷贝，避免写他人工作树） |
| 独立复核脚本 | `docs/board_audit/m20108/independent_heat_l2_check.py` → `…/independent_heat_l2_check.json` |
| 审计图 | `docs/board_audit/m20108/figures/m20108_heat_l2_crosscheck.png` |
| 复跑哈希台账 | `docs/board_audit/m20108/replay_hashes.txt` |
| 结论 | **L4 基准验证 · confirmed（approve with notes）**，附 1 项平台侧阻塞无关的元数据缺口 |

---

## ① 被审对象与"三件套"存在性

`git show --stat 8867c2a4` = 9 files / +3397，三件套逐个核到实物（非报告自述）：

| 三件套 | 实物指针 | 状态 |
|---|---|---|
| run_artifacts | `artifacts/CASE-FEM-HEAT-L2/evidence.json`（1558 行，sha256 `13517214…ad35`）、`artifacts/CASE-FEM-HEAT-L2/tables.md`（sha256 `c1127c42…97d9`）、`docs/assets/case_fem_heat_l2_convergence.png`、`docs/assets/case_fem_heat_l2_field.png` | 4/4 存在 |
| benchmark ref vs got | 目标阶 `p_ref=2.0` 由 `tensorfem.heat_l2_convergence.target_l2_order_from_manual()`（`src/tensorfem/heat_l2_convergence.py:557`）正则解析仓内先验表 `src/tensorfem/adaptive_poisson_2d.py:34`（**行号已核，逐字一致**）；got = 各算例 finest-pair L2 阶，见 §③ | 一致，见 §③ |
| repro_cmd | `PYTHONPATH=src python3 benchmarks/heat_l2_convergence.py`（`docs/CASE-FEM-HEAT-L2-steady-conduction.md` §⑥「一键复现」；入口 `benchmarks/heat_l2_convergence.py:491` `main()`） | **实跑 37.7 s，`verdict -> PASS (31 checks)`**，四件工件 sha256 与 commit blob **逐字节相同** |

字节级复跑证据（`replay_hashes.txt`）：

```
135172144362e8ef080bf461a3b5a2236d7b5296ac1a5bee8f8a8f7ced28ad35  evidence.json   (git show 8867c2a4:… 同值)
c1127c4279620bc1218a8302f7f30a4177aa7893af17131ff7c3e64de73797d9  tables.md       (git show 8867c2a4:… 同值)
01e3cac0…147b / cb629361…6fc77  convergence.png / field.png（重跑后同值，含 PNG）
```

报告生成器 `benchmarks/heat_l2_convergence.py:287 write_report()` 写回 `docs/CASE-FEM-HEAT-L2-steady-conduction.md`，重跑后该 md 亦逐字节不变 ⇒ **文档数值零手抄**（作者自述"数字由程序注入"成立）。

## ② 方法与独立性（对抗式复核）

审计不是复述，做了三层互相独立的检查：

1. **L1 — 复跑（replay）**：在 m20054 字节拷贝上执行其 repro_cmd。目的只验证"可复现"，不采信其闸门布尔值。
2. **L2 — 从原始数组重算**：直接读 `evidence.json` 的 `h` / `l2_relative` / `h1_relative` 数组，自行算 log-log 阶次与最小二乘斜率，与作者写入的 `l2_orders` 比对（差值全 0.0），再自行判"p≥2"。
3. **L3 — 独立重实现**（关键）：`independent_heat_l2_check.py` **不 import 被审模块** `tensorfem.heat_l2_convergence`，自建 Q4 有限元（自写 2×2 刚度装配、自写 8×8 Gauss 单元均值源、自写 4×4 Gauss 的 L2/H1 误差范数），对同一 MMS `T=sin(πx)sin(πy), q=2π²T` 跑 nx=8/16/32/64；并另用**仓内主干内核** `tensorfem.thermal.solve_steady_thermal`（`src/tensorfem/thermal.py:148`）配本审计自己的网格/源/误差范数再算一遍。三种源项语义都跑：单元常数源、精确积分载荷、仓内内核。

**独立性边界（如实声明）**：检查 3 的 `shipped_kernel` 变体与被审对象共用同一个求解内核（这正是要核的对象），但装配驱动与误差后处理全为审计者自有代码；`own_*` 变体与被审代码路径零共享。本轮**未**做任何实验/文献实测对标（该 anchor 属代码验证 code-verification，物理验证数据不存在，见 §⑦"未做的检查"）。

## ③ 数值裁定（基准判据 = 唯一裁判）

**基准表机器查询**（`runuser -u postgres psql -d tensorplatform`）：

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FEM-HEAT-L2';
-- (0 rows)
```

`tensor_benchmark_cases` 中该 anchor **无行**（全库 TensorFEM 有 11 行，均非本 anchor）。该 anchor 的判据只存在于算例池：

```sql
SELECT case_code,repo,tol_policy,status,mission_id FROM tensor_case_pool WHERE case_code='FEM-HEAT-L2';
-- FEM-HEAT-L2 | TensorFEM | 收敛率判据 | dispatched | 20054
```

⇒ 权威判据 = **收敛率判据**（tol_policy 为自由文本，无可机读 metric/tolerance，见 §⑤ F1）。

**got 数值 vs 判据**（阶次由审计者从 raw 数组重算，非抄表）：

| 算例 | 判别力 | finest-pair L2 阶 | 与 p=2 相对偏差 | 全阶梯 min 阶（coarse nx≥8） |
|---|---|---|---|---|
| sine_generation | 判别（带源项非多项式） | 1.9992 | 0.042% | 1.9866 |
| exp_trig_harmonic | 判别（调和、非多项式） | 1.9998 | 0.009% | 1.9973 |
| sine_aspect2 | 判别（2:1 域） | 1.9979 | 0.107% | 1.9653 |
| cubic_harmonic | **非判别**（节点精确，误差=纯插值） | 2.0002 | 0.008% | 2.0002 |
| quadratic_harmonic | **非判别**（同上） | 2.0000 | 0.000% | 2.0000 |
| bilinear_control | 控制例（可表示场） | 不主张阶次 | — | 最大绝对误差 3.553e-15 |

* H1 阶：1.0016–1.0021（理论 1），全部算例一致。
* 预登记闸门 31 项：审计复跑 **31/31 PASS**，无 failed 项。
* **是否满足"≥2 阶、相对偏差 <3%"：是**。判别算例 finest-pair 阶 1.9979–1.9998，最大相对偏差 0.107%（<3%，实际 <0.2%）；最差单对 1.9653 亦 > 预登记闸门 1.90。结论对"L2 阶次达 2 阶"成立。

**独立重实现交叉核**（`independent_heat_l2_check.json`，正弦算例 nx=8/16/32/64）：

| 变体 | L2 阶（8→16, 16→32, 32→64） | finest | 运行时 |
|---|---|---|---|
| 被审 evidence.json（含 4→8 段，故多一列） | —, 1.9866, 1.9967, 1.9992 | 1.9992 | — |
| 审计自有 numpy Q4（单元常数源） | 1.9866, 1.9967, 1.9992 | 1.9992 | 31.6 s |
| 审计驱动仓内内核 | 1.9866, 1.9967, 1.9992 | 1.9992 | 28.1 s |
| 审计自有 numpy Q4（精确积分载荷） | 1.9998, 1.9999, 2.0000 | 2.0000 | 35.6 s |

误差范数逐级比对：审计自有实现 vs 被审 `l2_relative`，相对差 **3.6e-15 / 3.3e-14 / 1.0e-12 / 2.3e-11**；仓内内核 vs 审计自有装配最大相对差 **2.3e-11**。⇒ 被审数字不是后处理假象，仓内 Q4 传导核在该 MMS 上确为 L2 二阶。

图：`figures/m20108_heat_l2_crosscheck.png`（A 面板 = 被审阶梯 + 三种独立变体 + O(h²) 参考线；B 面板 = 各算例逐对阶次 + 审计星标）。

## ④ 报告六段完整性（被审 `docs/CASE-FEM-HEAT-L2-steady-conduction.md`, 276 行）

| 六要素 | 位置 | 核验 |
|---|---|---|
| ① 案例介绍（物理背景/几何/参数） | §① 问题与参考解出处；§③ 参数表 | 有；物理背景落到船舶总体性能（结构热载荷/电子散热） |
| ② 怎么做（方法/流程/关键设置） | §② 方法与实现；§⑥ 一键复现 | 有；MMS 族、独立 Gauss 后处理、目标阶程序化解析 |
| ③ 计算结果（数字表） | §④ 对照表 + `tables.md` | 有；程序化注入（重跑逐字节一致） |
| ④ 结果图（云图 + 比较曲线） | 图 1 收敛曲线、图 2 FE/解析/误差云图 | 2 图存在且重跑可复现（PNG 哈希同值） |
| ⑤ 注意点（坑/局限） | §⑥ 失败模式 5 条 + 适用范围/局限 | 有；含"可表示场无信号"的负面知识、积分阶次污染、单元常数源、稠密规模上限、线程过订阅 |
| ⑥ 误差判定（逐项 + 是否<3%） | §④ 表尾 + §⑤ | 有；finest 1.999（偏差 0.04%），显式回答 <3% |

六段齐全、数值非手抄、图非占位。

## ⑤ 分级裁定与发现（按严重度排序）

**级别 = L4（基准验证：完整算例 + 误差分析 + 可复现）。** 理由：真代码输入→计算→输出（`src/tensorfem/heat_l2_convergence.py` 720 行，`solve_case:425`→`solve_steady_thermal:446`），解析参考解在线求值、误差范数为独立 Gauss 积分（`:193`），5 层网格阶梯 + 4/6/8 点积分敏感性 + Richardson，可一键复现且产物字节级可复现，17 项 pytest 真断言（非空跑）。不判 L5：无新物理/新方法，属验证工程。L3 只部分成立（复用主干 `tensorfem.thermal` 内核，但未接入主求解流程/平台 API，无 integration caller）。

| # | 严重度 | 发现 | 证据 | 影响/处置 |
|---|---|---|---|---|
| F1 | **中（平台侧，非生产者缺陷）** | `tensor_benchmark_cases` 对 `CASE-FEM-HEAT-L2` **0 行**；判据仅为 `tensor_case_pool.tol_policy='收敛率判据'`（自由文本，无 metric/tolerance）⇒ "got 是否在容差内"无法机读裁定，只能人工按文本判 | §③ 两条 SQL 输出 | 不阻塞本轮（生产者已在代码里预登记数值闸门 `GATE_L2_PAIR=1.90`、`GATE_L2_FINEST=(1.95,2.10)`，见 `heat_l2_convergence.py:137-143`；审计已独立复核）。**建议板端补一行机读基准**（metric="finest-pair L2 order of ‖e‖₀ on sine_generation MMS (32→64)", tolerance="[1.95,2.10]"）。审计者未代为写基准表——自证自判不合适 |
| F2 | 低 | 6 算例中 3 个对求解器**非判别**（cubic/quadratic_harmonic 为节点精确、误差=纯插值；bilinear_control 为可表示控制例）。"31 项闸门"不应被读作 31 个独立求解器检查，有效判别集 = 3 例 | §③ 表；`e_FE/e_interp=1.000000000` | 不降级：真实判据由 3 个判别例给出（1.9979–1.9998），且文档 §⑥ 第 1 条已主动披露该坑。建议后续表述改为"3 判别例 + 2 非判别例 + 1 控制例" |
| F3 | 低 | `tensorfem.thermal` 体源为单元常数 ⇒ 载荷一致性误差 O(h²)，与被测离散误差同阶，放大误差常数 | 审计独立测得：精确积分载荷同网格误差为常数源的 **0.386 倍**（2.6× 更小），阶升至 2.0000 | 不污染阶次判据（两变体同为 2 阶，审计已证）。文档 §⑥ 第 3 条如实记录为局限。含意：绝对精度类结论（如 0.08% 场误差）只在给定网格成立，不可外推 |
| F4 | 低 | "外部锚"为方法学节级引用（Salari & Knupp SAND2000-1009J；Roache 1994 JFE 116:405-413），离线无法核对全文；目标阶的主锚是**仓内**先验表 `adaptive_poisson_2d.py:34`（审计已逐字核对该行） | commit 与文档 §① | 可接受：数值参考本身是解析解（最强参考类型），外部引用仅承担方法学。记录为"锚点强度=方法学级，非数值级" |
| F5 | 低（测试质量） | `tests/test_heat_l2_convergence.py::test_predeclared_gates_all_pass_on_the_short_ladder` 只按前缀过滤 `failed` 名单，若该算例未产出任何 check 则**空真** | 测试 128-136 行 | 不阻塞：同文件另有直接断言同算例阶次/控制/积分独立性的测试。建议补 `assert any(name.startswith(case_key) for name in checks)` |
| F6 | 信息 | 审计时该 commit **未合入主干**（m20108 主树无 `src/tensorfem/heat_l2_convergence.py`），故 `integration_callers` 只能是 `heat_l2_convergence.py:446 → tensorfem.thermal.solve_steady_thermal`，无平台 API 调用者 | `ls` m20108 主树 + `git log` | 非缺陷（验收后由平台推送）；但据此**不主张 L3** |

## ⑥ 实际跑过的检查 / 未跑的检查

**跑过（全部可复现）**

1. `git show --stat 8867c2a4` — 9 文件 +3397 行，三件套齐全。
2. 复跑 repro_cmd（m20054 字节拷贝，`/tmp/audit20054_wt`）：37.7 s，`PASS (31 checks)`，4 工件 sha256 与 commit blob 逐字节相同（PNG 亦同）。
3. `PYTHONPATH=src python3 -m pytest tests/test_heat_l2_convergence.py -q` → **17 passed, 1 warning in 22.59 s**。
4. 从 `evidence.json` 原始数组独立重算阶次/斜率：与写入值差 0.0。
5. 独立 numpy Q4 重实现 + 仓内内核驱动（3 变体）：误差范数与被审值相对差 ≤2.3e-11，阶次逐位一致。
6. 程序化解析目标阶：`adaptive_poisson_2d.py:34` 行内容含 `L2 norm … O(h^2) … -1`（逐字核对）。
7. DB 两条 SQL（`tensor_benchmark_cases` / `tensor_case_pool`）。
8. 文档六段逐条存在性与数值一致性（重跑后 md 逐字节不变）。

**未跑 / 不能跑（标 unknown，不标 pass）**

* 物理实验/文献实测对标：该 anchor 为 MMS 代码验证，无对应实测数据集 ⇒ "外部物理验证" = **NOT_APPLICABLE**（不是 PASS）。
* nx=128 阶梯：稠密 4225→8129 未知量矩阵 ≈2.2 GB，作者已声明本机不可行；审计**未**对内存上限做压测 ⇒ 规模上限结论取作者记录，未独立验证。
* 奇异解（角点/裂纹）阶次退化：文档声明不在结论范围，审计未测。
* 平台 API 集成：不存在该能力的 API 端点，故"平台侧调用"= **NOT_SUPPORTED**。
* 未在 loong64 集群节点复跑（本机单次 <2 min，无需递计算）。

## ⑦ 落表（verdict）

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFEM', 'm20054 CASE-FEM-HEAT-L2', 20108, 4, '<见 verdict_evidence_m20108.json>', 'board-audit');
```

evidence 八键均指向真实指针（文件:行号 / 工件路径+sha256 / 测试名 / 复跑命令），见 `verdict_evidence_m20108.json`。

**Disposition：approve with notes（confirmed）。** 阻塞项 = 0；非阻塞项 = F1（板端补机读基准行）、F2/F5（表述与测试措辞）、F3/F4（已披露的局限与锚点强度）、F6（合入状态）。