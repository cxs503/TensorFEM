# 验证欠账清偿：三张「contradicted」锚点证书修复（fem_thermal / fem_sent_j_integral / fem_structural_acoustics）

【自找活 · verifier 席】｜派发锚点 `GAP-PP-4-opt-loop` 核对结果：见 §0。

## 0. 选题与锚点核对（诚实记录）

派发锚点为 `GAP-PP-4-opt-loop`（反问题优化基础设施 `tensorlbm.inverse`）。开工前核对：

1. 台账 `feature_ledger` 显示 `status=open`；
2. 但代码侧 **该能力已在 TensorLBM main 落地**：`src/tensorlbm/inverse/{optimizer,losses,convergence,seed}.py`（714 行，来自 m15399/m18039 合并），且 `tests/test_opt_loop_tau_recovery.py` + `tests/test_inverse_infra.py` 在 main 上 **31 passed**（本单只读复核，35.8 s）。TensorFEM 侧同族能力 `tensorfem.inverse` + `benchmarks/verified/fem_inverse_identification` 亦已证书化（3 种子、3 % 闸门、日志 sha256 重放）。
3. 故重做=违反 owner 2026-11-09 不重复令；本单改做本仓（TensorFEM）**验证欠账**：`tools/verified_anchor_audit.py` 报 `contradicted=3` —— 三张锚点证书自我矛盾或引用了不存在的工件，且 fail-closed 注册表测试在 main 上红。

## 1. 案例介绍（对象与缺口）

审计器 `tools/verified_anchor_audit.py` 是**不重跑求解器**的证书审计器：它从 `gates_verified.json` 重算每条闸门的 `value/operator/threshold` 判定，并按 `artifacts` 检查工件存在性与 git 跟踪状态。修复前三张锚点的缺陷：

| 锚点 | 审计缺陷 | 实情 |
|---|---|---|
| fem_thermal | `E_ARTIFACT_MISSING` | 证书声明 4 张收敛/场图，仓库里一张都没有提交 |
| fem_sent_j_integral | `E_ARTIFACT_MISSING`（+`W_ARTIFACT_AMBIGUOUS`） | 证书声明 6 张 SENT/J 积分图，仓库里没有 |
| fem_structural_acoustics | `E_SCHEMA`, `E_PASS_WITHOUT_CRITERION`, `E_GATE_VALUE_NOT_FINITE` | 私有 schema 标签 `...-partial.v1`；2 条闸门无数值判据；C5 的 `value` 为 null 却声明 `passed=true` |

## 2. 怎么做的

1. **fem_sent_j_integral**：`python benchmarks/verified/fem_sent_j_integral/run.py --stage assemble` 从已提交的 `_cache/{ladder,control,plane_strain}.json` 重建证据与图（8.9 s，零求解）。比对：17 条闸门的 `(value, threshold, passed)` **逐条不变**，`passed=true / failed_gates=[]` —— 仅补齐 6 张 PNG。
2. **fem_thermal**：`python benchmarks/verified/fem_thermal/run.py` 复算（42.3 s 求解，实际墙钟 51 s）。Case A/B 机器零（3.27e-13 / 1.18e-11），Case C 最粗网格 rel-L2 = 0.0059，三对加密的观测阶 **全部 0.0** → 证书维持**诚实 FAIL**（`passed=false`），只补 4 张图。
3. **fem_structural_acoustics**：修改 `make_report.py` 重发证书 ——
   - schema 改为规范信封 `tensorfem.verified-gates.v1`（该信封本就承载 `passed` + `failed_gates`；`fem_thermal` 就是 v1 里的 declared FAIL，私有 partial 标签没有必要）；
   - 每条闸门显式写入 `operator` + 数值 `threshold`（审计器无法重算「只存在于散文里的判定」）；
   - 三条闸门改写成它实际度量的**偏差量**：`B3 → |p − 2|`、`B4 → |σ − 1|`、`C5 → gap(L8)/gap(L2)`（数值全部由 `result.json` 原始表计算，未改任何一个原始观测量或阈值）；
   - `A2_plate_mesh_convergence`（从未声明过容差的粗网格诊断）降级为 `informational_checks`，**不再计为 pass**；
   - `figures` 声明改为「目录存在才列出」：该节点**从未提交过 PNG**，证书不再声明不存在的工件（空列表诚实，假路径不诚实）。
4. 刷新债务账 `tools/verified_anchor_audit.py --write`；`tests/test_verified_anchor_certificates.py` 中「注册表至少有一条非有限值闸门」这一**种群假设**已因 C5 数值化而失效，改为更强的真不变量：**已认证锚点不得带非有限值闸门**（非有限值必须进债务账）。

## 3. 计算结果（程序化，来自 `tools/verified_anchor_audit.py --json`）

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| anchors | 45 | 45 |
| certified | 37 | 40 |
| contradicted | 3 | 0（提交后为 0；未提交时 2 条 `E_ARTIFACT_UNTRACKED`） |
| uncertified | 5 | 5（`fem_axisym_shell`/`fem_elastoplastic_plate_buckling`/`fem_hertz_contact_outofsample`/`lbm_table_selfcheck`/`rainflow_fix_spec`） |
| gates | 504 | 503 |
| gates recomputed | 487 | 502 |
| unrecomputable gates | 17 | 2 |
| debt-diff unregistered | 3 | 0 |
| debt-diff stale | 0 | 0 |
| `tests/test_verified_anchor_certificates.py` | **1 failed**, 15 passed | 16 passed |

fem_structural_acoustics 保留两条**诚实失败**：C3 = 41.0 %、C4 = 30.3 % 对 Junger & Feit 无限柱闭包（阈值 15 %），修复没有放宽它们，也没有把它们移出 `gates`。

## 4. 结果图

本单无新物理场图；补齐的 10 张锚点图即证据（`benchmarks/verified/fem_thermal/fig_case_*.png`、
`benchmarks/verified/fem_sent_j_integral/sent_*.png`）。平台报告内嵌审计前后对比条形图（`contradicted/uncertified/unrecomputable` 三指标）。

## 5. 注意点与局限（读者不得外推）

1. 本单**没有新验证任何物理**：三张锚点的闸门数值、阈值、诚实 FAIL 全部原样保留；所做的是让证据与声明自洽、可被机器重算。
2. `E_ARTIFACT_UNTRACKED` 只能在 commit 之后消失 —— 审计器查的是 git 索引/跟踪状态，因此修复的最终态在提交里，不在工作树里。
3. 剩余 5 张 `uncertified` 锚点是真欠账（`E_NO_GATES`，根本没有闸门文件），本单只是把它们**如实登记**进债务账，没有伪造闸门；它们是下次自找活的首选线索。
4. fem_thermal 的 Case C 观测阶 = 0 是**真实未解问题**（5.9e-3 的 L2 残差在 4 套网格 × 时间步 10→640 加密下不变），已在该锚点 `limitations` 中记录为 honest FAIL，本单未掩盖。

## 6. 误差判定

- fem_thermal：Case A/B 机器零（3.27e-13 / 1.18e-11 vs 1e-10 阈值，**<3 %**）；Case C 最粗网格 0.59 %（**<3 %**）；观测收敛阶 0.0 vs 阈值 1.0 → **FAIL，不满足 <3 % 的可比口径（该闸门不是精度闸门）**。
- fem_sent_j_integral：17 条闸门全 PASS，形状因子 vs Tada 多项式与 J 积分路径无关性均在阈值内（重放后逐条相同，**<3 %** 口径由各闸门自身阈值给出）。
- fem_structural_acoustics：A1 模态 2.34 %（**<3 %**）、C1 脉动球 1.7e-5（**<3 %**）、B1/B2 σ 偏差 0.35 %/0.40 %（**<3 %**）；C3/C4 对 J&F 41.0 %/30.3 % → **不满足 15 % 阈值，诚实 FAIL**。
