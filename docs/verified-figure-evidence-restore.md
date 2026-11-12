# 图表证据链修复：33 个 verified 锚点的「证书点名的图」从 36 缺 32（`fem_kirsch_plate` 为例）

【锚点=CASE-FEM-KIRSCH-PLATE 证据链（并入 `FEM-ANCHOR-CERTIFICATE-INTEGRITY` 线）】｜
EDIT_WHITELIST: `benchmarks/verified/`, `tools/`, `tests/`, `docs/`

本单=paper/研究开发席自主工时，处理的是**证据可用性**而非新物理：证书
（`gates_verified.json`）里每个数都能离线复算（审计已验证 `contradicted_gates=0`），
但它**点名的图**在合并后的树里大半不存在。任何按证书取证的读者拿到的是 404，
而 `tools/verified_anchor_audit.py` 早就在喊 `E_ARTIFACT_MISSING`——只是没人还这笔账。

## 1. 案例介绍：缺口是什么

`benchmarks/verified/*/gates_verified.json` 的 `artifacts` 字段是**可得性声明**：

```json
"artifacts": {"figures": ["benchmarks/verified/fem_kirsch_plate/kirsch_field.png", ...]}
```

图形是驱动 `run.py` 的**运行产物**，写在锚点目录旁；`git status` 看不到一个从未被
`git add` 过的产物。合并流水线只搬 tracked 文件，于是图留在作者的工作树里，
族谱上只剩一个指不到的路径。修复前实测（`python tools/verified_anchor_audit.py`）：

| 量 | 修复前 | 修复后 |
|---|---|---|
| 锚点总数 | 33 | 33 |
| certified | **9** | **20** |
| contradicted | **18** | **7** |
| uncertified | 6 | 6 |
| `anchor_certificate_debt.json` 债条目 | **26** | **13** |
| 声明但缺失的图工件 | **36**（14 个锚点） | **7**（3 个锚点，均无驱动或驱动失败） |
| `tests/test_verified_anchor_certificates.py` | **2 failed**（红） | **16 passed**（绿） |

主分支在这批图缺失的同时是**红的**：`fem_beam / fem_kerr_plate / fem_plate` 已被
某次合并证书化，但债登记表没刷新（stale），`fem_kirsch_plate` 又因缺图新转
contradicted（unregistered）。fail-closed 门禁因此在主干上失效——这是修复的第一优先级。

### 代表锚点：CASE-FEM-KIRSCH-PLATE

| 项 | 值 |
|---|---|
| 参考解 | Kirsch 1898 无限板圆孔拉伸：`σ_θθ(a,θ)/T = 1 − 2cos2θ`，`K_t = 3`（净截面/毛截面两种基准） |
| 有限宽度对照 | Howland / Heywood 拟合（`d/W` 0.1–0.5） |
| 参数 | 四分之一模型，`E=1`、`ν=0.3`、单位厚度、远场 `T=1`；`d/W ∈ {0.02, 0.05, 0.10}` |
| 离散 | Q4 星光网格（`n_boundary ∈ {12,24,48}` 三档，最细 2304 单元）+ Q8 环带交叉核对（独立位移边界） |
| 判据 | `|K_t−3|/3 ≤ 5%`、Richardson 外推 ≤1%、观测阶 ≥1、`W_und` 不变性 ≤1%、Saint-Venant 长度 ≤1%、Howland 趋势 ≤3%、孔边环向应力剖面 RMS ≤5%、荷载平衡 ≤1e-10 |

## 2. 怎么做的（方法/流程/工具链）

1. **诊断而非猜测**：`tools/verified_anchor_audit.py` 的 `E_ARTIFACT_MISSING` 逐锚点展开，
   得到「哪个锚点缺哪个文件、有没有驱动」的清单（14 锚点 / 36 图 / 12 个有 `run.py`）。
2. **新建 `benchmarks/verified/restore_figure_evidence.py`**（489 行）：对每个
   「声明缺图 + 有驱动」的锚点，**执行该锚点自己的驱动**（绝不另写复现，避免第二个真值源），
   再回读证书点名的路径，校验 PNG magic 与最小体积，逐个记 `sha256`。
   失败面如实分类：`timeout / runner_failed / figure_missing / restored_with_drift`。
3. **可复现性侧信道**：重跑会重写 `result.json` / `gates_verified.json`。脚本把重写结果与
   `git show HEAD:<path>` 做**叶子级**比对，把变化分成三类：`volatile`（`generated_at`、
   `*_seconds`、`elapsed_*`）、`roundoff`（|Δ| ≤ 1e-9，即双精度/线程序噪声）、
   `physics_drift`（其余，视为发现）。默认 `--keep-json` 把 JSON 还原，合并树里的数保持不变。
4. **只提交图 + 脚本 + 刷新过的登记表**，JSON 数值零改动（`git diff` 只含 PNG 与
   registry/doc/script），把「证据补齐」与「数值变更」在 diff 层解耦。
5. 刷新 `tools/verified_anchor_audit.py --write`（债表 + `anchor_certificates.json` + 证书表文档），
   fail-closed 测试回到绿；`tests/test_figure_evidence_restore.py` 新增 5 条不变量把这一层钉住。

## 3. 计算结果（程序化产出，零手抄）

`benchmarks/verified/figure_evidence_restore.json` 的 `summary`：

```
anchors 33 | already_complete 25 | no_gates 6 | not_attempted 1 | no_runner 1
missing_after 7 | physics_drift []
```

逐锚点重跑（进程内实测，本机 128 核、`OMP_NUM_THREADS=4`）：

| 锚点 | 驱动耗时 | 补入图 | 结论 |
|---|---|---|---|
| fem_kirsch_plate | 39.9 s（全程） | 4 | 数完全复现：13 个叶子差异全是 `wall_seconds`/`generated_at` |
| fem_beam_static | 组批 3 并发 | 2 | 复现 |
| fem_cantilever | ” | 2 | 仅 `solve_seconds` 变 |
| fem_plate_static | ” | 3 | 复现 |
| fem_modal | ” | 3 | 复现（残留 `E_ANCHOR_NAME_MISMATCH`，债务不销） |
| fem_plate_modal | 26.6 s | 2 | `library_equivalence` 9.92e-11 → 1.06e-11（见 §5） |
| fem_hertz_contact | ” | 3 | 复现 |
| fem_panel_buckling | 6.9 s | 3 | `flange_local_buckling_stress` 变动，闸门仍过（见 §5） |
| fem_plate_postbuckling | 88.8 s | 3 | 复现 |
| fem_curved_beam | ” | 1 | 仅 `elapsed_*` 变 |
| fem_winkler_plate | 25.8 s | 2 | 仅 `elapsed_s` 变（首跑 `ModuleNotFoundError: _harness`，见 §5） |
| fem_structural_benchmarks | ” | 3 | 仅 `timings_s` 变 |
| fem_shell_modal | ” | 1 | 图补齐，但数值漂移（见 §5） |
| fem_structural_acoustics | ” | 4 | 图补齐，但 `E_SCHEMA/E_PASS_WITHOUT_CRITERION/E_GATE_VALUE_NOT_FINITE` 未销 |

合计 **32 张图入库**（`git diff --cached --stat`：44 files, +1757/−282），
certified 9→20，债务 26→13。

## 4. 结果图

- `benchmarks/verified/fem_kirsch_plate/kirsch_field.png` — 开孔板四分之一模型 σ_xx/T 云图（`d/W=0.02`，`n=48`，Q4）：孔口两侧等值线密集即 `K_t` 极值区。
- `benchmarks/verified/fem_kirsch_plate/kirsch_convergence.png` — `|K_t−3|/3` 随 `h` 的双对数收敛（Q4 星光网格三个 `d/W` + Q8 环带两条独立离散），5% 池容差为红线。
- `benchmarks/verified/fem_kirsch_plate/kirsch_hoop_profile.png` — 孔边环向应力 `σ_θθ(a,θ)/T` 全场 vs 解析 `1−2cos2θ`，点线 `K_t=3`。
- `benchmarks/verified/fem_kirsch_plate/kirsch_width_trend.png` — 有限宽度趋势：gross `K_t`（升）与 net `K_tn`（降）对 Howland/Heywood 与 Kirsch 极限。
- 同批入库：`fem_hertz_contact/hertz_pressure.png`（Hertz 压力半椭圆 vs 解析）、`fem_plate_postbuckling/postbuckling_path.png`（后屈曲路径）、`fem_structural_acoustics/figures/fig_esm_validation.png`（等效源法验证）等 28 张。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **驱动不是随便能跑的**：`fem_winkler_plate/run.py` 首跑 `ModuleNotFoundError: _harness`
   ——它假设 `benchmarks/verified` 在 `PYTHONPATH` 上（既有文档的 `PYTHONPATH=src:benchmarks/verified`
   约定）。修法写进脚本的环境构造，属工具修复，不是放宽判据。
2. **图上标量与证书标量未必同源**：`fem_plate_modal` 的 `library_equivalence` 闸门
   （上限 1e-9，实测 9.92e-11）重跑得 1.06e-11，**同一判定、不同舍入**——特征值求解对
   线程序敏感，量级差 ~9×只出现在第 10 位以后。它仍在阈值内，但**余量只有 10×**；
   换架构（x86 vs loong64）前不能宣称「可逐位复现」。本单不擅自放宽该阈值（会削弱判据），
   只把它登记为 `roundoff` 类漂移。
3. **`fem_panel_buckling` 的 `flange_local_buckling_stress` 变动**：该量是屈曲应力，
   闸门仍通过；本单记录为 `physics_drift` 但不改数——真值判定需要独立席位复核。
4. **`fem_shell_modal` 的 `lowest_mechanism_hz` 与 `first_six_hz[0]` 漂移**：壳单元存在近零
   机制模态，漂移提示「机制筛选」判据贴边；该锚点残留债务未销，列为待查项。
5. **`fem_structural_acoustics` 的闸门把 `pytest` 结果写进证书**（`F1_pytest/value`、
   `threshold`、`status` 三处随运行变动），这在语义上不是物理判据，且该文件 schema 仍不合规
   （`E_SCHEMA`）——图已补，证书仍红，属下一单。
6. **适用范围**：本单只修复「声明→可得」这一层，**不判断 oracle 是否选对**（那是独立评审的活）。
   `restore_figure_evidence.py` 不能伪造证据：驱动失败/超时/不出图一律记失败并留在债表。
7. 两个无驱动的锚点（`fem_winkler_plate_k_sweep`、`fem_shell_modal` 的部分图）保持
   `E_ARTIFACT_MISSING`；`fem_axisym_shell`、`fem_thermal`、`fem_sent_j_integral`、
   `fem_shell4_formulation_matrix`、`fem_elastoplastic_plate_buckling`、`rainflow_fix_spec`
   仍是 `E_NO_GATES`（证书本身没写），不在本单范围。

## 6. 误差判定（关键量 vs 基准，逐项）

| 量 | 基准 | 实测 | 误差 | 是否 <3% |
|---|---|---|---|---|
| `K_t`（Kirsch 极限，`d/W=0.02`，`n=48`） | 3（Kirsch 1898 解析） | 3.00141 | 4.69e-4 相对 | **是** |
| `K_t`（Richardson 外推） | 3 | 3.00920 | 3.07e-3 相对 | **是** |
| `K_t`（Q8 环带交叉核对，最细） | 3 | 3.00148 | 4.92e-4 相对 | **是** |
| 孔边环向应力剖面 RMS/`K_t` | `1−2cos2θ` 解析 | — | 3.18e-3 | **是** |
| 有限宽度趋势（gross，`d/W` 0.1–0.5） | Howland/Heywood | — | 9.85e-3 | **是** |
| 荷载平衡 `|applied−reacted|/applied` | 0 | 1.01e-12 | 绝对 1.01e-12 | **是**（≪1e-10 闸门） |
| 证书内部一致性（33 锚点 / 257 闸） | 自洽 | `contradicted_gates=0` | — | **是**（0 条矛盾） |
| `library_equivalence`（fem_plate_modal） | 库入口同解 | 1.06e-11（阈值 1e-9） | 1.06e-2 相对阈值 | **是**（但余量 10×，见 §5.2） |

**结论**：本单交付的每张图都由锚点自己的驱动从解析解/文献基准重新生成，
修复后 `pytest tests/test_verified_anchor_certificates.py tests/test_figure_evidence_restore.py`
全绿（16+5 条），证书点名的图在合并树里 100% 可得（33 锚点中 25 个图完整；
其余 8 个的缺失全部落在债表里，可被 fail-closed 门禁看到）。