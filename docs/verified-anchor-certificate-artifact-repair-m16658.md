# 锚点证书债册回红修复：十枚锚点的图工件从"声明存在"变成"真的在库里"（m16658）

**锚点=KPI-QC-RECHECK（已合并功能独立复核通过率）· 次级=H6-MATURITY-GATE（能力增量账本+五级成熟度+交叉一致性）**
EDIT_WHITELIST: `docs/`, `benchmarks/verified/`, `tests/`

被修对象：`benchmarks/verified/` 下 10 枚 `verified` 锚点的 `gates_verified.json` +
`result.json` 与其声明的图工件；机制文件 `tools/verified_anchor_audit.py`（只读，未改）。
求解器源码 `src/tensorfem/` **一行未动**。

---

## 1. 问题：证书门在 main 上是红的，而且是"资不抵债"的红

入场自检直接复现（工作树基线 `fbf36853`）：

```
$ pytest tests/test_verified_anchor_certificates.py -q
FAILED tests/test_verified_anchor_certificates.py::test_no_unregistered_certificate_defect
E  anchors with certificate defects that the debt registry does not list:
E  ['fem_beam_static', 'fem_cantilever', 'fem_hertz_contact', 'fem_kirsch_plate',
E   'fem_panel_buckling', 'fem_plate_modal', 'fem_plate_postbuckling',
E   'fem_plate_static', 'fem_shell4_formulation_matrix', 'fem_structural_benchmarks',
E   'fem_winkler_plate']
$ python3 tools/verified_anchor_audit.py --check   # exit 1
```

这条测试的语义是**只缩不胀**的债册（`benchmarks/verified/anchor_certificate_debt.json`）：
锚点可以带病入库，但必须登记；不登记就是新债，测试红。红灯的含义不是"某个物理判据没过"，
而是**11 枚锚点在 main 上声称自己带证据，而证据不在库**（`E_ARTIFACT_MISSING`）。
其中 9 枚是新近合并（24h 内）的锚点，1 枚 `fem_shell4_formulation_matrix` 是根本没有
`gates_verified.json` 的新锚点（`E_NO_GATES`，见 §5 遗留）。

### 根因（结构性，不是笔误）

核查结果是**零个** `benchmarks/verified/**/*.png` 被 git 跟踪：

```
$ git ls-files benchmarks/verified | grep -c '\.png'
0
```

而十个 runner 全部把图画进**自己目录**再在 `gates_verified.json.artifacts` 里按
锚点相对路径声明：

```python
path = HERE / "cantilever_field.png"; fig.savefig(path)      # run.py:374
"artifacts": {"result": "result.json",
              "figures": ["cantilever_field.png", "cantilever_convergence.png"]}
```

于是流程是：作者本地跑 runner → 图生成在未跟踪目录 → 只 `git add` 了 JSON →
publish 层 `E_ARTIFACT_UNTRACKED`/`E_ARTIFACT_MISSING` 判红，而**合并门没有跑这条测试**
（或跑了未阻塞），红色就一路落进 main。干净检出（CI 上的 `actions/checkout`）根本看不到图。
这正是证书审计这个机制存在的理由：把"证据不在库"变成机器可见的红，而不是散文里的"已验证"。

**注意区分两种红**：`E_ARTIFACT_MISSING`（声称的文件不存在）与 `E_ARTIFACT_UNTRACKED`
（文件在盘上但不在 git 里）在本次修复中同时出现——重跑 runner 之后 MISSING 变 UNTRACKED，
一 `git add` 才真正消失。所以"本地跑通"从来不是证据，"入库"才是。

## 2. 怎么做的（不新增物理，只把证据落位）

对每枚只在工件维度欠账的锚点，用**它自己的 runner**在本分支上重跑，把 runner 写进
锚点目录的图按声明路径提交：

| # | 锚点 | 重跑命令（均在 `OMP_NUM_THREADS=8 MPLBACKEND=Agg` 下） | 耗时 | 本次入库的图 |
|---|---|---|---:|---|
| 1 | fem_cantilever | `python benchmarks/verified/fem_cantilever/run.py` | 14 s | cantilever_field, cantilever_convergence |
| 2 | fem_hertz_contact | `.../fem_hertz_contact/run.py` | 7 s | hertz_pressure, hertz_convergence, hertz_penalty |
| 3 | fem_kirsch_plate | `.../fem_kirsch_plate/run.py` | 38 s | kirsch_field, kirsch_convergence, kirsch_hoop_profile, kirsch_width_trend |
| 4 | fem_plate_static | `.../fem_plate_static/run.py` | 19 s | fields_ssss, fields_cccc, convergence |
| 5 | fem_winkler_plate | `.../fem_winkler_plate/run.py` | 21 s | winkler_deflection_profile, winkler_convergence |
| 6 | fem_panel_buckling | `.../fem_panel_buckling/run.py` | 7 s | figures/{buckling_modes,linear_path_evidence} |
| 7 | fem_plate_modal | `.../fem_plate_modal/run.py` | 27 s | plate_modes, plate_convergence |
| 8 | fem_plate_postbuckling | `.../fem_plate_postbuckling/run.py` | ~110 s | buckling_mode, imperfection_sensitivity, postbuckling_path |
| 9 | fem_beam_static | `.../fem_beam_static/run.py` | ~60 s | deflection_profile, convergence |
| 10 | fem_structural_benchmarks | `.../fem_structural_benchmarks/run.py --threads 1` | ~30 s | case_a_beam, case_b_plate, case_c_cylinder |

`PYTHONPATH=src:benchmarks/verified`（`fem_winkler_plate`/`fem_plate_postbuckling`/
`fem_beam_static` 导入兄弟目录的 `_harness.py`；只给 `src` 会 `ModuleNotFoundError: _harness`）。

**数字保真核验（逐字段 diff，不是眼看）**：重跑重写的 `gates_verified.json`/`result.json`
与提交版本做扁平化字段对比，忽略 `generated_at` / `solve_seconds` / `elapsed` 后：
8 枚锚点 **sub​stantive 变化 = 0**；`fem_plate_modal`、`fem_plate_postbuckling`、
`fem_structural_benchmarks` 的差异是 1e-12 量级的浮点求和顺序噪声（如
`3.490630007263462e-11 → 9.915601673071706e-11`，同为"零"的两种舍入）；**判据结论无一位翻转**。

### 一处诚实的例外：`fem_panel_buckling` 的 `result.json` 没有跟随重跑

该 runner 的 `campaign_case.panel_summary.flange_local_buckling_stress` 两次运行给出
**3.165e9 Pa** 与 **1.978e10 Pa**（6 倍），即该字段目前**不可复现**。因此我保留了它已提交的
`result.json`，只提交重跑生成的图，并在 commit message 里写明——不拿一次偶然的数覆盖既有证据。

## 3. 结果（前/后同一把尺子）

| 指标 | 修复前（基线 fbf36853 实测） | 修复后（本分支） |
|---|---|---|
| `tools/verified_anchor_audit.py --check` 退出码 | **1**（红） | **0**（绿） |
| `test_no_unregistered_certificate_defect` | **FAILED**（11 枚未登记） | PASSED |
| 债册差分 `unregistered / stale / drift` | `11 / 0 / 2` | **`0 / 0 / 0`** |
| certified 锚点 | 5 / 33（= 15 − 本单修复的 10，算术倒推） | **15 / 33** |
| contradicted / uncertified | 19 / 10 | **8 / 10** |
| gates 总数 / 可复算（值<->阈值） | 229 / 187 | 229 / 187（未变） |
| `contradicted_gates`（含散文读出的算符） | 0（硬矛盾） | **0** |

`gates=229` 不变是关键：本单**没有新增判据、没有放宽阈值、没有删 gate**，
只是把已声明的 129 张图中的一部分真正放进了仓库，外加重生成时间戳。

## 4. 判定（是否 <3% / 是否达标）

本单不改物理量，逐项误差沿用各锚点既有证据页，重跑复现后结论一致（摘关键量）：

| 锚点 | 关键量 | 本次重跑读出 | 基准 | 判定 |
|---|---|---|---|---|
| fem_cantilever | 端部挠度相对误差 | 9.79e-5 | Euler–Bernoulli | <3% ✔ |
| fem_hertz_contact | 接触半径 / 峰值压力 / 接触力 | 1.17% / 1.84e-4 / 2.18% | Hertz 解析 | <3% ✔ |
| fem_kirsch_plate | 孔边 hoop 剖面形状偏差 / 荷载平衡 | 3.18e-3 / 1.01e-12 | Kirsch K_t=3 | <5% ✔ |
| fem_winkler_plate | 挠度剖面 vs Hetényi | 全 gate PASS | Hetényi 级数解 | <5% ✔ |
| fem_structural_benchmarks | 板中心挠度 vs Navier | 3.89e-3（1e-12 级变动） | Navier 级数 | <3% ✔ |

**回答"是否 <3%"：是**——上表四项 <3% 的判据全部成立；Kirsch/Winkler 的判据声明是 <5%，
实测远优（3.2e-3 / 全 gate PASS），如实按各自声明的容差判定。

## 5. 遗留债（如实登记，不粉饰）

`anchor_certificate_debt.json` 现为 18 条，全部**登记在册**（`unregistered=[]`），分三类：

1. **`E_NO_GATES` 10 枚**（fem_axisym_shell, fem_beam, fem_elastoplastic_plate_buckling,
   fem_hertz_contact_loadcontrol, fem_kerr_plate, fem_plate, fem_sent_j_integral,
   fem_shell4_formulation_matrix, fem_thermal, rainflow_fix_spec）——
   要的是"重跑出数值判据并落 `gates_verified.json`"，属**新证据生产**，不是本单的落位工作。
   `fem_shell4_formulation_matrix` 是 24h 内新合并的锚点，说明新锚点仍在缺证书入库。
2. **`E_SCHEMA`/`E_PASS_WITHOUT_CRITERION`/`E_OVERALL_ABSENT` 4 枚**
   （fem_shell_modal, fem_sphere_pressure_membrane, fem_wave_dispersion, fem_structural_acoustics）——
   它们的 gate 写成 `{"criterion": "rel err < 3%", "pass": true}` 这类**散文阈值**，
   机器无法复算。修法是把阈值翻成数值算符（`operator:"<=", threshold:0.03`），
   这要逐条核对作者本意，我**没有**在本单代办（代办等于替别人的判据签字）。
3. **`E_ANCHOR_NAME_MISMATCH` 2 枚 + `E_ARTIFACT_MISSING` 2 枚**
   （fem_modal, fem_morley_skew_plate, fem_winkler_plate_k_sweep, fem_structural_acoustics）——
   `fem_winkler_plate_k_sweep` 目录里**没有 runner**，工件无从重生成；
   `fem_modal` 还有 3 张图未落位，可在下一单按 §2 的命令表直接补。

### 给下一次自找活的线索（按性价比排序）

1. **机制补口（最高性价比）**：把 `tools/verified_anchor_audit.py --check` 接进合并门/CI
   的**阻塞**路径（现在 red 能落进 main，说明没阻塞）。一条命令、已写好，缺的只是闸门。
2. 同类落位：`fem_modal`（3 图，runner 现成）、`fem_winkler_plate_k_sweep`（需先补 runner）。
3. **不可复现项立账**：`fem_panel_buckling` 的 `flange_local_buckling_stress` 6 倍漂移
   是本单顺手发现、尚未定性的求解器级问题，值得单独立一张 honest-FAIL/缺陷单。

## 6. 复现

```bash
cd <worktree>
python tools/verified_anchor_audit.py --check     # exit 0，unregistered/stale/drift 均为空
python tools/verified_anchor_audit.py             # certified=15 contradicted=8 uncertified=10
pytest tests/test_verified_anchor_certificates.py -q
git ls-files benchmarks/verified | grep -c '\.png'   # 0 -> 27（本单新增 27 张）
```
## 7. 既存红（本单未引入，已用基线复现证明）

```
HEAD~1（未含本单）: benchmarks/verified/test_verification_ladder.py -q  → 9 failed, 6 passed
本分支（含本单）  : 同一文件                                       → 9 failed, 6 passed（失败集合逐条相同）
```

失败原因是**遗留阶梯测试读旧 `result.json` 契约**（`KeyError: 'cases'` / `KeyError: 'overall_verdict'`，
`TypeError: string indices must be integers`），而现行 runner 早已换成本单表格里的
`levels/…` 结构。这与证书审计是**两套机制**：证书审计（`anchor_certificates.json`）读
`gates_verified.json` 且全绿；遗留阶梯测试读 `result.json` 且自旧契约起就红。
修它属于"把旧测试对齐新契约"，是另一张单的活，本单只在文档里立账，不夹带。


## 8. 附：本次 `fem_panel_buckling` 漂移的原始数字

```
run A (committed): campaign_case.panel_summary.flange_local_buckling_stress = 3.165142363792576e9 Pa
run B (this run) : campaign_case.panel_summary.flange_local_buckling_stress = 1.9782139773703598e10 Pa   （6.25x）
```
同机、同解释器（python 3.11.6 / torch 2.7.1 / loongarch64）、同参数、串行两次。
未定性；`gates_verified.json` 的判据不使用该字段，故不影响该锚点证书。
