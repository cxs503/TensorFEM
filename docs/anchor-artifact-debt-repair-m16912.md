# 证书工件欠账清偿：声明工件重生成与门禁值可复现性核查（m16912）

【锚点=NEW:证书声明的图件从未入库，14 个 anchor 被判 E_ARTIFACT_MISSING（其中 9 个仅此一项缺陷）；
m16863 提交的 `fem_shell_modal` 因缺图未登记债务表，导致 main 上
`tests/test_verified_anchor_certificates.py` 红 —— 清偿工件欠账即修复该红测试，同时给出「重跑后门禁值是否复现」的核查证据】｜
EDIT_WHITELIST: `benchmarks/verified/`、`docs/`

## 1. 问题：证据是「声明的」，不是「存在的」

`benchmarks/verified/*/gates_verified.json` 的 `artifacts` 里逐条声明本次运行的图件（如
`deflection_profile.png`、`hertz_pressure.png`）。审计工具 `tools/verified_anchor_audit.py`
把「声明但磁盘不存在」判为**硬缺陷** `E_ARTIFACT_MISSING`，并把该类 anchor 记入只减不增的
债务表 `anchor_certificate_debt.json`。

实测（本次开工基线，`python3 tools/verified_anchor_audit.py`）：

```
anchors=33 certified=6 contradicted=18 uncertified=9
debt registry = 26 anchors
E_ARTIFACT_MISSING 的 anchor = 14 个
debt-diff unregistered=['fem_shell_modal']      ← 测试红
```

`git ls-files 'benchmarks/verified/**/*.png'` 返回**空**：整个 `benchmarks/verified/` 树里
没有一个图件被入库（仓库别处的 png 有 482 个，说明「图件入库」本来就是本仓惯例，只是这批 anchor 漏了）。
后果有三层：

1. **证书声明了无法查看的证据**——读者只能相信 `result.json` 里的数字，不能核对图；
2. **`fem_shell_modal` 让 main 变红**：m16863 把该 anchor 的不可伪证 PASS 改成诚实 FAIL 并从债务表
   移除（理由：它不再掩盖 PASS），但它的图件同样缺失 → 它有缺陷却不在债务表 → 只减不增的不变式被打破；
   修复方式有两条（把缺陷登记回债务表 / 补齐工件），本文选后者——**补证据比登记债务强**；
3. **重跑是否复现，此前无人核查**：工件缺失意味着没人能确认这些证书的数字来自哪一次运行。

## 2. 做法

对每个 `E_ARTIFACT_MISSING` 的 anchor，用它**自己的** runner 重跑一次，把它声明的图件真正生成出来
（`MPLBACKEND=Agg`，单进程，本机 load15>100 期间最多 2 并发，未起集群作业）：

```bash
export PYTHONPATH=src:benchmarks/verified MPLBACKEND=Agg
python3 benchmarks/verified/<anchor>/run.py            # 或 run_verification.py
```

**踩坑（会毁证据，记下来）**：目录同时有 `run.py` 和 `run_verification.py` 的 anchor，
两者写出的 `result.json` **schema 不同**（375 行 vs 69 行）。先跑错了 `fem_beam_static` /
`fem_plate_static` 的 `run_verification.py`，把已入库的富 schema `result.json` 覆盖成瘦 schema
（`git diff` 立刻暴露：`gates[2].value 6.2e-11 → 4.0`）。**判据：`result.json` 的 `generated_by`
字段指名了产出脚本**，按它选 runner；跑完必须 `git diff` 比对行数与门禁值，异常就 `git checkout` 还原。
（本单已还原重跑，见 §3 的 drift 全表。）

重跑后 `git add` 图件与刷新后的 `result.json`/`gates_verified.json`，再
`python3 tools/verified_anchor_audit.py --write` 刷新证书台账、债务表与 `docs/verified-anchor-certificate-table.md`。

## 3. 结果：债务 26 → 17，certified 6 → 16

| 项 | 基线 | 本单后 |
|---|---:|---:|
| certified anchors | 6 / 33 | **16 / 33** |
| 债务表 anchors（只减不增） | 26 | **17** |
| `E_ARTIFACT_MISSING` anchors | 14 | 4 |
| `tests/test_verified_anchor_certificates.py` | **红**（unregistered=`fem_shell_modal`） | **16 passed** |

清偿的 10 个 anchor（9 个从债务表移除 + `fem_shell_modal` 直接转 certified）：
`fem_beam_static`、`fem_cantilever`、`fem_hertz_contact`、`fem_panel_buckling`、`fem_plate_modal`、
`fem_plate_postbuckling`、`fem_plate_static`、`fem_shell_modal`、`fem_structural_benchmarks`、`fem_winkler_plate`。
新增入库图件 21 张，全部由各 anchor 自己的 runner 现场生成（非手绘、非复用）。

### 3.1 门禁值可复现性核查（本次最有信息量的一条证据）

对全部重跑 anchor，按 `id` 逐一比对工作树 `gates_verified.json` 与 `HEAD` 版本的 `value` / `passed`：

| anchor | 门禁值漂移 | PASS/FAIL 翻转 | 总判变化 |
|---|---|---|---|
| 9 个（beam_static / cantilever / hertz_contact / panel_buckling / plate_postbuckling / plate_static / shell_modal / structural_benchmarks / winkler_plate） | 无（相对差 ≤1e-9，仅 `generated_at` 变） | 0 | 无 |
| `fem_plate_modal` | 1 个：`library_equivalence` 9.9156e-11 → 6.4841e-11（相对 35 %，但两者都是 float 噪声量级） | 0 | 无 |

结论：**重跑复现了全部存量门禁的判定**（0 个门禁翻转、0 个总判翻转）。
`fem_shell_modal` 的诚实 FAIL 也当场复现：`G1.beam` 6.4e-6 %、`G2.plate` 6.8425 %、
`G3.shell_lowest_physical_vs_donnell` **50.356 %**（阈 15 %）、`G4.shell_mechanism_free_low_spectrum`
**8 个 rotation-only 机构模态**（阈 0）→ `passed:false`，与 m16863 交付的数字一致（plate 频率相对差 ~1e-9）。

### 3.2 未清偿（如实）

| anchor | 仍缺 | 原因 |
|---|---|---|
| `fem_kirsch_plate` | 4 张图 | 重跑 40 min 未收敛被我按容量纪律终止 |
| `fem_modal` | 3 张图 + `E_ANCHOR_NAME_MISMATCH` | 目录名 `fem_modal` vs 证书 `anchor` 字段不一致，属另一类缺陷 |
| `fem_structural_acoustics` | 图 + `E_SCHEMA`/`E_PASS_WITHOUT_CRITERION`/非有限门禁值 | 多重缺陷，非纯工件问题 |
| `fem_winkler_plate_k_sweep` | 3 张图 | 目录内**没有 runner**（`run.py` 不存在），只能补 runner 或降级引用 |

## 4. 注意点与局限

* **本单不改物理**：0 行求解器代码改动。修的是「证据是否存在 / 是否可复现」，不是「物理对不对」。
  物理侧的已知诚实 FAIL（shell modal 的 G3/G4，即 shell4 的旋转机构与模态精度）**未修**，
  根因（drill DOF 稳定化 / SS 约束集）仍是 `docs/fem-shell-modal-anchor-audit.md` §5 记的后续节点。
* **资产体积**：本单入库 21 张 png（合计 ~1 MB，130 dpi）。若后续 png 继续增长，建议改成
  「图件存 `docs/assets/`，证书按相对路径声明」——但那是全树改造，超出单干范围。
* **不可复现的隐患已记录但未定罪**：`fem_panel_buckling/result.json` 里
  `campaign_case.panel_summary.flange_local_buckling_stress` 3.1651e9 → 1.9782e10（6.2×）在一次重跑间跳变，
  而**它没有任何门禁覆盖**（`gates_verified.json` 逐字未变）。这条通道是「无门禁字段漂移」，
  下次选题建议给它补一个门禁或标记为 informational——否则该数字进了汇报也没人核。
* **容量纪律**：全程单进程、最多 2 并发（本机 load15 111–136 / nproc 128），未递交集群重跑。

## 5. 复现命令

```bash
python3 tools/verified_anchor_audit.py                 # 期望 certified=16, contradicted=8, uncertified=9
python3 tools/verified_anchor_audit.py --check         # 债务表与审计零漂移则 exit 0
PYTHONPATH=src python3 -m pytest tests/test_verified_anchor_certificates.py -q   # 16 passed
PYTHONPATH=src:benchmarks/verified MPLBACKEND=Agg python3 benchmarks/verified/fem_shell_modal/run_verification.py
```