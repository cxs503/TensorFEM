# CASE-FEM-SCORDELIS-LO-ROOF — 证书债务门收口：Scordelis-Lo 出债务册 + 缺失证据按 runner 复原（m17035）

【锚点=CASE-FEM-SCORDELIS-LO-ROOF】｜EDIT_WHITELIST: `benchmarks/verified/`, `tools/`, `tests/`, `docs/`
工作树：`/root/tp-agent/work/m17035`（分支 `agent/m17035`），基线 main `d1b1cf06`。

## 1. 为什么动这个锚点（缺陷来源，非自选美观）

本锚点的**物理**工作已经完成并通过（`benchmarks/artifacts/CASE-FEM-SCORDELIS-LO-ROOF/case_result.json` 11/11 门；
`docs/CASE-FEM-SCORDELIS-LO-ROOF.md`：圆柱壳元 16×16 = `-0.302469` vs 文献 `-0.3024` = `+0.023 %`）。但它在
**证据台账**上还挂着一笔欠账：fail-closed 门 `tests/test_verified_anchor_certificates.py` 在本工作树上是**红的**。

开工前实测（不是回忆）：

```
$ PYTHONPATH=src python -m pytest tests/test_verified_anchor_certificates.py -q
FAILED test_no_unregistered_certificate_defect
FAILED test_debt_registry_is_stale_free
2 failed, 14 passed
```

```
$ PYTHONPATH=src python tools/verified_anchor_audit.py
anchors=33 certified=10 contradicted=17 uncertified=6 gates=258 ...
debt-diff unregistered=[11 anchors] stale=['fem_shell_scordelis_lo'] drift=['fem_modal','fem_structural_acoustics','fem_wave_dispersion']
```

三类缺陷，各有独立根因：

| # | 现象 | 根因 | 我的判据 |
|---|---|---|---|
| D1 | `stale=['fem_shell_scordelis_lo']` | m16947 已把该锚点的判据修好（`gates_verified.json` 带 `operator`+数值阈值，live audit 判 **certified**），但 shrink-only 债务册没删条目 | 锚点级：live audit 判 certified 即必须出册（册子契约原文如此） |
| D2 | `unregistered=[11]` | 这些 `gates_verified.json` 的 `artifacts.figures` 声明的 png **从未提交**（repo 内 `find` 零命中，`git ls-files` 零命中） | 声明了证据却不存在 = 债务，不得静默 |
| D3 | `drift=[3]` | 证书在册，但缺陷码已随修复变化，册子是旧的 | 册子必须由 live audit 重生成 |

note：本锚点自身的度量（`w=-0.302469`，误差 +0.023 %，< 3 %）与本次改动无关——本次动的是**证据台账**，不是物理量。

## 2. 怎么做的

1. **先复原证据，而不是先登记债务。** D2 的 11 个锚点，其 runner 大多自己会写图（`savefig` 到锚点目录）。逐个在**独立工作树内**重跑 runner：

   ```
   PYTHONPATH=src python benchmarks/verified/<anchor>/run.py      # 单核，串行
   ```

   重跑前后逐文件 diff **只差 `generated_at` / `solve_seconds`（`elapsed_s`）**，物理数值逐位一致——所以这是**证据复原**（把已声明的图落盘），不是重新认证。
   唯一真正跑不动的：`fem_winkler_plate` 的 `run.py` 因 `from _harness import ...` 依赖 `benchmarks/verified/` 在 `sys.path` 上而直接 Traceback（rc=1）——
   即它文档里那条复现命令本身是坏的。用 `PYTHONPATH=src:benchmarks/verified` 复跑即恢复，并据此记录该 runner 的正确调用方式。
   超出预算的重锚点（`fem_kirsch_plate` 900 s 超时被杀，`fem_panel_buckling` / `fem_plate_postbuckling` / `fem_shell_modal` / `fem_structural_benchmarks` 未跑）**不假装恢复**，留给债务册如实登记。
2. **用机器重生成台账**，不手改 JSON：`PYTHONPATH=src python tools/verified_anchor_audit.py --write`（重算 gated 值、写 `anchor_certificates.json` / `anchor_certificate_debt.json` / `docs/verified-anchor-certificate-table.md`）。
   债务册条目全部来自 live audit 的复算结果，手工塞不进一条未经复算的条码（`debt_diff` 的 `stale`/`drift` 两把锁）。
3. `git add` 复原的 14 张 png（`E_ARTIFACT_UNTRACKED` 只有入库才消），再跑一次 `--write` 定版。

## 3. 结果（程序化，零手抄）

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| certified 锚点 | 10 | **16** |
| contradicted 锚点 | 17 | **11** |
| uncertified（无门文件） | 6 | 6 |
| 债务册条目 | 23（含 stale/drift） | **17** |
| `debt-diff` unregistered / stale / drift | 11 / 1 / 3 | **0 / 0 / 0** |
| `tests/test_verified_anchor_certificates.py` | 2 failed, 14 passed | **16 passed** |
| `gates_contradicted`（声明算子下自相矛盾的门） | 0 | 0 |

复原的图（14 张，全部来自重跑 runner，非手绘）：

| 锚点 | 复原的图 | runner 耗时 |
|---|---|---|
| `fem_beam_static` | `deflection_profile.png`, `convergence.png` | 11 s |
| `fem_cantilever` | `cantilever_field.png`, `cantilever_convergence.png` | ~470 s |
| `fem_hertz_contact` | `hertz_pressure.png`, `hertz_convergence.png`, `hertz_penalty.png` | 11 s |
| `fem_plate_modal` | `plate_modes.png`, `plate_convergence.png` | 54 s |
| `fem_plate_static` | `fields_ssss.png`, `fields_cccc.png`, `convergence.png` | 26 s |
| `fem_winkler_plate` | `winkler_deflection_profile.png`, `winkler_convergence.png` | 5 s（修 sys.path 后） |

本锚点：`fem_shell_scordelis_lo` 已出债务册（certified，21 门全部可复算），其三个配置的判定仍是

| 配置 | 16×16 `w` | vs −0.3024 | 判定 |
|---|---|---|---|
| 圆柱壳元（metric-consistent） | −0.302469 | **+0.023 %** | PASS（<3 %；Richardson 极限 −0.303913 → 0.517 %） |
| 平面面片 + 假定自然剪切（mitc） | −0.295310 | −2.345 % | PASS |
| 平面面片 + 默认 sri（反向门：必须差） | −0.437542 | +44.690 % | PASS（反向门，>20 % 仍成立） |

## 4. 图

无新增图：本次交付物是**台账/证据**层，图即上表 14 张复原 png（属各锚点自有剖面/收敛图，不重复贴）。审计汇总表见
`docs/verified-anchor-certificate-table.md`（机器生成）——用文字表代替重绘图，避免用绘图给台账工作化妆。

## 5. 注意点（踩坑与局限）

- **证据复原 ≠ 重新验证。** 重跑只证明 runner 仍能产出所声明的图，且数值与已提交证书逐位一致（diff 只差时间戳/耗时）。它不能替代对新单元/新工况的验证。
- **超时即如实留债。** `fem_kirsch_plate` 在本机（load15≈95，128 核，单核串行）900 s 未完成，我没有延长超时去硬凑，而是保留 `E_ARTIFACT_MISSING` 并在债务册标明补救：在集群上重跑该 runner 后 `git add` 图。
- **`fem_winkler_plate` 的复现命令是坏的**（`_harness` 不在 import 路径）：这说明"声明了证据但没跑过"。正确调用：`PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_winkler_plate/run.py`。建议后续把该 runner 的 sys.path 自愈（`sys.path.insert(0, str(Path(__file__).parents[1]))`），本次未改 runner 以免越界。
- **债务册仍是 17 条**，其中 6 条 `E_NO_GATES`（无门文件）、8 条真实缺图/命名不符。收缩只能靠逐锚点补证据，不能靠改判据。
- **`--write` 是吸收器**：它能一键把新债登记得"合法"。本次先复原再登记，顺序不能反；反了就是把可修的欠账变成永久债务。

## 6. 误差判定

| 关键量 | 值 | 基准 | 误差 | <3 %？ |
|---|---|---|---|---|
| Scordelis-Lo 自由边中点 `w`（圆柱壳元 16×16） | −0.302469 | −0.3024（Scordelis & Lo 1960） | +0.023 % | **是** |
| 同上，Richardson 极限（判据认证对象） | −0.303913 | −0.3024 | +0.517 % | **是** |
| 平面面片 + mitc 16×16 | −0.295310 | −0.3024 | −2.345 % | **是** |
| 平面面片 + 默认 sri 16×16（反向门） | −0.437542 | −0.3024 | +44.690 % | 否（**必须否**，否则反向门失效） |
| 台账一致性（本次主要交付） | 见 §3 | debt-diff 三空 + 16/16 测试绿 | 精确离散量，无插值误差 | 不适用（结构判据，非物理量） |
## 7. 附：同类未修红（如实留痕，非本单引入）

`tests/test_pinched_cylinder_maturity.py::test_run_artifacts_present_including_figures` 在本工作树**修复前即红**：
`benchmarks/verified/fem_pinched_cylinder/run_artifacts` 声明的 `case_convergence.png` / `case_field.png` / `case_sensitivity.png` 磁盘不存在。
它不在证书审计视野内（该锚点 `gates_verified.json` 未声明这三张图），所以 `debt-diff` 看不到——这是本类缺陷的第 12 例，说明"声明了图但没提交"是系统性欠账，
非个别。补救：在该锚点跑 `run.py` 生成图后 `git add`；本次未做（超本单范围，避免用未验证产物凑绿）。
