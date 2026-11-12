# CASE-FEM-SCORDELIS-LO-ROOF — 证书判据修复：把「无判据的 PASS」变成可复算的比较（m16947）

【锚点=CASE-FEM-SCORDELIS-LO-ROOF】｜EDIT_WHITELIST: `src/tensorfem/refinement_gate.py`, `tests/`, `benchmarks/verified/`, `docs/`
上游工作树：`/root/tp-agent/work/m16947`（分支 `agent/m16947`），基于 main `4277679e`。

## 1. 为什么动这个锚点（缺陷来源，非自选美观）

`tools/verified_anchor_audit.py`（只读审计）在 33 个 verified 锚点里把 `fem_shell_scordelis_lo`
判为 **contradicted: E_PASS_WITHOUT_CRITERION** —— 该锚点对某个量声明 `PASS`，却没有数值判据。
逐条落到两条闸门上（均出自共享判级库 `src/tensorfem/refinement_gate.py`）：

| # | 缺陷 | 旧的产物形态 | 后果 |
|---|---|---|---|
| D1 | `monotone_qoi` 无判据 | `actual=None, threshold=None, status=PASS` | 第三方**无法复算**该 PASS：审计只能记 `E_PASS_WITHOUT_CRITERION` |
| D2 | `asymptotic_range_consistency` 阈值=自身取值 | `actual=threshold=relative_gci` | `value <= threshold` 恒真 —— **自证判据等于没有判据** |
| D3 | 全库闸门缺 `operator` | 只有 `actual`/`threshold`，比较方向靠猜 | 审计需从散文猜方向（`W_OPERATOR_ABSENT`），证书不可机器校验 |

自查关键点：D1/D2 是**同一个根因**——判级库把"断言"当过关，把"测量值"当判据。

## 2. 改了什么（最小改动，共享库 + 一个锚点重生成）

1. `src/tensorfem/refinement_gate.py`
   - `GateVerdict` 增加显式 `operator`（`<=` / `>=`），并写进 `to_dict()`：闸门的比较方向成为契约的一部分（修 D3）。
   - `__post_init__` 硬拒绝「已判级（PASS/FAIL）但没有有限 `actual`/`threshold`」的闸门：把 D1 从"审计事后发现"变成"构造期不可能" ；`WITHHELD` 仍允许无判据（它不断言任何东西）。
   - `monotone_qoi` 改为可复算判据：**取值 = 单调性被破坏的步进转移数**（符号变化与平台步都计 1），阈值 0，`operator <=`。与旧实现 `all(s>0) or all(s<0)` 语义逐例等价（测试覆盖 4 个阶梯）。
   - `asymptotic_range_consistency` 改为 **|e_finest| / GCI ≤ 1**（阈值恒为 1），GCI 带塌到机器精度时退化为 |e_finest| ≤ 0；带不可用时保持 `WITHHELD` 且不写 NaN（修 D2）。
2. `benchmarks/verified/fem_shell_scordelis_lo/run_verification.py`
   - 闸门产物透传 `operator`；
   - 新增 `--from-dump`：从已提交的 `result.json` 重判证书（**零求解**），证书可离线复算。
3. 重跑该锚点全阶梯（8 次求解，flat facet 4 级 + cylindrical 4 级，单进程 ≈4 min）→ 重生成 `result.json` / `gates_verified.json` / 收敛图。

## 3. 结果（程序化注入，零手抄）

### 3.1 修复后的闸门表（`gates_verified.json` 原样）

| gate | status | operator | value | threshold |
|---|---|---|---:|---:|
| `flat_facet.accuracy_finest` | FAIL | `<=` | 0.494018 | 0.03 |
| `flat_facet.accuracy_asymptotic` | FAIL | `<=` | 0.531876 | 0.03 |
| `flat_facet.monotone_qoi` | FAIL | `<=` | 1 | 0 |
| `flat_facet.observed_order` | PASS | `>=` | 1.1665 | 1 |
| `flat_facet.asymptotic_range_consistency` | WITHHELD | `<=` | 15.5964 | 1 |
| `flat_facet.coarsest_sanity` | FAIL | `<=` | 0.629753 | 0.5 |
| `cylindrical.accuracy_finest` | PASS | `<=` | 0.00414701 | 0.03 |
| `cylindrical.accuracy_asymptotic` | PASS | `<=` | 0.00516661 | 0.03 |
| `cylindrical.monotone_qoi` | PASS | `<=` | 0 | 0 |
| `cylindrical.observed_order` | PASS | `>=` | 2.27601 | 1 |
| `cylindrical.asymptotic_range_consistency` | WITHHELD | `<=` | 3.26734 | 1 |
| `cylindrical.coarsest_sanity` | PASS | `<=` | 0.283298 | 0.5 |

`overall_verdict = WITHHELD`，`passed = False`。

### 3.2 Scordelis-Lo 阶梯（QoI = 自由边中点竖向位移，参考 −0.3024）

flat facet Shell4（Q4 膜 + Mindlin 板，一点剪切积分）：

| mesh | elements | w_probe | rel. error vs −0.3024 |
|---|---:|---:|---:|
| 4x4 | 16 | -0.492837 | 62.9753 % |
| 8x8 | 64 | -0.405557 | 34.1126 % |
| 16x16 | 256 | -0.437542 | 44.6897 % |
| 32x32 | 1024 | -0.451791 | 49.4018 % |

cylindrical shell4（精确柱壳度量）：

| mesh | elements | w_probe | rel. error vs −0.3024 |
|---|---:|---:|---:|
| 4x4 | 16 | -0.216731 | 28.3298 % |
| 8x8 | 64 | -0.296730 | 1.8751 % |
| 16x16 | 256 | -0.302469 | 0.0228 % |
| 32x32 | 1024 | -0.303654 | 0.4147 % |

- flat facet：**最细网格误差 49.40 %**，Richardson 极限 53.19 % → 判 FAIL（远超 3 %）。阶梯非单调（转移数 1）。
- cylindrical：最细网格误差 **0.4147 %**，**Richardson 极限 0.5167 %** → 两条精度闸门均 PASS（< 3 %）；**观察阶 2.276 ≥ 1**。
- 唯一未过的闸门 `cylindrical.asymptotic_range_consistency`：|e_finest|/GCI = **3.267 > 1** → 该 4 级阶梯在 0.13 % 量级上仍是前渐近的，GCI 带不保守，**最细网格值不可由本阶梯自证**——这是"证据不足"（WITHHELD），不是"超差"（FAIL）。诚实保留 `overall = WITHHELD`。

### 3.3 图

- `benchmarks/verified/fem_shell_scordelis_lo/scordelis_lo_convergence.png`：左=误差 vs 网格尺寸（对数，含 3 % 判据带），右=探针位移 vs 单元数（含 −0.3024 参考线）。

## 4. 修复前后的可复算性（可量化对照）

`PYTHONPATH=src python tools/verified_anchor_audit.py` 全库快照：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| certified 锚点数 | 6 | **7** |
| contradicted 锚点数 | 18 | **17** |
| 可机器复算的闸门数 / 总闸门 | 203 / 244 | **213 / 244** |
| `fem_shell_scordelis_lo` 缺陷 | `E_PASS_WITHOUT_CRITERION` | **无（debt 登记表删除该条）** |

## 5. 注意点 / 局限

1. **不改变物理结论**：cylindrical 仍 0.52 %、flat facet 仍 ~50 % 偏高，本单只把判据与可复算性修好；结论与上游一致。
2. `monotone_qoi` 的度量是"转移数"而非步长幅度：它对非单调极敏感（0/1/2…），但不区分"轻微抖动"和"来回震荡"。这是刻意的——单调性是定性前提，不应被幅度调参掩盖。
3. `asymptotic_range_consistency` 的比值依赖 GCI 的安全系数 Fs=1.25 与观察阶；比值 3.267 说明本阶梯前渐近，**不可**读作"参考值不可信"。
4. 顺带发现的**既存**债务（非本单引入）：`fem_shell_modal` 的证书声明图件 `shell_modal_audit.png` 不在仓内（`E_ARTIFACT_MISSING`，疑为上游门禁滤除二进制），已由 `--write` 诚实登记进 debt 表，未修。
5. 未重生成 `fem_pinched_cylinder` 等其他用同一判级库的锚点证书（需重跑各自求解器）；它们仍按旧产物审计通过，本单未触碰。

## 6. 验证（本会话实跑）

```
PYTHONPATH=src python -m pytest tests/test_refinement_gate.py tests/test_refinement_gate_criterion_semantics.py     tests/test_verified_anchor_certificates.py tests/test_benchmark_report_rendering.py -q
-> 37 passed
PYTHONPATH=src python benchmarks/verified/fem_shell_scordelis_lo/run_verification.py            # 全阶梯重算
PYTHONPATH=src python benchmarks/verified/fem_shell_scordelis_lo/run_verification.py --from-dump # 零求解重判
PYTHONPATH=src python tools/verified_anchor_audit.py --write   # debt-diff: unregistered=[] stale=[] drift=[]
```

新增回归测试 `tests/test_refinement_gate_criterion_semantics.py`（9 例）：判级闸门必须有有限判据与显式 operator、从自身数字可复算、
单调性度量逐例等价、GCI 判据随真误差成比例（证明阈值不再是取值副本）、前渐近必须 WITHHELD 而非 FAIL。
