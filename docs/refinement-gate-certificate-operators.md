# 细化阶梯锚点证书：让每一道门都能被机读复算（GAP 修复，TensorFEM）

锚点（feature_ledger）：`CASE-FEM-PINCHED-CYLINDER` / `CASE-FEM-SCORDELIS-LO-ROOF`
（两者均为 done 状态的既有能力，本单属**验证欠账**修复）。
受影响库：`src/tensorfem/refinement_gate.py`（`fem_pinched_cylinder` 与
`fem_shell_scordelis_lo` 两个锚点的公用判据生成器）。

## ① 案例介绍

两个经典壳体基准锚点用同一套细化阶梯判据（Richardson 观测阶 + Roache GCI）：

| 参数 | 夹持圆柱 pinched cylinder | Scordelis-Lo 圆柱屋盖 |
|---|---|---|
| 几何 | R=300, L=600, t=3（MacNeal–Harder） | R=25, L=50, 半角 40°, t=0.25 |
| 材料 | E=3e6, ν=0.3 | E=4.32e8, ν=0 |
| 载荷 | 反对称集中力 P=1 | 竖向面载 q=90 |
| 参考量 | 加载点径向位移 −1.8248e−5 | 自由边中点竖向位移 −0.3024 |
| 阶梯 | n = 4, 8, 16, 32 | 同 |
| 判据 | 相对误差 ≤3%（认证量=Richardson 极限） | 同 |

标定对象不是壳体单元本身（早已合并并验证），而是**证书本身的机读性**：
`tools/verified_anchor_audit.py`（shrink-only 债务登记簿）在这两个锚点上
判出 `E_PASS_WITHOUT_CRITERION`（fem_shell_scordelis_lo）与大量
"unrecomputable" 门 —— 即"证书说 PASS，但读者无法从它自带的数字复算"。

## ② 怎么做的

三处缺陷（全部在 `refinement_gate.py`，均为纯增量字段/语义修正，判据阈值不变）：

| # | 缺陷 | 修复 |
|---|---|---|
| 1 | `GateVerdict.to_dict()` 不输出 `operator`，12/6 道门只能靠散文推断比较符 | 新增 `operator` 字段，逐门显式声明（`<=` / `>=`） |
| 2 | 单调性门 `actual=None`（通过）/`1.0`（失败）、`threshold=None` —— 通过没有任何判据 | 改为**违反单调的阶梯步数**（符号反转 + 完全停滞）对阈值 0，`<=` |
| 3 | 渐近区间门把 GCI **带宽**同时写进 `actual` 与 `threshold`（自证同义反复） | `actual` = 实测最细网格真实误差，`threshold` = GCI 带宽，`<=`；并把带宽边界的 1e-12 估计松弛去掉，边界情况一律 `WITHHELD`（fail-closed） |

生成器同步（两个锚点的 `run_verification.py` 手抄 gate 字典，丢弃了新字段）：
`fem_shell_scordelis_lo` 转发 `operator`；`fem_pinched_cylinder` 转发 `operator`
并把 `failed_gates` 改成"数值判据未满足的门"（含 `WITHHELD` 但超阈者，
审计器按 `value <op> threshold` 复算要求该清单完全一致）。

复现链（零手抄）：

```
python3 benchmarks/verified/fem_pinched_cylinder/run_verification.py
python3 benchmarks/verified/fem_shell_scordelis_lo/run_verification.py
python3 tools/verified_anchor_audit.py --write     # 刷新证书 + 债务登记簿
python3 -m pytest tests/test_refinement_gate_certificate_operators.py \
                 tests/test_refinement_gate.py tests/test_verified_anchor_certificates.py -q
```

## ③ 计算结果（`tools/verified_anchor_audit.py` 前后对比，程序化读自
`benchmarks/verified/anchor_certificates.json`）

| 指标 | 修复前 | 修复后 | Δ |
|---|---|---|---|
| certified 锚点 | 6 | **7** | +1 |
| contradicted 锚点 | 18 | 17 | −1 |
| 可复算门（gates_recomputed）/ 243 | 201 | **216** | +15 |
| 不可复算门 | 42 | **27** | −15 |
| 无数字判据的 PASS 门 | 9 | **6** | −3 |
| 债务登记簿条目 | 27 | **26** | −1 |

逐门数字（Scordelis-Lo，修复后；来源 `gates_verified.json`）：

| 门 | 状态 | actual | threshold | 判据 |
|---|---|---|---|---|
| flat_facet.accuracy_finest | FAIL | 0.494018 | 0.03 | ≤ |
| flat_facet.accuracy_asymptotic | FAIL | 0.531876 | 0.03 | ≤ |
| flat_facet.monotone_qoi | FAIL | 1.0 | 0.0 | ≤ |
| flat_facet.observed_order | PASS | 1.16650 | 1.0 | ≥ |
| flat_facet.asymptotic_range_consistency | WITHHELD | 0.494018 | 0.0316752 | ≤ |
| cylindrical.accuracy_finest | PASS | 0.00414701 | 0.03 | ≤ |
| cylindrical.accuracy_asymptotic | PASS | 0.00516661 | 0.03 | ≤ |
| cylindrical.monotone_qoi | PASS | 0.0 | 0.0 | ≤ |
| cylindrical.observed_order | PASS | 2.27601 | 1.0 | ≥ |
| cylindrical.asymptotic_range_consistency | WITHHELD | 0.00414701 | 0.00126923 | ≤ |
| cylindrical.coarsest_sanity | PASS | 0.283298 | 0.5 | ≤ |

pinched cylinder（修复后）：G1 PASS 0.0181622≤0.03；G2 FAIL 0.0327193≤0.03；
G3 FAIL 1.0≤0.0；G4 PASS 1.09279≥1；G5 WITHHELD 0.0181622≤0.0178718；
G6 PASS 0.0285731≤0.5。

头部结论**未被放宽**：两锚点 `passed=False` 不变（Scordelis 平壳面 sri 阶梯仍
49.4% 超差，pinched cylinder 认证精度 3.272% 仍超 3% 门）；变的是"结论现在
能被第三方用证书自带数字复算"。

## ④ 图

两个锚点各带收敛曲线（`benchmarks/verified/fem_shell_scordelis_lo/scordelis_lo_convergence.png`、
`benchmarks/verified/fem_pinched_cylinder/convergence.png`），由各自的
`run_verification.py` 在本单重新生成，画的是相对误差 vs h 与位移 vs 单元数、
含 3% 参考线；本单改的是判据字段，不改曲线，故不重复嵌入。

## ⑤ 注意点（坑与适用范围）

- **门字段不是装饰**：`actual` 写成 `threshold` 的拷贝会让门"永远满足自己的阈值"，
  是自证同义反复的典型形态；审计器把它判为不可复算而不是 PASS，正是这个原因。
- `WITHHELD` 与 `FAIL` 是两条轴：证据不足 ≠ 违反容许度。修复后渐近区间门在
  数值上恰好也"不满足"，故 `failed_gates` 清单必须列入，否则审计器判
  `E_FAILED_GATES_LIST`（pinched cylinder 修复过程中真实踩到）。
- 生成器手抄 gate 字典会静默丢字段：新增字段必须同时改 `to_dict()` 与两个
  锚点的 `run_verification.py`，否则库里修好、产物里仍是旧形状。
- 适用范围：只对这一族（细化阶梯 + 参考值）锚点生效；`fem_winkler_plate` 等
  其它 `E_ARTIFACT_MISSING` 类债务不在本单范围。

## ⑥ 误差判定

| 量 | 值 | 基准 | 误差 | < 3 %？ |
|---|---|---|---|---|
| Scordelis-Lo 圆柱壳 16×16 自由边挠度 | −0.302469 | −0.3024（Scordelis & Lo 1960） | +0.023 % | **是** |
| Scordelis-Lo 平壳面 sri 默认规则 16×16 | −0.437542 | −0.3024 | +44.69 % | **否**（fail-closed，保留） |
| pinched cylinder 认证精度（Richardson 极限） | −1.88451e−05 对应 3.272 % | −1.8248e−05（MacNeal–Harder） | 3.272 % | **否**（fail-closed，保留） |
| 本单修复的判据字段 vs 其 detail 文本 | 0.494018 / 0.00414701 | detail 内 4.940e−01 / 4.147e−03 | 0（逐位一致） | 是（自洽性） |

即：本单的实质增量是**证书可复算性**（+15 门可复算、−1 债务、+1 certified），
不是精度提升；精度结论与修复前逐位相同。
