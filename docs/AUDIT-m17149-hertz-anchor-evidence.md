# AUDIT m17149 — CASE-FEM-HERTZ-CONTACT 证据链独立复核：主判据 `a` 的估计器缺陷（已修）

【锚点=CASE-FEM-HERTZ-CONTACT】｜仓=TensorFEM｜席位=tp-ver-6（verifier）｜skill=adversarial-review
结论速览：**发现 1 个实质缺陷（主判据接触半径的估计器在细网格上退化为"节点半径"，把网格伪影当作测量值入闸），已修并重新出证；**
另有 2 个证据表述缺陷（一份已发布报告的误差表基准错行且自相矛盾、一处不确定度被低估）如实登记。
修复后算例池判据仍全部 <3%，且 `a/R=0.10` 的受判误差由 **−1.750% 改善到 −0.491%**。

被审对象（全部为 HEAD 提交态工件，非分支态）：
`benchmarks/verified/fem_hertz_contact_loadcontrol/{run.py,result.json,gates_verified.json,gates_truncation.json,truncation_extrapolation.py}`、
`src/tensorfem/hertz_halfspace_loadcontrol.py`、`docs/CASE-FEM-HERTZ-CONTACT-{verification,truncation-extrapolation}.md`、
平台报告 16017 / 16052 / 16080。

## 1. 复核了什么（独立、不共享驱动代码）

新增 `tools/hertz_anchor_recompute.py`：用与驱动无关的算术重新导出

1. Hertz 闭式（从参数 `P,R,E,ν` 重算 `a=(3PR/4E*)^{1/3}`、`δ=a²/R`、`p0=3P/(2πa²)`，`E*=E/(1−ν²)`）→ 18 个算例的解析列；
2. 每一行的 `error_pct`（从 `numeric`/`analytic` 重算）；
3. 20 条证书闸门中可独立定义的 13 条的值（其余 7 条只校验"value<op>threshold 与 declared passed 是否自洽"，并**标注为 not recomputed 而不是假定正确**）；
4. 估计器卫生：(a) 主估计 `a_linear` 是否等于某个节点半径（退化），(b) 受判网格上两个独立估计器的散布。

配套回归测试 `tests/test_hertz_anchor_evidence.py`（7 passed），把"闸门不许与自己的数字矛盾"、"主判据不许退化"、"算例池 <3% 从工件读而不是从散文读"变成机器可判。

## 2. 发现（按严重度）

**F1（严重，已修）主判据 `a` 在 `a/R=0.10, h/a=0.0075` 上退化为节点半径**

* 证据（HEAD `result.json`）：该行 `contact_radius_linear = 0.9825000000146723`，而 `contact_radius_outer = 0.9825000000000000`——
  两者只差 `1.5e-11`，即压力零交叉**落在节点上**，主估计等于"最外接触节点半径"。同网格另两个独立估计器给
  `contact_radius_fit = −0.046%`、`contact_radius_fit_consistent = +1.501%`，与受判的 **−1.750%** 相差 1.7–3.2 pp。
* 机理：接触成员判据原为 `lam > 1e-10 * max(lam)`，而 ALM 在接触边缘留下的节点乘子在几何衰减后仍有 ~`2e-9` 峰值的量级，
  于是被计入接触，把接触斑**多扩一个面单元**；`pp[-1]/(pp[-2]−pp[-1]) ≈ 2e-9` 使零交叉退化为节点。该伪节点只在最细网格
  （`h/a=0.0075`，17040 单元）出现，因此表现为"加密反而误差三倍"：−0.814% → −0.375% → **−1.750%**（序列非单调、比值为 1.2）。
* 影响：证书的"主估计收敛故事"是假的——`a` 的受判值在最细网格上是网格伪影；3% 判据碰巧仍满足，属"结论对但理由错"。
* 修复（最小改动，`src/tensorfem/hertz_halfspace_loadcontrol.py`）：成员阈值改为**物理压力比**
  `p_i > 1e-6 · max(p)`（Hertz 边缘 `p/p0=√(1−(r/a)²)`，`p/p0=1e-6` 对应距真边缘 5e-13 a，物理上不可分辨），
  并把 `membership_ratio / edge_pressure_ratio / contact_radius_linear_interior_rel` 写进 `metadata` → `result.json`
  的每行 `diagnostics`，使"该估计是不是节点绑定"可离线审计。
* 重新出证（全量 13 次轴对称接触求解重跑，`run.py` 全扫描）：

| a/R | h/a | 修复前 a 误差 | 修复后 a 误差 | δ 误差 |
|---:|---:|---:|---:|---:|
| 0.05 | 0.0300 | −0.848% | −0.848% | −0.996% |
| 0.05 | 0.0150 | −0.532% | −0.532% | −1.011% |
| 0.05 | 0.0075 | −0.445% | −0.445% | −1.010% |
| 0.10 | 0.0300 | −0.814% | −0.814% | −0.996% |
| 0.10 | 0.0150 | −0.375% | −0.375% | −1.011% |
| **0.10** | **0.0075** | **−1.750%（退化）** | **−0.491%** | −1.010% |

  其余量不变：峰值压力 +0.630%、ρ 无关性散布 `a` 0.0186% / δ 0.00034%、截断阶梯 δ 误差 −2.900/−1.639/−1.007%、
  外推残差 0.42781%、观测阶 1.1826。证书 20/20 闸门 PASS（`gates_verified.json` 由 `run.py` 重写）。

**F2（中，报告缺陷，未改他人报告）报告 16052 误差表基准错行且判词自相矛盾**

* 证据：该报告 ⑥ 表写「压入深 δ（a/R=0.10，接受模型）| 基准 0.025000 | 实测 0.098990 | +295.9609% | **是**」。
  同仓 `result.json` 该行 `analytic.indentation = 0.1`（Hertz `δ=a²/R`, `a=1`），实测 `0.098990` ⇒ 真误差 **−1.010%**；
  报告的 0.025 是 `a/R=0.05` 行的基准，串行导致 +295.96%，而判词仍写"是（<3%）"——数字与判词互相矛盾。
  本仓闸门 `indentation_aR010 = 1.00978` 是对的，**错在报告的散文表**。
* 处置：不改他人已发布报告；由本单报告给出更正表，并由 `tools/hertz_anchor_recompute.py` + 测试保证
  "判据表的数只能来自工件、且判词必须与数值自洽"（工具会以 `VERDICT_CONTRADICTION` / `GATE_VALUE_MISMATCH` 失败关闭）。

**F3（轻，不确定度被低估）外推残差给的是点值，不是区间**

* 证据：`docs/…-truncation-extrapolation.md` 记"换三点灵敏度 ±0.03 pp"；但 HEAD 工件里存在**两套不同阶梯**：
  `gates_truncation.json` 的 96→768a 给 0.42781%，而 anchor dump 的 24/48/96a 阶梯独立外推给 **0.373%**，
  差 0.055 pp（≈1.8× 报告的不确定度）。m17059 分支的 5 档拟合给 −0.391%（该分支未并入 main，见 F4）。
* 处置：审计工具对该量改用**带宽容差**（0.10 pp），并在报告中把开放残差写成区间 **[0.37, 0.43]%**。

**F4（轻，证据可达性）报告引用的闸门来自未合并分支**

* 证据：报告 16080（m17059）称"证书 17/17 含 `truncation_extrapolated_bias`、`indentation_aR005_accepted_L384`"，
  但这三个 id 与 `artifacts/contact_hertz_loadcontrol/depth_bias_probe.{py,json}`、`tests/test_hertz_depth_bias_probe.py`
  仅存在于 `agent/m17059` 分支（`git branch --contains 513ae7f6`），main 的证书是 20 条闸门且不含它们。
* 影响：主线证据链上没有这条探针；引用时应写"分支态证据"。本单未合并该分支（跨单合并不是本单职权）。

## 3. 未做的检查（如实标注 not_run，不算通过）

1. `truncation_extrapolation.py`（96→768a 阶梯）**未用新估计器重跑**：它的 `truncation_insensitive_contact_radius_aR005 = 0.00105 pp`
   是"a 对外场范围不敏感"的判据，新旧估计器都远小于阈值，定性结论不变，但其中 a 的绝对数值仍是旧估计器产物。
2. `depth_bias_probe`（5 档远场）未重跑（证据只在未合并分支）。
3. 全仓测试套件未全量跑（本机 load15 117/128，只跑了本单 7 条 + 依赖的 `__pycache__` 级导入）；仅本单相关测试绿。
4. 低载失效探针 `a/R=0.02` 仍**不入闸**（现 a 误差 −0.53%、δ −1.011%），本轮未把它升格为闸门。
5. 未定位 δ 的 1/L 外推残差 [0.37, 0.43]% 的物理机理（前单已否证"远场分辨率"假设；Q4 面弦/δ 提取方式仍是候选）。

## 4. 处置（disposition）

**approve with changes** —— 修复已落地并经重跑与 7 条回归测试验证；F2/F3/F4 是表述与可达性问题，已在本单报告中更正/标注，
未改他人报告与未合并分支。建议下一手（可单干、与在飞单不重复）：
① 用新估计器重跑 96→768a 截断阶梯并把 `truncation_extrapolation.json` 的 a 数值刷新；
② 把 `a/R=0.02` 低载失效探针升格为"分辨率下限"闸门（`a/h ≥ N_nodes` 的适用域判据）；
③ 追查 δ 残余模型偏差机理（费时，属 S4 探索）。

## 5. 复现

```bash
PYTHONPATH=src python3 benchmarks/verified/fem_hertz_contact_loadcontrol/run.py   # 全扫描 + 重写证书（13 次求解，~4 min 单核）
python3 tools/hertz_anchor_recompute.py --json artifacts/hertz_anchor_recompute.json  # 独立重算，findings: none
PYTHONPATH=src python3 -m pytest tests/test_hertz_anchor_evidence.py -q            # 7 passed
git show HEAD:benchmarks/verified/fem_hertz_contact_loadcontrol/gates_verified.json  # 20/20 PASS
```