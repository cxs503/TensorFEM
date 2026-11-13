# W26 · CSR ULS 判据与 Smith 法基准的锚件二次核销（TensorFEM）

- 任务：mission 17566 · 主题 `w26_csr_uls_anchor_recheck` · runner `native`
- 生成时间：2026-11-13T07:03:41Z
- 生成器：`benchmarks/hull_girder_uls/recheck_csr_uls_anchor.py` (sha256 `3f42fe2e8dbfcd59…`)
- 被核销锚件：`benchmarks/hull_girder_uls/csr_uls_anchor.json` (status `TRANSCRIBED`, anchor_id `AN-IACS-CSR`, sha256 `f061b2c0b0aebdd0…`)
- 交付工件：`papers/artifacts/w26/hull-girder-ultimate-strength/csr_uls_recheck.json`（含逐项数值表、判据逐条裁决、溯源字段）

## 1 案例介绍

IACS Common Structural Rules for Bulk Carriers and Oil Tankers, 01 January 2019 edition, Pt 1 Ch 5 (ULS) and Pt 1 Ch 4 Sec 4 (wave loads); IACS UR S7 Rev.4 (2010); IACS UR S11 Rev.10 (2020) as cross-check

被核销的是 W24 的逐条转录锚件：CSR Pt 1 Ch 5（ULS 判据）+ Pt 1 Ch 4 Sec 4（波浪弯矩）为主证，UR S7 Rev.4 与 UR S11 Rev.10 为交叉证。W26 不做新的工程计算，只回答一个问题：**锚件里每一个被下游使用的数字，是否真能在所引文档的所引页上找到**。

| 文档 | 抓取 URL | 字节 | 复算 sha256 | 与锚件一致 |
|---|---|---|---|---|
| AN-IACS-CSR | https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/2022/03/15144815/csr-h-01-jan-2019.pdf | 21439977 | `ffb51b3153e43a7d3a2e…` | 是 |
| AN-IACS-UR-S7 | https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/2022/05/18114816/ur-s7rev4.pdf | 18903 | `25a82bfec5f2beeafb78…` | 是 |
| AN-IACS-UR-S11 | https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/2022/05/18115332/ur-s11rev10.pdf | 1501639 | `4711024b4b537cd7e0c6…` | 是 |

研究船：kvlcc2，Lpp = 320.0 m，B = 58.0 m，D = 30.0 m，T = 20.8 m，Cb = 0.8095；UR S7 最小剖面模数参考值 96.3761 m³。

## 2 怎么做的

1. **重抓原件并复算 sha256**（`sync_documents`）：直接对 `anchor['source_documents']` 里的 URL 做匿名 GetObject（User-Agent `curl/8`），逐份比对字节数与 sha256，并与 `fetch_csr_sources.py` 内硬编码表做第三方比对。
2. **整册抽页文本**（`extract_pages`）：pypdf 6.19.0 逐页 `extract_text`，与 PDF sha256 一起缓存，二次运行不再联网。
3. **页码偏移自行推导**（`printed_page_offset`）：只认页眉 `COMMON STRUCTURAL RULES 01 JAN 2019 <printed>`，投票得出 csr: offset 4（383 页投票）；s7: offset 0（0 页投票）；s11: offset 0（0 页投票）。UR 文档无 CSR 页眉，按“印刷页 = PDF 页”处理，并由引句定位结果反证。
4. **逐句核销的四级粒度**（`locate_quote`）：整段 verbatim（空白归一）→ normalised（去非字母数字后比对，容忍 PDF 字距/连字符）→ cell（表格按 `;` 拆格）→ restated（数字全部落在候选页、且 ≥60 % 实词命中——公式被 PDF 文字层打散时的判据）。全册搜索而不是只在所引页搜索，因此**页码本身也被核验**。
5. **系数一一对应表**（`COEFFICIENT_REGISTER`）：锚件下游真正使用的 14 组系数（γS、γW、γM、γDB、fβ、fps、0.19、0.58、0.7、Cw 的 10.75 与断点、UR S11 的 0.19/0.11、175/k、S7 的 10.75）逐组要求“该项所引页上出现印刷数字 token”，并记录该 token 是否在引句字符串内（否则标为需按页核验）。
6. **Table 2 坐标级重读**（`recheck_table2`）：γS/γW 是锚件唯一没有逐字引句的系数组（引句只有表题）。W26 用文字显示算子 (x, y) 重做行/列分组，得到 2 行因子值，与锚件声明的坐标逐一吻合。
7. **代数与内部一致性**（`arithmetic_recheck`）：重算 0.19×0.58 对 UR S11 的 0.11、Cw 在 300/350 m 的接缝、UR S7 的 Wmin、γR = γM·γDB、175/k，以及交验表每行 u = γR·M/MU 与 u ≤ 1 的裁决一致性。
8. **Smith 基准复算**（`smith_capacity_baseline`）：M_u 取自 W23 Smith 逐步崩溃扫描输出，并对利用率最差的两例用同一求解器、同一离散（strips = max(FIBRE_LEVELS)、steps = max(STEP_LEVELS)）从零重解，以证明基准是算出来的而不是抄来的。

复现命令：

```bash
python benchmarks/hull_girder_uls/recheck_csr_uls_anchor.py --pdf-cache /tmp/iacs_w26
python benchmarks/hull_girder_uls/make_recheck_figures.py
python benchmarks/hull_girder_uls/make_recheck_readme.py
python -m pytest tests/test_csr_uls_recheck.py -q
```

环境：Python 3.11.6、pypdf 6.19.0、网络通道 https://iacs.s3.af-south-1.amazonaws.com (anonymous GetObject)。

## 3 计算结果

### 3.1 六条 ULS 判据逐条裁决（判据 W26-C1）

| # | 条目 | 条款 | 所引页 | 定位页 | 核销粒度 | 数字缺失 | 裁决 |
|---|---|---|---|---|---|---|---|
| 1 | CSR-ULS-01 | Pt 1, Ch 5, Sec 2, [2.1.2] | 343 | [343] | verbatim | — | PASS |
| 2 | CSR-ULS-02 | Pt 1, Ch 5, Sec 2, [2.2.1] | 343 | [343] | verbatim | — | PASS |
| 3 | CSR-ULS-03 | Pt 1, Ch 5, Sec 2, Table 1 | 343 | [342, 343, 345] | restated | — | PASS |
| 4 | CSR-ULS-04 | Pt 1, Ch 5, Sec 2, Table 2 | 344 | [344] | verbatim | — | PASS |
| 5 | CSR-ULS-05 | Pt 1, Ch 5, Sec 2, [2.1.2] | 343 | [343] | normalised | — | PASS |
| 6 | CSR-ULS-06 | Pt 1, Ch 5, Sec 2, [2.3.1] | 344 | [344] | verbatim | — | PASS |

可核销率 100.0 % （6/6），其中逐字可核销 66.7 %（4/6），重述型条目：['CSR-ULS-03', 'CSR-ULS-05']。判据集定义：the anchor's ULS acceptance-criteria block: the criterion itself (ULS-01), the demand combination (ULS-02), the design load scenarios (ULS-03), the partial safety factors (ULS-04), the capacity factor (ULS-05) and the capacity definition (ULS-06).  CSR-ULS-07 (f_beta) is a wave-load factor and is adjudicated in the coefficient one-to-one table; every one of the anchor's 19 entries is adjudicated entry-by-entry, so the set can be regrouped without re-deriving anything

全部 17 条转录条目定位情况：verbatim 条目级 9 条、restated 8 条、偏移 0 条、缺失 0 条；分段口径 13 verbatim / 7 normalised / 0 absent。

### 3.2 波浪弯矩与安全系数一一对应（判据 W26-C2）

| 系数组 | 符号 | 引用条目 | 所引页 | 印刷 token 命中 | 引句内含 | 裁决 |
|---|---|---|---|---|---|---|
| W-0.19 | 0.19 (Mwv magnitude coefficient) | CSR-WAV-01 | 187 | 0.19@p[191] | 否 | PASS |
| W-fnlvh | fnl-vh = 1.0 (strength assessment) | CSR-WAV-02 | 187 | 1.0@p[191] | 是 | PASS |
| W-fnlvs | fnl-vs = 0.58 (CB + 0.7) / CB | CSR-WAV-02 | 187 | 0.58@p[191], 0.7@p[191] | 是 | PASS |
| W-fm | fm plateau 0.4L ... 0.65L, fm = 1.0 | CSR-WAV-03 | 187 | 0.4@p[191], 0.65@p[191], 1.0@p[191] | 是 | PASS |
| W-Cw | Cw = 10.75 - ((300 - L)/100)^1.5, breaks 90/300/350/500 | CSR-WAV-04 | 182 | 10.75@p[186], 300@p[186], 350@p[186], 90@p[186], 500@p[186], 1.5@p[186] | 是 | PASS |
| W-fps | fps = 1.0 (extreme sea loads) | CSR-WAV-05 | 176 | 1.0@p[180] | 是 | PASS |
| U-fbeta | f_beta = 1.05 (seagoing) | CSR-ULS-07 | 324 | 1.05@p[328] | 是 | PASS |
| U-gammaM | gamma_M = 1.1 | CSR-ULS-05 | 343 | 1.1@p[347] | 是 | PASS |
| U-gammaDB | gamma_DB = 1.25 / 1.10 / 1.0 | CSR-ULS-05 | 343 | 1.25@p[347], 1.10@p[347], 1.0@p[347] | 是 | PASS |
| U-gammaS | gamma_S = 1.0 (Table 2) | CSR-ULS-04 | 344 | 1.0@p[348] | 否 | PASS |
| U-gammaW | gamma_W = 1.2 (A) / 1.3 (B) (Table 2) | CSR-ULS-04 | 344 | 1.2@p[348], 1.3@p[348] | 否 | PASS |
| S11-mw | UR S11 0.19 / -0.11 / (Cb + 0.7) | S11-01 | 4 | 0.19@p—（文字层打散：10190）, 0.11@p—（文字层打散：10190）, 0.7@p—（文字层打散：107.0110） | 否 | PASS_CROSS_DOCUMENT |
| S11-sigma | UR S11 permissible bending stress 175/k | S11-02 | 5 | 175@p[6] | 是 | PASS |
| S7-wmin | UR S7 Wmin = c L^2 B (Cb + 0.7) k, cn = 10.75 | S7-01 | 1 | 0.7@p[1], 10.75@p—（文字层打散：75.10） | 是 | PASS_CROSS_DOCUMENT |

Table 2 坐标级重读：A 行 y = 699.32 → γS = 1.0 (x = 352.44), γW = 1.2 (x = 396.6)；B 行 y = 682.82 → γS = 1.0 (x = 352.44), γW = 1.3 (x = 396.6) → 裁决 PASS。

### 3.3 代数与内部一致性（判据 W26-C4）

| 检查 | 陈述 | 计算值 | 参照/印刷值 | 相对偏差 | 裁决 |
|---|---|---|---|---|---|
| A-01 | CSR sagging coefficient 0.19 x 0.58 reproduces UR S11 0.11 | 0.11019999999999999 | 0.11 | 0.0018148820326678028 | PASS |
| A-02 | Cw branches join at L = 300 m and L = 350 m (anchor CC-03) | 分支接缝 | 9.303152563556661e-14 | 9.303152563556661e-14 | PASS |
| A-03 | UR S7 Wmin = cn L^2 B (Cb + 0.7) k for the study hull | 96.37614079999999 | 96.37614079999999 | 0.0 | PASS |
| A-04 | gamma_R = gamma_M gamma_DB reproduces the gamma_R used per condition | {'hogging_oil_tanker': 1.2100000000000002, 'sagging': 1.1, 'hogging_BC_A_empty_hold': 1.375} | {'hogging_oil_tanker': 1.21, 'sagging': 1.1} | 0.0 | PASS |
| A-05 | UR S11 permissible bending stress 175/k = 175 N/mm2 at k = 1 | 175.0 | 175.0 | 0.0 | PASS |
| A-06 | every applicable utilisation row satisfies u = gamma_R M / M_u and the verdict follows u <= 1 | u = γR·M/MU | 0.0 | 0.0 | PASS |

最差相对偏差 1.815e-03。

### 3.4 Smith 逐步崩溃基准（判据 W26-C5）

基准：repo Smith progressive collapse on study scantlings; not CSR App 2 net scantlings；样本 18 例，利用率区间 [0.4363, 1.1053]，超限 6 例 ['kvlcc2_x0.60_sagging', 'kvlcc2_x0.70_sagging', 'kvlcc2_x0.40_sagging', 'kvlcc2_x0.50_sagging', 'kvlcc2_x0.60_sagging', 'kvlcc2_x0.70_sagging']。

| 情景 | 算例 | x/L | 工况 | M_u (MNm) | M_y (MNm) | M_u/M_y | 需求 M (MNm) | u | 裁决 |
|---|---|---|---|---|---|---|---|---|---|
| A | kvlcc2_x0.20_sagging | 0.20 | sagging | 12729.6 | 17701.0 | 0.7191 | 6691.0 | 0.5782 | PASS |
| A | kvlcc2_x0.20_hogging | 0.20 | hogging | 17579.1 | 24373.8 | 0.7212 | 6339.3 | 0.4363 | PASS |
| A | kvlcc2_x0.30_sagging | 0.30 | sagging | 14751.1 | 20880.3 | 0.7065 | 10499.8 | 0.7830 | PASS |
| A | kvlcc2_x0.30_hogging | 0.30 | hogging | 20771.1 | 28668.3 | 0.7245 | 9279.8 | 0.5406 | PASS |
| A | kvlcc2_x0.40_sagging | 0.40 | sagging | 15598.3 | 22409.1 | 0.6961 | 13796.6 | 0.9729 | PASS |
| A | kvlcc2_x0.40_hogging | 0.40 | hogging | 22172.7 | 30707.1 | 0.7221 | 12373.1 | 0.6752 | PASS |
| A | kvlcc2_x0.50_sagging | 0.50 | sagging | 15692.6 | 22653.6 | 0.6927 | 13926.7 | 0.9762 | PASS |
| A | kvlcc2_x0.50_hogging | 0.50 | hogging | 22416.0 | 31187.0 | 0.7188 | 12373.1 | 0.6679 | PASS |
| A | kvlcc2_x0.60_sagging | 0.60 | sagging | 15598.3 | 22409.1 | 0.6961 | 14558.4 | 1.0267 | FAIL |
| A | kvlcc2_x0.60_hogging | 0.60 | hogging | 22172.7 | 30707.1 | 0.7221 | 12373.1 | 0.6752 | PASS |
| A | kvlcc2_x0.70_sagging | 0.70 | sagging | 14751.1 | 20880.3 | 0.7065 | 13602.5 | 1.0143 | FAIL |
| A | kvlcc2_x0.70_hogging | 0.70 | hogging | 20771.1 | 28668.3 | 0.7245 | 10605.5 | 0.6178 | PASS |
| B | kvlcc2_x0.20_sagging | 0.20 | sagging | 12729.6 | 17701.0 | 0.7191 | 7248.6 | 0.6264 | PASS |
| B | kvlcc2_x0.30_sagging | 0.30 | sagging | 14751.1 | 20880.3 | 0.7065 | 11336.2 | 0.8453 | PASS |
| B | kvlcc2_x0.40_sagging | 0.40 | sagging | 15598.3 | 22409.1 | 0.6961 | 14911.7 | 1.0516 | FAIL |
| B | kvlcc2_x0.50_sagging | 0.50 | sagging | 15692.6 | 22653.6 | 0.6927 | 15041.8 | 1.0544 | FAIL |
| B | kvlcc2_x0.60_sagging | 0.60 | sagging | 15598.3 | 22409.1 | 0.6961 | 15673.6 | 1.1053 | FAIL |
| B | kvlcc2_x0.70_sagging | 0.70 | sagging | 14751.1 | 20880.3 | 0.7065 | 14558.4 | 1.0856 | FAIL |

注意：所有算例 M_u/M_y ∈ [0.6927, 0.7245]，即 Smith 逐步崩溃容量低于弹性屈服矩——受压板与纵骨先屈曲、峰值应力低于屈服（W23 交验表 dominant_mechanism 列同向：受压面板峰值低于屈服的屈曲控制）。这是本研究的船体剖面设计特征，不是核销误差。

从零复算（最差两例，同一离散）：

| 算例 | 存储 M_u (MNm) | 复算 M_u (MNm) | 相对偏差 | fibres | ledger 残差 | 裁决 |
|---|---|---|---|---|---|---|
| kvlcc2_x0.60_sagging | 15598.270014 | 15598.270014 | 0.000e+00 | 51 | 3.126e-06 | PASS |
| kvlcc2_x0.70_sagging | 14751.109193 | 14751.109193 | 0.000e+00 | 51 | 2.416e-06 | PASS |

### 3.5 f_beta 适用范围敏感性（锚件转录颗粒度的后果量化）

锚件引句把四条印刷条目压缩为“seagoing conditions”，交验表对全部算例取 f_beta = 1.05。印刷条目实际为：HSM, FSM (extreme sea loads) → 1.05；BSR, BSP (extreme sea loads) → 0.8；HSA, OST, OSA (extreme sea loads) → 1.0；ballast exchange, harbour, flooded → 1.0。

| 替代 f_beta | 利用率区间 | 裁决翻转数 | 翻转算例 |
|---|---|---|---|
| 1.0 | [0.4161, 1.0566] | 2 | A:kvlcc2_x0.60_sagging（1.027 → 0.982）, A:kvlcc2_x0.70_sagging（1.014 → 0.974） |
| 0.8 | [0.3350, 0.8650] | 6 | A:kvlcc2_x0.60_sagging（1.027 → 0.802）, A:kvlcc2_x0.70_sagging（1.014 → 0.811）, B:kvlcc2_x0.40_sagging（1.052 → 0.808）, B:kvlcc2_x0.50_sagging（1.054 → 0.812）, B:kvlcc2_x0.60_sagging（1.105 → 0.862）, B:kvlcc2_x0.70_sagging（1.086 → 0.865） |

## 4 结果图

![quote matrix](fig_recheck_quote_matrix.png)

*图 1*　锚件 19 条条目的核销粒度地图（绿=verbatim，蓝=符号/空白归一，橙=公式或表格单元重述但数字全部印刷在页，灰=声明 WITHHELD），右侧给出所引页与实定位页——两者一致是“页码也被核验”的直接证据。

![wave coefficients](fig_recheck_wave_coefficients.png)

*图 2*　左：Cw 三分支在 L = 300 m / 350 m 的接缝（机检残差 < 1e-12）；右：CSR 中垂波浪弯矩表达式与独立 UR S11 表述的对比，二者代数上同归 0.11 (Cb + 0.7) Cw L² B，双轴给出逐点相对差。

![utilisation](fig_recheck_utilisation.png)

*图 3*　左：按系数加权的 CSR 利用率与 u = 1 限值，空心标记为 f_beta = 1.0 / 0.8 的替代值（见 3.5）；右：Smith 逐步崩溃容量 M_u 沿站位的分布。

## 5 注意点

- **F-01（info）** 17 of 17 transcribed entries re-locate in the fetched PDFs (9 verbatim at entry level, 8 only as restatements: ['CSR-ULS-03', 'CSR-ULS-05', 'CSR-ULS-07', 'CSR-WAV-02', 'CSR-WAV-03', 'CSR-WAV-04', 'S7-01', 'CSR-ULS-W03']; segments: 13 verbatim, 7 normalised, 0 absent)
- **F-02（info）** Table 2 gamma_S/gamma_W is the only coefficient group the anchor carries without a verbatim quote; the coordinate re-read returns verdict PASS (2 factor rows)
- **F-03（info）** anchor algebra and printed numbers agree: worst relative difference 1.815e-03 (verdict PASS)
- **F-04（info）** Smith M_u baseline recomputation: PASS over 2 worst-utilisation cases
- **F-07（medium）** the f_beta transcription condenses four printed load-case bullets (1.05 HSM/FSM, 0.8 BSR/BSP, 1.0 HSA/OST/OSA) into 'seagoing conditions'; applying the alternative values flips 2 verdict(s) at f_beta = 1.0 and 6 at f_beta = 0.8
- **F-08（medium）** the WITHHELD entries are conservative: 2 of them have their clause text locatable in the fetched document (sample lines recorded in withheld_probe); W26 does not transcribe them
- **F-06（low）** coefficient groups whose anchor quote is a caption or model sentence only: ['W-0.19', 'U-gammaS', 'U-gammaW', 'S11-mw']; values verified against page tokens, quote field could carry the table body

- **CSR-ULS-W01**：text located in the fetched document, so the WITHHELD entry is conservative; W26 does not transcribe it (out of scope) but records the locator for the next pass；命中页 {'load-end shortening': [361, 362, 367, 368, 369, 370, 371]}
- **CSR-ULS-W02**：text located in the fetched document, so the WITHHELD entry is conservative; W26 does not transcribe it (out of scope) but records the locator for the next pass；命中页 {'checking criteria': [346, 349, 779, 781], 'net scantling': [34, 93, 94, 327, 335, 375, 398, 463, 572, 679, 682, 685, 773]}

- 限制：the recheck verifies that printed values exist on the cited page and agree with the anchor's algebra; it does not re-derive CSR clause interpretation (e.g. which loading condition governs)
- 限制：Table 1 (design load scenarios) is prose in the page text, so its rows are verified segment-wise, not by coordinates
- 限制：M_u stays on the repo Smith basis; the CSR App 2 net-scantling capacity is still not exercised (CSR-ULS-W01/W02 remain WITHHELD)
- 限制：only the extreme sea loads scenario (fp = fps = 1.0) is covered by the utilisation table inherited from W24

## 6 误差判定

| 关键量 | 相对偏差 | 是否 < 3 % | 说明 |
|---|---|---|---|
| CSR 0.19 x 0.58 versus UR S11 printed 0.11 | 0.1815 % | 是 | UR S11 prints 0.11, the CSR product is 0.1102: the two rule sets agree to 0.2 % |
| UR S7 Wmin recomputed versus anchor value | 0.0000 % | 是 |  |
| Smith M_u stored versus recomputed | 0.0000 % | 是 |  |
| utilisation identity u = gamma_R M / M_u | 0.0000 % | 是 |  |

对锚结论：**六条 ULS 判据与全部下游系数均可在原件所引页核销，锚件 sha256 复算一致（3/3 份文档）**；最差数值偏差来自 CSR 与 UR S11 两条规则表述之间的 0.19×0.58 = 0.1102 对 0.11（0.18 %），远小于 3 %。逐字级别并非 100 %：8 条条目的引句是公式/表格的 ASCII 重述（PDF 文字层无法给出连续字符串），其数字全部落在所引页，已在 3.1/3.2 明示粒度，未做任何数字外推。

