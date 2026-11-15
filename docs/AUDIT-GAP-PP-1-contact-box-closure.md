# GAP-PP-1-ICE-STRUCT-CONTACT-PATCH — 接触分析盒闭合性复核（verifier 席，m22228）

锚点：`GAP-PP-1-ICE-STRUCT-CONTACT-PATCH`（冰压斑载荷由 CZM 接触求解生成并喂给板格求解器，双向耦合）
复核对象（HEAD）：`src/tensorfem/ice_panel_patch_contact.py`（`czm_contact_patch` /
`coupled_patch_solve` / `nonuniformity_scan` / `default_configurations`）及其既有证据
`benchmarks/artifacts/GAP-PP-1-czm-contact-analytic/verification.json`、
`benchmarks/artifacts/GAP-PP-1-adversarial-verify/audit.json`、`docs/AUDIT-GAP-PP-1-adversarial-verification.md`、
`docs/GAP-PP-1-adversarial-verify-m22149.md`。

新复核工具（本单新增，可复跑）：

```bash
python tools/verify_gap_pp1_contact_box_closure.py \
    --out benchmarks/artifacts/GAP-PP-1-contact-box-closure
```

新复核测试（本单新增）：`tests/test_gap_pp1_contact_box_closure.py`（7 项）
工件：`benchmarks/artifacts/GAP-PP-1-contact-box-closure/closure.json` + `fig_closure_vs_headline.png`

---

## 1. 独立性局限（independence limitations）

| 项 | 说明 |
|----|------|
| 作者 vs 复核者 | 本席未参与该模块实现，但与实现者是同一 agent 席位（同模型、同源盲点） |
| 基准独立性 | 第一节的闭式解（抛物面压头 + 线性罚 Winkler 基础）是**解析式**，与被复核代码无共享代码；仅为**单向、无盖帽（p<σ_c）**口径 |
| 未覆盖 | 塑性/屈曲/大挠度、三维壳、真实冰压分布（campaign_nodes.id=29）；盖帽（p=σ_c）段的闭式解未构造，仅用"闭盒自洽"口径 |
| 未运行 | 全仓回归；全分辨率（nx=40/ny=35/grid=97）非均匀性扫描未在本单重跑（本单 D 节用 nx=24/ny=21/grid=61） |

---

## 2. 严重度排序的发现

### F1（Blocker，结论口径错误）生产算例的接触分析盒**截断了接触斑**，且从未被诊断

接触求解在矩形分析盒上做，盒半宽由 `edge["box_half"]` 指定、或以 `2*amplitude+0.05` 启发式给出。
生产配置一律 `box_half = 0.10 m`、法向力 40 kN。解析接触半径：

```
delta0 = sqrt(2·amp·F/(pi·k·R^2)) = 3.568e-4 m ,  r_c = sqrt(delta0 R^2/amp) = 0.1336 m
```

**r_c = 0.134 m > 盒半宽 0.10 m** ⇒ 盒面仍承压（实测 `p_boundary/p_peak = 0.478`），于是
`contact_area` 变成盒面积（0.040 m²，解析 0.0561 m²，**−28.7 %**）、`mean = F/area` 成为盒产物、
力平衡把峰值压力**抬高 +7.44 %**（1.5334 MPa vs 解析 1.4273 MPa）。

* 为什么之前没被发现：既有解析复核（`GAP-PP-1-czm-contact-analytic`）用的是
  **F = 7.07 kN、delta0 = 1.5e-4 m、r_c = 0.0866 m < 0.10 m** —— 接触**恰好**在盒内闭合，
  于是精度达到 1e-4。**被验证的口径和实际部署的口径不是同一个**，而代码里没有任何量把这件事说出来。
* 修复：`ContactPatch` 新增 `boundary_pressure_ratio` / `contact_closed`；`patch_from_contact` 之外
  的接触求解路径全部带上该诊断；`require_closed=True` 可把截断变成硬错误。

### F2（Blocker，锚点头号数字被掩盖）截断使"非均匀斑 vs 均匀斑"的峰值应力比几乎失效

同总力、同足迹、同网格（nx=24/ny=21，contact grid 61），球面冰棱 40 kN：

| 分析盒 | `contact_closed` | σ_Mx(非均匀)/σ_Mx(均匀) | 相对均匀上升 |
|--------|------------------|--------------------------|--------------|
| 0.10 m（原生产） | False（bnd 0.478） | **1.0774** | +7.7 % |
| 0.20 m（闭合） | True（bnd 0.0） | **1.7409** | +74.1 % |

即：原口径把"非均匀性使峰值应力上升"的幅度压小了近一个量级（超出量 0.077 → 0.741）。
原因是截断把压力场削成"平顶"（边缘 0.73 MPa ≫ 0），等于人为抹平了非均匀性。
**锚点验收判据 2（非均匀性影响扫描）的数值必须按闭盒重算**，否则报的是盒子的性质。

### F3（Major）双向耦合斑比单向更宽，因此基于单向（平板）尺度定的盒**必然**截断

板挠度使冰-板开口变小，但同时也让所需接近量 delta0 增大（k=4e9 Pa/m 的软界面 + w≈1.1 mm ⇒
delta0≈1.34 mm），净效应是**接触半径变大**：实测耦合斑在 0.10 m 盒内 `bnd=0.78`，
在 0.135 m 盒内仍 `bnd=0.43`，直到 **0.20 m** 才闭合。原 `coupled_patch_solve` 只在
单向启发式盒上求解（用 `2*amp+0.05=0.09`），因此 C3（双向球面）与既有耦合收敛性复核
（`tools/verify_gap_pp1_coupling_convergence.py`，`box_half=0.10`）**全程运行在截断斑上**。
* 修复：`coupled_patch_solve` 在盒为派生（无 `box_half`/`contact_box`）时，**在耦合不动点上**
  迭代扩张盒子直到闭合（≤6 次，每次 ×1.5）；显式盒照样尊重并如实上报。

### F4（Major）`nonuniformity_scan` 的默认 `max_coupling=12` 在收敛前被切断

C4（圆柱脊 + 偏心，盖帽到 σ_c=2 MPa）压力场相对变化序列为
2.72e-1, 1.26e-1, …, 1.55e-4（12 次）——**tol=1e-4 需要第 13 次**，于是扫描行报
`converged: False`，而同一配置在 `max_coupling=40` 下 13 次即收敛（末次 dp 7.5e-5）。
即"扫描没收敛"是默认预算问题，不是物理问题。
* 修复：`coupled_patch_solve` 默认 `max_coupling` 12 → 40；C4 现报 `converged: True`。

### F5（Minor）扁平冰棱（`flat`）的盒宽是**建模选择**，不是可闭合量

`flat` 边缘无浮雕 ⇒ 接触填满整个盒，"承压面积"由盒宽定义。因此诊断返回
`contact_closed = None`（不适用），`C1` 行的足迹应理解为"名义斑足迹"，不能当"解得接触面积"。
文档与 `as_dict()` 已标注。

---

## 3. 修复后自检（全部机读，`closure.json` → `E_verdict`）

| 判据 | 结果 |
|------|------|
| 求解器在**生产力 40 kN** 与闭式解一致（|误差| < 1 %） | peak −0.001 %/+0.000 %，area ≤ 0.1 % → **PASS** |
| 截断能被诊断捕获 | 0.10 m 盒 `contact_closed=False`、`require_closed` 抛错；0.20 m 盒 `True` → **PASS** |
| 截断确实抬高峰值压力（> 3 %） | +7.44 %（contact）/ +7.7 %（板应力比口径） → **PASS** |
| `default_configurations()` 四档全部闭合 | C1 None（flat，不适用）、C2/C3/C4 `True`，max bnd = 0 → **PASS** |
| 四档耦合全部收敛 | 12/13 次内 `converged=True`（默认预算改 40） → **PASS** |
| 锚点头号比值对盒子敏感（> 3 %） | 1.0774 vs 1.7409（+61 %） → **PASS（即原口径不可用）** |

测试：`python -m pytest tests/test_gap_pp1_contact_box_closure.py -q` → 7 passed；
与既有解析复核联跑 `tests/test_czm_contact_analytic.py` → 14 passed（17.6 s）。

---

## 4. 面向锚点验收的处置（disposition）

| 锚点验收判据 | 处置 |
|--------------|------|
| (1) `patch_from_contact()` 把接触面法向压力投影成板格载荷 | **成立**（本次未动；力守恒 1e-16 已由既有审计覆盖） |
| (2) 非均匀性对板格峰值应力的影响扫描（≥3 档构型） | **口径需重算**：F1–F3 修正后 C2/C3 用 0.20 m 盒、C4 用 0.12 m（脊长=盒宽，需声明）；原 0.10 m 结果作废 |
| (3) 与 JOUR-ICE-STRUCT 均匀斑基准同口径对照 | 均匀参考斑的足迹**取自解出的斑网格**，因此盒宽变化会同步改变对照足迹：报告必须写出足迹（0.40×0.40 m for C2/C3），不可只写比值 |
| 端到端算例 ≥1 可复现、pytest ≥5 绿 | 成立（本单再增 7 项，含盒闭合闸门） |
| 与 H3-ICE-PLATE-FEM 锚（0.339 % 级）同量级一致性 | **与本缺陷不冲突**：该项比较的是 FEM-vs-Navier 离散误差（收敛后 <1 %），不是接触盒；但必须与新盒子口径一并重述 |

**结论：request changes（本单已整改库侧与测试侧，扫描数值需按闭盒重跑后方可作为发表数字）。**
诚实边界：本单没有重跑全分辨率 `nonuniformity_scan`（预算），因此"D 节 1.7409"是
nx=24/ny=21/grid=61 的口径；发表的锚点数字应在此基础上做网格收敛后再引用。

---

## 5. 可复跑清单

```bash
python -m pytest tests/test_gap_pp1_contact_box_closure.py -q
python -m pytest tests/test_czm_contact_analytic.py -q
python tools/verify_gap_pp1_contact_box_closure.py \
       --out benchmarks/artifacts/GAP-PP-1-contact-box-closure
```