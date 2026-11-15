# GAP-PP-1-ICE-STRUCT-CONTACT-PATCH — 对抗性独立复核（verifier 席）

锚点：`GAP-PP-1-ICE-STRUCT-CONTACT-PATCH`（标题：冰压斑载荷由 CZM 接触求解生成并喂给板格求解器（双向耦合））
复核对象（HEAD）：`src/tensorfem/patch_from_contact.py`、`src/tensorfem/ice_panel_patch_contact.py`、
`examples/ice_struct_contact/end_to_end_demo.py`，以及它们的自证报告
`docs/VERIFY-GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.md` 与归档工件
`benchmarks/artifacts/GAP-PP-1-patch-from-contact-verify/verification.json`（mission 20624）。

复核工具（可复跑，本单新增）：`tools/verify_gap_pp1_adversarial.py`
复核测试（本单新增）：`tests/test_gap_pp1_adversarial_audit.py`

> 复核立场：**不信任被复核方的归档数字**。A/B/C/D/E 五节全部从 HEAD 重新计算，
> 只调用公开 API，被复核模块内部若有缺陷无法在审计脚本里"自我掩盖"。

---

## 1. 独立性局限（independence limitations）

| 项 | 说明 |
|----|------|
| 作者 vs 复核者 | 本复核者**未参与** m20624 / m21350 / m21619 的实现，但与它们是同一 agent 席位（同模型），存在同源盲点 |
| 基准的独立性 | Navier 双重级数（`IC.navier_rect_moments`）与被复核 FE 解**共享同一压力场对象**，压力场本身的构造误差无法被这条交叉校验发现（第五节 E 单独量了该项） |
| 未覆盖范围 | 双向耦合（`coupled_patch_solve`）的收敛性、塑性/屈曲/大挠度区间、三维壳、真实冰压分布；本复核只覆盖 **弹性小挠度** 口径 |
| 未运行检查 | 全仓回归（仅跑本包 + 既有 `tests/test_ice_panel_patch_contact.py`）；CZM 接触求解器本身的物理正确性（另属 H3/campaign_nodes.id=29 范围） |

---

## 2. 复核范围与所做检查

| 节 | 检查内容 | 方法 |
|----|----------|------|
| A | 网格收敛：FE（两条链）vs 独立 Kirchhoff Navier | 网格阶梯 nx = 16,24,32,48，Navier 在 FE 峰值点同点取值（避免"错位比较"） |
| B | 两条 FEM 链是否真的独立 | 检查两者装配调用链 + 同网格数值差 |
| C | 归档工件能否在 HEAD 复现 | 用当前代码重算归档行，比较漂移 |
| D | 端到端算例的小挠度适用性 / 比例的载荷无关性 | w/t 诊断 + 载荷缩放 1e-2 后的比例不变性 |
| E | `loaded_area` 语义 & "力误差"的两个口径 | 足迹面积 vs 承压面积；投影误差 vs 离散场-解析求积误差 |

---

## 3. 严重度排序的发现

### F1（Blocker，主张不成立）"FEM vs Navier 误差 0.41 %" 是**单网格择优值**，且不是收敛值

* 证据（HEAD 重算，`A_crosscheck_convergence`）：

  | 算例 | nx=16 | nx=24 | 收敛趋势 |
  |------|-------|-------|----------|
  | 均匀斑 0.4×0.4 m | **1.090 %** | **0.496 %** | 单调下降 |
  | 高斯斑 peak 2 MPa, σ=0.08 m | **1.687 %** | **0.735 %** | 单调下降 |

  也就是说：0.41 % 既不是 nx=16 的实测值（1.687 %），也不能单列出来当作"链的精度"，
  它只是某一行、某一网格的取数。真实可陈述的是"收敛后 < 1 %（nx≥24）"，仍满足 3 % 判据，
  但报告里"0.41 % 同量级于 H3 锚 0.339 %"的措辞把**离散误差**当成了**链的固有精度**。
* 影响：验收判据 4（H3 锚同量级一致性）本身可过，但支撑它的数字被美化了 4 倍。
* 修复：报告加勘误节（见 §5），审计工具固化网格阶梯，新增测试
  `test_fe_error_converges_toward_the_navier_series` 要求"单调下降 + 最细网格 < 1 % + 全部 < 3 %"。

### F2（Blocker，主张不成立）"双独立 FEM 路径一致性 < 10 %"不是独立校验

* 证据（`B_chain_independence`）：两条链（`patch_from_contact.contact_pressure_to_plate_load`
  → `plate_mindlin.solve_plate_on_winkler_foundation`；`ice_panel_patch_contact.solve_rect_panel`
  → `foundation_plate_bending.*`）**共用同一装配例程**，同网格同点数值差
  `1.6e-12`（相对）。两条"独立路径"的一致性是**代码路径同一性**，不能作为交叉验证。
* 影响：这条"PASS"在验收表里贡献了虚假的置信度。
* 修复：报告中该行改为"同路径一致性（1.6e-12，非独立校验）"；独立证据由 F1 的 Navier 级数承担。

### F3（Major）归档 verification.json 在 HEAD **不可复现**（漂移 10.4 %）

* 证据（`C_artefact_reproduction`，nx=16）：均匀斑 `mine` 归档 368.34 MPa → HEAD 330.03 MPa
  （漂移 **-10.4 %**）；高斯斑 `mine` 归档 842.11 MPa → HEAD 859.89 MPa（+2.1 %），
  后者与 `ic` 链 859.885 MPa 完全一致 —— 说明 m21350/m21619 合并后两条链已统一，
  而 `docs/VERIFY-...md` 与归档 json 仍是旧链的数字，**没有随代码更新**。
* 影响：报告引用的任何数字都不能假设"对着当前代码"。这类"工件不跟随代码"正是验证欠账的典型形态。
* 修复：审计工具每次重算归档行并输出漂移；测试 `test_archived_artifact_is_rechecked_not_trusted`
  要求 HEAD 数值仍在 3 % 内并显式给出可复现性裁决。

### F4（Major）端到端算例的**绝对应力不在小挠度适用域**（w/t ≈ 14）

* 证据（`D_validity`）：报告头号结论"非均匀斑使 σ_x 升高 +170.6 %"对应峰值挠度 139.1 mm，
  板厚 10 mm ⇒ **w/t = 14.0**，而小挠度 Mindlin 理论要求 w/t ≲ 0.5。
  线性理论里比例是载荷无关的（实测把载荷降 1e-2 后比例改变 < 1e-13），
  所以该 **比例** 是"线性模型的性质"，而非"收敛的物理量"；绝对应力 2493 MPa 更不能当设计值。
* 影响：结论方向（非均匀性显著提高峰值应力）成立，但量化口径被误当作物理事实。
* 修复：demo 增加 `small_deflection_validity` 诊断并被测试钉扎；报告写明适用域。

### F5（Major，MOCK 级）demo 里的 H3 锚一致性检查是**硬编码常量**

* 证据：`examples/ice_struct_contact/end_to_end_demo.py` 旧版
  `"gaussian_patch_fem_vs_navier_pct": -0.41`、`"same_order_of_magnitude": True` 为字面量，
  判据永远 PASS，属 MOCK 级"自证"。同时 demo 的产物目录 `outputs/` 被 `.gitignore` 排除，
  报告所引 4 张图与 `end_to_end_demo.json` 在仓库里根本不存在。
* 修复（本单已落地）：
  * `h3_anchor_check()` 从归档 json 读锚值与实测值，布尔量由数值派生（`|err| ≤ 3×anchor`），
    归档缺失时给出 `"consistency check NOT performed"` 而非静默 PASS；
  * 默认产物路径改为提交型 `benchmarks/artifacts/GAP-PP-1-end-to-end-demo/`，
    4 张图与 json 已随本单入库；
  * 新增测试 `test_demo_h3_check_is_derived_from_the_archive`。

### F6（Minor→已修）"投影总力 vs 解析总力 < 0.001 %"混淆了两个不同的求积口径

* 证据（`E_load_area`）：投影 vs **离散接触场**总力：`0.0`（机器精度，成立）；
  离散接触场 vs **解析** `2πσ²` 力：**0.55 %**（高斯斑，网格 + 3σ 截断）。
  CZM 斑的 `total_force` 与离散场自洽（5.1e-15），所以两个口径在那行才重合。
* 修复：报告勘误为"投影 vs 离散场 < 1e-12；离散场 vs 解析 0.18–0.60 %（均 < 3 %）"。

### F7（Minor→已修）`loaded_area` 是"足迹面积"，不是"承压面积"

* 证据：CZM / 高斯斑的网格内压力处处为正 ⇒ 足迹=承压（overstatement 1e-12，当前用法无误）；
  但**单侧接触**（部分网格零压，冰-结构实际形态）时 `loaded_area` 把承压面积高估 **100 %**，
  `F/loaded_area` 因此低估平均接触压力一倍。
* 修复：`patch_from_contact()` 的 `info` 新增 `pressurised_area`（严格 p>0 的子单元面积）与
  `footprint_area`（= 原 `loaded_area`，向后兼容），docstring 明确语义；测试
  `test_pressurised_area_is_exposed_by_the_projection` 与 `test_loaded_area_and_the_two_force_quadratures` 钉扎。

---

## 4. 未运行的检查（unknown，不是 pass）

* 双向耦合 `coupled_patch_solve` 的收敛性与固定点唯一性（既有测试只断言 `converged` 与会力，未做网格/容差敏感性）；
* 大挠度/膜力、塑性、屈曲区间（本链为线性理论，F4 已说明绝对应力不可用于该域）；
* 真实 CZM 接触压力分布（campaign_nodes.id=29）与高斯参数化分布的差异；
* 全仓回归（本单只跑 `tests/test_ice_panel_patch_contact.py` + 新增审计测试）。

---

## 5. 处置（disposition）与已落地整改

| 项 | 处置 | 落点 |
|----|------|------|
| F1 | 报告勘误 + 网格阶梯固化 | `docs/VERIFY-...md` 勘误节；`tools/verify_gap_pp1_adversarial.py` §A |
| F2 | 主张降级为"同路径一致性" | 同上勘误节；§B |
| F3 | 归档漂移显式化 | §C + `audit.json` |
| F4 | 适用域诊断入 demo 与报告 | `small_deflection_validity`；勘误节 |
| F5 | 去掉硬编码、产物入仓 | `examples/ice_struct_contact/end_to_end_demo.py`、`benchmarks/artifacts/GAP-PP-1-end-to-end-demo/` |
| F6/F7 | 口径分离 + API 补字段 | `src/tensorfem/ice_panel_patch_contact.py` |

**结论：request changes（已在同一 PR 内整改）**。锚点的功能链路（接触→投影→板格应力）
真实可用、力守恒到机器精度、误差在收敛后 < 1 %（< 3 % 判据）；
但原报告的 4 条"PASS"里 2 条（0.41 % 精度、双独立路径）属于**结论正确但证据表述不成立**，
1 条（H3 检查）是硬编码，1 条（产物）不存在。整改后上述各点均已变成可复跑的机读证据。

复跑命令：

```bash
python -m pytest tests/test_gap_pp1_adversarial_audit.py tests/test_ice_panel_patch_contact.py -q
python tools/verify_gap_pp1_adversarial.py --meshes 16,24,32,48 \
       --out benchmarks/artifacts/GAP-PP-1-adversarial-verify
```

工件：`benchmarks/artifacts/GAP-PP-1-adversarial-verify/audit.json`
（含 A–E 全部表格、`fig_convergence.png`、`fig_validity.png`）。