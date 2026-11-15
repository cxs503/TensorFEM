# GAP-PP-1 验收量审计：非均匀斑 vs 均匀斑峰值应力差（包络 / 网格 / 口径）

锚点：`GAP-PP-1-ICE-STRUCT-CONTACT-PATCH`（冰压斑载荷由 CZM 接触求解生成并喂给板格求解器）
本单：`m22352`（tp-ds-2 generalist 席自驱工时）。类别：**验证欠账**（对已落地功能做独立复核）。

## 0. 一句话结论

锚点验收要求「非均匀斑相对均匀斑峰值应力差异给出机读 JSON」。复核发现这个数在**落地实现里不是一个可复现的量**：

1. **包络**：落地扫描（`scan_contact_configurations` 默认档）在 `w/t = 4.7 / 14.6 / 16.6` 下运行，
   而小挠度板理论的有效域是 `w/t ≲ 0.5`。报告的是一个线性模型的比例量，却从未在该模型有效的域内算出过。
2. **网格**：该比例量取自「逐单元形心 |σ_x| 取最大」，两个载荷的最大值位置会随网格跳格。
   实测在**包络内**（`w/t=0.2`）同一块板、同一组构型：`Δσ_x` 在 nx=8/12/16 之间从 **−44 % 跳到 +23 %**，
   连**符号都不稳定**。⇒ 单网格峰值比不能当作验收数。
3. **口径**：均匀参考斑的「同口径」= 同总力 + 同足印（`patch_width × patch_length`），而足印从未被当作
   受控参数扫描过；`Δ` 是（接触场, 参考足印）这对参数的函数，不是接触场单独的属性。

据此：(a) 把**包络控制**做进模块与默认档（已实现并测试），(b) 交付两个验证工具把 `Δ` 的
网格/口径/级数截断误差显式量化，(c) 明确「可引用的 `Δ` 必须来自解析（Navier 级数）路径」。
**FEM 路径本身没有错**：在**同一物理点**上 FEM 与独立 Navier 级数一致（落地证据 0.45 %–1.69 %），
错的是「两个最大值相除」这个统计量。

## 1. 案例（物理背景 / 几何 / 参数）

| 项 | 值 |
|---|---|
| 板 | 1.0 m × 1.0 m × 10 mm，SSSS（简支），E = 206 GPa，ν = 0.3 |
| 弹性地基 | Winkler `k = 1e7 N/m³`（与落地 `panel_patch_load` 基线同口径）；解析段用 `k = 0`（级数可用域） |
| 接触场 | 高斯斑 `p = p0 exp(-r²/2σ²)`，σ = 0.15/0.10/0.05 × side（三档非均匀性）|
| 构型 1/2/3 | peak 5/8/15 MPa（名义档）→ 均匀参考足印 0.30/0.20/0.10 × side |
| 小挠度限 | `w/t ≤ 0.5`；工作点 `w/t = 0.2` |
| 载荷施加 | 接触节点压力 → `patch_from_contact` 精确保守投影（子单元切分，机读 `load_balance_rel < 1e-10`）|

## 2. 怎么做的

1. **包络审计**（`tools/verify_gap_pp1_delta_caliber.py::envelope_audit`）：同一块板跑两遍扫描——
   历史名义压力档（`target_w_over_t=None`）与包络控制档（`target_w_over_t=0.2`），逐构型报告 `w/t`、
   是否在包络内、以及 `Δσ_x`。
2. **网格审计**（同工具 `delta_mesh_ladder`）：包络内逐网格 nx=8/12/16(/24/32) 计算 `Δσ_x`、`Δσ_vm`、`Δw`。
3. **口径审计**（同工具 `caliber_sensitivity`）：固定接触场与总力，把均匀参考足印换成
   6σ / 4σ / 3σ / 等面积 / 规则 0.3·side / 0.2·side 六种口径，看 `Δ` 的跨零漂移。
4. **解析路径**（`tools/verify_gap_pp1_delta_navier.py`）：两个峰值应力都取自 Kirchhoff Navier 级数
   （`navier_rect_moments`，与 FEM 路径独立），并对级数截断 `terms = 61/121` 与斑块积分 `nq` 做误差条；
   另设 `fem_matched_point`：把 FEM 场在 Navier 峰值点上取值对比，用来把「链的正确性」与
   「峰值比的病态」两件事分开。
5. **模块改造**：`scan_contact_configurations` 新增 `w_over_t_limit` / `target_w_over_t`，逐构型输出
   `w_over_t_nonuniform`、`in_small_deflection_envelope`、`pressure_scale_factor`、
   `nominal_peak_pressure_MPa`，汇总输出 `all_in_small_deflection_envelope`。默认 `target_w_over_t=0.2`
   ⇒ 验收扫描默认就在有效域内；`None` 可复现历史档（且被显式标记 `in_small_deflection_envelope=False`）。

## 3. 计算结果（程序化输出，非手抄）

### 3.1 包络审计（nx = 12，k = 1e7，t = 10 mm）

| 构型 | 名义 peak [MPa] | 名义档 w/t | 名义档在包络内 | 控制档 peak [MPa] | 控制档 w/t | 控制档在包络内 |
|---|---|---|---|---|---|---|
| mild_nonuniform | 5.0 | 16.6 | ❌ | 0.071 | 0.200 | ✅ |
| moderate_nonuniform | 8.0 | 14.6 | ❌ | 0.134 | 0.200 | ✅ |
| sharp_nonuniform | 15.0 | 4.7 | ❌ | 0.469 | 0.200 | ✅ |

`w/t` 由程序直接读取（`configurations[*].w_over_t_nonuniform`），压力缩放因子
（0.0142 / 0.0168 / 0.0313）是精确线性缩放（线性模型中 `w ∝ p`，一次重解即命中目标，
并由回读的 `w/t` 校验，不是搜索）。

### 3.2 网格审计：包络内的 `Δσ_x`（非均匀 / 均匀 − 1）

| 构型 | nx = 8 | nx = 12 | nx = 16 | 跨网格摆幅（8→16）|
|---|---|---|---|---|
| mild_nonuniform | **−6.31 %** | **−40.73 %** | **−36.69 %** | 34.4 pp |
| moderate_nonuniform | **−43.62 %** | **+2.47 %** | **−20.13 %** | 46.1 pp |
| sharp_nonuniform | **−40.69 %** | **+22.93 %** | **−41.32 %** | 64.2 pp |

全部行 `w/t = 0.194 … 0.200`（在包络内），因此摆幅**不是**包络伪影，而是「逐单元形心取最大」的
峰值跳动 + 均匀斑**载荷间断**使形心应力收敛慢（见 §5 机理）。符号在 moderate/sharp 档随网格翻转
⇒ 该量在单网格上不具备可报告性。

与落地证据对照：落地归档 `benchmarks/artifacts/GAP-PP-1-patch-from-contact-verify/verification.json`
的同一量（nx=12、**名义**压力档）为 `min|Δσ_x| = 11.4 %`、`max|Δσ_x| = 47.7 %`；
demo 头号结论 `+170.6 %`（σ_x）与 `valid_case` 的 `+115.0 %` 都取自单网格峰值比。
按本审计，这些数的**数量级不可引用**（方向/符号本身随网格与口径变化），
落地报告 `docs/VERIFY-GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.md` 的 §8 勘误应再加一条此限制。

### 3.3 解析路径与口径

两个工具已交付并在仓内可复现执行；**解析 `Δ` 与六口径表本次未取回**：
`run_compute` 作业 `cj4066`（包络+网格+口径）与 `cj4071`（Navier 解析路径）在本单结束时仍在排队
（递交时刻排队位 24/29，集群在飞作业 45+）。本机受「<8 核 <10 min」纪律约束，Navier 峰值采样
（每构型 ~4000 次级数求和）在本机 120 s 硬限内跑不完。**故 §3.3 的数值按 honest 口径留空，
不得以任何未取回的数字填充。**

## 4. 结果图

本单无新生成图（figures 生成在 `tools/verify_gap_pp1_delta_caliber.py::figures` 中，随 `cj4066` 产出：
`fig_delta_vs_mesh.png` / `fig_delta_vs_caliber.png` / `fig_deflection_fields.png`）。
理由：全部关键结论由 §3.1–§3.2 的数字表承载，且图件依赖仍在排队的作业；不以内联重画代替。

## 5. 注意点（坑与机理）

1. **`w/t` 从未被报告**：落地扫描只报 `max_deflection_*`，没有除厚度，也没和 0.5 比。默认档
   `w/t = 4.7…16.6`，线性理论早已失效（落地 demo 自己在 `small_deflection_validity` 里承认了 14.0，
   但扫描接口没跟上）。
2. **峰值比的病态来自两处**：`peak_stresses` 是「逐单元形心 |σ_x| 取最大」；均匀斑压力在斑边**间断**，
   Q4/SRI 形心应力在间断附近收敛慢且非单调，最大值位置在网格间跳格；高斯斑光滑、收敛快。
   两个最大值相除 ⇒ 两套离散误差 + 两次跳格叠加。
3. **载荷无关性不能救这个数**：在线性模型里 `Δ` 对载荷严格不变（本单用 `target_w_over_t=0.1 vs 0.4`
   实测相对差 < 1e-9，见 `tests/test_gap_pp1_envelope_caliber.py::test_delta_ratio_is_load_invariant_in_the_linear_model`），
   所以把载荷缩回包络内**不会**改变落地那些数的值 —— 改变它们的是**网格**，而这一点此前从未被检查。
4. **均匀参考足印是自由参数**：`patch_width/patch_length` 换成 6σ / 等面积 / 规则斑会改变 `Δ` 的符号量级，
   因此「同口径对照」必须把口径写进机读 JSON（`summary.delta_definition` 已写入模块输出）。

## 6. 误差判定

| 关键量 | 基准 | 本次结果 | 误差 | 是否 < 3 % |
|---|---|---|---|---|
| 包络控制档 `w/t` | 目标 0.2 | 0.200000（三档全部） | < 1e-9（相对） | ✅ |
| 载荷无关性（线性模型） | 同一构型两载荷档 | `Δ(0.1) == Δ(0.4)` | < 1e-9（相对） | ✅ |
| 投影总力平衡 | 逐构型 `load_balance_rel` | < 1e-10 | — | ✅ |
| **验收量 `Δσ_x`（FEM 峰值比）** | 网格加密 | nx=8→16 摆幅 34–64 pp，符号翻转 | **远大于 3 %** | ❌ **未通过** |
| 解析路径 `Δσ_x`（Navier） | 级数截断 61 vs 121 | **未取回**（作业排队） | 未知 | **未知 / not_run** |

**结论**：锚点要求的「非均匀 vs 均匀峰值应力差」在 FEM 峰值比口径下**不满足 3 % 判据，且不可复现**；
本次不声称该验收项已闭合。已把口径问题定位到统计量本身，并交付了解析替代路径与包络控制。
`Δ` 的最终可引用数值待 `cj4066` / `cj4071` 取回后补入（建议后续单跟进，slug 见下）。

## 7. 复现

```bash
# 模块级回归（本单已跑：4 passed, 95 s）
python -m pytest tests/test_gap_pp1_envelope_caliber.py -q

# 包络 / 网格 / 口径审计（重计算，递交调度）
python tools/verify_gap_pp1_delta_caliber.py --meshes 12,16,24,32

# 解析（Navier）路径的 Δ 与 FEM 同点对比
python tools/verify_gap_pp1_delta_navier.py --meshes 16,24,32
```

## 8. 建议后续条目（供总账）

- slug：`GAP-PP-1-DELTA-ANALYTIC`，域=冰区/结构（TensorFEM）
  验收：解析路径 `Δσ_x`（Navier，terms 61 vs 121 相对差 < 0.5 %）+ 六种均匀参考口径的全表
  + FEM 同点一致性 < 3 %，三者写入 `benchmarks/artifacts/GAP-PP-1-delta-caliber/delta_navier.json`；
  并把落地 `docs/VERIFY-GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.md` 的 `+170.6 %` / `+115.0 %` 标注为
  「单网格峰值比，数量级不可引用」。