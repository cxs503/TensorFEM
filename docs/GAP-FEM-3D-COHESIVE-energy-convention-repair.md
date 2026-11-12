# GAP-FEM-3D-COHESIVE — 载荷约定/能量记账修复（验证欠账闭合）

锚点：`GAP-FEM-3D-COHESIVE`（in_progress）。本单闭合该锚点「注意点/局限」里明确列为
**仍开放**的两条 honest-FAIL：

| 原 honest-FAIL | 修前 | 修后 | 判据 |
|---|---:|---:|---|
| 能量平衡 `|W_ext − U_solid − W_int| / W_ext` | **49.9 %**（试样级）/ 46.7 %（小模型） | **2.7e-5**（试样级） | < 3 % |
| DCB 峰值载荷 vs LEFM | **+21.18 %** | **−14.31 %**（符号翻转） | < 3 %（仍不达标，但归因已正确） |

## ① 根因（两个消费者 + 一个约定错误）

`tensorfem.cohesive3d_dcb` 报出的 `load` 是**两支臂反力之和**（总载荷 `P = 2·P_arm`，
`_reaction()` 把上下加载面反力幅值相加），其共轭位移是 **`delta/2`**（每臂各带 `P/2`、
各走 `delta/2`）。模块 docstring 却把 per-arm 的因子用在了总载荷上：

1. **外功累加器**：`work += 0.5·(P_prev+P_cur)·Δδ` —— per-arm 因子 `1/2` 配总载荷 ⇒ 外功偏大 2×；
   另外首增量被 `if not curve: work += 0.0` 整段丢弃（线性弹性下少记 1/n 步）。
   实测 `W_ext / U_solid = 1.875 = 2 × (15/16)`（2× 来自约定，15/16 来自丢首步）。
2. **`beam_serr`**：`G_I = 6P²a²/(E b²h³)`。用总载荷 `P` 时正确的 Irwin-Kies 是
   `G = P²/(4b)·dC/da`（因为 `U = P²C/4`，而教科书 `P²/(2b)` 属于 per-arm 载荷），
   故少除了一个 2 ⇒ **`G_I` 偏大 2×**，`beam_peak_load` 因此**偏低 √2**：
   旧 `P = b h^1.5√(G_c E/6)/a` → 新 `P = b h^1.5√(G_c E/3)/a`。

## ② 独立判据（不含缺陷的 oracle）

- **约定无关的 SERR**：固定开口下 `G = −(1/b)·∂U/∂a|_{δ}`，`U = ½uᵀK_solid u` 来自线性解，
  与载荷约定无关。细长臂（`a/h≈38, b/h=3`）三个裂纹长度 `a = 40/38/36 mm` 中心差分得
  `G_FE = 7.179e-10 J/m²`：

| 公式 | `G` [J/m²] | `G_FE / G_formula` |
|---|---:|---:|
| FE 固定开口能量导数（oracle） | 7.179e-10 | 1.000 |
| 修后 `3P²a²/(E b²h³)` | 8.780e-10 | **0.818** |
| 修前 `6P²a²/(E b²h³)` | 1.692e-09 | **0.424**（被否证） |

残差 18 % 归因已定位且如实记录：有限元试件的键合韧带在裂尖后仍传力（等效裂纹更短）
+ Williams 根转角只是近似，故 FE 比理想悬臂梁模型**更刚**。

- **共轭恒等式**：线弹性解满足 `U = (P/2)·δ/2 = Pδ/4`（实测 rel < 5e-3）。
  修前该式给出 `U = Pδ/2`，与 ½uᵀKu 差 2×。

## ③ 改动（最小集，`src/tensorfem/cohesive3d_dcb.py`）

1. `solve_dcb()` 外功：`work += 0.25·(P_prev+P_cur)·Δδ`（含首增量，`P_prev=0`）。
2. `beam_serr()`：系数 `6 → 3`，docstring 写明总载荷约定的 `1/4` 因子。
3. `beam_peak_load()`：`/6 → /3`。
4. `_reaction()` 与模块 docstring：显式声明「`P` 的共轭位移是 `δ/2`」。

**未改动**：`load` 数值约定（仍是两臂反力之和）——因此载荷曲线、ASTM D5528 柔度标定
（`C = δ/P = 4(a+Δ)³/(Ebh³) + a/(kGbh)`）与既有 `peak_load_N` 全部**不变**，
D5528 柔度包夹带判据仍通过（`compliance_calibration.all_passed = True`）。

## ④ 试样级复跑（`benchmarks/cohesive3d_dcb_displacement_control_probe.py`）

```
nx60_peak_load_N = 34.779 N      lefm_peak_load_N = 40.587 N   rel = −14.31 %
energy_balance_rel = 2.73e-05    (修前 0.499)
declared criteria (A,B,C) passed = True ; peak_load_vs_lefm.passed = False（诚实 FAIL 保留）
runtime 42.8 s
```

峰值仍不达 3 % 线，但**符号与归因已纠正**：修前「FE 比 LEFM 高 21 %」把参考值的
√2 缺陷误记成「过程区欠分辨/强度控制」的 FE 偏差；修后 FE 比 LEFM **低 14.3 %**，
与粗网格 CZM（过程区 22.5 mm ≈ 0.45a，强度控制）应给出的「低于 LEFM」一致。
honest FAIL 未被粉饰：`peak_load_vs_lefm.passed` 仍为 False。

## ⑤ 测试

```
PYTHONPATH=src python -m pytest tests/test_cohesive3d_dcb_load_convention.py -q     # 5 passed (103.6 s)
PYTHONPATH=src python -m pytest tests/test_cohesive3d_dcb.py \
    tests/test_cohesive3d_dcb_solver_guards.py -q                                    # 12 passed (8.2 s)
```

新增 5 条判据（`tests/test_cohesive3d_dcb_load_convention.py`）：
1. `test_elastic_external_work_closes_on_strain_energy` — 弹性段 `W_ext = U_solid + W_int`
   （残差 < 0.5 %，修前 0.467）；
2. `test_reported_load_is_total_of_both_arms_and_conjugate_to_half_opening` — `U = Pδ/4`；
3. `test_beam_serr_matches_convention_free_fe_energy_release_rate` — 固定开口 `G` oracle
   0.75 < `G_FE/G_修后` < 1.05 且**否证**修前公式（0.424）；
4. `test_beam_peak_load_inverts_beam_serr_and_is_sqrt2_of_old_formula` — 自洽求逆 + √2；
5. `test_compliance_full_energy_identity` — `U = P²C/4` 且 FE 柔度 ≤ 梁论（更刚，如实记录）。

## ⑥ 局限 / 适用范围

- `G` oracle 只覆盖细长臂、单一 `b/h = 3`；宽试件（`b/h ≥ 6`，平面应变）与厚臂
  (`a/h < 10`) 的根转角/韧带柔度模型未复核，仍是欠账。
- 载荷约定保留「两臂反力之和」；凡对外引用 ASTM D5528 单臂载荷的场合须除以 2
  （docstring 已写明）。
- 本单全部数字由控制节点单进程真跑（39–104 s），未用集群；架构 x86_64。

复现：`PYTHONPATH=src python scratch/make_evidence.py`（本单脚本）
+ `PYTHONPATH=src python benchmarks/cohesive3d_dcb_displacement_control_probe.py`。
证据：`docs/GAP-FEM-3D-COHESIVE-energy-convention-evidence.json`，
图：`docs/assets/gap-fem-3d-cohesive-energy-convention.png`。