# 平板剪切屈曲：参考式与双离散交叉审计

**结论先行**：合并通道的 `classical_shear_coefficient` 在 `a/b < 1` 时使用了 `a/b >= 1` 的那一支，
与"同一块板旋转 90°"的精确对偶 `k(a)·a² = k(1/a)` 冲突，`a/b = 0.5` 处偏低 **18.5%**。
`benchmarks/plate_shear_buckling_evidence.json` 的 `aspect_sweep` 里 `a/b = 0.5` 那行报出的
**22.7% 误差是参考式错误**，不是求解器误差——同一位移解上，两族独立离散只差 0.670%。
其次 Timoshenko–Gere 拟合式本身在 `a/b ≈ 2` 处比收敛值低 **3.16%**，因此它不能当 3% 门限的参考式。

## ① 案例介绍（物理背景与参数）

- 问题：四边简支矩形板在面内均布剪力 `N_xy` 下的弹性屈曲，无量纲系数
  `k_s = N_xy·b²/(π²D)`，`b` = 板宽（本次固定 `b = 1 m`，只变板长 `a`）。
- 参数：`E = 210 GPa`、`ν = 0.3`、`t = 12 mm`，`D = Et³/[12(1-ν²)]`。
- 文献锚点：`k_s = 9.34 @ a/b = 1`；`k_s → 5.34`（`a/b → ∞`）（Timoshenko & Gere 1961, Table 9.1）。
- 审计对象是**参考式与 sweep 判据**，不是新增求解器能力。

## ② 怎么做的（方法/流程）

1. **对偶恒等式（无需文献值的硬判据）**：系数按板宽 `b` 归一化；把同一块板旋转 90°，
   剪力状态不变而 `b → a`，故精确解必须满足 `k(a)·a² = k(1/a)`。
   该恒等式在 `tests/test_shear_buckling_reference_audit.py` 中 fail-closed 锁死。
2. **两族独立离散**：
   - 族 A：生产 Ritz 正弦积基（`ritz_inplane_buckling_load`，本审计 `basis = 14`；
     与 `basis = 18` 在 `a/b = 1` 处差 < 2e-4）；
   - 族 B：新增 **BFS 双三次 Hermite C1 协调元**（`src/tensorfem/plate_buckling_hermite.py`），
     单元 16 自由度 `w, w_,x, w_,y, w_,xy`，弯曲阵用二阶导、剪力几何阵取 `2∫w_,x w_,y dA`，
     简支边界为 `w = 0` 且切向斜率为零（法向斜率自由）；
     广义特征问题经 `K_b` 正交特征基约化，避开薄网格上广义 LAPACK 驱动中止。
   - 两族**不共享代码与逼近空间**，族间差即离散误差量级。
3. 命令：`python scripts/audit_shear_buckling_aspect_sweep.py`（支持 `--aspects` 分块续跑，
   每算完一个 `a/b` 立刻落盘）→ `outputs/plate_shear_buckling_aspect_audit.json`，
   图 → `docs/assets/plate-shear-buckling-aspect-audit.png`。

## ③ 计算结果（程序化注入，零手抄）

| a/b | Ritz k_s | C1 Hermite k_s | 两族互差 | 参考值 | 两支式经典 | 审计前单支式 | 经典式偏差 | 单支式偏差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 26.1848 | 26.1881 | 0.013% | 26.1865 | 25.3600 | 21.3400 | -3.16% | -18.51% |
| 0.75 | 13.2863 | 13.2715 | 0.111% | 13.2789 | 13.4933 | 12.4511 | +1.62% | -6.23% |
| 1.00 | 9.3248 | 9.3035 | 0.229% | 9.3141 | 9.3400 | 9.3400 | +0.28% | +0.28% |
| 1.25 | 7.7668 | 7.7149 | 0.670% | 7.7409 | 7.9000 | 7.9000 | +2.06% | +2.06% |
| 2.00 | 6.5462 | 6.5470 | 0.013% | 6.5466 | 6.3400 | 6.3400 | -3.16% | -3.16% |

- 交叉校核覆盖 `a/b ∈ {0.5, 0.75, 1, 1.25, 2}`，两族最大差 **0.670%**。
- 未做交叉校核的 `a/b`（需要 > 700 节点才收敛，本单算力内不报未收敛 FE 数）。
- 审计前单支式对偶破坏（`a/b < 1` 侧）达 **15.8%**（`k(0.5)·0.25/k(2)`）；
  修正后的两支式对偶残差 **< 1e-12**。
- 经典拟合式最差偏差出现在 `a/b = 2`：**-3.16%**。

## ④ 结果图

`docs/assets/plate-shear-buckling-aspect-audit.png`：左图 Ritz / C1 Hermite / 两支式三条曲线
在 `a/b ≥ 0.5` 全段重合到线宽内，点线（审计前单支式）在 `a/b < 1` 明显掉离；右图为偏差百分比，
红虚线为 3% 门限：两支式在 `a/b ≈ 2` 越界（-3.16%），单支式在 `a/b < 1` 越界约 18%，
两族互差（三角）始终远小于门限。

## ⑤ 注意点（坑与适用范围）

- **归一化口径决定分支**：`k_s` 按板宽 `b` 归一化时，`a/b < 1` 必须是 `4 + 5.34/(a/b)²`。
- **拟合式 ≠ 精确解**：`5.34 + 4/α²` 在 `a/b ≈ 2` 低 3.16%、在 `a/b ≈ 1.25` 高 2.06%。
  把 3% 门限套在它上面测到的是"拟合精度"，不是"离散精度"。
- **Hermite 网格下限**：跨度方向 < 6 个单元时简支约束集下弯曲算子可能出零能边界模态，
  已用 `ValueError` 显式拒绝（绝不出数）；因此交叉校核只覆盖 `a/b ≤ 2.6`。
- **未改动**：`benchmarks/plate_shear_buckling_evidence.json` 不在本单 EDIT_WHITELIST
  （仅放行 `docs/ src/ tests/ examples/ outputs/`），其 `aspect_sweep` 的 `a/b < 1` 行
  与 `relative_error_vs_classical` 仍为被推翻的旧口径，需一次重新生成把参考换成两支式。

## ⑥ 误差判定：关键量 vs 基准，是否 <3%

| 关键量 | 基准 | 偏差 | < 3%？ |
|---|---|---|---|
| 生产 Ritz vs C1 Hermite（全部交叉校核点） | 彼此 | ≤ 0.670% | ✅ |
| Ritz `k_s(1.0) = 9.3248` | 文献 9.34 | -0.16% | ✅ |
| 两支式 `k_s(1.0) = 9.34` | 两族均值 9.314 | +0.28% | ✅ |
| 两支式 `k_s(0.75) = 13.49` | 两族均值 13.28 | +1.62% | ✅ |
| 两支式 `k_s(1.25) = 7.90` | 两族均值 7.74 | +2.06% | ✅ |
| 两支式 `k_s(0.5) = 25.36` | 两族均值 26.19 | -3.16% | ❌ 拟合式极限 |
| 两支式 `k_s(2.0) = 6.34` | 两族均值 6.546 | -3.16% | ❌ 拟合式极限 |
| 审计前单支式 `k_s(0.5) = 21.34` | 两族均值 26.19 | **-18.5%** | ❌ 已修正 |
| 限定项 `a/b = 6`（旧口径比渐近 5.34） | 有限展弦闭式 5.4511 | 0.52%（原报 2.62%） | ✅ |

**回答"是否 <3%"**：求解器层面 **<3% 成立**——生产 Ritz 与独立 C1 Hermite 互差 ≤ 0.670%，
Ritz 对文献锚点 9.34 偏差 -0.16%。参考式层面 **<3% 不成立**：经典拟合式在
`a/b ∈ [0.5, 2]` 与收敛值差 3.16%，故 3% 门限只能用于文献锚点或收敛值，不能全域套在拟合式上；
这正是本单纠正的判据用法。

## 复现

```bash
python scripts/audit_shear_buckling_aspect_sweep.py --aspects 0.5,1.0,2.0
python -m pytest tests/test_shear_buckling_reference_audit.py -q
```

## 第 2 轮：PR #104 CI 红修复记录

本轮不新增物理结论，只把 PR #104 的 CI 信号修绿，并留下归因证据。

### 红因 1：新增文件自带 4 条 ruff F401（本单修复）

| 文件 | 未使用导入 | 处理 |
|---|---|---|
| `tests/test_shear_buckling_reference_audit.py` | `math`、`numpy` | 删除（本测试全部用 `pytest.approx` 与纯标量，无 numpy 依赖） |
| `scripts/audit_shear_buckling_aspect_sweep.py` | `math`、`plate_buckling_hermite.hermite_shear_coefficient` | 删除（脚本只用 `hermite_shear_coefficient_ladder`） |

修复后 `ruff check` 对本 PR 的四个文件 **All checks passed**（`ruff 0.15.0`，配置 `line-length=100`）。
`lint (advisory)` 作业整体仍红，因为主干本身带着 **4219** 条历史告警（作业被显式设为 advisory，
本单未扩大范围去清主干欠账；只保证"本 PR 不新增告警"）。

### 红因 2：`test (3.11)/(3.12)` 的唯一失败与本 PR 无关（归因证据）

全量 1406 条用例在本容器（loong64/128 核、load ≈ 460 的高竞争环境）上分块跑，唯一失败为：

```
tests/test_finite_strain_curved_friction_contact3d.py::
  test_real_curved_nonmatching_tet4_friction_level_one_closure
```

该用例只触碰 `finite_strain_curved_friction_contact3d` 通路，与本 PR 修改的
`plate_shear_buckling.py`／新增 `plate_buckling_hermite.py` 没有任何 import 或数值耦合；
在 `main` 基线工作树上单独复跑同一用例同样无法通过（见 `artifacts/ci-cure/`）。
结论：这是基线既有的、对线程数/负载敏感的 3D 摩擦接触闭包用例，不是本 PR 引入的回归；
本轮不在本 PR 里修改它（越界且无法在本单算力内定位），如实上报。

### 本 PR 自身用例

```
python -m pytest tests/test_shear_buckling_reference_audit.py -q     # 8 passed
python -m pytest tests/test_plate_shear_buckling.py -q               # 见 artifacts/ci-cure/
```

### 修复后实测（本容器，python 3.11.6，loong64）

| 项 | 命令 | 结果 |
|---|---|---|
| 新增审计用例 | `pytest tests/test_shear_buckling_reference_audit.py -q` | **8 passed**（92.87 s） |
| 既有剪切屈曲用例 | `pytest tests/test_plate_shear_buckling.py -q` | 通过 |
| 两文件合计（去导入后复跑） | `pytest tests/test_shear_buckling_reference_audit.py tests/test_plate_shear_buckling.py -q` | **23 passed in 269.08 s** |
| 红测归因（main 基线） | `pytest ...::test_real_curved_nonmatching_tet4_friction_level_one_closure`（`/root/tp-agent/work/m4138`，main 线 `9b1e6fa`） | **1 failed in 387.54 s** → 既有失败，复现于未含本 PR 的基线 |
| 红测（本分支） | 同一用例 | 同样 FAILED（`artifacts/ci-cure/chunkA-branch.log`） |
| Lint | `ruff check`（PR 四个文件） | All checks passed |
