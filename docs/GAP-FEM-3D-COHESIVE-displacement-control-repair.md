# GAP-FEM-3D-COHESIVE · 返修轮：位移控制求解器缺陷修复 + honest 部分验收（m15256）

本页是 feature-ledger 锚点 `GAP-FEM-3D-COHESIVE`（"3D cohesive zone 单元 + 验证算例"）返修轮的
证据页。**结论先行：单元级（COH3D8 + BK）能力为 REAL 且已用可复现运行工件背书；试件级（DCB
位移控制 + 梁理论对标）**未达标**，本页如实记录失败量级与开放缺陷，不作粉饰。**

## 1. 案例介绍（物理背景 / 几何 / 参数）

DCB（ASTM D5528 双悬臂梁，Mode I 层间断裂）用于检验 3D 内聚单元在真实三维应力状态下
能否给出可信的载荷-张开响应。几何：L=60 mm，b=3 mm，每臂厚 h=3 mm，初始裂纹 a0=30 mm，
结构化 HEX8 adherends + 零厚度 COH3D8 韧带。材料/内聚参数：E=70 GPa，ν=0.30，
k_n=1e12 Pa/m，k_s=4e11 Pa/m，t_n0=5 MPa，t_s0=3 MPa，G_Ic=300 J/m²，G_IIc=800 J/m²，BK η=2.2。
解析基准（独立闭式）：`C = 4a³/(Ebh³) + a/(5/6·G·bh)`，含 Williams(1989) 根部转角修正 Δ；
LEFM 峰值 `P = b h^1.5 sqrt(G_Ic E / 6) / (a+Δ)`（ASTM D5528 / Kanninen 梁理论谱系）。

## 2. 怎么做的（方法 / 流程）

1. **前轮死法定位**：m14987/m15001 的 `benchmarks/cohesive3d_dcb_3d.py` 从未产出运行工件。
   复现后定位到两个**求解器缺陷**（均已在 `src/tensorfem/cohesive3d_dcb.py` 修复）：
   - **缺陷 A（物理错误）**：位移控制面上被加载的 dof（x=L, z）没有进 `fixed_dofs`，落进自由集。
     Newton 于是把这些 dof 的残差也压到 0 → 反力恒为 0 → 试件根本不张开、损伤恒 0、
     所有峰值判据相对误差退化为 -1。
   - **缺陷 B（不终止）**：`while delta < delta_max*(1+1e-12)` 在 `delta == delta_max` 后恒真，
     载荷步走完后以零长度步无限循环（历史上"DCB 跑不完"的直接原因，控制节点上残留 24 min 僵尸进程实证）。
   修复：加载 dof 并入约束集；终止条件改 `delta_max*(1-1e-9)` + `target<=delta` 断路 + 步数上限。
   行为由 `tests/test_cohesive3d_dcb_solver_guards.py`（3 个新测试）钉住，并修正
   `tests/test_cohesive3d_dcb.py` 中记录旧（错误）契约的边界条件断言。
2. **单元级验证（真跑）**：`python benchmarks/cohesive3d_verification.py` → all_passed True
   （element_mesh_objectivity / coupled_block_mode_i / coupled_block_mode_ii / coupled_block_mesh /
   coupled_block_energy 全 PASS），工件 `benchmarks/cohesive3d_verification.json`。
3. **试件级探测（真跑）**：新增 `benchmarks/cohesive3d_dcb_displacement_control_probe.py`
   （nx=24, ny=1, nz=2，机器可判、0.3 s），用**修正后的位移控制**量测柔性并与梁理论对标。
4. 全量内聚 pytest：`61 passed, 2 xfailed`（2 个 xfail 即下述开放缺陷，strict=True 不掩盖）。

## 3. 计算结果（数字表，程序化提取自 JSON 工件）

试件级（`benchmarks/cohesive3d_dcb_displacement_control_probe.json`）：

| 量 | 基准(ref) | 本实现(got) | 相对误差 | 判据 | 判定 |
|---|---|---|---|---|---|
| DCB 柔性 C [m/N]（vs Williams 根部转角梁） | 2.35674e-05 | 1.32143e-04 | +460.7% | ≤10% | **FAIL** |
| DCB 柔性 C [m/N]（vs 无转角修正梁） | 1.91962e-05 | 1.32143e-04 | +588.4% | — | **FAIL** |
| 峰值反力非零（缺陷 A 回归） | >0 | 0.15135 N | — | >0 | PASS |
| 终止于 delta_max（缺陷 B 回归） | delta_max | 2.0e-05 (=目标) | 0 | 精确 | PASS |
| 收敛 | — | converged_all=True, 40 步, 16 Newton 迭代 | — | — | PASS |

单元级（`benchmarks/cohesive3d_verification.json`）：objectivity / patch / mode-I / mode-II /
mesh-objectivity / energy 六项 all_passed=True（本页不重复其数字，工件为准）。

宽度分辨基准（`benchmarks/cohesive3d_dcb_3d.json`，quick 模式，runtime 135.7 s）：
`all_passed=False`；其中 `compliance_bracket` 的 `got=0.0` 是该脚本内 `linear_compliance()`
仍在用"加载 dof 自由 + 单位力"的旧写法（修正后 prescribe 面上受力→位移恒 0），属**待迁移的陈旧助手**，
真实可判数字以 probe 的 +460.7% 为准；`peak_load_vs_lefm` got=0.0572 N vs ref=28.699 N（-99.8%）、
`energy_balance_rel`=0.936、`width_objectivity` got=0.719 vs ref=30.085 —— 均为同一根因（试件过柔）。

## 4. 结果图

`docs/assets/cohesive3d-dcb-3d/load_opening.png`（载荷-张开曲线）、`damage_field.png`（损伤场）、
`rcurve.png`（R 曲线）：由 `benchmarks/cohesive3d_dcb_3d.py` 真跑生成，图中峰值/损伤的异常形态
正是上述"试件过柔"的直观体现。平台报告内嵌同一 PNG。

## 5. 注意点 / 局限

- **开放缺陷（未解决）**：修正位移控制后正常加载，FE 试件柔性是梁理论的 ~5.6–6.9 倍
  （nx=24 时 5.61×；nx=50 时约 11×），量级接近 2³，**首要怀疑有效自由跨度为 2a**
  （韧带/夹持区定位或 adherends 与内聚面带耦合位置待核），已写入 2 个 strict-xfail 测试与
  `negative` 记录；未在本轮预算内定案。
- 该缺陷使**试件级** DCB 不能作为验收算例；因此 `GAP-FEM-3D-COHESIVE` **未闭合**，
  仅单元级能力可用（BK 律 + COH3D8 + 目标性 + patch + 双材料块）。
- 小应变运动学、无摩擦/接触/循环退化（既有限制，未变）。
- 集群 cj1024（全量 DCB）因队列深度 238 未取回，本轮结论一律基于控制节点真跑工件。

## 6. 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 单元级 objectivity/patch/mode-I/mode-II/energy | 闭式 | ≤1e-10（工件 all_passed） | **是** |
| DCB 柔性 vs 根部转角梁 | 2.35674e-05 m/N | +460.7% | **否** |
| DCB 峰值载荷 vs LEFM | 28.699 N | -99.8% | **否** |
| 能量平衡 \|W-U-D\|/W | 0 | 0.936 | **否** |

## 复现命令（repro_cmd）

```bash
python benchmarks/cohesive3d_verification.py            # 单元级：all_passed True
python benchmarks/cohesive3d_dcb_displacement_control_probe.py   # 试件级：compliance passed=False（honest）
python -m pytest tests/test_cohesive3d.py tests/test_cohesive3d_dcb.py \
    tests/test_cohesive3d_dcb_solver_guards.py tests/test_cohesive3d_dcb_verification.py -q
# -> 61 passed, 2 xfailed
```
