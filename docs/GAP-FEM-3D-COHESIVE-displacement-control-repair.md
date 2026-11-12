# GAP-FEM-3D-COHESIVE · 返修第 2 轮：位移控制加载面 prescription 缺陷（根因 + 修复）（m16164）

本页是 feature-ledger 锚点 `GAP-FEM-3D-COHESIVE`（"3D cohesive zone 单元 + 验证算例"）的**返修第 2 轮**
证据页，接续第 1 轮页（m15256，见 git 历史）。**结论先行：第 1 轮记录为"开放缺陷"的
「FE 试件比梁理论柔 5.6–11 倍 / 怀疑有效自由跨度 = 2a」已被定位为**加载面位移 prescription 的缺陷**并修复；
根因不是几何、不是自由跨度、也不是内聚律。修复后试件级**柔性标定判据 PASS**
（a = 20/30/40 mm 全部落在 [plain beam, 1.10×Williams 根部转角梁] 包夹内）。
**峰值载荷 vs LEFM 仍为诚实 FAIL**（+21.2%），但本轮给出可判的机理证据（网格收敛、罚刚度无关、
强度相关），不再是与几何混淆的遗留量。**

## 1. 案例介绍（物理背景 / 几何 / 参数）

DCB（ASTM D5528 双悬臂梁，Mode I 层间断裂）用于检验 3D 内聚单元在真实三维应力状态下能否给出可信的
载荷-张开响应与柔性。几何：L = 60 mm，b = 3 mm，每臂厚 h = 3 mm，初始裂纹 a0 = 30 mm（本轮标定另取
20/40 mm），结构化 HEX8 adherends + 零厚度 COH3D8 韧带（韧带占 0 ≤ x ≤ L−a0）。
材料：E = 70 GPa，ν = 0.30；内聚：k_n = 1e12 Pa/m，k_s = 4e11 Pa/m，t_n0 = 5 MPa，t_s0 = 3 MPa，
G_Ic = 300 J/m²，G_IIc = 800 J/m²，BK η = 2.2。闭式基准（与 FE 无关）：`C = 4a³/(Ebh³) + a/(5/6·G·bh)`，
a → a+Δ（Williams 1989 根部转角）；LEFM 峰值 `P = b h^1.5 sqrt(G_Ic E/6)/(a+Δ)`。

## 2. 怎么做的（方法 / 流程）

1. **把"物理"与"数值"分开**：先用**未损伤切线刚度**做线性求解（不与内聚软化耦合），量测
   `C = δ/P`。结果：nx=24/nz=2 的线性柔性是梁理论的 0.934 倍，nx=48 是 1.118 倍 —— 几何/自由跨度**没有** 2 倍误差。
   第 1 轮的"2a 自由跨度"假设**被证伪**。
2. **量级对齐**：路径跟随器在同一网格上给出 `C = 1.32143e-04`，恰为线性值的 **6.000 倍**；该试件加载面
   每条臂有 6 个节点（nz=2, ny=1）→ 6 = 每臂加载节点数。查证：`solve_dcb` 的 prescription 把 ±0.5 除以了
   加载 dof 总数（`0.5/n_loaded_dofs`），即真实施加开口 = δ/n_face。**nz=4 时 n_face=20**，故第 1 轮
   记录的"a=50mm 约 11 倍"同样是该因子（0.55×20 ≈ 11）。
3. **修复**：新增 `opening_prescription(model, delta=None)`（`src/tensorfem/cohesive3d_dcb.py`）——
   刚体平移式加载面：上臂每节点 +δ/2、下臂每节点 −δ/2，总开口恒等于 δ，**与加载面节点数无关**；
   `solve_dcb` 改为调用它。同族缺陷一并修：`benchmarks/cohesive3d_dcb_3d.py::linear_compliance`
   的陈旧助手（单位力 RHS 落在已约束的 dof 上→柔性恒 0，且同样除以 n_face）。
4. **判据前移（先声明后测量）**：试件级验收改用 ASTM D5528 的**柔性标定法**：三个裂纹长度的
   `C(a)` 必须被 [plain beam, 1.10×root-rotation beam] 包夹；另加两条网格/罚刚度稳健性判据。
   工具链：`benchmarks/cohesive3d_dcb_displacement_control_probe.py`（39.6–40.1 s，机器可判，退出码非 0 即失败），
   数字全部写入 `benchmarks/cohesive3d_dcb_displacement_control_probe.json`；并把这 3 行
   标定并入任务交付物 `benchmarks/cohesive3d_verification_tables.py`（现 27 PASS / 0 FAIL / 2 KNOWN_LIMIT）。

## 3. 计算结果（数字表，程序化抽取自 JSON 工件，零手抄）

### 3.1 试件级柔性标定（`compliance_calibration.rows`）

| a [mm] | C_FE [m/N] | C_beam plain | C_beam +root rotation | vs root rotation | 包夹内 |
|---|---|---|---|---|---|
| 20 | 7.931011e-06 | 5.643739e-06 | 7.759486e-06 | +2.2% | yes |
| 30 | 2.202375e-05 | 1.904762e-05 | 2.356742e-05 | −6.6% | yes |
| 40 | 4.715234e-05 | 4.514991e-05 | 5.297703e-05 | −11.0% | yes |

拟合指数 `C ~ a^n`：FE 2.568 vs 梁基准 2.769（小 a 处 Δ 修正使基准本身偏离 3）。

### 3.2 网格 / 罚刚度稳健性（`mesh_independence`、`peak`）

| 量 | 值 |
|---|---|
| C(nz=2, 12 加载 dof) | 2.202375e-05 m/N |
| C(nz=4, 20 加载 dof) | 2.245717e-05 m/N（相对差 **1.97%**；修复前该比值是 20/6 = 3.33） |
| C(nx=48) vs 梁基准 | +9.22%（仍在 ±15% 内） |
| 峰值：nx=60 vs nx=24 | 34.779 N vs 37.924 N（差 9.0%） |
| 峰值：k_n=1e12 vs 1e14 | 34.779 N vs 34.851 N（差 0.2%） |

### 3.3 峰值载荷 vs LEFM（诚实 FAIL）

| 量 | 基准(ref) | 本实现(got) | 相对误差 | 判据 | 判定 |
|---|---|---|---|---|---|
| DCB 峰值载荷（nx=60, t_n0=5 MPa） | 28.699 N | 34.779 N | **+21.2%** | ≤5% | **FAIL** |
| 峰值位置 | — | δ = 1.344e-3 m | — | — | — |
| 峰值处过程区跨度 | — | 22.5 mm（= 0.75 a） | — | — | — |
| 能量平衡 \|W−U−D\|/W | 0 | 0.500 | — | ≤1% | **FAIL（另案）** |

强度扫描（同网格 nx=60，同一进程内真跑）：t_n0 = 5 / 10 / 20 MPa → 峰值 34.779 / 37.251 / 38.677 N，
**随强度升高而升高**，不向 LEFM 28.699 N 收敛。结合"网格收敛 + 罚刚度无关"，说明该参数区间的峰值是
**强度/过程区控制**而非韧性控制：内聚高斯点位于裂尖后方半个单元处，粗网格下其分离只有宏观开口的
~4e-4 倍，故数字化过程区被严重欠分辨（δ_n0 = 5 µm 对 h = 3 mm）。要让峰值成为可判验收量需
dx ≲ 0.1–0.2 mm（数千单元，集群级），本轮按实记录为开放项。

## 4. 结果图

![DCB compliance calibration and load-opening](../docs/assets/cohesive3d-dcb-3d/specimen_compliance_and_peak.png)

`docs/assets/cohesive3d-dcb-3d/specimen_compliance_and_peak.png`（由 `benchmarks/cohesive3d_dcb_specimen_figures.py` 同进程真跑生成，数字零手抄）：
(a) 三个裂纹长度的 FE 柔性与梁理论包夹带；(b) a0 = 30 mm 的载荷-开曲线 + LEFM 峰值水平线，
直观显示线性段与峰值超调。

## 5. 注意点 / 局限

- **修复了什么**：加载面 prescription 的网格依赖（本轮）；加载 dof 未进约束集、路径跟随器不终止（第 1 轮，已修）。
- **仍是开放项**：①峰值载荷 vs LEFM（+21.2%，机理已定位：过程区欠分辨 / 强度控制）；
  ②能量平衡 49.9% 缺口（`diss_path` 梯形积分的记账问题，与本轮 prescription 无关；逐 GP 面积已核对
  Σarea = 单元面积，故不是面积因子）；③宽规格基准 `cohesive3d_dcb_3d.py --quick` 的 `delta_max=4e-5`
  在该参数下整段落在线性支上（峰值/能量判据只能是诊断量），真实峰值需集群级运行。
- 小应变运动学、无摩擦/接触/循环退化（既有限制，未变）。
- 本轮所有数字均由控制节点真跑（单线程/4–8 线程，39.6–125 s），集群未参与；判读注架构 x86_64。

## 6. 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 单元级 objectivity/patch/mode-I/mode-II/energy（`cohesive3d_verification_tables.json` 27 PASS） | 闭式 | ≤1e-10 | **是** |
| 试件柔性 C(a) @ a=20 mm | 7.759486e-06 m/N | +2.2% | **是** |
| 试件柔性 C(a) @ a=30 mm | 2.356742e-05 m/N | −6.6% | **否（<10%，在包夹带内）** |
| 试件柔性 C(a) @ a=40 mm | 5.297703e-05 m/N | −11.0% | **否（在包夹带内）** |
| 加载面 prescription 网格无关性 | 1.0 | 1.97% | **是** |
| DCB 峰值载荷 vs LEFM | 28.699 N | +21.2% | **否（诚实 FAIL）** |
| 能量平衡 | 0 | 0.500 | **否（诚实 FAIL，另案）** |

## 复现命令（repro_cmd）

```bash
python benchmarks/cohesive3d_dcb_displacement_control_probe.py   # 试件级：declared PASS / peak honest FAIL
python benchmarks/cohesive3d_verification_tables.py               # 27 PASS, 0 FAIL, 2 KNOWN_LIMIT
python -m pytest tests/test_cohesive3d.py tests/test_cohesive3d_dcb.py \
    tests/test_cohesive3d_dcb_solver_guards.py -q                  # 65 + 4 passed, 无 xfail
```