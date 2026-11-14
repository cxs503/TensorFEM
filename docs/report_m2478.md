# 自由边界板振动验证报告（Mission 2478 补报）

**Status:** honest-FAIL（验证工作已完成，但 SRI Q4 单元在自由边界下存在已知沙漏模态，需 MITC4 清偿）  
**Repository:** TensorFEM · **Module:** `tensorfem.plate_vibration`（`boundary="free"` 路径）  
**Evidence:** `benchmarks/free_plate_vibration.py`、`benchmarks/free_plate_vibration_evidence.json`、`tests/test_free_plate_vibration.py`（18 项）  
**后续清偿:** `docs/free-plate-mitc-vibration-validation.md`（MITC4 消除沙漏）、`docs/free-plate-winkler-foundation-validation.md`（自由边+Winkler 地基）

---

## 1. 案例介绍（物理背景/几何/参数表）

**物理问题：** 完全自由（无任何位移/转动约束）矩形板的首阶弯曲振动。自由板没有闭式频率解析解，因此验证策略是构造**独立离散参考解**（全局 Rayleigh–Ritz，Legendre 多项式基），与有限元解交叉比对。

**几何与材料参数：**

| 参数 | 符号 | 取值 |
|---|---|---|
| 板长 / 板宽 | a, b | 1.0 m × 1.0 m（方板）；2.0 m × 1.0 m（矩形板 a/b=2） |
| 板厚 | h | 0.01 m（a/h = 100，薄板极限） |
| 弹性模量 | E | 200 GPa |
| 泊松比 | ν | 0.30 |
| 密度 | ρ | 7850 kg/m³ |
| 抗弯刚度 | D = Eh³/(12(1−ν²)) | 1.8315×10⁴ N·m |
| 面密度 | ρh | 78.5 kg/m² |
| 边界 | — | 完全自由（无任何约束） |
| 单元 | — | Q4 选择性减缩积分（SRI）Mindlin 板 + 一致质量（含转动惯量） |

**无量纲频率参数：** λ = ω a² √(ρh/D)，与 Leissa *Vibration of Plates* (NASA SP-160, 1969) 对标。

---

## 2. 怎么做的（方法/流程/工具链）

### 2.1 独立参考解：全局 Rayleigh–Ritz

新模块 `plate_ritz_reference.py`，Kirchhoff 应变能与动能：

$$U = \frac{D}{2} \int \left[ w_{xx}^2 + w_{yy}^2 + 2\nu w_{xx} w_{yy} + 2(1-\nu) w_{xy}^2 \right] dA, \quad T = \frac{\rho h}{2} \int \dot{w}^2 dA$$

取张量积 Legendre 多项式试函数 $w = \sum c_{pq} P_p(\xi) P_q(\eta)$，一维积分用 Gauss–Legendre **精确**求积，四个算子矩阵用 Kronecker 积拼装。三个刚体模态（1, x, y）用 M-正交补投影剔除后解广义特征值问题（Cholesky 归约 + `numpy.eigvalsh`）。

**关键：** 该方法与 FE 不共享单元、不共享积分规则、不共享自由度布局，是真正独立的参考解。

### 2.2 被验证对象：`tensorfem.plate_vibration` 的 `boundary="free"` 路径

新增"自由边界＝无消元"分支，并把零能量特征对在弹性模态之前显式剔除：`rigid_body_relative_tolerance`（默认 1e-9，相对最大特征值）以下的特征对被计数并记入 `PlateVibrationResult.rigid_body_modes`。

### 2.3 数值陷阱（本单实测）

- `∫ w_xx w_yy` 的 Kronecker 拼装必须用 `kron(C2x, C2y.T)`；写成 `kron(C2x, C2y)` 在 order 8 上把耦合计错 5 倍并让刚度阵变为不定。模块内建自检 `energy_quadrature_check`，相对偏差 2.28×10⁻¹⁴。
- `scipy.linalg.eigh(K, M)` 在本环境对 Legendre 质量阵直接 segfault，故一律走 Cholesky 归约。

### 2.4 工具链

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/free_plate_vibration.py
OMP_NUM_THREADS=8 PYTHONPATH=src python -m pytest -q tests/test_free_plate_vibration.py
```

---

## 3. 计算结果（数字表，脚本直出，零手抄）

### 3.1 参考解自身：收敛 + 与文献一致

| Ritz 阶次 | λ₁ | λ₂ | λ₃ | λ₄ |
|---:|---:|---:|---:|---:|
| 8 | 13.46824 | 19.59614 | 24.27021 | 34.80107 |
| 12 | 13.46820 | 19.59614 | 24.27020 | 34.80090 |
| 18 | 13.46820 | 19.59614 | 24.27020 | 34.80089 |
| **Leissa SP-160 (1969)** | **13.468** | **19.596** | **24.272** | — |

λ = ω a²√(ρh/D)。序列随阶次单调下降（Rayleigh–Ritz 是上界），8 阶已收敛到 1e-5 量级；与文献值偏差 0.0015% / 0.0007% / 0.0074%。刚体约束残差 `rigid_body_defect < 1e-12`。

### 3.2 自由方板 FE 网格研究（前 4 阶弹性模态）

| 网格 | 零能量模态数 | λ₁ | λ₂ | λ₃ | λ₄ |
|---|---:|---:|---:|---:|---:|
| 6×6 | 4 | 13.6215 | 20.2136 | 25.4840 | 35.5991 |
| 8×8 | 4 | 13.5413 | 19.9493 | 24.9591 | 35.2842 |
| 10×10 | 4 | 13.4978 | 19.8222 | 24.7101 | 35.0980 |
| 12×12 | 4 | 13.4704 | 19.7518 | 24.5732 | 34.9823 |
| 16×16 | 4 | 13.4409 | 19.6809 | 24.4358 | 34.8517 |

### 3.3 与独立参考的相对误差（基准 = Ritz 阶次 14）

| 模态 | 12×12 计算 | 12×12 误差 | 16×16 计算 | 16×16 误差 |
|---|---:|---:|---:|---:|
| 1 | 13.4704 | +0.016% | 13.4409 | −0.203% |
| 2 | 19.7518 | +0.794% | 19.6809 | +0.432% |
| 3 | 24.5732 | +1.248% | 24.4358 | +0.682% |
| 4 | 34.9823 | +0.521% | 34.8517 | +0.146% |

### 3.4 自由矩形板 a/b = 2（16×16）

| 模态 | FE | Ritz 参考 | 相对误差 |
|---|---:|---:|---:|
| 1 | 21.5779 | 21.46315 | +0.535% |
| 2 | 26.4469 | 26.57490 | −0.482% |
| 3 | 58.4587 | 58.48195 | −0.040% |
| 4 | 60.8126 | 59.60608 | +2.023% |

### 3.5 结果图

**说明：** 本补报单未重新生成图形（原单工件目录未留档 png 文件路径），理由如下：
- 原单 `benchmarks/free_plate_vibration.py` 应生成 `docs/assets/free_plate_vibration_convergence.png` 与 `docs/assets/free_plate_vibration_mode_shapes.png`，但当前 worktree 内 `docs/assets/` 目录未包含这两个文件（可能未被 git 跟踪或已被清理）。
- 补报单纪律要求"不改业务代码、不重跑计算"，因此不重新执行 benchmark 脚本生成新图。
- 数字表已从 `benchmarks/free_plate_vibration_evidence.json` 程序化提取，零手抄。

如需复现图形，执行：
```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/free_plate_vibration.py
```

---

## 4. 误差判定（逐项，显式回答是否 < 3%）

| 关键量 | 基准 | 相对误差 | < 3%？ |
|---|---|---:|:--:|
| Ritz λ₁（方形） | Leissa 13.468 | 0.0015% | 是 |
| Ritz λ₂ / λ₃ | Leissa 19.596 / 24.272 | 0.0007% / 0.0074% | 是 |
| FE 自由方板 λ₁，16×16 | Ritz 13.4682 | 0.203% | 是 |
| FE 自由方板 λ₂，16×16 | Ritz 19.5961 | 0.432% | 是 |
| FE 自由方板 λ₃，16×16 | Ritz 24.2702 | 0.682% | 是 |
| FE 自由方板 λ₄，16×16 | Ritz 34.8009 | 0.146% | 是 |
| FE 自由矩形板 λ₁，16×16 | Ritz 21.4631 | 0.535% | 是 |
| FE 自由矩形板 λ₄，16×16 | Ritz 59.6061 | 2.023% | 是（余量最小） |
| 拼装算子 vs 直接二维求积 | 精确恒等式 | 2.28×10⁻¹⁴ | 是 |

**结论：全部关键量 < 3%，最差项 2.02%（自由矩形板第 4 阶）。** 但见 §5 第 1 条——零能量模态的**个数**不达标，这是本单的 honest 限制，不是可被频率误差掩盖的细节。

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

### 5.1 SRI Q4 在自由边界下多出 1 个零能量（沙漏）模态

自由板物理上应有 3 个刚体模态（两个平移 + 一个转动），实测每个网格都是 **4** 个零能量模态（`rigid_body_modes == 4`）。这是选择性减缩积分剪切项秩亏的经典后果。

**本单的处理：** 显式计数并记录（不静默丢弃），测试 `test_free_plate_has_exactly_three_rigid_body_modes_plus_one_hourglass` 锁定该事实。如果逐阶取"前 3 个非零模态"而不计数，就会把沙漏模态当成首阶物理模态（实测会被误读成 λ≈70，即 5 倍高估）。

**清偿状态：** 该项已由 `docs/free-plate-mitc-vibration-validation.md` 清偿——`solve_plate_vibration` 新增 `shear_treatment="mitc"`，自由板刚度零空间由 4 维回到物理上正确的 3 维。

### 5.2 自由边界要求 fail-closed 而不是 fail-open

支撑边界一旦出现零能量模态说明单元集仍含机构，模块直接抛 `ValueError`；自由边界的零能量模态必须显式计数，二者共用同一相对容差（默认 1e-9 × 最大特征值，按 ω² 相对量取）。容差取 1e-6 会错删物理模态（最高模态被转动惯量撑到极高频，相对尺度被拉大），这是本单实测到的定标坑。

### 5.3 参考解不是"真值"而是上界序列

Ritz 给出的是上界，FE 的 SRI 单元在弯曲上通常略软，两者误差在 12×12 → 16×16 不是单调同号（λ₁ 由 +0.016% 变 −0.203%），说明两级误差量级同阶、存在误差交叉，不能据此声称观察到收敛阶。

### 5.4 模型层面仍是 Kirchhoff/Mindlin 薄板

a/h = 100，一阶剪切修正使频率偏低约 0.03%（既有文档已量化）；对 h 加厚或高阶模态，Ritz（Kirchhoff）与 FE（Mindlin）会因模型差异而非离散误差分叉，不可继续用 3% 门限互相判分。

### 5.5 不覆盖的内容

- 预紧力、附加质量/流固耦合、阻尼、几何缺陷、各向异性/层合板
- 自由边+弹性地基组合（Winkler 项对自由边非精确特征函数）——该项已由 `docs/free-plate-winkler-foundation-validation.md` 清偿

### 5.6 稠密代数上限

16×16（867 自由度）单次约 3.5 s；24×24 在本单实测超时（>120 s，含转动惯量的高频端使稠密 eigh 代价显著），工程板格必须换稀疏特征值求解器。

---

## 6. 复现与证据

| 产物 | 路径 |
|---|---|
| 参考解模块 | `src/tensorfem/plate_ritz_reference.py` |
| 被验证模块 | `src/tensorfem/plate_vibration.py`（`boundary="free"` 新增） |
| 测试（18 项） | `tests/test_free_plate_vibration.py` |
| 证据脚本 | `benchmarks/free_plate_vibration.py` |
| 原始数据 | `benchmarks/free_plate_vibration_evidence.json` |
| 后续清偿（MITC4） | `docs/free-plate-mitc-vibration-validation.md` |
| 后续清偿（Winkler） | `docs/free-plate-winkler-foundation-validation.md` |

测试断言全部对文献值、独立离散参考或精确矩阵恒等式；无一条对存储快照。

---

## 7. 结论（honest-FAIL 声明）

**验证工作已完成，频率误差全部 < 3%。** 但 SRI Q4 单元在自由边界下存在 1 个多余沙漏模态（物理 3 个刚体模态，实测 4 个零能量模态），这是单元秩亏的已知缺陷，不能通过频率误差达标来掩盖。

**清偿路径：**
1. MITC4 假定剪应变消除沙漏模态 → `docs/free-plate-mitc-vibration-validation.md`
2. 自由边 + Winkler 地基验证 → `docs/free-plate-winkler-foundation-validation.md`

本补报单忠实记录原单（mission 2478）的验证工作与 honest 限制，不改写结论，不粉饰缺陷。
