# m14670 GAP-FEM-WINKLER 交差审计核验单

**审计对象**：m14670（GAP-FEM-WINKLER·冰板弯曲 FEM 真实现（替 Winkler 占位）·TensorFEM，已 succeeded）
**审计人**：board-audit（本任务 m14965）
**审计日期**：2026-11-11

---

## ① 取证（逐项核验）

### 1.1 实现文件（impl_files）——真实存在

| 文件 | 大小 | 内容 |
|---|---|---|
| `src/tensorfem/foundation_plate_bending.py` | 41KB | Q4 Mindlin-Reissner 板单元 + MITC 剪切处理，Winkler/Pasternak/Kerr 地基耦合 `K_w = k ∫ Nᵀ N dA`（2×2 Gauss），含无限板 Hetényi 闭式解、Navier 双级数、浮力自洽锚 |
| `src/tensorfem/hetenyi_reference.py` | 26KB | Hetényi (1946) 解析参考：无限板 Kelvin-kei 解、无限梁形状函数 A/B/C、有限简支梁傅里叶正弦级数 |
| `src/tensorfem/beam_foundation.py` | 46KB | Euler-Bernoulli 梁 + 一致地基阵 `k ∫ Nᵀ N dx` |
| `src/tensorfem/winkler_foundation.py` | 4KB | Winkler 地基 |
| `src/tensorfem/element/foundation.py` | 11KB | 地基单元 |

### 1.2 入口（entry）

`benchmarks/fem_winkler_plate_k_sweep.py`（603 行）——主基准脚本，三族算例（A 无限板、B 有限梁、C 有限板）+ 硬门判据。

### 1.3 运行工件（run_artifacts）——全部存在且有效

| 工件 | 大小 | 内容 |
|---|---|---|
| `result.json` | 111KB | 全量数值（每族每 k 的 FEM/参考值/误差/平衡残差/ndof） |
| `gates_verified.json` | 9KB | **27/27 门通过**，`passed=True` |
| `winkler_k_sweep_beam.png` | 116KB | 梁挠度对比图 |
| `winkler_k_sweep_error.png` | 70KB | 误差 vs k 双对数图 |
| `winkler_k_sweep_profile.png` | 65KB | 板剖面 w(r)/w(0) vs r/l 对比图 |

### 1.4 测试（tests）

`tests/test_fem_winkler_plate_k_sweep.py`（161 行）——**19 个测试全部通过**（18 快 + 1 慢全量扫描），含：
- 变异探针 `test_gate_detects_a_mutated_foundation_stiffness`：k 注入 25% 误差 → 门必须咬住（证明验收链非惰性）
- 无量纲塌缩 `test_plate_nondimensional_collapse_over_three_decades`：w₀·√k 跨 3 量级不变
- 零地基退化 `test_zero_foundation_is_not_a_foundation`：k→0 不静默复用

### 1.5 集成调用方（integration_callers）

- `benchmarks/pasternak_ice_sheet_bending.py:32`（冰板 Pasternak 应用）
- `benchmarks/verified/fem_winkler_plate/run.py:70`（既有 Winkler 锚）
- `tests/test_foundation_plate_bending.py:24`、`tests/test_foundation_reference_audit.py:27`
- **已并入 main**（merge #632，commit 0955ebb2）

### 1.6 基准数值（benchmark ref vs got）

| 族 | k | w_FEM | w_参考 | 误差% |
|---|---|---|---|---|
| A 无限板 | 1000 | 509.4338 mm | 509.9020 mm (Hetényi) | 0.0918 |
| A 无限板 | 1e4 | 161.1320 mm | 161.2452 mm | 0.0701 |
| A 无限板 | 10055.2 | 160.6889 mm | 160.8016 mm | 0.0701 |
| A 无限板 | 1e5 | 50.9891 mm | 50.9902 mm | 0.0021 |
| A 无限板 | 1e6 | 16.1588 mm | 16.1245 mm | 0.2124 |
| B 有限梁 | 1e4 | 688.6410 mm | 688.6410 mm (级数) | 0.000006 |
| B 有限梁 | 1e6 | 20.7395 mm | 20.7395 mm | 0.000194 |
| C 有限板 | 1e4 | 76.1629 mm | 76.1581 mm (Navier) | 0.0063 |
| C 有限板 | 1e6 | 6.0427 mm | 6.0335 mm | 0.1533 |

**最差误差 0.2124%**（A 族 k=1e6），全部 <5% 硬门，且全部 <3% 更严读数。

### 1.7 复现命令（repro_cmd）——已验证可执行

- `python benchmarks/fem_winkler_plate_k_sweep.py`（全量）→ **ALL GATES PASSED**（27/27）
- `python benchmarks/fem_winkler_plate_k_sweep.py --quick`（粗网格）→ **ALL GATES PASSED**
- `python -m pytest tests/test_fem_winkler_plate_k_sweep.py -v` → **19 passed**

### 1.8 外部锚（external_anchor）

- Hetényi, M. (1946) *Beams on Elastic Foundation*, Univ. of Michigan Press
- Timoshenko & Woinowsky-Krieger (1959) *Theory of Plates and Shells*, §62

### 1.9 报告

已发布至 tensor_reports API（report id 13795，mission_id=14670，含 3 张图）。

---

## ② 判级（L0-L5）

**级别 = L4（基准验证：完整算例 + 误差分析 + 可复现）**

理由：
- **L0 文档成果**：否——有真实代码（~600 行实现 + 161 行测试 + 603 行基准）
- **L1 数学/单元验证**：是——Q4 Mindlin 单元 + MITC 剪切处理，地基耦合 `K_w = k∫NᵀNdA` 经 3 量级 k 扫描验证
- **L2 独立功能实现**：是——`foundation_plate_bending.py` 是真实 FEM 代码（输入→计算→输出），非 MOCK/STUB
- **L3 系统集成**：是——已并入 main（merge #632），被 `pasternak_ice_sheet_bending.py` 等下游调用
- **L4 基准验证**：**是**——完整三族算例（无限板/有限梁/有限板）× 3 量级 k 扫描，27/27 硬门通过，最差误差 0.2124%，含误差分析、平衡残差、无量纲塌缩、变异探针，且**可复现**（全量/粗网格/测试均验证通过）
- **L5 科研工程成果**：本节点是验证/加固型基准（非新理论贡献），L4 已充分覆盖

**不虚标**：本节点是「验证/加固」型任务（k 扫描硬化），非「新理论/新方法」型——L4 是功能实质的准确描述。

---

## ③ 落表

已 INSERT INTO `tensor_maturity_verdicts`：
```
repo='TensorFEM', scope='m14670 GAP-FEM-WINKLER', mission_id='14965',
level=4, evidence=<JSON>, verdict_by='board-audit'
```

---

## ④ 结论

**m14670 GAP-FEM-WINKLER 验收通过**：三件套（run_artifacts / benchmark ref vs got / repro_cmd）全部真实存在且可复现。27/27 硬门通过，最差误差 0.2124%（<3% 更严读数），19 测试全绿，含变异探针证明门非惰性。**confirmed=消账**。
