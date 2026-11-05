# Cook membrane: Gauss-point (Barlow) stress recovery promoted to the production path

基线：`docs/benchmark-qualification-status.md` 把 Cook 膜的应力资格长期挂在 `WITHHELD`——理由是
「生产 Q4 应力是逐单元常数场、5 个规范探针全部落在单元界面上、单点取值不唯一」，并明确记下
「恢复场尚未进入生产 `cook_fields`」。本单闭合的正是这句欠账：把 Barlow 恢复做成**生产代码**，
并用同一份独立 Q8 参考逐级复核。原始逐单元协议 `stress_accuracy_qualified` **保持 `False`**，
不做任何放宽。

## 1. 案例与参数

* 几何：Cook 膜，48×44 梯形板（`continuum_benchmarks.cook_membrane`），单位厚度，平面应力。
* 材料：`E = 1`、`ν = 1/3`（即 `cook_membrane` 组装用的无量纲口径）。
* 载荷/边界：右端均布剪切 6.25/单位长度，左端固支。
* 参考解：独立 Q8（serendipity，3×3 积分）三级网格（12/24/48）Richardson 外推，
  逐点保存于 `benchmarks/cook_membrane_stress_reference.json`，本轮**未重新求解参考**。
* 5 个规范探针（父坐标 → 物理坐标）：(0.5,0.5)→、及 `xi=0.5`、`eta=0.25/0.5/0.75` 一等，
  与历史合并通道完全一致；参考 von Mises 量级 0.0687–0.1565。

## 2. 方法（生产路径）

新增 `src/tensorfem/stress_recovery.py`（Torch，独立于审计用的 NumPy 实现）：

1. 在 2×2 Gauss 点（Q4 的超收敛点）求 `σ = D B u`，槽序 eta 最快；
2. 每单元按一维两点外推 `(0.5 ± 1/(2√3))` 的张量积把 4 个 Gauss 值线性外推到 4 个节点；
3. 共享节点按**单元面积加权**平均 → 连续、单值的节点应力场；
4. 探针处对该节点场做双线性插值（界面处天然单值，不再依赖 floor/ceil 平局裁决）。

`additional_benchmark_fields.cook_fields(n, reference=None)` 因此新增两个字段：
`nodal_stress`（nnodes×3）与 `stress_recovery`（方法、逐点值、逐点误差、最差站点、界面积分值离散度、
`passed`）。带 `reference` 时按「最差站点归一化 von Mises 误差 < 3%」给出判据；不带参考时
`passed=None`，只给场不给结论。装配、求解器与 `stress_accuracy_qualified` 一律不动。

证据全部由 `scripts/cook_recovery_production_evidence.py` 机器产出 →
`docs/cook-recovery-evidence.json`；零手抄。

## 3. 结果（数字表）

**恢复场逐点误差 vs 独立 Q8 参考**（归一化口径：`|Δ| / max(|ref|, 0.05·scale)`，
`scale = 0.15650`，与 `cook_stress_reference.normalized_error` 相同）

| n | 节点数 | 恢复场最差站点误差 | 原始逐单元场（最好..最坏可取值） | 位移误差 | 界面离散度 |
| --- | --- | --- | --- | --- | --- |
| 8 | 81 | 8.0231% | 21.11% .. 29.78% | 7.8498% | 6.9e-17 |
| 16 | 289 | **1.9514%** | 11.75% .. 15.78% | 2.2103% | 8.3e-17 |
| 32 | 1089 | 0.4838% | 6.70% .. 7.94% | 0.5942% | 8.6e-17 |
| 64 | 4225 | **0.1202%** | 3.51% .. 4.02% | 0.1481% | 9.4e-17 |

* 观测收敛阶：恢复场 **p = 2.009**（二阶），原始逐单元场 **p = 0.981**（一阶）——恢复把
  收敛阶从一阶抬到二阶，这正是欠账的物理来源。
* 5 个探针在**每一级**网格上都落在单元界面（`ambiguous_sites = 5/5`），这解释了为什么
  原始协议必须报区间而不是单值；恢复场在这些点上的「多单元插值离散度」≤ **9.4e-17**，
  即单值性达到机器零（理论应为 0）。
* n=64 逐点（探针 1..5）：0.0075%、0.0335%、0.0568%、0.1159%、0.1202%。
* 参考自身的 GCI（细网格相对）为 4.3e-5 .. 6.8e-4（≈0.007%–0.068%），因此 0.1202% 应读作
  **「≤0.12%」**，真实值被参考地板部分遮挡。

**实现交叉校验**（两条互不依赖的实现路）

| 校验 | 结果 |
| --- | --- |
| Torch 应变矩阵 vs NumPy `cook_stress_reference.q4_strain_matrix`（倾斜单元、4 个 Gauss 点 + 2 个任意点） | 最大差 **2.78e-17** |
| Torch 恢复场 vs NumPy 审计恢复 `cook_probe_protocol.nodal_recovered_stress`（n=64，4225 节点） | 最大差 **3.22e-14**（相对 4.7e-14） |

图：`docs/figures/cook_recovery_error_ladder.png`（误差-网格 log-log，恢复场 vs 原始区间 + 3% 门限）、
`docs/figures/cook_recovery_field.png`（n=64 恢复场 vs 原始逐单元场的 von Mises 云图，红叉为 5 个参考探针）。

## 4. 误差判定：是否 < 3%？

| 量 | 基准 | 最好误差 | < 3%？ |
| --- | --- | --- | --- |
| 常量应力 patch（线性位移场）恢复 | 解析常数应力 | 6.6e-19（机器零） | **是** |
| Torch 应变矩阵 | NumPy 独立实现 | 2.8e-17 | **是** |
| Torch 恢复场 | NumPy 审计恢复 | 4.7e-14（相对） | **是** |
| 恢复场最差探针 n=16 | Q8 Richardson 参考 | 1.9514% | **是** |
| 恢复场最差探针 n=32 | 同上 | 0.4838% | **是** |
| 恢复场最差探针 n=64 | 同上 | 0.1202%（≤0.12%，受参考地板 0.068% 限制） | **是** |
| 原始逐单元场最差可取值 n=64 | 同上 | 3.51%（区间 3.51–4.02%） | **否**（fail-closed 保留） |
| 位移响应 n=64 | 已发表标量 23.96 | 0.1481% | **是** |

结论：**恢复场口径下 5 个探针逐点在 n≥16 已全部 < 3%**，n=64 达 0.12% 且二阶收敛；
原始逐单元协议在 n=64 仍 3.5%+，故 `cook_fields['stress_accuracy_qualified']` 继续为
`False`，两个口径并存、互不覆盖。

## 5. 注意点 / 局限

1. **恢复不是「换尺子让它过关」**：逐单元常数场的 5.02% 早已被 `cook-probe-protocol-audit.md`
   归因为协议病态（探针全在界面上）；本单给出的是教科书标准后处理（Barlow 外推 + 节点平均），
   并要求它在**每一级网格**上都可复现、单调、二阶收敛、且与另一条独立实现一致到 1e-14。
2. **参考地板**：Q8 参考只到 n=48，细网格 GCI 0.068%，因此 0.12% 是上界而不是精确值；
   再往下压误差需要更细的参考，本单没有做。
3. **适用面**：只覆盖 Cook 膜平面应力（Q4、单位厚度、5 个声明探针点）。不构成三维厚度应力
   认证，不覆盖加筋板/屈曲，也不是工业验收；`nodal_stress` 是后处理量，未回写装配或求解器。
4. **面积加权**是结构化补丁的标准选择；对强畸变网格应改用更保守的权重（本单未测畸变敏感性）。
5. 曾踩的坑：应变矩阵链式法则必须写 `dN/dx = dN/dξ · inv(J)ᵀ`（`J[a,b]=dx_b/dξ_a`）；
   漏转置在矩形上「碰巧正确」、在 Cook 的梯形映射上立刻错 0.2（相对 15%+），
   与 `cook_stress_reference._strain_matrix` 文档记录的同一历史缺陷同形。该约定现已被
   `test_strain_matrix_matches_numpy_audit_formula` 用倾斜单元锁死。

## 6. 复现

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/cook_recovery_production_evidence.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/test_stress_recovery.py -q   # 6 passed
```

字段说明：`docs/cook-recovery-evidence.json`（ladder / raw_protocol_band / cross_checks / status / scope），
代码 `src/tensorfem/stress_recovery.py`、`src/tensorfem/additional_benchmark_fields.py`（`cook_fields`），
测试 `tests/test_stress_recovery.py`。