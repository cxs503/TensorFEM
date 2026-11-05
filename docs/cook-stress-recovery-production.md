# Cook 膜应力恢复进入生产路径：把 3.687% 的逐点欠账压到 0.121%

本单结清 `docs/cook-membrane-pointwise-protocol.md` 与 `docs/benchmark-qualification-status.md`
留下的最后一条 Cook 应力欠账：**「同一位移解经 Gauss 点+节点恢复后 n=64 为 0.120%，但恢复场
尚未进入生产 `cook_fields`」**。恢复场现已由生产模块
`tensorfem.stress_recovery` 实现，`cook_fields(...)['stress_nodal']` 直接给出，
并在冻结的 Q8 独立参考上通过 3% 逐点门限。

**结论先行**（数字全部程序化注入，来源 `benchmarks/cook_stress_recovery_evidence.json`）：

| 场 | n=8 | n=16 | n=32 | n=64 | 实测收敛阶 | 3% 门限 |
|---|---:|---:|---:|---:|---:|---|
| 生产**恢复**节点场（新） | 8.1759% | 1.9868% | 0.4910% | **0.1210%** | 2.04 / 2.02 / 2.02 | **通过**（n=16 起） |
| 存盘**单元中心**场（未改） | 24.2409% | 10.9460% | 5.9268% | **3.0530%** | 0.75 / 0.89 / 0.87 | 不通过 |

`stress_accuracy_qualified` 对存盘单元中心场**保持 `False`**（fail-closed 不动）；新增的
认证只针对恢复场，判据、口径与适用范围在下文逐条写明。

## 一、案例与问题定义

线弹性平面应力 Cook 梯形膜（与既有 Cook 报告同一算例）：

| 项 | 值 |
|---|---|
| 四角 | (0,0) (48,44) (48,60) (0,44) |
| E / ν / 厚度 | 1.0 / 1/3 / 1.0 |
| 载荷 | x=48 边均布竖向面力，合力 1.0（边长 16） |
| 约束 | x=0 边全约束 |
| 已发表标量参考 | 加载边中点 (48,52) 挠度 23.96 |
| 门限 | 逐点 von Mises 相对误差 3%（含 5% 量级下限） |

生产场：`additional_benchmark_fields.cook_fields(n)`；独立参考：`cook_stress_reference`
自写 Q8 serendipity + 3×3 Gauss + n=12/24/48 Richardson，冻结在
`benchmarks/cook_q4_pointwise_protocol.json`（5 个非对齐探针，GCI 0.002%–0.066%）。

**欠账的物理本质**：`continuum.q4_stiffness` 只返回单元中心 B 矩阵，所以存盘应力是
`C @ B(0,0) @ u`，每个单元一个常数值，理论上只有一阶精度；而 Q4 的超收敛点在 2×2 Gauss
点（Barlow）。恢复场 = Gauss 点应力线性外推到节点 + 面积加权节点平均 + 双线性插值，
是 Zienkiewicz–Zhu 的标准做法，二阶。此前它只在审计脚本里存在，用户拿不到。

## 二、怎么做的

1. **生产模块** `src/tensorfem/stress_recovery.py`（纯张量代数，无装配/无求解器）：
   - `q4_gauss_stresses`：2×2 Gauss 点 B 矩阵与应力（Jacobian 非正即抛错）；
   - `recover_nodal_stress`：4×4 外推矩阵（±1/√3 → ±1 线性外推）+ 面积加权 `index_add_`
     节点平均；任何节点无贡献即抛错；
   - `locate_point`：**多起点 Newton**（父方框中心 + 4 个角点）反演双线性映射，校验映射
     残差，未收敛/出界抛 `ValueError`——直接吸收 m2571 记录的「裸 Newton 在斜单元上不全局
     收敛」缺陷，不再静默取错单元；
   - `interpolate_nodal_stress`：返回插值应力、候选单元数与 `element_spread`（声称节点场
     却在界面处给出不同值 = 恢复缺陷，被显式暴露而不是隐藏）；
   - `recovery_verdict`：fail-closed 判据，NaN/负值/空阶梯直接抛错，非单调阶梯判不通过。
2. **生产场接入**：`cook_fields(n)` 新增 `stress_nodal`、`von_mises_nodal`、`recovery`
   三个键；`stress`（单元中心）与 `stress_accuracy_qualified=False` 原样保留。附加成本
   ≤1 次 4 点 B 矩阵，n=64 求解 14.0 s 中恢复部分可忽略。
3. **认证脚本** `scripts/audit_cook_stress_recovery.py`：n=8/16/32/64 阶梯，
   每个探针同时记录恢复场、含该点的单元的原始常应力（raw 的最好情况）、
   `element_spread`、以及实测收敛阶，输出 `benchmarks/cook_stress_recovery_evidence.json`
   与 fail-closed verdict（不合格时退出码 1）。
4. **图** `scripts/plot_cook_stress_recovery.py`：
   `docs/assets/cook_recovery_convergence.png`（误差-网格 log-log + O(h²) 参考斜率 + 3% 门限）、
   `docs/assets/cook_recovery_field.png`（n=32 单元中心阶梯场 vs 恢复节点场）。
5. **测试** `tests/test_stress_recovery.py`（12 项）：外推矩阵线性再现性、斜单元常应变
   patch 精确性、**与审计 numpy 实现逐位一致（1e-12）**、节点场界面单值性、
   畸变单元反演往返、域外点拒采、verdict fail-closed、生产键位存在且原始场未被洗白、
   恢复场 3% 门限、冻结证据阶梯单调性。

## 三、计算结果（程序化注入）

| n | 节点/单元 | 求解 s | 位移响应误差 | 恢复场最大误差 | 原始场最大误差 | 界面散布 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 81 / 64 | 0.13 | 7.8498% | 8.1759% | 24.2409% | 0.0 |
| 16 | 289 / 256 | 0.33 | 2.2103% | 1.9868% | 10.9460% | 0.0 |
| 32 | 1089 / 1024 | 1.69 | 0.5942% | 0.4910% | 5.9268% | 0.0 |
| 64 | 4225 / 4096 | 13.99 | 0.1481% | 0.1210% | 3.0530% | 0.0 |

n=64 五个探针的恢复场误差：0.0079% / 0.0265% / 0.0702% / 0.1205% / 0.1210%（最差为探针 5）。
所有探针 `element_spread` 恒为 0：恢复节点场在界面处严格单值，插值与所选单元无关。

### 误差判定（逐项 vs 基准，显式回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---:|---|
| 恢复场逐点 von Mises，n=8 | Q8 独立外推 | 8.1759% | **否**（粗网格不达标，保留记录） |
| 恢复场逐点 von Mises，n=16 | 同上 | 1.9868% | **是** |
| 恢复场逐点 von Mises，n=32 | 同上 | 0.4910% | **是** |
| 恢复场逐点 von Mises，n=64 | 同上 | 0.1210% | **是** |
| 原始单元中心场逐点 von Mises，n=64 | 同上 | 3.0530% | **否**（恰在门限外，与冻结协议 n=128 的 3.687% 同向） |
| 位移响应（标量 23.96），n=64 | 已发表标量 | 0.1481% | **是** |
| 恢复算子 vs 审计实现 | 逐节点应力 | 6.7e-15（相对） | **是**（同解一致性，非精度认证） |

## 四、注意点与局限

1. **认证的是恢复场，不是原始场**。恢复场用了「Gauss 点外推 + 节点平均」，在
   单元边界上把不连续场抹平；它不能用来声称**单元中心常数场**精度合格，也不能用来
   反推原始场的界面跳变被消除。`stress_accuracy_qualified` 保持 `False` 且由测试锁死。
2. **原始场对照口径**：本报告的原始场取「包含该探针的单元的常应力」（对原始场最有利的
   采样），n=64 为 3.053%；冻结协议（floor tie-break 选元）在 n=128 为 3.687%。两者
   不矛盾：都说明一阶场在可负担网格上贴不住 3%。
3. **恢复场的二阶性来自光滑区**：夹持角点与加载边端点是奇异区，不在探针范围内，不能外推。
4. **多起点 Newton 不是全域保证**：4 个角点 + 中心共 5 个起点覆盖了 Q4 双线性映射的
   实际分支结构（映射是一元二次，至多 2 个原像），残差校验保证不会静默返回错单元；
   极其扭曲（接近退化）的单元仍可能拒采，这是有意的 fail-closed 行为。
5. **适用范围**：仅线弹性平面应力、E=1、ν=1/3、厚度 1、单位合力剪切载荷的 Cook 梯形膜；
   不覆盖几何非线性、塑性、板壳、接触。
6. **未做**：未把恢复场接入其他单元族（CST/HEX8/壳）——它们的超收敛点不同，需各自认证；
   未改 `stress_accuracy_qualified` 的语义；未推进 n=128 恢复阶梯（n=128 原始求解约 85 s，
   本单在 load15>100 的负载纪律下只跑到 n=64，n=64 已达 0.121%，无必要）。

## 五、复现

```
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src python scripts/audit_cook_stress_recovery.py
#  → benchmarks/cook_stress_recovery_evidence.json（verdict.qualified = true）
OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/plot_cook_stress_recovery.py
#  → docs/assets/cook_recovery_convergence.png, docs/assets/cook_recovery_field.png
OMP_NUM_THREADS=4 PYTHONPATH=src python -m pytest tests/test_stress_recovery.py -q
#  → 12 passed（约 20 s）
```

## 六、给下次自找活的线索

1. 同一恢复算子的下一站：**CST 三角元无超收敛点**，应在单元内做 patch 恢复或改用
   整体 L2 投影；HEX8 走 2×2×2 Gauss（与 Q4 同构，是低风险扩展）。
2. 该恢复场可直接喂给 `cook_stress_reference.strain_recovery_agreement` 与
   `docs/benchmark-qualification-status.md` 的壳体条目——**开孔半球壳缺独立应力参考**
   仍是开放欠账，且壳元的超收敛点尚未审计。
3. 若要把「逐点 3%」的原始场也做到合格，唯一诚实的路径是提高网格或换单元；
   不要再改探针协议来凑数（本仓已两次踩到该坑）。