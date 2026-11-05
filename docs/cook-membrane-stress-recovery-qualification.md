# Cook 膜应力恢复场进入生产路径：5.02% 欠账的点式结清

**结论先行**：同一份生产位移解，从「单元中心常数应力」改成「Gauss 点外推 + 面积加权节点
平均」的恢复场后，5 个已发布探针点的 von Mises 最大误差在 n=64 为 **0.120%**（合并协议
同点最大 10.97%、单侧散布最大 19.92%），在 n=32 为 **0.484%**，n=16 为 1.951% 但最差
**分量**误差 8.947% 仍超门限，n=8 为 8.023% 超门限——粗网格失败项保留不删。恢复场均满足
单值性（探针处两侧插值差异 ≤ 7e-17）。生产 `cook_fields` 存档的单元中心场**未改动**，
其 `stress_accuracy_qualified` 仍为 `False`（fail-closed 不动）。

## 1. 案例介绍（物理背景 / 几何 / 参数）

Cook 膜是平面应力有限元的经典检验题：右端固支的梯形膜，左端承受单位合力剪切载荷，
由于夹持端存在应力集中与粗网格畸变，位移与应力收敛速度不同，长期被用作「位移软件」
与「应力软件」的分水岭。

| 项 | 取值 |
|---|---|
| 几何 | 梯形膜，左边界 x=0，y∈[0,44]；右边界 x=48，y∈[0,60] |
| 材料 | E=1、ν=1/3、平面应力、单位厚度 |
| 载荷 | 左端沿 y 方向的单位合力剪切（consistent edge traction） |
| 约束 | 右端 (x=48) 全部固支 |
| 主参考量 | 加载边中点位移 u_y = 23.96（文献标量，生产报告中位移误差 0.594%@n=32） |
| 应力参考 | 独立编写的 Q8 平面应力解 + Richardson 外推，冻结于 `benchmarks/cook_membrane_stress_reference.json` |
| 探针点 | (ξ,η)=(0.5,0.5),(0.3,0.5),(0.7,0.5),(0.5,0.25),(0.5,0.75)，与 `cook_stress_reference.SAMPLE_POINTS` 一致 |
| 判定门限 | 相对误差 < 3%，误差归一化用参考尺度 5% 下限 `|Δ|/max(|ref|,0.05·scale)` |

背景欠账：合并的 `cook_fields` 只存**单元中心常数应力**（O(h)），并在 n=64 上给出 5.02%
点式残差，因此 `stress_accuracy_qualified=False`。m2430 已证明 5.02% 是**测量协议**缺陷
（5 个标准点全部落在 Q4 单元边界上，取值不唯一），并给出恢复场的测量值 0.120%，但
**恢复场从未进入生产路径**（`docs/benchmark-qualification-status.md`「尚未闭环」一节点名）。

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **生产化恢复内核** `src/tensorfem/stress_recovery.py`（新增）：2×2 Gauss 点应力 →
   张量积线性外推（Barlow 1976）→ 按单元面积加权节点平均（Zienkiewicz–Zhu 1987）→
   节点场双线性插值（单值，处处良置）。同时提供 `recovery_error`（与 Cook 审计同口径的
   reference-scaled 5% 下限）。
2. **单一真源**：测量模块 `cook_probe_protocol` 删掉自己的重复实现，改为从
   `stress_recovery` 再导出同一批函数对象——审计数字与生产数字不可能再漂移
   （测试 `test_single_source_of_truth_with_the_audit_module` 用 `is` 断言钉死）。
3. **资格记录** `src/tensorfem/cook_recovery_qualification.py`（新增）：
   `cook_recovery_case(n)` = 生产 `cook_fields(n)` 位移 → 恢复场 → 5 点误差 + 分量误差 +
   单值性 + 与单元中心场对照；`cook_recovery_study(levels)` = 细化阶梯 + 单调性 + fail-closed 理由。
4. **证据与图**：`scripts/build_cook_recovery_qualification.py` 生成
   `benchmarks/cook_recovery_qualification.json` 与 3 张图（收敛曲线、恢复场云图、探针条形图）。
5. **测试**：`tests/test_cook_recovery_qualification.py`（13 项）——均匀应力 patch 精确性、
   单值性、探针协议缺陷仍保留、最细网格达标、粗网格失败保留、证据 JSON 防漂移、存档场标志不动、
   以及 3 个负控（场乘 1.05 必须被拒 / 参考乘 1.2 必须被拒 / 参考缺失必须 fail-closed / 篡改点集必须报错）。

复现：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python3 scripts/build_cook_recovery_qualification.py
OMP_NUM_THREADS=2 PYTHONPATH=src python3 -m pytest tests/test_cook_recovery_qualification.py tests/test_cook_probe_protocol.py -q
```

## 3. 计算结果（程序化注入，无手抄）

细化阶梯（每级 5 点取最大误差）：

| n | 恢复场 von Mises 最大误差 | 恢复场最差分量误差 | 存档单元中心场最大误差（对照） | 该级是否达标 |
|---:|---:|---:|---:|---|
| 8 | 8.02311% | 32.97504% | 104.04430% | 否（von Mises 超门限） |
| 16 | 1.95144% | 8.94680% | 50.19415% | 否（分量超门限） |
| 32 | 0.48379% | 2.39274% | 23.30393% | 是 |
| 64 | 0.12019% | 1.06786% | 10.97387% | 是 |

n=64 逐探针（恢复场 vs 冻结 Q8 参考）：

| (ξ,η) | 恢复 vm | 参考 vm | 误差 | 单值性散布 | 单元中心场单侧散布 |
|---|---:|---:|---:|---:|---:|
| (0.50,0.50) | 0.068690 | 0.068700 | 0.00748% | 6.9e-17 | 19.919% |
| (0.30,0.50) | 0.073210 | 0.073190 | 0.03354% | 1.4e-17 | 4.545% |
| (0.70,0.50) | 0.086750 | 0.086800 | 0.05680% | 1.4e-17 | 4.932% |
| (0.50,0.25) | 0.156320 | 0.156500 | 0.11585% | 2.8e-17 | 4.163% |
| (0.50,0.75) | 0.083360 | 0.083460 | 0.12019% | 5.6e-17 | 7.523% |

图：

- ![收敛曲线](../assets/cook-recovery/cook-recovery-convergence.png)
  `docs/assets/cook-recovery/cook-recovery-convergence.png`——恢复场（von Mises 与最差分量）
  与存档单元中心场的四级收敛，红线为 3% 门限。
- ![恢复场云图](../assets/cook-recovery/cook-recovery-field.png)
  `docs/assets/cook-recovery/cook-recovery-field.png`——n=64 恢复 von Mises 云图，标注 5 个
  探针点及其误差。
- ![探针条形图](../assets/cook-recovery/cook-recovery-probes.png)
  `docs/assets/cook-recovery/cook-recovery-probes.png`——逐探针「恢复值 vs 单侧误差带」，
  对数纵轴，展示良置与不良置取值的量级差。

## 4. 注意点（坑 / 局限 / 适用范围）

- **恢复场不是原存档场**。`cook_fields` 存的是单元中心常数应力（O(h)），恢复场是同一位移解的
  二阶后处理。引用 0.120% 必须同时说明协议；生产 `stress_accuracy_qualified` 保持 `False`
  就是为了不让后处理冒充原始场精度（测试 `test_stored_production_field_still_carries_no_stress_claim` 钉死）。
- **分量比 von Mises 更严**。n=16 的 von Mises 已 1.95% 达标，但最差分量 8.95% 不达标；
  因此本单把门限同时加在分量上，n=16 判失败。只报 von Mises 会高估精度。
- **参考有地板**。冻结 Q8 参考的逐点 GCI 为 0.004%–0.068%（von Mises 相对量），n=64 的
  0.120% 仍高于地板故可判读；更细网格继续下降将进入参考精度地板，届时 0.120% 应读作「≤」而非精确值。
- **不覆盖奇异区**。夹持角点与加载边端点的峰值应力（载荷不连续 + 角点奇异性）逐点不收敛，
  不在覆盖范围，不能用本结论外推。
- **恢复是局部有界的**。它不消除真实的应力集中、不修复奇异峰值、不做全局平滑；只解决
  「单元边界上取值不唯一」这一测量缺陷。
- **方法范围**：只对 4 节点双线性四边形 + 平面应力的 2×2 Gauss 恢复做过验证；HEX8/TET4/
  壳体单元的等价恢复（按单元类型各自的超收敛点）未在本单验证，不能默认成立。

## 5. 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 计算值 | 基准 | 相对误差 | 是否 <3% |
|---|---|---:|---:|---|
| 恢复场 5 点最大 von Mises 误差，n=64 | 0.12019% | 冻结 Q8 Richardson 参考 | 0.12019% | **是** |
| 恢复场最差分量误差，n=64 | 1.06786% | 同上 | 1.06786% | **是** |
| 恢复场 5 点最大 von Mises 误差，n=32 | 0.48379% | 同上 | 0.48379% | **是** |
| 恢复场最差分量误差，n=32 | 2.39274% | 同上 | 2.39274% | **是** |
| 恢复场最差分量误差，n=16 | 8.94680% | 同上 | 8.94680% | **否**（阶梯失败项，保留） |
| 恢复场 5 点最大 von Mises 误差，n=8 | 8.02311% | 同上 | 8.02311% | **否**（阶梯失败项，保留） |
| 存档单元中心场 5 点最大误差，n=64 | 10.97387% | 同上 | 10.97387% | **否**（本次未认证、也不改动其标志） |
| 探针单值性，n=64（两侧插值相对差） | ≤ 7e-17 | 单值要求 | — | 单值（不适用 3% 判据） |

**回答**：本单交付的恢复场在最细网格 n=64（及 n=32）上 von Mises 与全部分量**均 < 3%**，
达 3% 判据；n=16/n=8 的失败项如实保留，不删除、不外推。存档单元中心场仍未 < 3%，其
fail-closed 标志保持 `False`——两条结论互不替代。

## 6. 未做 / 边界

- 未改动 `cook_fields` 的返回结构、`stress_accuracy_qualified`、`benchmarks/` 下已发布的
  `results.json` 与证据哈希清单，因此既有报告与校验脚本的语义完全不变。
- 未把恢复场接入其他 benchmark（Mindlin 板、开孔半球壳等）；开孔半球壳仍缺独立应力参考，
  该缺口保持不变。
- 未做壳体单元（Shell4/MITC4）的等价恢复验证。