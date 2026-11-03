# Cook 膜应力场独立审计（验证欠账清单条目）

## 结论摘要

`docs/benchmark-qualification-status.md` 记录的欠账是「Cook 膜位移响应通过、缺独立应力参考」。
本单为该条目建立独立审计通道，结论如下（honest-FAIL，不做粉饰）：

| 项 | 结论 | 证据 |
|---|---|---|
| 已发表参考位移 23.96（加载边中点 48,52） | **确认**：两套独立离散均落在 3% 内；Q4 四级网格 Richardson 外推 23.967（0.03%） | `benchmarks/cook_membrane_stress_reference.json` → `platform.response`、`reference.probe` |
| 独立 Q8 参考解自收敛性 | **通过**：五个内部采样点 von Mises 的 GCI(最细) 0.049% | `reference.points[*].von_mises.gci_fine_relative` |
| 生产 Q4 应力场精度认证 | **仍未闭环（WITHHELD）**：审计自带的点式 Q4 应力恢复与 `cook_fields` 存盘单元中心应力不一致，点式比较不可认证 | `recovery_cross_check` |
| 加载边（y=44..60）位移型线 | **未解释分歧**：Q4 近水平（23.2→25.0），Q8 强倾斜（7.6→52.2），两法应收敛到同一解 | `loaded_edge_profiles` |

因此本次审计**没有**把 `cook_fields` 的 `stress_accuracy_qualified` 翻成 True，也没有改动
`docs/benchmarks/tutorials/cook_membrane_full.md` 及其哈希清单：报告正文的「无独立应力参考」表述
在同一次审计中依然成立。

## 方法与工具链

1. **独立参考实现**（`src/tensorfem/cook_stress_reference.py`，纯 NumPy，不复用 `tensorfem.continuum`
   的装配与求解代码）：
   - 8 节点 serendipity（Q8）平面应力单元，3×3 Gauss 积分，等参双线性映射；
   - 自建 COO 汇总→CSR 结构，Jacobi 预处理共轭梯度，容差 1e-13；
   - 加载边按二次边形函数施加均布竖向面力，合力恒为 1（核验 `load_resultant`）；
   - 固定左边界 x=0 全部自由度。
2. **收敛与不确定度**：网格 n=12/24/48，对五个固定材料点（父坐标 (0.5,0.5)、(0.3,0.5)、
   (0.7,0.5)、(0.5,0.25)、(0.5,0.75)）的应力分量做 Richardson 外推 + Roache GCI（安全系数 1.25）。
   采样点全部位于内部，远离夹持角点与加载边端点处的载荷间断奇异区。
3. **生产代码侧**：`tensorfem.additional_benchmark_fields.cook_fields(n)`，n=8/16/32/64，
   原样调用，不修改生产路径。
4. 复现：`OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_cook_membrane_stress.py`，
   产物即上表引用的 JSON。

## 计算结果（程序化注入，零手抄）

Q8 独立解（容差 1e-13）：

| n | 节点 | 单元 | CG 迭代 | 相对残差 | 合力 |
|---:|---:|---:|---:|---:|---:|
| 12 | 481 | 144 | — | <1e-13 | 1.000000 |
| 24 | 1825 | 576 | — | <1e-13 | 1.000000 |
| 48 | 7105 | 2304 | — | <1e-13 | 1.000000 |

具体迭代数、逐级应力、Q4 四级位移与五个采样点的参考/计算对照表全部写在
`benchmarks/cook_membrane_stress_reference.json`（含环境版本、问题定义、样点父坐标与单元定位），
本文件不重复抄写以免出现手抄漂移；表格由 `scripts/audit_cook_membrane_stress.py` 生成。

## 注意点与局限

- **加载边型线分歧未定论**：Q4 在 n=16 与 n=64 上的型线都接近水平，Q8 在 n=12/24/48 上持续强倾斜，
  二者是同一边值问题的两套协调位移法，理论上必须收敛到同一解。审计未能在本单预算内定位该分歧
  （候选原因：加载边端点载荷间断的边界层、Q4 完全积分在弯曲主导区的剪切锁定、以及审计自身 Q4
  点式恢复的未验证性）。该项作为开放问题保留，**不得**据此宣称任一方「更准」。
- **点式恢复交叉校验失败**：审计用自写 `q4_strain_matrix` 复算 `cook_fields` 存盘的单元中心应力，
  相对差达数百 %，说明审计自身的点式恢复尚未通过验证；因此五点的 Q4 vs Q8 对照（JSON 中
  `comparison`）只能作为「分歧存在」的证据，不能作为 Q4 误差量。任何把该对照当作误差百分比的引用
  都是误用。
- **适用范围**：本审计只讨论线弹性平面应力、E=1、ν=1/3、厚度 1、单位合力剪切载荷下的 Cook 梯形膜；
  不涉及几何非线性、塑性、板壳或接触。采样点结论不覆盖夹持角点与加载边端点的峰值应力。
- **奇异区**：夹持边与自由边的交界 (0,44)、以及加载边端点的载荷间断处应力奇异/不连续，点式值随网格
  发散，任何「峰值应力误差 x%」的说法在这些位置都无意义。

## 后续建议（给下次自找活）

1. 用有限差分对 `field_stress`/`q4_strain_matrix` 的 B 矩阵做独立的数值微分校验（`u` 插值后中心差分），
   一次性判定 Q4 点式恢复与 Q8 应力回代孰对孰错——这是解开型线分歧的最短路径。
2. 分歧定位后，再把 `cook_fields` 的 `stress_accuracy_qualified` 与
   `scripts/validate_additional_benchmark_reports.py` 的硬性禁止条款一并改为「有证据才放行」的门禁。
