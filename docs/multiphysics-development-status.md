# 多物理场开发状态与验收记录

本轮完成公共接口和首个二维双向耦合验证。TensorLBM–TensorDEM 已实际交换载荷与运动；TensorFEM 船体接收器和 TensorFVM 通道验证分别运行。四个项目尚未组成统一的三维湿态上浮破冰求解器，物理精度尚未认证。

## 已完成的交付

| 项目 | 实际交付 | 验证 |
| --- | --- | --- |
| TensorDEM | SI 状态/载荷接口、冲量与离散能量账本、碎片统计、完整重启 | 39 项测试及 12 个子测试；均匀载荷时间加密呈一阶误差下降 |
| TensorLBM | D2Q9/BGK 与 DEM 双向交换、保守插值/分配、子步、交换中途重启、六个实际算例 | 70 项测试；六组原始场与源文件哈希审计通过 |
| TensorFVM | SIMPLE/MAC 通道三网格验证、独立原始场审计和 LBM 参考 | 12 项测试；三网格解析验证通过 |
| TensorFEM | 当前 Q4 表面力/力矩/虚功映射、显式二维到三维嵌入、带附体 SUBOFF 线性船体载荷接收器 | 12 项新测试加 27 项既有测试通过；两组动态原始场审计通过 |

### 二维湿态耦合

干态峰值 6.818504 N，湿态峰值 11.162610 N。时间步减半的峰值变化为 0.09654%，交换间隔减半为 0.11697%。颗粒加密变化为 **10.16594%，未达到 3% 判据**；弹簧参数未随颗粒重新标定，不能作为连续体网格收敛结论。

水和工具作用、冰与工具接触分别记账。最大线动量残差约 1.48e-11 kg m/s；湿态 DEM 离散能量残差为 0.001624 J，界面离散功残差为 0.000209 J，均未隐藏。重启后的轨迹按位一致。

该算例为周期全液体域，采用人工运动黏度 0.02 m²/s。点摩擦正则化界面允许滑移，液体仍存在于颗粒内部；它没有解析不可渗透边界、自由液面、重力浮力或真实海冰标定。不能视为实际海水破冰预测。

### FEM 船体接口

带附体 SUBOFF 的帆罩施加人为指定的 10000 Pa 向上牵引，真实组装壳体刚度并积分响应。力映射残差不超过 2.28e-13 N，力矩残差不超过 1.14e-13 N m，功率残差不超过 1.12e-16 W。时间步减半后的最大应力变化为 0.9951%，重启按位一致。

此处没有流体和冰求解器参与，属于船体接口验证。表面面积是中面面积，厚度偏置只提供力臂；小应变壳接收器不能用于大转动碰撞预测。详见 [表面接口](surface-coupling-interface.md) 和 [计算记录](assets/hull-coupling/study.json)。

### 流体独立验证

FVM 的 36×12、72×24、144×48 通道速度相对 L2 误差分别为 0.634358%、0.161971%、0.042848%；压力梯度误差分别为 1.369853%、0.344019%、0.078365%。

LBM 最细 H=48 通道在 30000 步内未满足稳态判据，保留失败记录。FVM 和 LBM 的入口条件、域长和数值精度不同，只分别对同一解析解验证，不能作为完全相同边界条件下的直接比较。FVM 移动冰边界尚未验收。

## 版本与复现

本次独立项目版本及详细记录：

- [TensorDEM fe724a0](https://github.com/cxs503/TensorDEM/tree/fe724a0b3bd7099a2f0269bb2d3ba8738305616b)，`docs/coupling/interface.md`。
- [TensorLBM 04b2474](https://github.com/cxs503/TensorLBM/tree/04b247489b87c7f38a4cd95ae64d79903556cebf)，`docs/tensordem_coupling_p1.md`。
- [TensorFVM 54d1aa3](https://github.com/cxs503/TensorFVM/tree/54d1aa3617855b6b629a853b44e1a55836696f80)，`docs/suite-channel/README.rst`。

统一使用 SI 单位。DEM/LBM 为 global_xy；FEM 为 global_xyz，必须显式指定嵌入平面。载荷携带采样时间、保持区间和几何版本；冰的所有权只允许一个求解器持有。解析流体参与时，不得重复加入同一水作用的简化载荷。

在安装 PyTorch 的 Python 环境下，设 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。在各项目根目录执行以下原始场审计：

```bash
# TensorLBM，DEM_SRC 指向 TensorDEM/src
PYTHONPATH="src:${DEM_SRC}" python examples/ice_coupling/audit_evidence.py
# TensorFVM
PYTHONPATH=src python scripts/validate_suite_channel.py
# TensorFEM
PYTHONPATH=src python scripts/run_hull_surface_coupling.py
PYTHONPATH=src python scripts/validate_hull_surface_coupling.py
```

各项目的详细记录包含完整生成参数和测试命令。这里的通过表示明确列出的数值/接口检查通过，不表示工业软件等级或实验精度达标。

## 下一阶段的执行顺序

1. 先解决有限尺寸冰颗粒的流体边界：检验不可渗透性、滑移和载荷，再开展流体网格、域尺寸、界面系数及颗粒加密；同时收敛离散能量残差。
2. 标定冰的刚度、破坏与耗散，加入需要的转动和邻域搜索；独立验证弯曲与断裂，再扩展三维。
3. 将真实耦合载荷接入 FEM 船体，先做刚体湿态接触，再做柔性船体；逐项通过载荷、运动和重启验收。
4. 在自由液面、浮力及冰材料验证后开展三维带附体 SUBOFF 上浮破冰。Glacier 几何数字化及试验对比保留为独立路线。

既有 FEM 冰模型作为独立参考，避免与 DEM 重复计算冰体。本轮没有完成上述后续阶段。
