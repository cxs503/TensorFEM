# 艇体表面载荷与运动接口

本轮交付是车辆壳体的接口和线性动力学验证。没有在 FEM 中装配冰层，也未把合成载荷标记成 CFD/DEM 计算结果。完整四项目耦合、有限转动壳动力学和真实湿态精度仍待验证。

## 几何、单位与力方向

`QuadSurfaceExchange` 使用当前壳中面节点，global_xyz，SI 单位，每个 Q4 四个 Gauss 样点。牵引力为作用在 FEM 艇体上的 Pa，乘样点面积后才得到 N。正压力沿连接法向的反方向。

样点面积是中面面积；`face_offset_m` 定义力臂，不代表重建真实曲面外皮面积。偏置产生的力矩以 `r × F` 传给节点转动自由度，速度使用 `v + ω × r`。同一形函数的转置同时传递力与速度，因此总力、关于同一原点的力矩和虚功守恒。

当前适配器在 CPU 上运行；输入复制到 CPU。几何更新必须重建映射。折叠 Q4、错误单位、非有限数据、过期时刻、不匹配几何哈希和被修改的映射均拒绝。几何哈希不认证结构精度。

`PlanarEmbedding` 显式声明二维 DEM/LBM 坐标在三维中的原点和两条正交轴。例如 axes=[[1,0],[0,0],[0,1]] 将二维正 y 映射为三维正 z；没有隐式翻转。反向载荷投影拒绝丢失面外分量。二维展向厚度仍需在力/质量/面积转换时一致声明。

## 物理归属

- 联合模型由 TensorDEM 拥有冰力学；现有 FEM 黏聚冰算例作为独立参考。
- 不允许在同一区域同时装配两套冰质量、刚度和断裂。
- 不允许把已解析的水动力和未声明分区的浮力/附加质量/阻力简化项重复计入。
- TensorLBM/TensorFVM 来源标签仅说明接口来源，不自动赋予物理验证资格。

## 艇体响应

`LinearHullReceiver` 仅装配 SUBOFF 艇壳、围壳与四舵的线性壳矩阵和物理质量。外部载荷在每个显式步内保持常量；步长由保守谱界限制，运动、外功和冲量累计保存在完整 JSON 重启中。运动输出携带时间、位置、速度、法向、面积、样点编号和几何哈希。

此响应只适用于小应变、小转动；没有把现有有限转动模块的独立能力冒充为本适配器能力。当前恒载步接口不包含耦合子迭代或任意历史插值。

## 实际验证与重现

解析矩形牵引、刚体旋转速度、旋转协变、偏置力矩及虚功有独立测试。车辆算例将合成 10000 Pa 的竖直牵引施于围壳，保存实际动态响应；基准与减半时间步均记录力/力矩/功映射、动量、离散能量和应力，全状态重启逐位一致。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/run_hull_surface_coupling.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/validate_hull_surface_coupling.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_surface_coupling.py
```

[机器可读结果](assets/hull-coupling/study.json) 和原始场记录明确保留 `physical_accuracy_qualified=false`。这不是完整 CFD–DEM–FEM 联算。
