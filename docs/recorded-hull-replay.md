# 实际 LBM–DEM 载荷历史到 SUBOFF 船体的单向验证

本轮将之前实际运行的二维湿态 LBM–DEM 载荷历史接入 FEM 带附体 SUBOFF 线性壳模型，覆盖原来的 0.15 s 时间范围。载荷来自 TensorLBM `04b2474` 的 `docs/assets/ice-coupling/wet.json`，完整文件 SHA 和每个时间区间的力/力矩分解保存于证据中。

## 载荷与空间映射

水作用、冰接触和其他载荷分别读取，只相加一次，且必须与原总载荷一致。原 global_xy 正 y 映射至 FEM global_xyz 正 z；面外轴向力矩沿两根嵌入轴的叉积转换。嵌入原点明确选在参考帆罩样点面积重心，并执行原点平移的 `o × F` 力矩项。

原始载荷是圆形工具的合力/合力矩，没有 SUBOFF 表面压力场。因此使用面积加权最小范数分配，在帆罩上保持全部六个合力/合力矩分量。**该分配不能恢复真实压力分布，所得局部应力只是接口响应示例。** 没有把圆形工具与 SUBOFF 的几何差异隐藏为实际艇体破冰计算。

## 时间积分与几何

源历史每行代表前一个 0.5 ms 区间的载荷；采用区间常量，不将结束时刻误当作下一时刻的采样载荷。基准和时间减半计算均保留每个源区间，不压缩时间、不缩放载荷。

固定参考几何的线性模型采用隐式中点法：

`(M + dt² K / 4) v_mid = M v_old + dt (F − K q_old) / 2`。

`q_new = q_old + dt v_mid`，`v_new = 2 v_mid − v_old`，离散外功为 `F · (q_new − q_old)`。独立受迫振子测试验证二阶时间误差和离散能量恒等式。完整状态重启含质量/刚度哈希，拒绝同尺寸不同材料的恢复。

这是 CPU 稠密线性验证工具，不是可扩展生产求解器。冻结几何、小应变、固定载荷分配，不包含船体运动向流体/冰的反馈，也不包含自由液面或新冰材料。时间步对比只验证这次记录载荷下的敏感性，不认证真实碰撞应力。

## 实际计算结果

- 原载荷时长 0.15 s，基准 300 步，时间减半 600 步。
- 力/力矩分配残差约 3.23e-14；功率映射残差约 6.94e-18 W。
- 基准动量残差最大约 1.01e-10 kg m/s；离散能量相对残差约 4.78e-11。
- 基准示例峰值 von Mises 应力约 1.828 MPa；时间减半差异约 0.00742%。
- 最大位移约 0.183 mm，最大节点转角约 6.07e-5 rad，完整重启按位一致。

确切数值及门槛见 [study.json](assets/recorded-hull-replay/study.json)。两个计算保存完整最终位移、速度、应力和源区间历史；独立审计核对原文件哈希、源载荷分解、嵌入力矩、冲量、最终应力及能量。`physical_accuracy_qualified=false` 始终保留。

## 复现

在安装 PyTorch 的 Python 环境下，于 TensorFEM 根目录运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/run_recorded_hull_replay.py \
  --source /path/to/TensorLBM/docs/assets/ice-coupling/wet.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/validate_recorded_hull_replay.py \
  --source /path/to/TensorLBM/docs/assets/ice-coupling/wet.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q \
  tests/test_recorded_hull_loads.py tests/test_surface_coupling.py
```

下一接口阶段必须获得真实空间载荷、共同几何和运动反馈，再资格化双向流体–冰–艇体耦合。
