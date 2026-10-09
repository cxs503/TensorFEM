# 保留作用点的外部载荷与运动接口

本接口将带位置的点力传给 Q4 壳节点，并输出同一映射下的点速度。它补充了此前仅凭合力重分配的接口，不替代接触搜索、真实压力积分或双向联算。

## 映射与约束

每个源点对选定的 Q4 承载面进行有界 Gauss–Newton 投影，得到形函数 N 和锚点 x。源点 p 不被投影位置覆盖，而以显式力臂 r=p−x 保留。节点平移载荷为 N F，转动载荷为 N(r×F)；返回速度为 ΣN(v+ω×r)。因此原作用点的合力、力矩和虚功均保持。

投影限制在每个单元自然坐标 [-1,1] 内；这是小型 CPU 承载面搜索，不是全局接触算法。用户必须指定最大允许力臂，超限即拒绝。折叠单元、非有限坐标、过期载荷、错误来源、载荷不一致及映射变异均拒绝。需要移动几何时必须重新构建映射，当前原始验证采用参考几何。

## 实际源数据验证

输入为 TensorDEM 新材料湿态驱动 `wet_12.json` 中实际保存的 24 个工具周界流体点力。读取流体作用在工具上的力，即 `−marker_force`，源位置和载荷保持时刻一起保存；冰接触力未冒充为本记录的一部分。

XY 正 y 显式映射到 XYZ 正 z，并将圆形工具中心注册到 SUBOFF 帆罩承载面中心。记录完整平移与嵌入轴，保持原点平移产生的力矩。**这只是载荷与承载面的明确注册，不是 SUBOFF 外流场或压力重建。** 原工具流体域仍采用可渗透点摩擦模型。

实际结果：

- 流体作用合力 z 分量约 0.596884 N。
- 力和源合力/力矩一致性残差约 1.11e-16，力矩残差约 2.17e-19 N m；虚功映射残差为 0。
- 最大承载面力臂约 1.5 mm，低于显式 80 mm 允许界限。
- 载荷采样时刻为 0.029 s，源状态结束时刻为 0.030 s，明确保留载荷保持区间。

返回速度用独立刚体运动探针检查，经过 XYZ→XY→XYZ 往返一致。探针和原流体工具速度最大差约 0.138 m/s，未将该速度回写流体；`feedback_applied_to_fluid=false`、`motion_probe_only=true`、`physical_accuracy_qualified=false`。本记录没有求解新的艇体应力或宣称闭环功率一致。

[原始记录](assets/point-load-exchange/case.json) 保存全部点力、点位置、承载单元、形函数、力臂、节点力/偶矩、探针速度和源文件哈希。7 项测试及源数据到映射/运动的完整重放审计通过。

## 复现

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/run_point_load_exchange.py \
  --source /path/to/TensorDEM/docs/calibrated-wet/wet_12.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/run_point_load_exchange.py \
  --source /path/to/TensorDEM/docs/calibrated-wet/wet_12.json --audit
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_point_surface_coupling.py
```

下一接口门是共同几何、实际运动回写与载荷更新时序，在闭环运行中重新验证冲量和外功。
