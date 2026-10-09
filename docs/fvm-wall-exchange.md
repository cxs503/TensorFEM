# FVM 实际壁面载荷到 FEM 的作用点映射

本轮读取 TensorFVM 新守恒动量后端实际 50 步 CPU/CUDA 案例的 96 个 staircase 壁面作用点。使用来源中的压力和黏性分力，分别及合计映射到 Q4 载体；保留法向、面积验证、作用点产生的力矩和虚功关系。

`read_wall_loads` 拒绝错误 SI 单位、重复力所有者、牵引/分力不一致、总力/力矩不一致、非法法向/面积及错误时钟。源压力在步末 0.25 s 求值，显式黏性在步初 0.245 s 求值，两个时间分别保留，没有伪装成同步瞬时应力。

## 验证

- 15 项加载与既有作用点接口测试通过。
- CPU/CUDA 两组实际载荷与 Q4 映射完整重放通过，保存源文件/原场 SHA256。
- 独立 NumPy 从双线性权重、偏置力偶及原始压力/黏性分力重建节点载荷、总力、原点力矩与功：最大力误差 8.44e-15 N、力矩误差 7.11e-15 N·m、功率误差 3.34e-16 W。
- 返回点速度与给定刚体虚速度场误差 <5.56e-17 m/s。

[来源和映射证据](assets/fvm-wall-exchange/study.json)，[独立审计](assets/fvm-wall-exchange/audit-report.json)。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/run_fvm_wall_exchange.py --fvm /path/to/TensorFVM --audit
python scripts/audit_fvm_wall_exchange.py --fvm /path/to/TensorFVM
```

## 范围

这是**实际牵引的单向接口验证**。载体承接力与偏置力偶，虚速度仅用于运动/功率检查，未回写 FVM、未解柔性壳体响应或声称双向 FSI。压力和黏性同属 fluid 力所有者，分量相加不会重复施力。

源 FVM 是 cell 动量/face 质量混合原型；中心/面速度仍不相容，尚未认证整体不可压输运、能量或 LES。黏度为合成 Re3900，不能当作校准海水。这里也没有 SUBOFF 自由液面或冰破坏。

下一步应先完成可信流体边界/运动与重启回滚，再把同样的空间载荷接入真实柔性壳体。
