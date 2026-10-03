# 多构件船体梁渐进强度资格原型

`hull_girder_progressive` 用可审计的截面部件模型追踪弯矩—曲率路径。板、纵骨或其他纵向构件以面积和形心高度表示；求解器采用平截面假设，并在每个曲率点求解零轴力对应的轴向应变。

部件应力为理想弹塑性。拉伸上限为材料屈服强度；受压上限可另外指定为较低的 `compression_strength`，用于表达已由独立板屈曲计算确定的降效强度。报告保留每个部件的应变、应力、轴力、弯矩贡献和状态（弹性、拉伸屈服、受压屈服或屈曲限幅）。

## 资格案例

内置案例含板、纵骨、舷侧和内构件共四对镜像部件。独立参考值直接按镜像部件闭式求和：

`M(kappa) = sum(A_i * min(sigma_yi, E_i*kappa*abs(y_i)) * abs(y_i))`。

资格门禁要求整条 M–κ 曲线归一化误差严格小于 3%，并同时检查轴力平衡、弯矩单调、外功非负单调以及多个部件事件。测试另设非对称屈曲降效案例，检查承载路径下降、屈曲状态可追溯及轴力平衡。

```python
from tensorfem.hull_girder_progressive import HullComponent, solve_component_section

parts = [
    HullComponent("deck", 0.02, 1.0, 210e9, 355e6),
    HullComponent("bottom", 0.02, -1.0, 210e9, 355e6, 220e6),
]
curve = solve_component_section(parts, [0.0, 5e-4, 1e-3, 2e-3])
```

## 能力边界

这是 TensorFEM 截面级资格原型，不是完整船级社渐进破坏方法。屈曲强度是输入的部件限幅值，而不是本模型内的壳屈曲解；当前不含壳体失稳相互作用、初始缺陷、残余应力、卸载、循环损伤、断裂或空间有限元。它不依赖也不调用 TensorLBM。
