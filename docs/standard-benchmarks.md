# 标准标模与结果字段

公共闭式标模位于 `tensorfem.standard_benchmarks`，用于校验求解器和后处理，而不是替代有限元计算。

```python
from tensorfem.standard_benchmarks import (
    cantilever_beam, cantilever_beam_field,
    simply_supported_plate_center, hertz_contact_force, convergence,
)
print(cantilever_beam())
print(cantilever_beam_field(0.5, 0.1))  # 位移场与弯曲应力场样本
print(simply_supported_plate_center())  # Navier 中心挠度
print(hertz_contact_force(1e-4))        # Hertz 力-压入量参考曲线
print(convergence([(4, 1.0), (8, 1.1)], 1.2))
```

每个标模应在算例报告中记录：单位、边界条件、网格规模、位移/应力场、反力、参考值、相对误差和网格收敛趋势。接触、屈曲和极限强度算例必须额外记录接触压力或载荷-位移峰值；任何未达到 3% 门槛的结果必须标记为 `blocked`。
