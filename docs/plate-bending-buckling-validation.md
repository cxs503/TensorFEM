# 线性边缘应力梯度下的板格屈曲（非均匀预屈曲应力闭环）

`marine_plate_buckling`（单向压缩，Navier 单模态精确）与 `plate_shear_buckling`
（面内剪切 / 压剪组合）只覆盖**均匀**面内应力；两个模块的 scope 段与
[benchmark-qualification-status.md](benchmark-qualification-status.md) 的
"尚未闭环的工作"都把 *非均匀预屈曲应力* 列为欠账。本文档关闭该欠账中与船体结构
最直接的一类：**线性变化的边缘应力梯度**，即船体梁总纵弯曲把弯曲应力梯度叠加到
板格名义压应力上的情形。

实现：`src/tensorfem/plate_bending_buckling.py`；
证据：`benchmarks/plate_bending_buckling.py` → `benchmarks/plate_bending_buckling_evidence.json`；
测试：`tests/test_plate_bending_buckling.py`（36 项）。

## 1 载荷定义

四边简支、等厚、单位宽度 `b` 的平板，面内载荷沿宽度线性变化

```
N_x(y) = N1 [1 + (psi - 1) y / b],     -1 <= psi <= 1,   N1 > 0 为压
```

| psi | 物理情形 | 长板古典系数 `k` |
|---:|---|---:|
| `+1` | 均匀压缩 | 4.00 |
| `0` | 三角形压缩（一边零应力） | 7.81 |
| `-0.5` | 压-弯组合（另一边半拉） | — |
| `-1` | 纯面内弯曲（半边受拉） | 23.9（`a/b>=2/3`）；25.6（`a/b=1`） |

`k` 一律以**峰值边缘压应力**为参照：`N_cr = k pi^2 D / b^2`。

## 2 为什么必须换一套基

对均匀压缩，`w = sin(mπx/a) sin(nπy/b)` 使 y 方向正交、几何矩阵对角，单模态即精确。
线性梯度破坏可分离性： destabilising 能量 `∫ N_x(y) w,x^2` 的 y 耦合为

```
G_nq = b [ delta_nq / 2 + (psi - 1) ∫_0^1 xi sin(n pi xi) sin(q pi xi) d xi ]
∫_0^1 xi sin(n pi xi) sin(q pi xi) d xi = -(1/pi^2) [ 1/(n-q)^2 - 1/(n+q)^2 ]    (n+q 奇数)
```

两项事实（均有测试锁死）：

* 非对角项**只在 `n+q` 为奇数**时非零，所以临界模态必然是奇偶混合；
* `psi = -1` 时**整个对角恒为零**（`1/2 + (-2)(1/4) = 0`），
  即单模态 Navier 解在纯弯曲下根本给不出屈曲载荷。这正是必须用耦合基的原因。

## 3 四条验证腿

1. **文献锚点**：Timoshenko & Gere, *Theory of Elastic Stability* (1961), Table 9.1
   （亦见 Allen & Bulson, *Background to Buckling*, 1980）；
2. **退化极限**：`psi = 1` 必须由同一套通用装配精确回到 `k = 4`
   （强内一致性门，实测舍入级 `2.2e-16`）；
3. **独立积分**：对同一能量泛函做 2D Gauss–Legendre 数值积分，
   不复用任何闭式积分（实测相对差 `2.7e-15` / `1.8e-15`）；
4. **独立离散化**：强形式 `D lap^4 w + N_x(y) w,xx = 0` 的有限差分解，
   除板模型本身外不共享任何东西。

## 4 结果

| 工况 | psi | a/b | 文献 k | 计算 k | 相对误差 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| uniform_limit | +1 | 6 | 4.00 | 4.000000 | 2.2e-16 | 通过（舍入级） |
| pure_bending_long_panel | -1 | 6 | 23.9 | 23.8818 | 0.076 % | 通过 |
| pure_bending_square | -1 | 1 | 25.6 | 25.5283 | 0.280 % | 通过 |
| triangular_long_panel | 0 | 6 | 7.81 | 7.8120 | 0.025 % | 通过 |

独立腿：

| 检验 | 结果 | 门限 |
|---|---|---|
| 有限差分 `psi=-1` 方板（div=32，1024 未知量） | `k = 25.4440`，vs 变分 0.330 % | 5 % |
| 有限差分 `psi=+1` 方板（同一节点求解器） | `k = 3.9968`，vs 4.00 0.080 % | 5 % |
| 闭式矩阵 vs 数值积分 | `2.7e-15`（弯曲）/ `1.8e-15`（几何） | `1e-10` |
| 基 12 → 16 漂移 | `6.6e-9` | `5e-3` |
| `k(psi)` 单调性 | `4.000 → 5.319 → 7.812 → 13.397 → 23.882` | 单调 |

后两条的物理含义：以峰值压应力为参照，梯度越"弯"越不危险；但注意这只是
**弹性临界载荷**的结论，含缺陷/后屈曲的极限强度不受此保证。

图件：`docs/assets/plate_bending_buckling/k_vs_psi.png`（`k(psi)` 与三个锚点）、
`docs/assets/plate_bending_buckling/mode_and_aspect.png`（方板纯弯曲临界模态场与
`k_b(a/b)`）。

## 5 两个实现陷阱（可复用）

1. **等节点数网格欠分辨**：有限差分若令 `nx = ny`，长板的 x 向按长宽比被欠分辨，
   网格能承载的半波数被卡死，系数会系统性算错（本单先是得到 `k=8.58/7.50/6.70`
   而对 4.00 不收敛）。API 因此以 `divisions`（每宽单元数）生成**等尺寸**网格。
2. **归约辅助函数的对角前提**：`plate_shear_buckling` 的
   `_smallest_positive_from_symmetric_indefinite` 用两次 `solve` 实现
   `R = F^-1 G F^-T`，实际得到 `F^-T G F^-T`；只有在 `F` 为对角时二者才等价。
   正弦基下弯曲矩阵恒为对角，故该模块自身正确；对非对角的节点弯曲矩阵会给出
   错误特征值（本单若不换求解器，`psi=+1` 方板得 `k=5.69` 而非 4.00）。
   本模块自带 `_smallest_positive_generalized`，并在 docstring 记录该前提。

## 6 适用范围（不覆盖）

* 平板、等厚、四边简支、无面内剪切；`k` 以峰值边缘压应力为参照；
* **不覆盖**：壳单元几何刚度、离散加强筋的 tripping、后屈曲（有效宽度 / 张力场）
  强度、塑性、初始缺陷敏感性、`D16/D26` 弯扭耦合、压 + 剪 + 梯度三分量组合、
  双向梯度载荷、峰值应力位置的逐点场量认证。

正交异性（纵向加筋等效刚度）情形只报自洽趋势（同 `plate_shear_buckling` 的既有
做法）：加筋提高临界载荷，但以纵向刚度归一后的 `k` 不一定提高，故入库证据只记录
载荷比较，不宣称文献对标。

## 7 复现

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/plate_bending_buckling.py
OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_plate_bending_buckling.py -q
```

注意：本机 BLAS 若按默认线程数运行，在负载较高时会因线程超订把该脚本拖到超时；
复现时固定 `OMP_NUM_THREADS=1`。