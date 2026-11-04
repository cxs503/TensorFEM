# Issue #8 结清：有限转动分层 Shell4 的「零应力参考」门限改为刚度导出舍入底噪

**结论**：`tests/test_finite_rotation_layered_shell4.py` 里那条
`strict=False` 的 `xfail`（issue #8）已删除。它掩盖的不是「机器相关的浮动误差」，
而是三个互相矛盾的门限口径：

| 位置 | 原判据 | 本机实测残差 | 余量 | 性质 |
|---|---|---:|---:|---|
| 浅翘曲自由参考 `stress` | `< 1e-10` | 4.149e-11 | 2.4× | 过紧，换个 BLAS 就红 |
| 浅翘曲自由参考 `force` | `< 1e-10` | 4.150e-11–1.66e-12 | 2.4×–60× | 同上 |
| 带初始翘曲缺陷参考 `stress` | `< 1e-10` | **1.757e-10** | **0.57×（已越界）** | 严重超门——同一构建族里另一条测试只是还没红 |
| 刚体转动后参考 `force`/`stress` | `< 4e-7` | 4.86e-12 / 6.58e-11 | ~6e3×–1e4× | 过松，放大 1e4 倍仍绿 |

`xfail(strict=False)` 的副作用是**这条断言永远不可能失败**：真出现客观性回归时，
pytest 只会把它记成「预期失败」。这正是 TensorDG 侧记录过的「判据无牙齿」问题在
TensorFEM 的同一形态。

## 1. 案例与判据

被检验对象是已合并的有限转动分层壳单元
`assemble_finite_rotation_layered_shell4`（corotational 框架 + 2×2×层内 J2
平面应力）。物理命题只有一句：**整体刚体运动必须产生零应力、零内力**。

- 几何：4 节点单个 Shell4 面元，边长 1 m，厚度 t = 0.08 m，3 层。
- 参考构型：平面（精确零）与浅翘曲 `z = [0, 0.012, −0.007, 0.004]` m。
- 材料：E = 2×10⁵，ν = 0.3，屈服 250，硬化 1400（仅用于 `LayeredShell4Model`）。
- 刚体运动：轴 (0.2, −0.7, 0.4)/‖·‖，转角 0.83 rad，平动 (0.6, −1.2, 2) m。
- 判据：`max|stress| < floor.stress` 且 `max|force| < floor.force`，两个量都是
  由模型自身刚度标度导出的舍入底噪，不是手挑的绝对值。

## 2. 方法：把门限写成舍入量的函数

对刚体运动，corotated 变形量 q 恒为 0，但装配过程仍在乘加量级为 E 的量
（膜刚度 `E t/(1−ν²)`、弯刚度 `E t³`、以及只有 ε 精度的 SVD 框架）。
所以一个**数值上正确**的实现返回的不是精确零，而是 `O(ε E)` 的应力残差与
`O(ε E t A)` 的内力残差。新增公开工具函数
`roundoff_floor(model, safety=64)`（`src/tensorfem/finite_rotation_layered_shell4.py`）
就按这两条标度给出门限，同时导出 `shell_section_area(model)` 作为内力标度中的面积项。

为什么 `safety = 64` 不是「随手放宽」：

1. **残差确实随 E 线性缩放**（纯舍入的签名）。同一浅翘曲参考在
   E = 2×10⁴…2×10⁷ 上测得残差 4.149e-12 / 4.149e-11 / 4.149e-10 / 4.149e-9，
   即 `stress/(εE) = 0.934` 恒定不变；十倍刚度点上比值 10.000000（相对差 < 1e-9，
   由 `test_roundoff_floor_is_stiffness_scaled_and_bounds_the_free_reference` 锁死）。
   若残差来自离散误差，它不会随 E 变化。
2. 实测归一化比值 0.93（浅翘曲自由参考）、1.48（刚体转动后），
   内力项 ≤ 0.037（`force/(ε E t A)`）。safety = 64 给出 43–68× 余量。
3. 门限仍比可分辨的物理量低 12 个数量级：同一算子能把叠加在刚体运动上的
   **1e-12 膜应变**解成 1.533e-7 应力，即 54× 底噪、3.4e3× εE。

## 3. 计算结果（程序化生成，非手抄）

`docs/assets/finite_rotation_roundoff_gate.png`（左：归一化残差 vs E；右：门限牙齿）。

刚体运动上叠加膜应变 ε̄ 时的最大应力（E = 2×10⁵）：

| ε̄ | max\|stress\| | / 底噪 2.842e-9 |
|---:|---:|---:|
| 0 | 6.583e-11 | 0.023 |
| 1e-14 | 1.521e-09 | 0.535 |
| 1e-13 | 1.545e-08 | 5.4 |
| 1e-12 | 1.533e-07 | 54 |
| 1e-11 | 1.533e-06 | 539 |
| 1e-10 | 1.533e-05 | 5393 |

## 4. 新判据与负控（都在测试里，可复算）

1. `test_roundoff_floor_is_stiffness_scaled_and_bounds_the_free_reference`
   —— 两条参考残差都 < 底噪；底噪/E = 64ε < 1e-12（是舍入声明而非放宽容差）；
   残差与 `force/(ε E t A)` 在 E→10E 上比值均为 10.000（rel 1e-9）。
2. `test_objectivity_gate_keeps_its_teeth_below_the_old_absolute_threshold`
   —— (a) 1e-12 膜应变必须被解出（> 20× 底噪，实测 54×）且 < 1e-6（是物理不是舍入）；
   (b) 1e-6 rad 的转角失配必须被**内力门**抓到（实测 5.05e-3 vs 底噪 2.27e-10，
   即 2.2e7×），并记录应力门看不到该失配（此时应力仍 ≈ 底噪）。
3. `test_free_reference_gate_fails_on_a_non_objective_operator`
   —— 实现级负控：monkeypatch 给 `_corotated_deformation` 注入 1e-9 的虚假变形
   （相当于客观性 bug 泄漏），零参考残差必须越门（> 1e3× 底噪才通过）。
   旧 `xfail(strict=False)` 会把这种回归静默记成 xfail，现在它是红。

## 5. 注意点与局限

- 底噪常数（0.93 / 1.48 / 0.037）是 float64 + 本机 BLAS 的测量值；safety = 64
  的吸收能力约 40–70×，远大于跨构建的舍入差异（通常 < 3×），但不是数学上界。
  若将来装配路径引入更多近消去（例如数值 Jacobian 路径成为默认），比值可能上升，
  此时应重新测 `stress/(εE)` 而不是盲目调大 safety。
- 底噪只覆盖「参考构型 + 刚体运动」这一类零应力命题。它**不能**替代真实载荷下的
  精度验收，也不认证有限转动面的物理范围（仍为 finite-rotation/small-strain）。
- `shell_section_area` 用对角线叉积，对平面四边形精确、对翘曲面元只是 O(面积) 标度，
  仅用于内力门限，不参与任何刚度计算。
- 本轮未改动任何求解器数值路径（只新增工具函数与测试口径），因此不需要重新认证
  已有算例；`_corotated_deformation` / 装配实现未变。
- 运行环境注意：本机 `site-packages` 里 `tensorfem` 指向别的 worktree
  （`/root/tp-agent/work/m3182/src`），直接 `python3 script.py` 会 import 到旧代码；
  `pytest` 因 `pythonpath = ["src", "."]` 正常。跑脚本请显式
  `PYTHONPATH=<this worktree>/src`。