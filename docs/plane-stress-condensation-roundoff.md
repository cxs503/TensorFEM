# 平面应力 σ_zz 凝聚的舍入级收敛判据（issue #8 收口）

状态：**已修复**（`src/tensorfem/layered_shell4_plasticity.py::plane_stress_j2_update`）
证据：`docs/assets/plane_stress_condensation/evidence.json`、
`docs/assets/plane_stress_condensation/condensation_roundoff.png`、
`scripts/audit_plane_stress_condensation.py`、`tests/test_plane_stress_condensation_roundoff.py`

## 1. 症状

分层 Shell4 的材料点在局部用标量 Newton 迭代求 ε_zz，使 σ_zz = 0（平面应力凝聚）。
连续化（arc-length）求解器在**未收敛的 Newton 试算步**上会给出巨大的膜应变
（本例 `ε = (3.369, 10.019, 1.115)`，即 1000% 量级的试探值，属正常的中途迭代）。
此时局部凝聚抛出

```
RuntimeError: plane-stress local condensation did not converge
```

并把整个求解链路打断——而这条试算路径本来只需要"保持有限、继续回溯"。三处测试因此
长期挂着 `xfail`，理由写成"machine-dependent（不同 CPU/BLAS 的归约顺序）"，掩盖了真实
机理：`tests/test_shell4_arc_line_search.py`、`tests/test_v1_evidence_aggregator.py`（两条）。
另有 `tests/test_finite_rotation_layered_shell4.py` 的一条 xfail 属**另一问题**
（应力自由参考的绝对阈值 1e-10），本单未动。

## 2. 根因（程序化复现的迭代史）

局部 Newton 的接受判据是**绝对阈值**：

| 判据 | 旧值 | 本例能否达到 |
|---|---|---|
| `abs(sigma_zz) <= 1e-15 * max(E, σ_y, 1)` | 2.0e-10 | 否：残差 4.86e-10 / 5.71e-10 交替 |
| `abs(step) <= 1e-17 * max(1, |ε_zz|)` | 1.4e-16 | 否：步长停在 ε_zz 的 1 ulp ≈ 3.6e-15 |

σ_zz 是 O(E·ε)≈O(1e4) 量级应力的差（K·tr(ε) 与偏量部分相消），双精度下可达的残差下限
约 `eps·|σ| ≈ 2.2e-16 × 2e4 ≈ 4e-12`；旧阈值 2e-10 只比该下限松 50 倍，一旦应变放大到
10⁻¹ 量级就**低于可达精度**，迭代在两个相邻浮点数之间来回振荡，40 次耗尽后抛错。
复现的迭代史（`evidence.json` 的 `old_residual_history`）：

```
it=1 zz=-13.312457 resid=7.43e+00
it=2 zz=-13.312501 resid=-4.86e-10   ← 已在工作精度内
it=3 zz=-13.312501 resid= 5.71e-10   ← 振荡，永远不满足 2e-10
...
it=40 残差仍在 ±5e-10，抛 RuntimeError
```

## 3. 修复（最小改动，物理不变）

把两个绝对阈值改为**按舍入水平与参与相消的应力量级缩放**：

```python
unit_roundoff = torch.finfo(strain.dtype).eps   # 2.22e-16 (float64)
accept = 256. * unit_roundoff                   # 5.7e-14 相对
stress_scale = max(max(E, σ_y, 1), float(sigma.abs().max()))
abs(residual) <= accept * stress_scale          # 应力残差接受
abs(step)     <= accept * max(1., abs(zz))      # 步长落到 ulp 时接受并重算一次
# 新增 fail-closed：dσ_zz/dε_zz 非有限或 ≤ 0 时仍抛错
```

同时保留原有 fail-closed 语义：40 次未达判据、或 σ_zz 切线非有限/非正，仍抛 `RuntimeError`。
返回的应力/切线/状态一律由**收敛点重算**，不引入任何近似分支。

## 4. 结果与误差判定

| 量 | 旧实现 | 修复后 | 独立基准 |
|---|---|---|---|
| 局部迭代次数（振荡例） | 40（抛错） | 3 | — |
| 最终 |σ_zz| | 5.71e-10（未接受） | 4.86e-10（接受） | 独立 200 步二分根：4.86e-10 |
| 接受阈值 | 2.0e-10 绝对 | 1.14e-8 绝对（= 5.7e-14 × |σ|_max） | — |
| ε_zz 根 | — | -13.312501442657762 | 二分根 -13.312501442657762 |
| 平面应力路径 σ_y(ε_y) 与二分参考最大偏差 | — | 4.92e-11 MPa | 应力 O(1e3) MPa |

**是否 < 3%**：是。关键量是凝聚后的面内应力，与独立二分根的对差为 `4.92e-11 MPa`
对 O(1e3) MPa 应力，即 **5e-14 相对**，远优于 3%；σ_zz 残差 4.86e-10 相对应力幅值 2e-14。
不足 3% 的判据不靠自述：由 `tests/test_plane_stress_condensation_roundoff.py` 的
10 组应变×硬化参数独立二分复核锁死（`1e-9` 相对门限）。

测试面：原先 3 条 `xfail`（arc-length 1 条 + v1 证据 2 条）在修复后**实测通过**，
已删除过期标记（`--runxfail` 下 2 passed / 2 passed，见提交说明），新增 13 条材料点级用例。

## 5. 注意点与适用边界

- 放大容差**不是**放宽物理精度：接受阈值相对应力幅值恒为 5.7e-14，远小于任何工程容差；
  它只避免"要求低于浮点可达精度"的无解判据。
- 该修复使**未收敛的试探迭代**不再中断求解，但不会让未收敛的载荷步变成收敛步：
  收敛判据（残差范数）仍在求解器层，arc-length 的 `converged` 语义未改。
- 不能推广为"所有材料点迭代都可松容差"：`plasticity.py` 里 3D 回映映射的切线仍是解析式，
  与此处的标量根求混用需分别设判据。
- 若将来切换到 float32，`accept` 会自动随 `finfo.eps` 放大（~1.2e-7 相对），
  但 float32 下 shell 塑性路径不应作为验收证据。