# MITC 剪切场斜网格不变性（缺陷修复记录）

## 缺陷

`src/tensorfem/plate.py` 的 `q4_mindlin_stiffness(..., shear_treatment="mitc")`
在单元四条边的中点采样**物理**横向剪应变 γ_xz / γ_yz 后线性插值。该写法只在矩形
（J 对角）上与 Dvorkin–Bathe 假定自然剪应变场等价。平行四边形上任取精确的
"常曲率、无剪切"状态，插值后的 γ 不再为零，产生寄生剪切能：

| 斜角(°) | 修复前 mitc 寄生剪切能/弯曲能 | 同期 sri |
|---|---|---|
| 90 | 7.86e-15 | 9.17e-15 |
| 80 | 2.49e-01 | -4.60e-15 |
| 70 | 1.06e+00 | -2.53e-14 |
| 60 | 2.67e+00 | -2.56e-14 |
| 45 | 8.01e+00 | -8.35e-14 |
| 30 | 2.40e+01 | -2.75e-13 |
| 20 | 6.05e+01 | -7.19e-13 |

（修复前数据存于 `benchmarks/verified/fem_morley_skew_plate/artifacts/prefix_mitc_skew_defect.json`，
在父提交 df250c73 上测得；修复后该列不可再生成。）

工程后果：所谓"无锁死"的 MITC 在斜网格上比 SRI 锁死更严重——Morley 30° 斜板
（L/h=1000, n=32）中心挠度只有 0.0155（参考 0.41，差 26 倍），而 SRI 给出 0.384。

## 修复

改为协变分量采样：γ_ξ = w,ξ + x,ξ θx + y,ξ θy，在 (ξ=0, η=±1) 采样后按 η 线性插值；
γ_η 同理在 (ξ=±1, η=0) 采样后按 ξ 插值；再用局部 J⁻¹ 变换回物理分量
（γ_cov = J γ_phys），2×2 高斯积分。矩形上与旧写法逐点等价（J 对角）。

## 证据（`gates_verified.json` G9 与测试）

修复后任意斜角（90…20°）寄生剪切能比：

| 斜角(°) | 后向不变性残差 |
|---|---|
| 90.0 | 3.10e-15 |
| 75.0 | -2.68e-14 |
| 60.0 | -3.78e-14 |
| 45.0 | -8.93e-15 |
| 30.0 | -8.57e-15 |
| 20.0 | -2.21e-13 |

* 零能模态数保持恰 3（无伪零能机构），SRI 仍为 5（2 个伪机构，如其文档所述）。
* `tests/test_plate_mitc_skew_invariance.py` 断言 6 个斜角上寄生剪切能 < 1e-10 相对、
  秩恰为 3、刚体模态零能；`tests/test_bending_patch.py` 中原来"斜 patch 上 MITC 有
  非零伪剪切能（<2%）"的记录性断言升级为"精确为 0"。
* 环形回归：`tests/test_plate_mitc_skew_invariance.py tests/test_bending_patch.py
  tests/test_element_rank_audit.py tests/test_plate_bfs.py` → 58 passed。
