# 加筋板格极限强度与 Smith 法 M--chi 验证面

Status: **evidence package（含一处 honest FAIL）**. 模块
`tensorfem.panel_mchi_qualification`；证据由
`benchmarks/panel_mchi_qualification.py` 生成到
`artifacts/panel_mchi_qualification/evidence.json`，本文件的全部数字由
`benchmarks/panel_mchi_qualification.py --doc` 从该 JSON 注入，无手抄值。

物理背景：船体总纵崩溃分析（Smith 法）逐单元消费板格的弯矩--曲率关系。本
包提供板格级 $M$--$\chi$ 面（中性轴逐增量再平衡、超过极限的纤维卸载、能量
账本、极限点唯一性），并与 TensorFEM 的梁/板特征值、弹塑性 Shell4 弧长
路径互证。

### 板格截面与 M--chi 极限状态（M_u 单位 kN m）

| 案例 | beta | lambda | sigma_u/sigma_y (chain) | M_u plateau | chi_u (1/m) | M_u/M_p | M_u softened | energy residual |
|---|---|---|---|---|---|---|---|---|
| square_bay_tee | 2.25 | 0.15 | 0.7553 | 73.63 | 0.04364 | 0.9834 | 66.63 | 1.01e-05 |
| square_bay_flatbar | 2.25 | 0.19 | 0.7490 | 50.20 | 0.04161 | 0.9695 | 45.02 | 3.56e-06 |
| ice_belt_deep_web | 1.35 | 0.47 | 0.9345 | 86.22 | 0.03754 | 0.9359 | 78.04 | 1.42e-05 |

### 增量细化（曲率步长无关性）

| 案例 | 步数 | M_u (kN m) | chi_u (1/m) | 能量残差 | M_u 相对差 |
|---|---|---|---|---|---|
| square_bay_tee | 100 | 73.6297 | 0.043640 | 5.59e-05 | 0.00e+00 |
| square_bay_tee | 200 | 73.6297 | 0.043640 | 1.01e-05 | 0.00e+00 |
| square_bay_tee | 400 | 73.6297 | 0.043640 | 2.81e-06 | 0.00e+00 |
| square_bay_tee | 800 | 73.6297 | 0.043640 | 9.12e-07 | 0.00e+00 |
| square_bay_flatbar | 100 | 50.1982 | 0.041611 | 4.02e-05 | 0.00e+00 |
| square_bay_flatbar | 200 | 50.1982 | 0.041611 | 3.56e-06 | 0.00e+00 |
| square_bay_flatbar | 400 | 50.1982 | 0.041611 | 1.43e-06 | 0.00e+00 |
| square_bay_flatbar | 800 | 50.1982 | 0.041611 | 4.10e-07 | 0.00e+00 |
| ice_belt_deep_web | 100 | 86.2181 | 0.037541 | 5.94e-05 | 0.00e+00 |
| ice_belt_deep_web | 200 | 86.2181 | 0.037541 | 1.42e-05 | 0.00e+00 |
| ice_belt_deep_web | 400 | 86.2181 | 0.037541 | 1.30e-06 | 0.00e+00 |
| ice_belt_deep_web | 800 | 86.2181 | 0.037541 | 5.47e-07 | 0.00e+00 |

### 与 TensorFEM 弹塑性 Shell4 Riks 路径对比

| 案例 | 网格 | 自由度 | 弧长步 | sigma_u/sigma_y (FE 末点) | sigma_u/sigma_y (闭式链) | 相对差 | 捕捉极限点 | 证据等级 |
|---|---|---|---|---|---|---|---|---|
| square_bay_tee | coarse | 36 | 60 | 0.0251 | 0.7553 | 96.7 % | **no** | lower bound at the executed end shortening |
| square_bay_tee | fine | 95 | 60 | 0.9854 | 0.7553 | 30.5 % | **no** | lower bound at the executed end shortening |
| square_bay_tee | medium | 73 | 60 | 0.9709 | 0.7553 | 28.5 % | **no** | lower bound at the executed end shortening |
| square_bay_tee | medium_long | 73 | 120 | 1.0031 | 0.7553 | 32.8 % | **no** | lower bound at the executed end shortening |
| square_bay_flatbar | medium | 73 | 60 | 0.9697 | 0.7490 | 29.5 % | **no** | lower bound at the executed end shortening |
| ice_belt_deep_web | medium | 73 | 60 | 1.0019 | 0.9345 | 7.2 % | **no** | lower bound at the executed end shortening |

### beta 参数谱（方板 T 型加筋，变板厚）

| t_p (mm) | beta | Faulkner | von Karman 1/beta | 板极限/sigma_y | 板格链 sigma_u/sigma_y |
|---|---|---|---|---|---|
| 8 | 3.38 | 0.5045 | 0.2961 | 0.5045 | 0.6456 |
| 10 | 2.70 | 0.6032 | 0.3701 | 0.6032 | 0.6988 |
| 12 | 2.25 | 0.6910 | 0.4441 | 0.6910 | 0.7553 |
| 14 | 1.93 | 0.7678 | 0.5181 | 0.7678 | 0.8101 |
| 16 | 1.69 | 0.8337 | 0.5921 | 0.8337 | 0.8604 |
| 20 | 1.35 | 0.9325 | 0.7402 | 0.9325 | 0.9408 |

### 弹性锚的有限元互证

| 案例 | FD 网格 | k (FE) | 相对 k=4 误差 |
|---|---|---|---|
| square_bay_tee | 6x6 | 3.9333 | 1.67 % |
| square_bay_tee | 12x12 | 3.9806 | 0.49 % |
| square_bay_tee | 24x24 | 3.9947 | 0.13 % |
| square_bay_flatbar | 6x6 | 3.9333 | 1.67 % |
| square_bay_flatbar | 12x12 | 3.9806 | 0.49 % |
| square_bay_flatbar | 24x24 | 3.9947 | 0.13 % |
| ice_belt_deep_web | 6x6 | 3.9517 | 1.21 % |
| ice_belt_deep_web | 12x12 | 3.9821 | 0.45 % |
| ice_belt_deep_web | 24x24 | 3.9948 | 0.13 % |

| 案例 | 单元数 | P_cr,FE (MN) | P_cr,Euler (MN) | 相对误差 |
|---|---|---|---|---|
| square_bay_tee | 8 | 128.3235 | 128.3193 | 3.28e-03 % |
| square_bay_tee | 16 | 128.3196 | 128.3193 | 2.06e-04 % |
| square_bay_tee | 32 | 128.3193 | 128.3193 | 1.29e-05 % |
| square_bay_tee | 64 | 128.3193 | 128.3193 | 8.01e-07 % |
| square_bay_flatbar | 8 | 76.3495 | 76.3470 | 3.28e-03 % |
| square_bay_flatbar | 16 | 76.3471 | 76.3470 | 2.06e-04 % |
| square_bay_flatbar | 32 | 76.3470 | 76.3470 | 1.29e-05 % |
| square_bay_flatbar | 64 | 76.3470 | 76.3470 | 8.05e-07 % |
| ice_belt_deep_web | 8 | 20.2992 | 20.2985 | 3.28e-03 % |
| ice_belt_deep_web | 16 | 20.2986 | 20.2985 | 2.06e-04 % |
| ice_belt_deep_web | 32 | 20.2985 | 20.2985 | 1.29e-05 % |
| ice_belt_deep_web | 64 | 20.2985 | 20.2985 | 8.05e-07 % |

### 判据表（机器判定，落 evidence.json）

| 判据 | 结果 |
|---|---|
| energy_residual_below_1e_4 | PASS |
| energy_residual_decreasing_with_refinement | PASS |
| peak_unique_all_cases | PASS |
| monotonic_to_peak_all_cases | PASS |
| moment_below_plastic_bound | PASS |
| plate_eigen_k4_within_0_5pct | PASS |
| column_euler_fem_within_0_1pct | PASS |
| softened_paths_have_limit_point | PASS |
| fe_sigma_u_within_15pct | **FAIL** |
| panel_chain_equals_section_axial | PASS |

### 误差判定（逐项显式回答是否 <3%）

| 关键量 | 基准 | 相对误差 | <3%? |
|---|---|---|---|
| 长板弹性屈曲系数 k | 解析 k=4 | 0.13 % (24x24) | 是 |
| 柱临界荷载（梁单元） | Euler | 8.01e-07 % (64 单元) | 是 |
| 弹性极限 M=EI chi | 解析 | <1e-6（pytest 断言） | 是 |
| 能量账本残差 | 自身 int M dchi | 1.42e-05 (200 步) | 是 |
| 板极限应力 vs Faulkner | Faulkner 公式 | 0.00e+00 %（同一实现） | 是 |
| 截面惯性矩 vs 构件链 | StiffenedPanel.column_inertia | 0.00 % (见 pytest) | 是 |
| 板格 sigma_u vs FE 单胞 | Shell4 Riks 末点 | 32.8 %（冰带案例 7.2 %） | **否**（且 FE 极限点未捕捉，判据 FAIL） |

FE 极限点捕捉情况：全部未捕捉，FE 值是已执行端部缩短处的下界。

## 注意点与局限

- 软化参数是调用方输入，不是标定值；平台律下 $M(\chi)$ 渐近于 $M_p$ 而无
  极限点，代码用 `saturating` 显式标注，禁止冒充极限点。
- 板的自身惯性矩 $bt^3/12$ 被忽略（约 0.3 % 的 $I$）；腹板/翼板以厚度分层
  恢复（单点集总会丢约 11 % 的 $I$）。
- 纤维本构是变形理论：能量账本验证积分与平衡一致性，不验证本构正确性。
- 已执行的 FE 互证是单胞**轴向压缩** Riks 路径；弹塑性壳的**弯曲** $M$--$\chi$
  路径未执行，$M$--$\chi$ 的有限元互证为 WITHHELD，仅给出 $M_y$ 下界 /
  $M_p$ 上界与能量、收敛证据。
- 本环境无文献库访问权限，未数字化任何已发表试验序列，也不引用记忆数值；
  对标限于已发表闭式模型（Euler、Johnson--Ostenfeld、Faulkner、von Karman、
  EN 1993-1-1）与 TensorFEM 自身离散。

## 复现

```
python benchmarks/panel_mchi_qualification.py            # 闭式 + 特征值，约 20 s
python benchmarks/panel_mchi_qualification.py --doc      # 重写本文档
python benchmarks/panel_mchi_qualification.py --fe square_bay_tee --fe-mesh medium
python -m pytest tests/test_panel_mchi_qualification.py -q
```

图：`docs/assets/panel_mchi_qualification/*.png`（M--chi 面、beta 谱、能量/柱
曲线、FE 对比）。
