# Benchmark 验收状态与纠错记录

本次核验对象是 `/data/TensorFEM`，不是主目录中未更新的早期副本。
当前已生成七个独立问题的研究报告：五个在声明的位移与应力范围内通过，两个仅通过位移响应验收。注册表的 55 个标量检查不能作为 55 个完整 FE 案例或全域应力精度的证明。项目仍未获得 1.0 验收。

## 逐点精度优化与当前边界

新增 [SUBOFF 人工壳体初始上浮撞冰研究案例](benchmarks/tutorials/suboff_upward_ice_collision.md)：
几何来自 TensorLBM，包含可分离冰片、黏聚断裂、动态接触、能量/动量收支和六组敏感性计算。
该案例尚无独立物理参考，网格及接触敏感性失败均保留，未获得破冰精度验收；不计入以上七项精度报告的通过数量。

新增 [带围壳与四尾舵的 SUBOFF / 可配置冰层案例](benchmarks/tutorials/suboff_appended_ice_simulation.md)：
六个实际算例通过能量、动量、质量和适用性检查，具备变厚度、开水区、预裂缝、浮力和碎片统计。时间步减半通过，但结构网格加密的力/应力变化超过 3%；未认证完整破冰或物理精度，不增加上述精度报告通过数量。

厚球壳和船体梁已增加逐点最大误差门槛并细化网格，所有粗网格失败保留。

| 案例 | 最细网格 | L2 与逐点结果 | 范围 |
|---|---|---|---|
| 三维厚球壳内压 | 64×64×6，147456 HEX8 | 径向应力 L2 0.415483%；径向逐点最大 2.979011%；位移逐点最大 0.152505% | 全节点位移、全部单元中心张量/径向/环向/von Mises 的 L2 与逐点范数均通过；边界外推峰值不在验收内 |
| 船体 Frame 梁 | 96 单元 | 纤维应力 L2 0.004256%；非零纤维应力逐点最大 1.802389% | 全长预定采样的弯矩/非零纤维应力通过；不是三维壳应力 |
| Q4 悬臂梁 | 80×16 | 中部 sigma_x 最大 0.635538%，tau_xy 最大 1.794898% | 预定中部通过；全域及固支峰值未认证 |
| Mindlin 板 | 32×32 | 非零节点挠度最大 0.054139%，中心弯曲应力向量最大 0.134510% | 声明模型与采样范围内通过 |
| 等效膜力 patch | 40×20 | 位移与中心应力达到舍入精度 | 通过 patch 验证，不能认证显式加筋板 |
| Cook 膜与开孔半球壳 | 32×32 / 40×40 | 位移响应通过；Cook 独立 Q8 应力参考已建立，点式探针协议未良置（5/5 点在单元边界，单侧散布最大 19.919%），见 [cook-probe-protocol-audit.md](cook-probe-protocol-audit.md)；非对齐探针口径 n=128 最大 3.687%、外推一致 2.233%（[协议报告](cook-membrane-pointwise-protocol.md)） | Cook：`stress_accuracy_qualified` 仍为 False，理由「探针协议未良置」；开孔半球壳缺独立应力参考 |

原先 20×20×4 球壳径向逐点误差 12.257342%，24 单元船体梁逐点误差 7.223673%；两者现在由正式门禁标记失败。球壳 32×32×4 和 40×40×4 仍失败，船体梁 48 单元仍失败，均未删除。

球壳全部 L2/原有指标下降；逐点最大误差在 20/32/40/64 细化段下降。8/12 粗网格的环向逐点最大误差由 0.953% 增至 1.672%，非单调记录保留，不能宣称所有指标在全部粗细网格上都单调。

[逐点审计 JSON](benchmarks/tutorials/pointwise-audit.json) 记录当前误差、最差采样位置和数据哈希。
执行 `scripts/audit_pointwise_benchmark_errors.py` 可用独立 NumPy 公式重算。

## 新增：曲面壳体单元的应力级否定审计

[闭合球壳内压膜态审计](sphere-membrane-validation.md)：用闭合球壳内压这一**存在精确无边界层
膜态解**的曲面算例，复核已合并的 Shell4 平面映射面元。结果未通过 3% 门限：216 面元时平均
膜力误差 1006%（相对 `pR/2`）与 1022%（相对刻面参照 `p*rho/2`），主膜力各向异性 437%，
位移非径向残差 94%；拟合收敛阶约 2.34，外推需约 2e4 面元才能达 3%。同一面元的平面膜 patch
恢复逐位精确，故失败源于「薄+曲面+压力」下折角弯曲主导，而非应力恢复。该结论以 fail-closed
测试锁死，不得据此宣称壳体族认证。

## 已修复的证据问题

| 问题 | 修复后的行为 |
|---|---|
| 只根据端位移误差标记通过 | 位移、应力、载荷、反力和三网格趋势共同参与验收；误差由原始字段重算 |
| 梁报告未检查应力误差 | 检查轴向、剪切、von Mises 和应力张量；应力比较位置与范围明确声明 |
| 两篇梁报告是同一模型 | 船体案例替换为 120 m 简支 Frame 梁和均布线载荷；与 1 m 固支 Q4 悬臂模型独立 |
| patch test 被误解为真实加筋板验证 | 报告明确只验证等效厚度均匀膜力，不覆盖加筋板屈曲或极限强度 |
| 缺少 Word、PDF、HTML 结果交付 | 三种导出均由同一完整 Markdown 正文生成，包含结果表、云图、误差曲线及能力边界 |
| 斜网格 Mindlin 坐标导数错误 | 使用正确逆转置转换，新增刚体倾斜零能量测试；半球旧数值记录作废 |
| Cook 参考值采样位置错误 | 23.96 对应加载边中点，修正右上角采样；重新运行三级网格 |
| 字段为空也可被视为通过 | 空证据、字段缺失、NaN、重复网格、场数据与坐标不符及反力不平衡均触发门禁 |
| 标量报告被渲染成完整通过 | 船海标量报告使用 scalar_status 与 reference-only，渲染器不能仅凭一个小误差提升为完整报告 |
| 秩审计把 3×3×3 的 `hex20` 记作「2×2×2 积分」 | 标注错误已更正；代码一直是 3×3×3。同轮补齐畸变/曲面 20 节点体的精确性与秩证据，并判定 2×2×2 变体含 6 个伪零能模态、不可采用，见 [hex20-distortion-qualification.md](hex20-distortion-qualification.md) |

## 新增：TET10 积分精度（Keast 五点规则的欠账量化）

[TET10 积分精度审计](tet10-integration-accuracy.md) 清偿「Keast 只精确到 3 次、精度损失未量化」这条
欠账。独立 Duffy 参考（$24^3$ 点）先对闭式单项式自验证到 1e−16、并与生产 torch 实现互差 ≤2.7e−16，
再判生产规则：直边几何误差 3.6e−15（生产规则在仿射几何上精确，故既有 TET10 悬臂基准不受污染，
其 2.44 % 残差来自离散而非积分）；曲边几何 0.05/0.11 边长达 0.428 %/2.551 %，曲率 0.12 处
3.29 % 首次破 3 % 门。误差来源是 $oldsymbol B$ 的五点采样（仿射场应变能误差 ≤3.1e−15 排除
$\det J$ 嫌疑）。五点规则不能用于一致质量阵（Frobenius 19.7 %、对角最大 70.8 %–81.9 %），
故日后实现 `tet10_mass` 时必须换规则。`tet10_stiffness(..., rule=...)` 提供 opt-in 高阶通道，
默认路径逐位不变。

## 新增：HEX20 畸变与曲率精确性鉴定

[HEX20 畸变与曲率鉴定](hex20-distortion-qualification.md) 把 20 节点体的证据从「规则立方体 affine
patch 测试」扩到畸变（二次翘曲、渐变剪切）与真实曲率（厚壳扇区，R=1 起）网格：线性场位移残差
≤5.6e−16、形心应力 ≤4.5e−13、常应变能量恒等式 ≤3.6e−14，全部达到舍入精度；秩恒为 6 零能/0 伪零能。
同一组门禁对「单元中心雅可比复用到 27 个 Gauss 点」的经典实现 bug 给出 2.955e−3（R=1）的拒绝
响应，故门禁非空转；该对照在仿射（等雅可比）网格上天然失效，此局限已写入报告。曲面块体积对照
解析扇区体积 L·W·(H+H²/2R)：R=1 时 2 单元 0.202%、4 单元 0.013%，为几何表示误差而非积分误差。
本节仅为精确性/秩证据，不构成曲梁弯曲精度或非线性路径认证。

## 当前闭环案例与验收范围

| 案例 | 位移误差 | 应力误差 | 范围 |
|---|---:|---:|---|
| Q4 悬臂梁，80×16 | 端位移 0.084699% | sigma_x L2 0.635530%；tau_xy L2 0.312720%；von Mises L2 0.631762% | 应力仅为预定中部 0.2L≤x≤0.8L 的单元中心；端部峰值未认证 |
| 120 m 船体梁，96 单元 | 全长采样位移 L2 0.000000209% | 全长纤维 sigma_x L2 0.004256% | Euler–Bernoulli 梁纤维；不是三维壳应力 |
| 等效厚度均匀膜力，40×20 | 全节点位移及端位移达到舍入精度 | 全域单元中心应力达到舍入精度 | Q4 patch test；不是显式加筋板 |

L2 是声明采样集上的相对范数误差，不表示每个节点的逐点误差均小于 3%。云图平滑插值不参与数值验收。梁端参考存在模型差异，不能把中部应力通过外推为固支端峰值通过。

## 新增案例与证据范围

| 案例 | 最细网格结果 | 认证范围 |
|---|---|---|
| 三维厚球壳内压，64×64×6，147456 HEX8 | 位移向量 L2 0.104625%；应力张量 L2 0.082488%；径向应力 L2 0.415483% | 三维 Lamé 精确场比较；L2 和逐点最大误差均低于 3%；全部原有指标及细化段新增指标下降 |
| Mindlin 简支正弦载荷板，32×32 | 挠度 L2 0.054139%；表面弯曲应力 L2 0.134510%；横向剪力 L2 0.000052% | Mindlin 模型独立精确场，不涵盖三维厚度应力 |
| Cook 膜，32×32 | 加载边中点位移误差 0.594182% | 位移响应；独立 Q8 应力参考已建立（GCI 0.068%，见 [Cook 膜应力审计](cook-membrane-stress-audit.md)）：存盘单元中心场 n=64 p95 0.467%、尺度归一化最大 2.910%，但五固定样点的任意点提取判据仍 6.375%>3%，故应力不认证 |
| 18° 开孔半球壳 | 40×40 默认位移误差 2.035156%；参数误差 2.919689%/2.035156%/1.479970% | 位移响应及力/力矩平衡；表面应力已由独立三维 HEX20 参照判定为**不通过**（内部平均 5.09%，见 [oracle 报告](hemisphere-hex20-stress-oracle.md)） |

新增报告和三种导出由同一正文生成，保留失败粗网格。新增文件哈希记录在
[additional-evidence-manifest.json](benchmarks/tutorials/additional-evidence-manifest.json)。

## 完整报告与导出

| 案例 | 正文 | 网页 | PDF | Word |
|---|---|---|---|---|
| 悬臂梁 | [Markdown](benchmarks/tutorials/cantilever_beam_full.md) | [HTML](benchmarks/tutorials/exports/cantilever_beam_full.html) | [PDF](benchmarks/tutorials/exports/cantilever_beam_full.pdf) | [Word](benchmarks/tutorials/exports/cantilever_beam_full.docx) |
| 船体梁 | [Markdown](benchmarks/tutorials/hull_girder_longitudinal_bending_full.md) | [HTML](benchmarks/tutorials/exports/hull_girder_longitudinal_bending_full.html) | [PDF](benchmarks/tutorials/exports/hull_girder_longitudinal_bending_full.pdf) | [Word](benchmarks/tutorials/exports/hull_girder_longitudinal_bending_full.docx) |
| 膜力 patch | [Markdown](benchmarks/tutorials/stiffened_panel_fe_full.md) | [HTML](benchmarks/tutorials/exports/stiffened_panel_fe_full.html) | [PDF](benchmarks/tutorials/exports/stiffened_panel_fe_full.pdf) | [Word](benchmarks/tutorials/exports/stiffened_panel_fe_full.docx) |
| 三维球壳内压 | [Markdown](benchmarks/tutorials/sphere_pressure_full.md) | [HTML](benchmarks/tutorials/exports/sphere_pressure_full.html) | [PDF](benchmarks/tutorials/exports/sphere_pressure_full.pdf) | [Word](benchmarks/tutorials/exports/sphere_pressure_full.docx) |
| Mindlin 板 | [Markdown](benchmarks/tutorials/mindlin_navier_full.md) | [HTML](benchmarks/tutorials/exports/mindlin_navier_full.html) | [PDF](benchmarks/tutorials/exports/mindlin_navier_full.pdf) | [Word](benchmarks/tutorials/exports/mindlin_navier_full.docx) |
| Cook 膜（响应） | [Markdown](benchmarks/tutorials/cook_membrane_full.md) | [HTML](benchmarks/tutorials/exports/cook_membrane_full.html) | [PDF](benchmarks/tutorials/exports/cook_membrane_full.pdf) | [Word](benchmarks/tutorials/exports/cook_membrane_full.docx) |
| 开孔半球壳（响应） | [Markdown](benchmarks/tutorials/hemisphere_hole_full.md) | [HTML](benchmarks/tutorials/exports/hemisphere_hole_full.html) | [PDF](benchmarks/tutorials/exports/hemisphere_hole_full.pdf) | [Word](benchmarks/tutorials/exports/hemisphere_hole_full.docx) |

HTML 内嵌全部 PNG，可独立打开；Word、PDF 也包含图像。原始 FE 字段、图片与各格式文件的 SHA-256 位于 [evidence-manifest.json](benchmarks/tutorials/evidence-manifest.json)。哈希用于检测文件漂移，数值是否通过由原始字段重新验证。

## 尚未闭环的工作

Cook 膜已有独立 Q8 应力参考（m2430），其点式 5.02% 残差的归因由 [cook-probe-protocol-audit.md](cook-probe-protocol-audit.md) 定谳：全部 5 个规范采样点落在 Q4 单元边界上，取值不唯一且该协议逐点不收敛；同一位移解经 Gauss 点+节点恢复后 n=64 为 0.120%、二阶收敛，但恢复场尚未进入生产 `cook_fields`，故 `stress_accuracy_qualified` 保持 False。开孔半球壳（此前判定为「仍缺独立应力参考」）的独立应力参照**已补并给出否定结论**：[三维 HEX20 实体 oracle](hemisphere-hex20-stress-oracle.md) 逐站位复算后，内部区域表面 von Mises 相对误差平均 5.09%、最大 9.00%（孔边 6.87%、点载附近 7.56%），不满足 3% 门限，`stress_accuracy_qualified` 维持 `False`；oracle 自身先在 Lamé 圆环（最大 8.6e-7）与薄板弯曲（0.26%）上过标；局部板屈曲的**单轴压缩**已闭环，面内**剪切与压-剪组合**亦已按文献锚点（k_s=9.34@a/b=1、5.34 长板渐近，误差 0.17%/2.6%）闭环，见 [marine-plate-shear-buckling-validation.md](marine-plate-shear-buckling-validation.md)；仍缺：壳单元几何刚度、非均匀预屈曲应力、Hertz 接触及其他经典壳体案例的完整场验证与相同质量的报告，不能列为完整报告通过。

既有 `.qualification/v1-readiness/aggregate-v045.json` 记录的 1.0 阻塞项仍为三维双变形体接触，以及加筋板 4/8/12 网格的峰值和峰后响应收敛。本次补齐线性 benchmark 场验证及报告，没有重新认证这两个工业能力，也没有提升版本到 1.0。


非均匀预屈曲应力（线性边缘应力梯度：纯弯曲 / 三角形压缩 / 压-弯组合）已闭环，见 [plate-bending-buckling-validation.md](plate-bending-buckling-validation.md)：对 Timoshenko & Gere (1961) Table 9.1 的 4.00 / 7.81 / 23.9 / 25.6 四个锚点误差 0.076% / 0.025% / 0.076% / 0.280%，`psi=1` 退化极限舍入级回到 4.00，独立有限差分解一致到 0.33%；范围仅限四边简支平板线性梯度与**弹性临界载荷**，不含壳单元几何刚度、后屈曲强度与逐点场量认证。

扭转（Saint-Venant 三个截面）已由闭式解独立判定，见 [saint-venant-torsion.md](saint-venant-torsion.md)：最细网格刚度与中跨剪应力均 < 3%，但逐点剪应力峰值需 `n >= 24` 才进 3%（2.886%），粗网格失败项保留。

### 开孔应力集中：从圆孔推广到椭圆孔（Inglis，m7524）

[椭圆开孔验证](elliptic-hole-inglis-validation.md)：新增 `tensorfem.elliptic_hole`，用推导出的
Kolosov–Muskhelishvili 势（`m` 家族，`m→0` 即 Kirsch）给出椭圆孔的精确应力/位移场，孔边无面力
≤3e-14、两个尖点恒等式（载荷轴 `σ_yy=-T`、横向尖点 `K_t=1+2·semi_y/semi_x`）残差 <1e-12、
`m→0` 复现 Kirsch 到 4.6e-6。生产 Q4 平面应力链在共焦椭圆外边界（精确面力，零截断）下实测：
长圆孔 `K_t=2.333` 的尖点峰值 `n=40/48` 为 2.74%/2.29%（**进 3% 门限**，Richardson 更低）；
尖椭圆孔 `K_t=7`（尖点曲率半径 `semi_x/3`）在 4802 自由度内为 8.87%、观测阶 0.95，
**honest FAIL 3% 门限**，即该形状必须局部加密/奇异单元后方可引用峰值。位移场二阶收敛
（7.7e-4→1.9e-4），力平衡 1e-14。范围仅限单轴远场、平面应力、小应变、各向同性；
双轴/剪切远场、多孔干涉、裂纹孔、正交异性板与直边有限宽（Howland 口径）不在本单内。

### 壳单元几何刚度：已复核但仍未认证（m3652）

曲面壳几何刚度已用轴向受压圆柱壳的经典解 `N_cr = E t^2/(R sqrt(3(1-nu^2)))` 独立复核，见 [圆柱壳几何刚度审计](cylindrical-shell-geometric-stiffness-audit.md)。结论：环向膜刚度与解析值一致（误差 ≤0.1%），但屈曲载荷在可负担网格内误差自 +110.2% 单调降至 −2.50% 并穿越零点，轴向族最细网格仍 +23.5%，且临界模态在 `n_theta≥48` 失去轴对称性，故该项**保持未认证**（`certified=False`），并以 fail-closed 测试锁死。

## 复现

在项目 Python 环境安装 `.[dev,reports]` 后运行：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/rebuild_benchmark_reports.py
PYTHONPATH=src .venv/bin/python scripts/validate_published_benchmark_reports.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/rebuild_additional_benchmarks.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/validate_additional_benchmark_reports.py
```

第一条命令重新求解三组案例、生成全部报告格式并执行验证。第二条检查原始场、网格收敛、独立案例数量及文件哈希，不会把缺失证据的其余 benchmark 标记通过。

## 新增：Cook 膜恢复场已进入生产路径（原始协议仍 fail-closed）

[恢复场生产化报告](cook-recovered-stress-production.md)：Cook 膜的 Barlow（Gauss 点外推 + 面积加权节点平均）
恢复已落在 `src/tensorfem/stress_recovery.py`，并由 `cook_fields(n, reference=...)` 输出 `nodal_stress` 与
`stress_recovery` 两个字段。逐点最差误差 n=8/16/32/64 为 8.0231%/1.9514%/0.4838%/0.1202%（二阶，p=2.009），
5 个规范探针在界面上的插值离散度 ≤9.4e-17（单值）；原始逐单元协议同期仍为 3.51%–4.02%（一阶，p=0.981），
故 `cook_fields['stress_accuracy_qualified']` **保持 `False`**，两个口径并存互不覆盖。证据
`docs/cook-recovery-evidence.json`（含 Torch vs NumPy 两条实现的应变矩阵 2.8e-17、恢复场 4.7e-14 交叉校验）。
Cook 膜已有独立 Q8 应力参考（m2430），其点式 5.02% 残差的归因由 [cook-probe-protocol-audit.md](cook-probe-protocol-audit.md) 定谳：全部 5 个规范采样点落在 Q4 单元边界上，取值不唯一且该协议逐点不收敛；同一位移解经 Gauss 点+节点恢复后 n=64 为 0.120%、二阶收敛，但恢复场尚未进入生产 `cook_fields`，故 `stress_accuracy_qualified` 保持 False。开孔半球壳的独立应力参照**已补并给出否定结论**：[三维 HEX20 实体 oracle](hemisphere-hex20-stress-oracle.md) 逐站位复算后，内部区域表面 von Mises 相对误差平均 5.09%、最大 9.00%（孔边 6.87%、点载附近 7.56%），不满足 3% 门限，`stress_accuracy_qualified` 维持 `False`；oracle 自身先在 Lamé 圆环（最大 8.6e-7）与薄板弯曲（0.26%）上过标；局部板屈曲的**单轴压缩**已闭环，面内**剪切与压-剪组合**亦已按文献锚点（k_s=9.34@a/b=1、5.34 长板渐近，误差 0.17%/2.6%）闭环，见 [marine-plate-shear-buckling-validation.md](marine-plate-shear-buckling-validation.md)；仍缺：壳单元几何刚度、Hertz 接触及其他经典壳体案例的完整场验证与相同质量的报告，不能列为完整报告通过。

## 新增：TET10 一致质量阵（Keast 欠账已清偿）

[tet10 一致质量阵报告](tet10-consistent-mass.md) 清偿「五点 Keast 规则不能用于一致质量阵」这条欠账：
新增 `tet10_consistent_mass / tet10_lumped_mass / tet10_mass_quadrature / tet10_volume`（坍缩 Gauss，
order=4 对直边 TET10 精确），并把 `element_type="tet10"` 接入 `solid_mass.assemble_solid_mass /
mass_properties` 与 `solid_mass_audit.axial_rod_case`。独立验证：直边/畸变/曲边三几何的质量阵与
24³ Duffy 参考 Frobenius 2.2e−15，行和命中闭式 −ρV/20、+ρV/5，总质量误差 8.3e−17；物理锚
固定–自由杆 ω₁ 在 4 单元 0.0016 %、16 单元 5.0e−6 %（同网格 TET4 为 0.64 %）。出厂 Keast 规则
的 19.677 % Frobenius / −70.83 % 顶点对角误差以 fail-closed 测试锁死，子午面上不得回退。
范围仅限质量阵与轴向动力学：不含弯曲精度认证，也不改变 `tet10_stiffness` 默认路径（逐位不变）。
