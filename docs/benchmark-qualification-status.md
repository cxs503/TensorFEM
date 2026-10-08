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
| Cook 膜与开孔半球壳 | 32×32 / 40×40 | 位移响应通过；缺独立应力参考 | 完整应力精度验收未完成 |

原先 20×20×4 球壳径向逐点误差 12.257342%，24 单元船体梁逐点误差 7.223673%；两者现在由正式门禁标记失败。球壳 32×32×4 和 40×40×4 仍失败，船体梁 48 单元仍失败，均未删除。

球壳全部 L2/原有指标下降；逐点最大误差在 20/32/40/64 细化段下降。8/12 粗网格的环向逐点最大误差由 0.953% 增至 1.672%，非单调记录保留，不能宣称所有指标在全部粗细网格上都单调。

[逐点审计 JSON](benchmarks/tutorials/pointwise-audit.json) 记录当前误差、最差采样位置和数据哈希。
执行 `scripts/audit_pointwise_benchmark_errors.py` 可用独立 NumPy 公式重算。

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
| Cook 膜，32×32 | 加载边中点位移误差 0.594182% | 位移响应；独立应力参考和原始数值表溯源待补 |
| 18° 开孔半球壳 | 40×40 默认位移误差 2.035156%；参数误差 2.919689%/2.035156%/1.479970% | 位移响应及力/力矩平衡；表面应力仅展示，独立应力认证待补 |

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

Cook 膜与开孔半球壳尚缺独立应力精度参考。局部板屈曲、Hertz 接触及其他经典壳体案例仍缺完整场验证与相同质量的报告，不能列为完整报告通过。

既有 `.qualification/v1-readiness/aggregate-v045.json` 记录的 1.0 阻塞项仍为三维双变形体接触，以及加筋板 4/8/12 网格的峰值和峰后响应收敛。本次补齐线性 benchmark 场验证及报告，没有重新认证这两个工业能力，也没有提升版本到 1.0。

## 复现

在项目 Python 环境安装 `.[dev,reports]` 后运行：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/rebuild_benchmark_reports.py
PYTHONPATH=src .venv/bin/python scripts/validate_published_benchmark_reports.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/rebuild_additional_benchmarks.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/validate_additional_benchmark_reports.py
```

第一条命令重新求解三组案例、生成全部报告格式并执行验证。第二条检查原始场、网格收敛、独立案例数量及文件哈希，不会把缺失证据的其余 benchmark 标记通过。
