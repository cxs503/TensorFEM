# 场结果来源

`cantilever-fe/`、`hull-girder/`、`stiffened-panel/` 是当前研究报告使用的实际 FE 求解与恢复场，包含原始 JSON、连续云图和三网格误差图。位移和应力验收范围由报告正文及 JSON 声明，图像插值不参与误差判定。

本目录根部的 `cantilever_displacement.png`、`cantilever_bending_stress.png` 和 `hertz_contact_pressure.png` 是解析参考图，来源见 `cloud-metadata.json` 的 `source` 字段，不能当作实际 FE 结果。

七个真实案例的完整 Markdown、Word、PDF、HTML 位于 `docs/benchmarks/tutorials/`；`evidence-manifest.json` 记录当前原始场、图像与导出的文件哈希。已移除旧的同模型 Q4 船体梁和过时悬臂梁云图。

新增 `sphere-pressure/` 和 `mindlin-navier/` 使用独立精确位移与应力场。
`cook-membrane/` 和 `hemisphere-hole/` 的真实 FE 应力只用于展示，
尚无独立应力精度认证。新增四个报告的哈希见
`docs/benchmarks/tutorials/additional-evidence-manifest.json`。
