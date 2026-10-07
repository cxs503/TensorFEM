# 船舶与海洋工程标模案例说明书

使用 `scripts/run_marine_benchmark_report.py` 可生成可审计的 JSON 案例册：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_marine_benchmark_report.py \
  --output results/marine-benchmarks.json
PYTHONPATH=src .venv/bin/python scripts/validate_marine_benchmark_report.py \
  results/marine-benchmarks.json
```

案例覆盖船体梁、加筋板、局部板屈曲、船体梁极限强度与渐进屈服、疲劳、裂纹尖端、壳屈曲、静水力学、波浪 Morison 载荷等公开可复核问题。每个条目包含问题类型、SI 输入约定、计算过程、计算量、独立参考解、相对误差、应力/位移场字段摘要和网格收敛状态。

报告的 3% 是硬门禁：误差超过 3%、缺少独立参考值或缺少真实 FE 场数据时，状态必须为 `blocked`。当前报告中的 `qualified` 表示可执行解析/筛选模型通过参考值校验，不等同于船级社认证；要做完整 FE 认证，应在对应条目的 `field_summary` 和 `mesh_convergence` 中附加节点应力、位移、反力、云图文件及跨网格结果。

生成的 JSON 可作为后续 HTML/PDF/Word 渲染器的单一数据源，哈希字段用于发布前完整性检查。
