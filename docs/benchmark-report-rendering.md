# Benchmark 报告生成与验收

完整研究报告使用同一 Markdown 正文导出 HTML、PDF 和 Word，包含问题、条件、过程、真实计算表格、误差和连续云图。安装 `.[reports]` 后运行：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/rebuild_benchmark_reports.py
PYTHONPATH=src .venv/bin/python scripts/validate_published_benchmark_reports.py
```

前者重新求解已有三篇报告的案例，检查位移、应力、反力和三网格趋势，再生成正文及各格式；后者从完整原始字段重算指标并检查报告资产的 SHA-256。

导出位于 `docs/benchmarks/tutorials/exports/`。HTML 内嵌 PNG，PDF 和 Word 嵌入同样的云图，所有格式保持相同的数值表格与能力边界。缺失图片导致生成失败，不能以空白图或仅有摘要的文件冒充完整报告。

JSON 通用摘要渲染命令 `scripts/render_benchmark_report.py` 用于浏览供给的原始证据，不作为完整研究报告交付。标量误差小于 3% 不会自动成为完整场验证通过；船海标量报告独立记录 `scalar_status`，完整场仍标记为 `reference-only`。严格误差门禁是 `<0.03`，等于 3% 不通过。

梁应力验收范围必须在报告中声明。中部梁理论指标通过不能认证固支端峰值，均匀膜力 patch test 通过不能认证真实加筋结构屈曲和极限强度。详见 [验收状态与下载](benchmark-qualification-status.md)。
