# Benchmark 报告生成

将算例运行器输出的 JSON 报告渲染为网页、PDF 和 Word：

```bash
PYTHONPATH=src .venv/bin/python scripts/render_benchmark_report.py \
  results/standard-benchmarks.json --output-dir results/benchmark-report
```

报告应包含问题说明、单位和计算条件（由算例 JSON 的 `input`、`reference` 字段记录）、场变量采样（`field_samples`）、参考值、FE 值及 `relative_error`。提供 `stress` 或 `displacement` 场采样时，HTML 会生成彩色散点云图；PDF 和 DOCX 会包含结果摘要与可审计原始字段。缺少可选库时：HTML/DOCX 仍可生成，PDF 明确报错而不生成伪文件。

误差门禁是 3%：`qualified` 仅表示有 FE 结果且相对误差不大于 0.03；超限结果始终为 `blocked`，只有参考值的案例为 `reference-only`。渲染不会改变原始报告哈希或篡改资格状态。
