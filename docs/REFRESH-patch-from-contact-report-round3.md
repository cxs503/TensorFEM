# REFRESH — patch_from_contact 交付物第 3 轮（mission 23259）

锚点：`GAP-PP-1-ICE-STRUCT-CONTACT-PATCH` / 原 PR#1121（owner 2026-11-05 裁定关闭）经 m20207 重开第 3 轮。
对象：`docs/fig1_pressure_distributions.png`、`docs/fig2_deflection_comparison.png`、`docs/fig3_plate_deflection.png`、`docs/patch_from_contact_report.json`。
本文件由 `tools/gen_patch_from_contact_report.py` 生成，数字全部注入自 `artifacts/patch_from_contact_report_numbers.json`。

## 1. 本轮结论（三条，带数字）

1. **交付物此前不可复现**：三个 png 在仓库里没有任何生成脚本（`git log --all -- docs/fig1_pressure_distributions.png` 只有二进制变更），报告 json 的数值来自 m20624 之前的代码路径。本轮补上生成器，一条命令重出全部四个文件。
2. **报告里 −99.5% 的「非均匀 vs 均匀峰值挠度差」在当前代码下不可复现**：力匹配基准给 +12.04%…+86.97%，旧 Gauss 基准给 +4.54%…+86.45%，符号全为正。该数字连同其结论句作废，替换为本轮可复现数值。
3. **main 上仍存在的缺陷：均匀（IACS）基准未力匹配**。台阶压力交单元 2×2 Gauss 时间断被全有全无采样，本算例 nx=32 下施加力偏差 -2.07%…+8.51%，Δ 相对力匹配基准移动 0.52…7.50 pp（sharp 构型 +4.54% → +12.04%）。本轮在报告链路上用精确子单元分裂的力匹配基准（误差 1.7e-16）替代，并把旧基准保留为负对照。

## 2. 复现方式

```bash
python tools/gen_patch_from_contact_report.py       # 约 1.5 min，单核
python tools/gen_patch_from_contact_report.py --doc-only   # 只用已有 json 重出文档
python -m pytest tests/test_patch_from_contact_report_artifacts.py -q
```

## 3. 关键数字

| 构型 | 非均匀峰值挠度 [mm] | 均匀(力匹配) [mm] | Δw [%] | 非均匀 σx [MPa] | 均匀 σx [MPa] | Δσx [%] | 旧基准 Δw [%] |
|---|---|---|---|---|---|---|---|
| mild_nonuniform | 2.0000 | 1.0697 | +86.97 | 38.442 | 12.880 | +198.46 | +86.45 |
| moderate_nonuniform | 2.0000 | 1.4396 | +38.93 | 48.097 | 21.371 | +125.06 | +40.59 |
| sharp_nonuniform | 2.0000 | 1.7851 | +12.04 | 67.419 | 41.999 | +60.53 | +4.54 |

| nx | Δw [%] | Δσx [%] | 投影误差 σx vs 解析场 [%] |
|---|---|---|---|
| 16 | +86.21 | +189.55 | -0.21 |
| 24 | +86.77 | +196.10 | -0.22 |
| 32 | +86.97 | +198.46 | -0.22 |

## 4. 误差判定

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| 投影后总力 | 接触网格积分力 | 4.9e-15 | 是 |
| 力匹配均匀基准施加力 | 名义 p·W·L | 1.7e-16 | 是 |
| 板心挠度（满布均匀压力，nx=32，k=0） | Kirchhoff 级数 2.1534 mm（FEM 2.1539 mm） | +0.020% | 是 |
| 线性缩放恒等式（mild/nx=16） | 同算例直接重解 | 挠度 2.1e-14 / σx 1.5e-14 | 是 |
| Δw 网格自收敛（16→32） | 网格梯子 | 移动 0.76 pp | 是 |
| 旧 Gauss 采样均匀基准施加力 | 名义 p·W·L | 最大 8.5% | **否**（已废弃） |

## 5. 未做 / 局限

* 未改 `src/tensorfem/patch_from_contact.py`：`scan_contact_configurations` 内部的均匀基准仍是未力匹配的旧实现（该文件是与在途 mission m23210 分支重叠的热文件，按交付纪律本轮不动，缺陷与修正只在报告链路上隔离展示）。
* 旧版（m20207）产物未逐项复原，故「−99.5% 的成因分解」不做定量归因。
* 均匀基准落区取 ±3σ 采样窗口（6σ×6σ），比压力核心宽约 3 倍；换等面积口径 Δ 会显著变小。
