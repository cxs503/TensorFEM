# 证书图件欠账闭合：从 committed result.json 零求解重绘（m18409）

## 1. 案例介绍
`benchmarks/verified/*/gates_verified.json` 的 `artifacts` 段声明证据，审计器
`tools/verified_anchor_audit.py` 按 git index 解析每条声明，缺失即 `E_ARTIFACT_MISSING`。
推包器 `tp_pusher.py` 丢弃二进制（`\.(png)$`）唯独放行 `docs/` 前缀。两者叠加成复发的赤潮：
图放 `benchmarks/verified/<anchor>/` → 被滤除 → 审计转红 → 再补。
开工基线（main `90699c60`）：audit certified=28 / contradicted=4 / uncertified=5；
三个锚缺 8 图（`fem_inverse_identification` 1、`fem_sent_j_integral` 4+2 越界、
`fem_winkler_plate_k_sweep` 3）；`certificate_renderables.py --check` violations=1；
`tests/test_anchor_figure_gate_safety.py` 红（新增越界 2 锚 + 14 条陈旧冻结行）。

## 2. 怎么做
不重跑求解：三个锚的 `result.json` 已提交所有被绘制的量。新工具
`tools/anchor_figure_rebuild.py` 只读 `result.json`，零求解重绘 10 张图到
`docs/assets/verified/<anchor>/`（推包器放行的路径类），证书直接命名该路径；
渲染剥除 PNG 元数据、固定 dpi，`figures_manifest.json` 用 sha256 逐图绑定数据与笔迹。
同时修掉回滚隐患：`fem_inverse_identification/run.py` 原先写图到锚目录并据此写证书，
跑一次该锚测试就会把证书改回被滤除路径；现改为写 `docs/assets/verified/...`。
`FROZEN_OFFENDERS` 由 17 行缩到 0 行，`anchor_certificate_debt.json` 由
`verified_anchor_audit.py --write` 重写（shrink-only 语义保持）。

## 3. 计算结果（工具直出，零手抄）
| 指标 | 前 | 后 |
|---|---|---|
| certified | 28 | 31 |
| contradicted | 4 | 1（仅 `fem_structural_acoustics`：E_SCHEMA/E_PASS_WITHOUT_CRITERION/E_GATE_VALUE_NOT_FINITE，非图件问题） |
| E_ARTIFACT_MISSING（3 锚） | 3/3 | 0/3 |
| renderables violations | 1 | 0 |
| gate 复算 | 349/368 | 349/368（未改任何 gate 数值） |

## 4. 结果图
![Winkler k-sweep 误差](figures: winkler_k_sweep_error.png)

重建的 `winkler_k_sweep_error.png`：中心挠度误差与剖面最大误差对 k（1e3…1e7 N/m³）双对数曲线，
含 5 % 门限线。误差 1e-2…1e-1 % 全部低于门限，k 最大两点略升为真实点集形状，未平滑。

## 5. 注意点
- 画图不代替测量：`plane_strain.cases` 只存规范站点 + `path_spread`，故平面应变路径无关性
  绘制为「站点值 ± 实测散布」，不伪造 contour 曲线（函数 docstring 明示）。
- 曾用错横坐标（平面应变 ladder 键为 0.20/0.30/0.50，而平面应力为 0.20/0.35/0.50），
  `KeyError: '0.30'` 当即被拦；改为按 key 匹配各自 case 集。
- 仍红 1 项为 `fem_structural_acoustics` 的证据语义问题，需给 C5/A2/B3 判据或降 informational，
  属另一修复线；本单未动其 gate 数值（不得覆盖 honest FAIL）。
- 锚点 `XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO` 属 TensorLBM 域，本工作树为 TensorFEM，
  仓不一致，按诚实申报改做本仓验证欠账；建议另派 TensorLBM 单。

## 6. 误差判定
本单不产生新物理量，声称的改进均为可复算审计计数：certified 28→31、contradicted 4→1、
E_ARTIFACT_MISSING 3→0、renderables violations 1→0、gate 复算数不变。关键量（上述计数）
误差为 0（工具直出）。图与数据的绑定由 sha256 manifest 保证，
`tests/test_anchor_figure_rebuild.py::test_render_is_deterministic_and_data_driven`
在临时目录重渲染并逐字节比对。

复现：`python tools/anchor_figure_rebuild.py --write && python tools/verified_anchor_audit.py --write
&& python tools/certificate_renderables.py --check && python -m pytest tests/test_anchor_figure_rebuild.py
tests/test_anchor_figure_gate_safety.py tests/test_certificate_renderables.py
tests/test_verified_anchor_certificates.py tests/test_inverse_anchor_certificate.py -q`（44 passed）
