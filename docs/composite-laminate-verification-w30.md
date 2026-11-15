# W30 · 层合板验证：外部锚核销与闭合残差表

任务：`composite-laminate-verification`（mission 21513），仓 TensorFEM，分支 `agent/m21513`。

产物：

| 路径 | 内容 |
|---|---|
| `papers/artifacts/w30/composite-laminate-verification/laminate_anchor_closure.json` | 主工件：锚件对表、逐点闭合残差、判定、根因、能力缺口 |
| `papers/artifacts/w30/composite-laminate-verification/fig_*.png` | 三张图（ABD 残差 / 指数 parity / 弯曲 CLT-FSDT-FE） |
| `benchmarks/anchors/laminate-failure-criteria-anchor.json` | 锚件台账（逐条分级 + 检索日志 + sha256） |
| `benchmarks/composite_laminate_w30/symbolic_anchor.py` | 锚件生成器（60 位 mpmath） |
| `benchmarks/composite_laminate_w30/clt_independent.py` | 独立 CLT + 三种失效判据实现 |
| `benchmarks/composite_laminate_w30/run_closure.py` | 闭合跑批 + `--selftest` |
| `benchmarks/composite_laminate_w30/publish_report.py` | 报告发布（tensor_reports id 19445） |

复现：

```bash
python3 benchmarks/composite_laminate_w30/symbolic_anchor.py --digits 60
PYTHONPATH=src python3 benchmarks/composite_laminate_w30/run_closure.py --selftest
python3 benchmarks/composite_laminate_w30/publish_report.py
```

## 1 判定摘要

| 判据 | 门限 | 实测 | 结论 |
|---|---|---|---|
| ABD 矩阵闭合残差 | ≤ 0.1 % | TensorFEM 4.34e-14 %；独立实现 3.34e-14 % | PASS |
| 失效指数逐点偏差 | < 15 % | 240 点最差 2.44e-13 %，中位 7.5e-14 % | PASS |
| 判据恒等式（许用值处指数≡1） | ≤1e-12 | 4.44e-16 | PASS |
| 覆盖度 | ≥2 铺层 × ≥3 工况 | 4 铺层 × 4 膜工况 + 2 横向工况 × 3 判据 | PASS |
| 剪切变形对指数的影响 | < 15 % | a/t=100：0.08 %；a/t=10：9.28 % | PASS |
| **外部规范锚（DNV/CSR 层合板条款）** | 必须逐条核销 | **WITHHELD / NOT_SUPPORTED** | **未闭合，如实记账** |

## 2 方法：三条互相独立的路径

1. **ANCHOR**（`symbolic_anchor.py`，60 位 mpmath）：`Q = S⁻¹`、`Q̄ = T⁻¹QR`、厚度精确积分；
   60 位与 90 位之间相对偏差 3.4e-62（精度稳定性自检）。
2. **IND**（`clt_independent.py`，双精度 numpy）：教科书幂次公式，不 import tensorfem。
3. **TF**：TensorFEM `Laminate.abd()`（旋转三维刚度 + 凝聚）与 `ply_stiffness_3d` 的 Q。

铺层：`[0/90/±45]s`、`[±30/0]s`（任务要求）+ `[0/45/-45/90]`（B≠0）、`[0/45]`（A16≠0，用于耦合项根因）。
膜工况：单轴拉伸 1e6 N/m、双轴 5e5/5e5、纯剪 5e5、单轴压缩 −1e6。
横向工况：交叉铺层 `[0/90/0/90]s`，q0 = 1e5 Pa，a/t = 100 与 10。

## 3 关键结果

- ABD 三路一致到 1e-16~1e-14 相对量级；`As`（未加 5/6 修正）同样一致，含不平衡铺层 `[0/45]`。
- 失效指数 240 点全部在门限内，偏差量级为双精度舍入（1e-13 %）。
- 弯曲：a/t=100 时 FSDT 与 CLT 挠度差 0.235 %、指数差 0.08 %；a/t=10 时挠度差 23.2 %、
  Tsai-Wu 指数差 9.28 % —— 这是**剪切变形理论差异**，不是实现缺陷，且指数仍在 15 % 带内。
- Q4 Mindlin FE（MITC）与解析 FSDT：n=8/16/24 时挠度偏差 −1.18 % / −0.29 % / −0.13 %（单调收敛）。

## 4 过程中发现并修正的问题

1. **本次实现自身的符号错误（已修）**：横向剪切耦合 `Q45` 初版写成 `(G23−G13)cs`，与 TensorFEM
   的三维旋转路线差 12 %。用应变能 `S45 = cs(1/G13 − 1/G23)` 显式展开定符号后确认
   **TensorFEM 正确、本工件初版错误**，已改为 `(G13−G23)cs`，修正后三路 1e-16 一致。
   该结论已写入工件 `root_causes`，避免把自身错误误报为平台缺陷。
2. **[0/90/±45]s 不是特殊正交铺层**：A16=A26=0（平衡）但 D16/D11 = 0.0341，TensorFEM 闭式
   Navier 参考解要求 D16=D26=0，因此弯曲表改用交叉铺层。这是真实弯曲-扭转耦合，非噪声。
3. **归一化陷阱**：对称铺层 B 块与部分 D16/D26 是精确零（60 位下只剩 1e-37 级舍入），
   逐块归一化会把 1e-45 的舍入放大成 100 %。工件改为按该铺层 A/B/D 的最大元素归一化。

## 5 能力缺口（本单只记录，不改 src/）

- `src/` 无层合板级失效判据（只有 `cohesive_failure.py` 的界面牵引律），求解器无法输出首层失效裕度。
- `plate_fem` / `fsdt_plate_fem` 只返回挠度，docstring 承诺的应力恢复缺失。
- `_fsdt_rigidity` 拒绝 `A45 ≠ 0`，不平衡角铺层无法进入 FSDT 板路径。

## 6 局限（未完成即未完成）

- **DNV / CSR 层合板条款值取不到**：DNV 旧 PDF 路径 301 到 JS 页面、`rules.dnv.com` 为 SPA、
  NTRS 403、MIL-HDBK-17 403、OSTI 429、HAL 有反爬墙。台账 `LAM-EXT-RETRIEVAL-LOG` 逐条记录
  URL 与 HTTP 状态。故本单数值锚是**经典解 + 判据恒等式**，不是规范条款转录。
  CSR-H 是钢结构规范，无层合板判据 → `NOT_SUPPORTED`（依据仓内 w28 转录件 sha256 4c6a6761…）。
- 强度许用值（Xt/Xc/Yt/Yc/S）是**声明输入**，无出处（`LAM-STRENGTH-INPUTS` = WITHHELD）；
  所有对比量对其零次或一次齐次，不影响 PASS/FAIL。
- 弯曲表的 FE 层应力是派生估计（FSDT 精确曲率按 FE/精确挠度比缩放），非独立 FE 应力恢复。
- 三维/层间应力（`LayeredExact3D`、`layered_hex20_fem`）本单 **未跑**：状态 `not_run`，不算通过。