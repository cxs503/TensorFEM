# GAP-PP-4-opt-loop: `tensorfem.inverse` 反问题优化基础设施 —— 验收证据

> 本文件由脚本从工件程序化生成（零手抄）：数据源 `runs/inverse-identification-benchmark/benchmark.json`、`runs/inverse-identification-benchmark/*/summary.json`、`.../convergence.jsonl`。

## 1. 案例介绍（物理背景/几何/参数）

反问题基础设施服务于「不是求解、而是标定」的工程流程：已知观测反推材料/工况参数。本仓用两个独立真值案例验证：

- **decay（单指数衰减）**：`T(t) = Tinf + A exp(-t/tau)`，与 LBM 速度衰减信号同构（单指数+常数偏置）。真值 `tau = 4.0 s, A = 60.0, Tinf = 20.0`，41 个采样点 `t ∈ [0,20] s`，观测含 1% 相对噪声。
- **axial-bar（真实 FE 求解器在环）**：4 单元轴杆 `TrussModel`（L=2 m, A=1e-3 m², F=1e4 N），自由节点位移为观测量，噪声 0.5%，真值 `E = 210 GPa`。每次前向评估都组装并求解一次有限元系统。
- **axial-bar-product（负控/可辨识性）**：同时反演 `E` 与 `A`，但数据只含乘积 `E·A`。正确的基础设施必须报告 `identifiable: False` 与弱方向，而不是给出两个「自信」的数。

## 2. 怎么做的（方法/流程/工具链）

- 优化器：`src/tensorfem/inverse.py` 中的 Levenberg–Marquardt，有界参数空间，正定量走对数坐标；带信赖域/增量比接受判据，终止判据 gtol/xtol/ftol/loss-plateau。
- loss 组合：`loss = Σ data residuals² + Σ Tikhonov 项`，每项逐迭代分项记录（`loss_components`），正则权重可审计。
- 机读收敛日志：每次尝试一行 JSONL（schema `tensorfem.inverse-convergence/1`），summary 里钉 `convergence_log_sha256`，可重放可复核而非「信任」。
- 种子控制：`SeedControl(seed, starts)`，多起点抖动来自 `torch.Generator().manual_seed(seed + 1009 i)`，起点全记录。
- 可辨识性诊断：加权 Jacobian 的秩/条件数/奇异值/线性化相对标准误/零空间方向；秩亏时 `converged` 与 `identifiable: False` 同时出现。
- 入口：库 `tensorfem.inverse_cases.identify()`；CLI `python -m tensorfem.cli inverse-identify --case ... --seed ...`；注册表 `tensorfem.benchmark_registry.run_registered_benchmarks()`（`inverse.axial_bar.modulus`，`tensorfem verify` 路径）；基准脚本 `benchmarks/verified/fem_inverse_identification/run.py`。

## 3. 计算结果（数字表，程序化注入）

| 检查 | 参考值 (独立真值) | 反演值 | 相对误差 | 判据 | 结论 |
|---|---|---|---|---|---|
| decay seed=1 | 4 | 4.0339289 | 0.8482% | < 3% | PASS |
| decay seed=2 | 4 | 4.033929 | 0.8482% | < 3% | PASS |
| decay seed=3 | 4 | 4.0339289 | 0.8482% | < 3% | PASS |
| axial-bar seed=0 | 2.1e+11 | 2.1065342e+11 | 0.3112% | < 3% | PASS |
| axial-bar-product (E,A 同反演) | rank 1 / identifiable False | rank 1 / identifiable False | 乘积 E·A 0.3112% | 判据反转：必须判为不可辨识 | PASS |
| 同种子日志重放 | 哈希逐位相同 | identical=True, 不同种子哈希互异=True | — | 精确 | PASS |


整体门：**PASS**（tolerance = 3%）。环境：loongarch64, Python 3.11.6, git 442d393a。


复现命令：`PYTHONPATH=src python benchmarks/verified/fem_inverse_identification/run.py`

## 4. 结果图

`runs/inverse-identification-benchmark/inverse-identification-benchmark.png`：左=观测/真值/三种子反演曲线；中=逐迭代 loss（日志重放）；右=参考值 vs 反演值相对偏差条形图与 3% 接受带（E+A 负控标注秩与可辨识性）。

## 5. 注意点（坑与局限）

- 迭代次数/收敛路径对**初值**敏感（bar 案例 5 次迭代、decay 22 次），但三种子 tau 一致到 1e-9 —— 说明结果是数据而非起点主导。
- 日志哈希含 provenance（脚本路径 + git commit），跨 commit 不可比；做重放对比时须固定 provenance（本基准用固定字典）。
- 可辨识性诊断是**线性化**结论：非线性强耦合下秩/条件数只是局部判据。
- 噪声模型假设同方差（`sigma` 给定）；异方差数据应改用加权残差。
- 负控 `plate.foundation.pasternak_zero_reduces_to_winkler`（非本单产物）经审计实为同一调用自比较（恒等式），不能发现缺陷——本单在 `registry_evidence_audit.EVIDENCE_CLASSES` 中据实标注为 `identity`。

## 6. 误差判定

- 关键量逐项误差见上表；最大相对误差 **0.8482%**（tau，三种子同值）。
- 是否 < 3%？**是** —— 全部标量在 3% 判据内；负控以「必须判不可辨识」为判据，实测 rank=1 / identifiable=False，符合预期。
- 该系统误差来源为观测噪声（1% / 0.5%）与最小二乘估计量的统计涨落，报告值（`reported_relative_stderr`）与实测偏差同量级（tau 0.75% vs 0.85%，E 0.32% vs 0.31%），说明误差已被正确估计而非掩盖。

