# ANCHOR `fem_inverse_identification` — 反问题识别锚点证书化（原 `E_NO_GATES`）

【自找活·infrastructure 席】｜EDIT_WHITELIST: `benchmarks/verified/fem_inverse_identification/`, `src/tensorfem/inverse.py`, `tests/`, `docs/`

## 0. 为什么做这一件（选题依据，非锚点直接指派）

派发锚点 `CASE-FEM-HERTZ-CONTACT` 经核对**已有双重证书**（`benchmarks/verified/fem_hertz_contact` + `..._loadcontrol`，`docs/CASE-FEM-HERTZ-CONTACT-verification.md`、`...-truncation-extrapolation.md`，a/δ 误差 −0.445%/−1.010%），重做=违反 owner 2026-11-09 不重复令；故本单改做同族**验证欠账**：`benchmarks/verified/fem_inverse_identification` 有驱动无证书，`tools/verified_anchor_audit.py` 判 `E_NO_GATES`，而 fail-closed 注册表测试 `tests/test_verified_anchor_certificates.py` 在 main 上是**红的**（`unregistered=['fem_inverse_identification']`）。修复它既闭证又让 CI 门从红转绿。

## 1. 案例介绍（物理背景 / 模型 / 参数）

`tensorfem.inverse` 是求解器在环（solver-in-the-loop）的参数反演基础设施。两个案例各带**独立由构造给出的真值**：

| 案例 | 正演 | 未知量 | 真值 | 观测噪声 |
|---|---|---|---|---|
| decay | 单指数+偏置 `T(t)=Tinf + A exp(-t/tau)`（与 LBM 速度衰减同构） | tau, A, Tinf | tau=4.0 s, A=60.0, Tinf=20.0 | 1% 相对 |
| axial-bar | 真实 `tensorfem.model.TrussModel` 4 单元轴向求解 | E | E=210 GPa | 0.5% 相对 |
| axial-bar-product（负控） | 同上，但同时求 E 与 A | E, A | 数据只含 E·A=2.1e8 N | 0.5% 相对 |

多起点：3 个种子 × 3 个 start；收敛容差与判定沿用 `tensorfem.inverse_cases.TOLERANCE = 0.03`（平台 3% 标准）。

## 2. 怎么做的（方法/流程/关键设置）

1. 复算驱动 `benchmarks/verified/fem_inverse_identification/run.py`（单进程 ~9.8 s，`git_commit=c8b9889c`）。
2. 给驱动加**证书出口**：写 `result.json`（机器可读 dump）+ `gates_verified.json`（schema `tensorfem.verified-gates.v1`，经 `benchmarks/verified/_harness.py::gate_manifest`）+ 图 `inverse_identification_recovery.png`，三者都在锚点目录内并入库（审计查存在性 + git 跟踪）。
3. 阈值**先于**运行写死在 `CRITERIA`：3%（标量恢复，平台标准）、3%（E·A 乘积）、1e-6（多起点散布）、0（未收敛数）、1.0（日志重放一致）。
4. `--from-dump` 支持从已提交 `result.json` 重建证书（零算力），证书本身的生成路径可被测试复算。
5. 刷新审计总账 `tools/verified_anchor_audit.py --write`（`certified 27 → 28`，门数 314 → 325，债务漂移清零）。

## 3. 计算结果（全部从 `result.json` 程序化注入，零手抄）

| 检查 | 参考值（独立） | 反演值 | 相对误差 | 闸门 | 判定 |
|---|---:|---:|---:|---|---|
| tau（seed 1） | 4.0 s | 4.033928945 s | +0.8482% | <3% | PASS |
| tau（seed 2） | 4.0 s | 4.033928965 s | +0.8482% | <3% | PASS |
| tau（seed 3） | 4.0 s | 4.033928945 s | +0.8482% | <3% | PASS |
| E（轴向杆） | 210 GPa | 210.653422 GPa | +0.3112% | <3% | PASS |
| E·A（负控，E 与 A 皆未知） | 2.1e+08 N | 210653421.9 N | +0.3112% | <3% | PASS |
| 可辨识性（负控） | 应报 not identifiable / rank 1 | identifiable=False, effective_rank=1 | — | rank ≤1 且 flag=0 | PASS |
| 多起点散布（3 种子） | 与起点无关 | (max−min)/tau_true = 4.874e-09 | — | ≤1e-6 | PASS |
| 同种子日志重放 | 逐位一致 | sha256 `2b2551c7b4228abc…` 重复 | — | 一致 | PASS |
| 异种子日志 | 必须不同 | seed2 `090ee75ad965f843…` | — | 不同 | PASS |

证书：`benchmarks/verified/fem_inverse_identification/gates_verified.json`（11 条闸门，failed_gates=[]，overall passed=True）。
条件数（负控）= 4.453e+11（无容差口径的硬事实：两个未知量只由一个乘积约束）。

## 4. 结果图

`benchmarks/verified/fem_inverse_identification/inverse_identification_recovery.png`（随报告内嵌）：
左=观测/真值/三种子反演曲线；中=逐迭代归一化 loss（日志重放）；右=参考 vs 反演相对偏差条形图与 ±3% 接受带（标注负控秩与可辨识性）。

## 5. 注意点（坑 / 局限 / 适用范围）

**D1（已修，缺陷）** `src/tensorfem/inverse.py` 的不可辨识告警只打印**数值秩**：负控实测输出 「(rank 2 of 2, condition 4.45e11)」——满秩却宣称不可辨识，读者无法判断触发条件。根因：`identifiable = effective_rank == n and condition <= cond_limit`，但消息没给 effective_rank 与 cond_limit。改为打印 `numerical rank / effective rank of n / condition vs cond_limit`，并加断言测试。
**D2（已修，本单引入后自查修掉）** 证书写盘路径原为模块常量，`tests/test_inverse_identification_benchmark.py` 只 monkeypatch `OUT`，导致测试会**覆写仓库内已入库证据**（测试非自封闭）。改为 `ARTIFACT_DIR` + `artifact_paths()` 调用时解析，测试同时重定向该目录，并验证跑完测试后 `result.json` 的 md5 不变。
**L1（诚实记录）** 两案例都是严格意义的 inverse crime：观测由**同一正演模型**生成（单指数 / 4 单元桁架），所以这里的误差是优化器的误差，不含模型误差；正演离散误差分离研究未做。
**L2** 噪声只取单次实现（decay 1%、杆 0.5%），报出误差是单 realisation，不是集合平均；未做噪声水平的敏感性扫描。
**L3** 日志重放只在**同进程**内验证（同解释器、同 torch 线程调度）；跨进程确定性未断言。
**L4** `effective_rank` 依赖奇异值截断容忍（s_max/cond_limit），是约定而非定理；条件数是容差无关的替代陈述。
**L5（未修，如实登记）** 总账仍余 8 条债务（6 `E_NO_GATES`、2 `E_ARTIFACT_MISSING`）。其中 `fem_sent_j_integral` 原先在债务表里写的是 `E_NO_GATES`，实际缺陷是声明的 6 张 PNG 从未入库（`E_ARTIFACT_MISSING`）——本单只把**代码订正为真相**（`--write`），没有伪造图片；恢复该锚点需从其 `result.json` 重建 6 张图或重跑求解器，建议单独派单。

## 6. 误差判定（是否 <3%，逐项显式回答）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---:|---|
| tau（seed 1） | 生成真值 4.0 s | +0.8482% | **是** |
| tau（seed 2） | 生成真值 4.0 s | +0.8482% | **是** |
| tau（seed 3） | 生成真值 4.0 s | +0.8482% | **是** |
| E（轴向杆） | 生成真值 210 GPa | +0.3112% | **是** |
| E·A（负控） | 生成真值 2.1e8 N | +0.3112% | **是** |
| 可辨识性判定（负控） | 应报 rank 1 / not identifiable | 完全一致（rank 1, flag False） | 是（离散判定，无百分比） |

结论：所有闸门 PASS，最大标量误差 **0.848%（tau）/ 0.311%（E）**，均远小于 3%；唯一超过 3% 的量不存在（负控乘积误差 0.311%）。诚实边界见 §5 的 L1–L5。

## 7. 复现
```
PYTHONPATH=src python benchmarks/verified/fem_inverse_identification/run.py        # ~9.8 s, 写 dump+证书+图
PYTHONPATH=src python benchmarks/verified/fem_inverse_identification/run.py --from-dump  # 零算力重建证书
python tools/verified_anchor_audit.py            # 该锚点应 certified
python -m pytest tests/test_inverse_anchor_certificate.py tests/test_verified_anchor_certificates.py -q
```
