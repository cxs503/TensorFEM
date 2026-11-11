# 交差审计：m14683 GAP-FEM-ADAPTIVE（TensorFEM 自适应网格）→ 判级 L4

## ① 案例介绍
对象 m14683（已 succeeded）：TensorFEM 的 Zienkiewicz-Zhu(SPR) 误差指示子 + 指示子驱动 h 自适应细化，
验证算例为 270° 凹角 L 形板（Williams 奇异性）。物理参数（`artifacts/audit_m14683/repro_study.json` →
`parameters`）：平面应变、E=1、ν=0.3、单位厚度、A=1、半边长 H=1、起始网格
16×16、Dörfler 标记比例 0.5、
细化轮数 2。精确解为 Williams 特征函数（λ=0.5444837367825），
外边界施加其迹作为 Dirichlet 数据，缺口面自由，从而能量范数有闭式值 5.342707。

## ② 怎么做的
1. 取证：读 m14683 工作树源码/工件/测试与 run 日志，核对 `artifacts/adaptive_zz/{study.json,summary.md,benchmark_table.md,evidence.json}`、
   `runs/m14683_adaptive_zz/*.log`、5 张图，均存在且非空。
2. 独立复现：把 src/scripts/tests 复制到 `/tmp/repro_m14683`，按 mission 的 repro_cmd 跑
   `adaptive_zz_study.py`（35 s）→ 得 `repro_study.json`；`adaptive_evidence_pack.py` 重生成 evidence.json
   （仅绝对路径不同）；`pytest tests/test_adaptive_zz.py` → 15 passed（本工作树再跑一次同为 15 passed）。
3. 机器比对 22 个标量判据，最大相对差 0.0e+00（最差键 `adaptive_rate`），逐位一致。
4. 判级按 maturity_gate 判据方案 L0–L5 机器口径逐条核 evidence；本地预跑 judge() 返回 gate_level=4（无缺项）。

## ③ 计算结果（ref vs got，程序化生成）
| quantity | reference | measured | rel. error | tolerance |
|---|---|---|---|---|
| Williams corner exponent lambda | 0.5444837368 | 0.5444837368 | 0.00 % | 0 |
| uniform ladder global rate | -0.2722418684 | -0.2487655684 | 8.62 % | 15% |
| uniform ladder tail rate (last 3 intervals) | -0.2722418684 | -0.2599424287 | 4.52 % | 15% |
| adaptive ladder tail rate (last 6 intervals) | -0.5 | -0.4933364176 | 1.33 % | 15% |
| ZZ effectivity, worst of all levels | 1 | 1.046718105 | 4.67 % | 20% |
| final adaptive relative energy-norm error | 0 | 0.1266586332 | 12.67 % | 15% |

补充：效果因子 θ 均匀阶梯 0.9925–1.0327，
自适应阶梯 0.9909–1.0467（判据 θ∈[0.8,1.2]：
True）；354 DOF 自适应误差 0.67670 对比 770 DOF 均匀
0.73777（2.175× 自由度换 9.0% 精度），
等精度省自由度 2.80×。

## ④ 结果图
左：复现的 L 形收敛阶梯（均匀 vs 自适应）；右：验收表 ref-vs-got 相对误差与声明容差对照。
（图由 `scripts/audit_m14683_report.py` 从 JSON 程序化绘制，无手抄。）

## ⑤ 注意点 / 局限
- 公开 API 缺口（medium）：`src/tensorfem/__init__.py` 导出 0 个 adaptive_zz 符号，两个兄弟模块也未导出，无 CLI/benchmark_registry 入口；
  集成仅到模块级（`adaptive_1d.py:67`、`adaptive_poisson_2d.py:65` import 共享函数）。
- evidence.json 的 manifest 记的是交付提交的父提交 330f1504（生成于 commit 前，dirty=11），实际交付提交 172895c7（合并 d32fe045）；
  文件内容与合并后主树逐位一致，属 provenance 轻微不精确。
- benchmark_table 第 5 行把「能量范数值接近」写成 0.07%，易被误读为精度；真实末级相对误差是 12.67%（预渐近），已在 summary 中体现但未换算成百分比。
- 均匀阶梯整体率 -0.2488 与理论 -0.2722 差 8.6%、尾段差 4.52%，均 >3%；模块自己把这当作 bracket 并如实记录，3% 只有自适应尾段率（1.33%）达到。
- 1D P1 奇异场效果因子 θ=0.233（偏 76.7%）以 honest-FAIL 记录在表内，未被掩盖；2D 范围不受影响。

## ⑥ 误差判定（逐项，显式回答是否 <3%）
- Williams λ 复现：ref 0.5444837367825 vs got 0.5444837367825 → 6.61e-12% **<3%**。
- 自适应尾段收敛率：-0.493336 vs 理论上界 -0.500000 → 1.33% **<3%**。
- 效果因子最差值：1.04672 vs 1 → 4.67% **不 <3%**（判据是 θ∈[0.8,1.2]，满足）。
- 均匀阶梯整体率：-0.248766 vs -0.272242 → 8.62% **不 <3%**（预渐近，尾段 4.52% 亦不 <3%）。
- 末级相对能量范数误差：12.67% **不 <3%**（预渐近；验收对象是收敛率与 θ，非绝对精度）。
结论：L4 的两项核心判据（尾段率、θ 区间）达标/成立；绝对精度在 15% 平台红线内且如实披露，不作粉饰。

## 判定
**L4（基准验证）**，gate_level 预演 4（无缺项），disposition：approve with notes。
未达 L5：单一 2D 线弹性经典问题的方法验证，无实验/跨问题泛化或发表级新贡献；不把「百页报告」当级别。
