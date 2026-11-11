# 【交差审计】CASE-NAFEMS-LE1 椭圆膜应力集中（TensorFEM）：裁定 L3，非 L4

## ① 案例介绍
NAFEMS LE1 = 含椭圆孔平板单轴拉伸的应力集中基准（Inglis 1913 闭式 K_t=1+2b/a）。审计对象按锚点锁定其真实载体 mission 15238；
**任务单点名的 m15261 实测是 CASE-NAFEMS-LE11（椭球壳内压，task_key=case-nafems-le11，工件 runs/nafems_le11/*）——scope 绑定错位，已写入 verdict。**

| 参数 | 值 |
|---|---|
| semi_x / semi_y | 1.0 / 0.5（K_t=2.0） |
| 外边界 | 共焦椭圆 ρ₀=5.0（施加精确面力） |
| 单元/网格 | Q4，(θ,ρ) 结构网格 n=16…48 |
| 容差 | <5%（本单独立复算 K_t=2.0 一致） |

## ② 怎么做的（审计方法）
三件套取证：工件 sha256 + git 血缘（PR #686 / merge 8b7bf3e3 / 是否在 main）、报告表 vs JSON 逐格比对、注册裁判行 SQL、
独立闭式复算、报告 §⑤ 三处数字的对抗探针（grading=1.0 实跑 n=24、Richardson 复算、观测阶复算）、重跑 repro。

## ③ 计算结果（实测数字）
| 项 | 实测 |
|---|---|
| tensor_benchmark_cases 中 anchor=CASE-NAFEMS-LE1 行数 | 0（全表 14 行无 NAFEMS 条目） |
| 最细网格 n=48 尖点误差 | 1.7635%（<5% ✓） |
| 粗网格诚实 FAIL | n=[16]（5.29%/4.21% 附近，未粉饰） |
| 载荷轴尖点 | -1.0（理论 -1.0，误差 1.85%） |
| 报告表 vs JSON | 6/6 行一致（四舍五入容差内） |
| 独立解析复算 | K_t=2.0（自写公式）vs 模块 exact=2.0 ✓ |
| grading=1.0 @ n=24 | 12.05% vs grading=3.0 的 3.51%（×3.44，报告称"~50%"） |
| Richardson（p=1, n=40/48） | 1.999494（报告称"2.000000"） |
| 观测阶复算 | [1.0179, 1.0054, 0.9961, 0.9896, 0.987]（报告称 0.95，实为另一算例 sharp 值 0.9531） |
| 重跑复现 | NOT_COMPLETED（22:05:20 启动，审计窗口内 0B 输出、工件 mtime 未变） |
| mission 15238 平台报告行 | 0，report_owed=f |

## ④ 结果图
图1 收敛性（重绘自工件，非手抄）；图2 报告声明 vs 实测探针。

## ⑤ 注意点（坑与局限）
- 外边界用**精确 Inglis 面力**（elliptic_hole.py:406），截断误差被消除 → 这是 patch test 口径，1.76% 是插值/应力恢复误差，不是对 NAFEMS 有限板原算例的复现；
- 引用 "NAFEMS (1990) Test LE1" 但无原表原页，实际目标值自推自算（外部文档锚点 WITHHELD）；
- 无 LE1 专用回归测试；artifacts 未进 main（门禁滤除）；
- 无外网/知识库无 NAFEMS 源 → 手册数值本身本单无法核实。

## ⑥ 误差判定（是否 <3%？）
- 尖点环向应力 vs 闭式：**1.76% < 3% ✓**（最细网格 n=48；粗网格 n=16 为 5.29%，诚实 FAIL 未隐藏）；
- 载荷轴尖点：1.85% < 3% ✓；位移最大相对误差 0.016% ✓；力平衡残差 4.5e-15 ✓；
- **但**该"基准"没有注册裁判行（tensor_benchmark_cases 0 行），手册锚点未取得 → 结论限于"与 Inglis 闭式一致"，不外推为"通过 NAFEMS LE1 验收"。

**裁定：L3**（L1/L2/L3 实测成立：模块门 + 349 行真基准脚本 + 生产路径 ContinuumModel/solve_continuum 调用方且已合入 main）；
L4 不记：注册裁判行缺位 + 手册锚点 WITHHELD + 复现性 NOT_COMPLETED + F4-F6 三处报告数字错标/超范围。
verdict 已落 tensor_maturity_verdicts id=283（repo=TensorFEM, level=3, verdict_by=board-audit, gate_state=pending）。
