## ① 案例介绍

3D 厚壁球壳（内半径 a=8 m，外半径 b=10 m，
壁厚 t=2 m，b/a=1.25）受内压
p=1e+06 Pa，外表面自由，线弹性小变形单材料。
E=2.1e+11 Pa，ν=0.3。
参照=Lamé 解（Lamé 1852；Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed., 1970, §5.6）：
σ_r=A−B/r³，σ_t=A+B/(2r³)，u_r=[(1−2ν)Ar+(1+ν)B/(2r²)]/E，A=pa³/(b³−a³)，B=Ab³。
解析应变能 26833.4 J、壳体积 2044.13 m³（程序化读自工件）。
本单是**交差审计**（非重做）：核验 m15656 三件套（run_artifacts / ref-vs-got / repro_cmd）与六段报告，判级落表。

## ② 怎么做的（审计方法）

1. 取证：`tensor_dev_missions` id=15656 的 result JSON（head_commit=null、fail_stage=`budget_exhausted+forced_finish`、
   git_status 全 `??`）；m15656 工作树 6 个交付文件；工件 JSON（sha256 入证）。
2. 独立复算：在 m15656 工作树执行 `PYTHONPATH=$PWD/src python3 benchmarks/case_hollow_sphere_3d.py --out /tmp/repro2`
   ，逐阶梯比对 worst 误差（表 ④），确认不是手抄。
3. 测试复跑：`python3 -m pytest tests/test_hollow_sphere_3d.py -q` → **1 failed, 11 passed in 172.23s**（红）。
4. 真伪核查（对抗式）：读 `solve_hollow_sphere` 全路径，确认无解析位移注入——刚度由生产核
   `hex8_stiffness`/`hex20_stiffness` 装配（:317/:320），载荷为一致面载荷（:195），约束仅 6 个切向 gauge
   （:248 `_gauge_dofs`），解由 Jacobi-PCG 求得（:328/:383），应力由 σ=D·B·u 后处理（:332）。
5. 裁判核查：`tensor_benchmark_cases WHERE anchor_slug='CASE-HOLLOW-SPHERE-3D'` → **0 行**（裁判缺位）；容差只能取
   自算例池 id=48 `tol_policy='<5%'` 与驱动 `TOLERANCE=0.05`。

## ③ 计算结果（ref vs got，程序化注入）

选定族 = HEX20（曲边界投影），最细网格 n=6（432 单元 / 7158 dof）：

| 探针 | 参照 (Lamé) | FE got | 相对误差 | <5% |
|---|---|---|---|---|
| radial_displacement_inner | 6.67291e-05 | 6.67196e-05 | 0.014% | ✅ |
| radial_displacement_outer | 5.2459e-05 | 5.24223e-05 | 0.070% | ✅ |
| radial_stress_inner | -704941 | -731292 | 3.738% | ✅ |
| hoop_stress_inner | 1.92624e+06 | 1.89387e+06 | 1.680% | ✅ |
| von_mises_inner | 2.63118e+06 | 2.62517e+06 | 0.229% | ✅ |
| hoop_stress_outer | 1.67599e+06 | 1.65213e+06 | 1.424% | ✅ |

次级探针（σ_r 在外表面 →0，按 p 归一）：

| 探针 | 参照 | FE got | 归一误差 |
|---|---|---|---|
| radial_stress_outer | -204439 | -225017 | 2.058% (of p) | ✅ |

全阶梯：

| 族 | n | 径向层 | 节点 | 单元 | dof | worst 探针误差 | 主控探针 | 应变能误差 | 网格体积误差 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| HEX8 | 4 | 2 | 294 | 192 | 882 | 28.182% | radial_stress_inner | 9.071% | 7.116% | FAIL |
| HEX8 | 6 | 2 | 654 | 432 | 1962 | 15.350% | radial_stress_inner | 4.670% | 3.284% | FAIL |
| HEX20 | 4 | 2 | 1066 | 192 | 3198 | 2.875% | hoop_stress_inner | 0.185% | 0.136% | PASS |
| HEX20 | 6 | 2 | 2386 | 432 | 7158 | 3.738% | radial_stress_inner | 0.041% | 0.028% | PASS |

复算对照（本审计独立重跑 vs 交付工件）：

| 阶梯 | 工件 worst | 重跑 worst | 绝对差 |
|---|---|---|---|
| HEX8 n=4 | 28.1823% | 28.1823% | 0.00e+00% |
| HEX8 n=6 | 15.3504% | 15.3504% | 0.00e+00% |
| HEX20 n=4 | 2.8751% | 2.8751% | 0.00e+00% |
| HEX20 n=6 | 3.7381% | 3.7381% | 0.00e+00% |

自由残差 / gauge 残差 / 闭合面载荷合力·合力矩：分别 ~1e-12 / ~1e-13 / ~1e-16（工件 `equilibrium`、`loads` 字段）。
HEX8 基线如实失败：n=4 28.18%、n=6 15.35%——算例没被几个三线性单元做出来，这点是诚实的。

## ④ 结果图

三幅图均由工件 JSON 程序化生成（零手抄）：探针误差条形图（含 5% 验收线）、worst 探针与能量误差随 dof 的收敛图、
hex20 最细网格 got-vs-ref 一致性图（含 ±5% 带）。图中可见 **HEX20 worst 探针非单调**
（n=4 2.875% → n=6 3.738%），而能量范数误差单调下降。

## ⑤ 注意点（坑与局限）

- **交付物无提交身份**：mission 因预算耗尽 forced_finish，head_commit=null，6 个交付文件全部 untracked，
  `origin/main` 里查不到 `src/tensorfem/hollow_sphere_3d.py`——一次 `git clean` 就归零。
- **自带测试红**：1 failed, 11 passed in 172.23s；失败项 `test_refinement_reduces_energy_error` 用 `solve_hollow_sphere(2,2,order=2)`
  触发退化 HEX20 雅可比异常（模块本身拒绝 n=2 网格是对的，是测试基准取错）。文档却声称"15 invariants"（实为 11 个测试函数/12 项）。
- **无集成入口**：`benchmark_registry.py`/`cli.py` 均无 hollow 相关注册（grep 0 命中），没有验证门会跑这个算例。
- **裁判缺位**：`tensor_benchmark_cases` 无本锚点行；`tensor_case_pool` id=48 仍 `dispatched`（mission_id=15431 未更新）；
  `tensor_reports` 无 mission 15656 的报告行（六段报告从未上平台）。
- **文档与工件不同步**：docs §4 把 **n=4（最粗）** 标成"finest selected mesh"（192 单元/3198 dof、worst 2.875%），
  而工件最细是 n=6（432 单元/7158 dof、worst 3.738%）；§5 收敛叙述"0.0 in 1/n"是空话。
- 探针方法学：层体积平均 vs 层加权质心半径处的 Lamé 值，误差含探针位置偏移；外表面 σ_r 用 |Δσ_r|/p 归一（合理且已声明）。

## ⑥ 误差判定

关键量 vs 基准（解析 Lamé）逐项误差见 ③ 表：**6 个主探针 + 1 个 p 归一探针全部 < 5%**，
最差 3.738%（radial_stress_inner）→ **是 < 5%（算例容差），且 < 3%？否**（3.738% > 3%，按 3% 尺子不合格；
说明：算例池判据是 <5%，3% 是平台通用精度尺，此算例未达 3%，如实记录）。
能量范数误差 0.041%、网格体积误差 0.028%。

**裁定 L2**：数值实质（真 FE + 解析基准 + 完整误差分析 + 可复现）已达到 L4 级别的证据密度，
但 L3（系统集成）这一级未落地——无 registry/CLI 调用者、无基准表行、无提交身份、自带测试红——故不越级认证；
返修项=F1/F2/F3/F4/F5/F6/F7（清单见 `artifacts/audit-m15809/evidence.json`）。


---

## 附：审计发现（严重度排序）与处置

| 编号 | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | high/blocking | 交付物无提交身份：`head_commit=null`，6 个文件全 untracked，`origin/main` 无 `src/tensorfem/hollow_sphere_3d.py` | `tensor_dev_missions` id=15656 result；`git cat-file -e origin/main:src/tensorfem/hollow_sphere_3d.py` 失败 |
| F2 | high/blocking | 自带测试红：1 failed, 11 passed in 172.23s，失败项 `test_refinement_reduces_energy_error` | 审计复跑 `/tmp/pytest_hollow.log` |
| F3 | high（封顶 L2） | 无集成调用者：`benchmark_registry.py`/`cli.py` grep 'hollow' 0 命中 | 源码 grep |
| F4 | medium | 基准裁判缺位：`tensor_benchmark_cases` 无 `CASE-HOLLOW-SPHERE-3D` 行 | psql |
| F5 | medium | 算例池未销账：`tensor_case_pool` id=48 status=dispatched mission_id=15431 | psql |
| F6 | medium | 平台报告缺失：`tensor_reports` 无 mission 15656 行 | psql |
| F7 | medium | 文档与工件不同步：docs §4 把 n=4（最粗）标成 finest（2.875%），实为 n=6 worst 3.738%；§5 收敛叙述空话 | `docs/case-hollow-sphere-3d.md` vs 工件 JSON |
| F8 | low | 探针方法学：层体积平均对比层质心半径处解析值，误差含位置偏移；外表面 σ_r 用 |Δσ_r|/p 归一（已声明、可接受） | `probe_metrics` |

## 附：检查项与未做项

已做：missions 记录取证 / benchmark_cases 与 case_pool 与 reports 查表 / 独立重跑算例并逐阶梯比对 / pytest 复跑 /
`origin/main` 提交存在性核查 / 求解路径源码对抗式审阅（确认无解析位移注入）。

未做：集群作业 cj1187（完整阶梯+pytest on loong64）在审计窗口内仍在排队（queued_ahead=371），本地节点完成复算；
GitHub CI（x86）未核；n=2 退化网格是否应支持未判（模块拒绝是合理设计，问题在测试基准选择）。

## 附：处置

**request changes（返修）**，level=L2。返修清单：F1 把交付文件提交到分支并合入 main；F2 修正
`test_refinement_reduces_energy_error` 的基准（用 n=4 vs n=6，而非 n=2），使 suite 全绿；F3 在
`benchmark_registry.py` 注册 `shell.hollow_sphere.*` 证据项（可复用 `benchmark_evidence` 模式）；F4 登记
`tensor_benchmark_cases` 锚点行（metric=探针相对误差，tolerance=<5%）；F5 关闭 case_pool id=48；F6 上平台六段报告；
F7 重生成文档使其引用工件最细网格数值。
