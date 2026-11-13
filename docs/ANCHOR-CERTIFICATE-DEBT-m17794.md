# 锚点证书债务收账（m17794）：fem_thermal 驱动修复 + fem_sent_j_integral 证据补齐

锚点指派为 `CASE-LAX`（domain=TensorFVM）。核对台账时该锚**已在 TensorFVM main 完成**
（`src/tensorfvm/compressible/lax.py`，Merge #425/m15239；报告 `docs/lax-shock-tube/report.md`，
星区 p*/u*/ρ* 两套独立精确解互证 ≤ 1e-15，Rankine-Hugoniot 残差 ≤ 2e-16）。按不重复纪律不重做，
改做本仓（TensorFEM）自身的验证欠账——`benchmarks/verified/anchor_certificate_debt.json` 的
**证书债务账本**（shrink-only，`tests/test_verified_anchor_certificates.py` 对其 fail-closed）。

## 1. 发现的两个真实缺陷

| # | 锚点 | 缺陷 | 根因 |
|---|---|---|---|
| D1 | `fem_thermal` | `E_NO_GATES`：证书从未能产出 | 驱动与 `tensorfem.thermal` 现行 API 脱节——`solve_transient_thermal(model, T_init, t_final, n_steps=…)` 签名已变为 `(model, time, initial_temperature, theta)`；且 `rect_mesh_2d` 的 Q4 连接顺序是顺时针（相对参考单元），`assemble_thermal` 抛 `Q4 element has non-positive Jacobian`。即：该锚点**根本无法运行**，证书缺口不是"忘了提交"，是驱动坏死 |
| D2 | `fem_sent_j_integral` | `E_ARTIFACT_MISSING`：证书声明了 6 张 PNG，仓库里一张都没有 | 证书由 branch 生成后未提交图件；`--stage assemble` 只重建"组装"层，图件随之丢失。审计因此把该锚点从 certified 降级为 contradicted，而债务账本仍按旧的 `E_NO_GATES` 记录 → `defect_drift`，main 上 `test_no_unregistered_certificate_defect` 变红 |

## 2. 修复

1. **D1**：修 `benchmarks/verified/fem_thermal/run.py`
   - 时间推进改用现行 API：`time = linspace(0, t_final, n_steps+1)`，`result.temperature[-1]`；
   - Q4 连接改为相对参考单元的逆时针 BL, BR, TR, TL（`[n0, n0+ny+1, n0+ny+2, n0+1]`），Jacobian 转正；
   - 证书改由共享 `_harness.gate_manifest` 发规范信封（`schema/anchor/title/criteria/gates/passed/failed_gates/artifacts/limitations`），闸门改为 `Gate` 对象（显式 `operator`、`reference` 写下解析解出处），消除 `E_SCHEMA/W_OPERATOR_*` 风险；
   - case C 阶梯改为**时空同步加密** `n_steps = 10·(nx/4)²`（dt ∝ h²），使后向 Euler 的 O(dt) 误差不与 O(h²) 空间误差互相遮蔽。
2. **D2**：`python benchmarks/verified/fem_sent_j_integral/run.py --stage assemble` 从**已提交缓存**（`_cache/{ladder,control,plane_strain}.json`）零求解重建 6 张图（11 s，纯 matplotlib），并入库；证书闸门复算 `passed=true, failed_gates=[]`（17 条闸门，最严阈值 1e-09，K 相对误差最大 0.0104627）。

## 3. 机读结果（数字全部由工件读出，零手抄）

### fem_thermal 证书（`benchmarks/verified/fem_thermal/gates_verified.json`）

| 闸门 | 参考（独立解析解） | 值 | 判据 | 结论 |
|---|---|---|---|---|
| case_a_machine_zero | T(x) = T0 + (TL - T0) x / L (LINE2 nodally exact) | 3.2685e-13 | <= 1e-10 | PASS |
| case_b_machine_zero | T(x) = T0 + q_dot/(2k) (L^2/4 - x^2) | 1.18234e-11 | <= 1e-10 | PASS |
| case_c_coarse_below_3pct | T(x,y,t) = T0 S(x,t;a) S(y,t;b), Carslaw & Jaeger Sec. 3.4 | 0.00590278 | <= 0.03 | PASS |
| case_c_convergence_order | Q4 backward-Euler consistent-capacity conduction: order >= | 0 | >= 1 | FAIL |

整体 `passed = False`（`failed_gates = ['case_c_convergence_order']`）。

- Case A/B：LINE2 两端固定温度的线性/二次温度场是**节点精确**的，实测最大绝对误差
  3.27e-13 K 与 1.18e-11 K（阈值 1e-10 K），即组装（导热阵、源项、容量阵）被机器精度证实。
- Case C 粗网格相对 L2 = 0.005903（判据 <3%，**通过**）。

### case C 收敛阶：honest FAIL（本轮最有价值的否证）

阶梯（每级同时加密 h 与 dt）：4x3: relL2=0.005903 (10 steps); 8x5: relL2=0.02427 (40 steps); 16x10: relL2=0.02088 (160 steps); 32x20: relL2=0.02262 (640 steps)。

- 三级细化对的观测阶**全为 0.0**，相对 L2 在 4 套网格、dt 细化 8 倍范围内**几乎不动**（5.9e-3 量级）。
- 因此该残差**不是求解器的离散误差**（离散误差会随 h、dt 缩小），候选解释是：初值在 Dirichlet 边界上不连续（h 无关），
  以及双 Fourier 参考解截断在 80 个奇模态（Gibbs 摆动 ~4/(nπ)≈1.6%，dt 无关）。
- 该闸门**保持失败**，不作调参掩盖：这是本锚点当前的真实状态（`passed=false`），并已写入证书 `limitations`。

## 4. 债务账本变化与门禁

修复后 `python tools/verified_anchor_audit.py`：36 个锚点，`certified` 数量上升，`debt-diff unregistered=[] stale=[] drift=[]`，
`tests/test_verified_anchor_certificates.py` 16 passed（main 上修复前为 **1 failed**：`fem_inverse_identification` 未登记的
新增债务 + `fem_sent_j_integral` 缺陷漂移）。

> 说明：`fem_inverse_identification` 的证书由在飞分支 `agent/m17690`（commit 537d100b，2026-11-13 07:59）产出，未合入 main；
> 本轮按 fail-closed 规则把该债务**登记**进账本（而非伪造证书），待该分支合入后由它从账本删除。

## 5. 图件（png，均在 worktree 内）

- `fem_thermal`：fig_case_a_convergence.png, fig_case_b_convergence.png, fig_case_c_convergence.png, fig_case_c_field.png
- `fem_sent_j_integral`（本轮重建）：sent_convergence.png, sent_path_independence.png, sent_plane_strain_convergence.png, sent_plane_strain_path_independence.png, sent_plane_strain_shape_factor.png, sent_shape_factor.png

## 6. 注意点 / 局限

1. `fem_thermal` 的 A/B 闸门是"节点精确"型判据（1e-10 K），它证明的是装配正确性，不是渐近阶；
   真正的阶只由 case C 承担，而 case C 当前 honest FAIL。
2. case C 的参考解截断未单独设闸（截断误差未量化），这是本轮**未完成**的部分：要把 0.59% 残差归因到
   参考截断，需要在同一节点集上把级数项数扫到收敛并单独出闸——属下一步（已在 limitations 中登记）。
3. `--stage assemble` 重建图件**不重跑求解器**，所以图件与既有 `result.json` 严格同源；证书 `generated_at` 因此被刷新为重建时刻。
4. 本机 load15≈102（nproc=128），本轮全部为轻量活：一次 51 s + 一次 97 s 的热传导基准、一次 11 s 的图件重建，无重算力进程。

## 7. 给下次自找活的线索

- `fem_winkler_plate_k_sweep`：`E_ARTIFACT_MISSING`（证书引用的工件既不在库里也未在 `_cache`），需重跑或改证书；
- `fem_structural_acoustics`：`E_SCHEMA/E_PASS_WITHOUT_CRITERION/E_GATE_VALUE_NOT_FINITE`，是"证书自相矛盾"型缺陷，
  可零求解修复（重发信封 + 给闸门数值阈值），是同类活里最便宜的一个；
- `fem_axisym_shell` / `fem_shell4_formulation_matrix` / `fem_elastoplastic_plate_buckling`：`E_NO_GATES`，
  三者都已有 `result(s).json`，缺的是带解析基准的闸门层——`fem_axisym_shell` 最接近可收账（有 result.json）；
- `rainflow_fix_spec`：无 `gates_verified.json`，雨流计数是可对 ASTM E1049 解析算例设闸的；
- case C 参考截断收敛性研究（见局限 2）。
