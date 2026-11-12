# Audit: Bloch--Floquet 归约建在节点空间（m16896）

**锚点** `fem_wave_dispersion`（benchmarks/verified/fem_wave_dispersion）

## 1 发现问题
审计工具 `tools/verified_anchor_audit.py` 把该锚点判为 `contradicted`：
旧证书是裸字典（无 envelope、无 operator/passed），`E_SCHEMA` + `E_OVERALL_ABSENT`，
9 条门禁值全部不可重算。同时文档 §6 的总判定是一句**硬编码字符串**
「门禁 9/9 PASS；…最大误差 556.8130% < 3%」——同一页的表格里却有 4 条 FAIL，
即文档自相矛盾（证书完整性缺陷，非物理缺陷）。

## 2 根因（独立复现）
1. `bloch_cell_matrices` 用**节点号**做约化变换矩阵的列索引：`kh=0.2`、`8×4` 单胞
   90 个自由度只有 40 列，而正确维数是 80（每保留节点 2 个未知量）。于是不同节点的
   `u_x`/`u_z` 被混叠进同一列，归约后的算子与真实 Bloch 问题无关。
   旁证：Γ 点只剩 1 个零模态（应为 2 个刚体平移）。
2. 因此 FEM 相位速度与 `kh` 几乎无关（`kh=0.2/0.5/1.0` 均 1.2474），
   却仍给出收敛阶 ≈2 —— 收敛到一个**错误的算子**。旧文档的诚实负面结论与
   `test_bloch_floquet_reduction_defect_is_open` 的 fail-closed 门是对的，
   但其猜测的根因（相位因子方向/周期拓扑环）不对。
3. 次要缺陷：(a) S0 按序号 `omega[1]` 取，`kh=0.2` 时 S0 实际是第 3 个模态（A1 支插入其间）；
   (b) `s0_plate_velocity_extrapolated` 与 `s0_longitudinal_limit` 两侧量纲不一致
   （归一化相位速度 vs 有量纲波速）；(c) S0 的高 `kh` 渐近是 `c_R` 而非 `c_L`，原门禁判据本身错；
   (d) `fem_convergence_order` 的 `value` 被写成 `max(0, 1.5-order)`、门限 0，恒真不可证伪；
   (e) Γ 点「5 个零模态」的期望值本身错（周期条件下刚体转动不相容，正解为 2）。

## 3 修复
- 归约改在**自由度空间**：`trans[2·node+d, 2·reduced[node]+d]`，维数 2(n_nodes−n_right)。
- 新增 `bloch_cell_modes`（返回未归约自由度空间的模态）与 `midplane_family_fraction`
  （按中面奇偶性分类 S/A 族），**分支识别不看解析根**，避免用 oracle 挑自己。
- 门禁重写为 `tensorfem.verified-gates.v1` 十项，含两项结构性门禁
  （`gamma_zero_mode_count == 2`、`floquet_reduction_dofs == 2(n_nodes−n_right)`）——
  这两项正是能抓住本次缺陷的门禁。
- 文档总判定改为由门禁值程序化生成，硬编码的「9/9 PASS … <3%」被删除。

## 4 结果（`benchmarks/verified/fem_wave_dispersion/result.json`）
- 审计评级 `contradicted` → **certified**（10/10 门 PASS，0 E_ 缺陷），
  已从 `anchor_certificate_debt.json` 中删除（shrink-only 账本只减不增）。
- 32×16 单胞、`kh=0.2–1`：A0/S0 相位速度最大相对误差 0.3339%（门槛 3%），
  收敛阶 1.971（门槛 ≥1.5），一致质量阵守恒 0（1e-16 量级）。
- 遗留（未修，已写进文档 §5 与 fail-closed 测试）：`n_roots≥2` 的多根扫描会返回
  残差 O(0.1) 的候选（`q=0` 退化为伪根等），本锚点只使用每族最低根。
