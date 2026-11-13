# Board Audit: m18054 — XVAL-LBM-MOVING-WALL-HALFWAY-BB-FLOAT64

## 审计概要

| 项目 | 值 |
|---|---|
| 被审任务 | m18054 (TensorFEM) |
| 锚点 | XVAL-LBM-MOVING-WALL-HALFWAY-BB-FLOAT64 |
| 审计任务 | m18153 |
| 审计人 | board-audit (m18153) |
| 判定级别 | **L4** (基准验证) |
| 门状态 | **confirmed** (消账) |

## 取证过程

### 1. 验收三件套核验

| 要件 | 状态 | 证据 |
|---|---|---|
| run_artifacts 文件 | ✅ 存在 | `docs/assets/lbm_moving_wall_couette/results.json` (23KB, 12 cases), `profile_errors.png` (106KB), `error_ladder.png` (89KB) |
| benchmark ref vs got 数值 | ✅ 全部在容差内 | 最差误差 2.32e-04 (C-P)，远优于 <15% 验收线（余量约 600 倍） |
| repro_cmd 可执行 | ✅ 已验证 | `PYTHONPATH=src pytest tests/test_lbm_couette_moving_wall.py -q` → 8 passed (13.6s)；`PYTHONPATH=src python3 examples/lbm_couette_moving_wall_validation.py` → 12 算例数字与报告一致 |

### 2. 报告六段核验

| 段 | 状态 |
|---|---|
| ① 案例介绍 | ✅ Couette 流物理背景、几何参数表完整 |
| ② 怎么做的 | ✅ Ladd 闭合式、周期掩膜修正、验证脚本说明 |
| ③ 计算结果 | ✅ 4 分辨率梯子 + 4 粘度 + C-P + float32 对照，数字表完整 |
| ④ 结果图 | ✅ profile_errors.png + error_ladder.png |
| ⑤ 注意点 | ✅ 速度项符号陷阱、周期掩膜泄漏陷阱、未覆盖范围 |
| ⑥ 误差判定 | ✅ 5 项关键量 vs 基准逐项列出，均 <3% |

### 3. 代码实质核验

- **实现文件**: `src/tensorfem/lbm_channel2d.py` 616 行（main 上 528 行，moving-wall 新增 ~88 行）
- **关键代码段**:
  - `_shift_periodic_x` (L93-100): x 周期流式
  - `u_wall_bottom/u_wall_top/periodic_x` 配置字段 (L130-132)
  - `_wall_links` 构建 (L188-205): Ladd 运动壁 half-way BB 链接
  - `_stream` 中 moving-wall BB 应用 (L310-320)
  - `wall_force()` 动量交换力 (L488-516)
- **测试**: `tests/test_lbm_couette_moving_wall.py` 166 行，8 个测试全部通过，断言不变量（非仅执行）
- **验证脚本**: `examples/lbm_couette_moving_wall_validation.py` 541 行，12 算例全自动扫描

### 4. 数值独立复现

| 关键量 | 报告值 | 复现值 | 一致? |
|---|---|---|---|
| Couette rel L-inf (ny=10) | 1.04e-11 | 1.042e-11 | ✅ |
| Couette rel L-inf (ny=18) | 8.25e-11 | 8.249e-11 | ✅ |
| Couette rel L-inf (ny=26) | 2.25e-10 | 2.252e-10 | ✅ |
| Couette rel L-inf (ny=34) | 3.95e-10 | 3.949e-10 | ✅ |
| 壁面位置 offset | ≤8.3e-11 dx | ≤8.3e-11 dx | ✅ |
| 剪应力 rel err (ny=34) | 1.22e-09 | 1.22e-09 | ✅ |
| C-P rel L-inf | 2.32e-04 | 2.321e-04 | ✅ |
| 动量平衡 | 1.1e-12 | 1.104e-12 | ✅ |
| float64/float32 误差比 | 2.5e5 | 2.5e5 | ✅ |

### 5. 集成状态

- `lbm_channel2d.py` 基础模块已在 main 上（通过 m8136 合并）
- `fsi_lbm_strong_coupling.py` 导入 `ChannelConfig`/`ChannelLBM`（在 main 上）
- **moving-wall 新增部分**（`u_wall_*`, `periodic_x`, `_wall_links`, `wall_force`）在 `agent/m18054` 分支（commit `08290da8`），**尚未合并到 main**
- 回归测试 `test_fsi_lbm_strong_coupling.py` 12 passed（无破坏）

## 判级理由

| 级别 | 是否满足 | 理由 |
|---|---|---|
| L0 文档成果 | ✅ | 有完整文档和报告 |
| L1 数学/单元验证 | ✅ | 8 个测试断言数学不变量 |
| L2 独立功能实现 | ✅ | 真代码：ChannelConfig → ChannelLBM → step → wall_force |
| L3 系统集成 | ⚠️ PARTIAL | 基础模块在 main 上并被 FSI 求解器导入，但 moving-wall 新增部分未合并到 main |
| **L4 基准验证** | **✅** | **完整 12 算例基准 + 误差分析 + 可复现，所有误差远优于 <15% 验收线** |
| L5 科研工程成果 | ❌ | 尚未合并到 main，无下游使用 |

**最终判定: L4** — 技术实质完整，基准验证充分，可复现。未合并到 main 是流程问题而非技术问题。

## 过度声明检查

无实质性过度声明。报告正确指出 Couette 线性解下 half-way BB 本身精确，只能验证壁面闭合式和位置，不能测空间阶数。

## 未执行检查

- loong64 集群复跑（本审计范围不需要）
- 第三方 LBM 代码对表（解析解基准不需要外部参考实现）

## 处置

**confirmed** — 消账。m18054 交付物真实有效，L4 级别成立。
