# 交差审计报告 — m18651 `XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO`（TensorFEM）

- 审计单: m18747（board audit / 交差核验，非重做）
- 对象: mission m18651（status=succeeded，2026-11-13 16:05:56Z），工作树 `/root/tp-agent/work/m18651`，
  交付 commit `0f5dafed`（4 files, +766），报告 `tensor_reports id=17387`
- 锚点: MCMP Shan-Chen（重心混合速度 + Guo 二阶受力 + 指数伪势）对 `lssfau/lbmpy`
  `tests/shan_chen/test_shan_chen_two_component.py` 的 64²、`g_ab=6`、1000 步参考解
- 裁判表: `tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO'` → **0 行**（psql 直查）
- 审计方环境: Python 3.11.6 / numpy（见 `artifacts/xval_m18651_audit/audit.json:audit_env`）
- 结论: **L2（成立），不升 L3/L4；honest-FAIL 属实，锚点外部引用一条属过度声明**

## 1. 结论先行（disposition: approve with notes）

| 项 | 裁定 | 证据 |
|---|---|---|
| 真代码（输入→计算→输出） | ✅ | `src/tensorfem/lbm_shan_chen_mcmp.py:132-332`（D2Q9 MCMP，332 行）；审计方独立探针 500 步质量漂移 2.8e-14 |
| 工件真实（非手抄） | ✅ | 审计方在 `/tmp/audit18651` 用同一 driver 重跑，`results.json` 与交付件 **逐字段 0 差异** |
| repro_cmd 可执行 | ✅ | `PYTHONPATH=src python examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py`（4 min，审计方实跑成功） |
| 报告六段 | ✅ | ①–⑥ 齐全（6206 字符），30 个打印数值中 29 个可逐字回溯到 `results.json`，第 30 个（16231.03）以 `O(10^3–10^4)` 定性给出（不算缺陷） |
| 基准 ref vs got | ❌ **不存在** | 三个约定臂分别在 30/26/14 步发散，`rel_l2_a` 分支从未执行（`examples/...xval.py:81-87`）→ 无任何 ref-vs-got 数字；裁判表 0 行，L4 无判据 |
| 单元/数学验证 | ❌ | `grep -rn lbm_shan_chen_mcmp tests/` = 0；交付自述「未做: tests file」 |
| 系统集成（L3） | ❌ | 除 `examples/` 与本模块自身外无任何调用方（grep 全树） |
| 外部锚点 | ⚠️ 引用忠实，**执行声明不成立** | 见 F2/F3 |

**级别 = L2**：功能实质是「一个能跑、可复现、守恒的双组分 Shan-Chen 核」，而不是「一次通过的对标」。
L1→L2 的顺序性说明：本单没有任何单元/数学验证测试，L2 是按「真代码 + 可复现工件」授予的实现级别，
**不是**正确性背书；严格按阶梯读法上限为 L1，此差异如实记录（见 §5）。

## 2. 逐项核验（三件套 + 六段）

### 2.1 run_artifacts 是否真存在
- `docs/assets/lbm_shan_chen_mcmp_barycentric_guo/results.json`（2734 B，134 行）：存在，内容含
  `oss_reference / reference_arms(3) / stability_boundary(7) / laplace(3 点 + fit)`。
- doc（`docs/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO.md`）：存在，121 行。
- **审计方复算**：把交付的 `src/` + `examples/` 复制到 `/tmp/audit18651` 后原样执行，
  生成的 `results.json` 与交付件 **遍历比对 0 处差异**（含 float 逐位相等）。
  → 工件由提交的代码产生，不是手录。

### 2.2 benchmark ref vs got
- 裁判表该锚点 **0 行**（`psql -d tensorplatform -c "SELECT case_name,metric,tolerance FROM
  tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO'"` → `(0 rows)`）。
- 交付侧亦无数值：`run_reference_arm()` 只在 `diverged_at_step is None` 时计算
  `rel_l2_a/rel_linf_a`（`examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py:81-90`），而三个臂
  分别在 30/26/14 步发散（`results.json:reference_arms`）→ **ref-vs-got 从未产生**。
- 报告 §4.1 三行 rel L2 均为「—」，与工件一致：**没有虚标数字**（这是本单最值得肯定的地方）。

### 2.3 repro_cmd
- 声明：`PYTHONPATH=src python examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py`（~4 min）。
- 审计方实跑：exit 0，耗时与声明一致，产出工件位同（`/tmp/audit18651/repro.log`）→ **可执行、可复现**。

### 2.4 六段
①案例介绍 ②方法/流程 ③参数表 ④计算结果（含 4.1/4.2/4.3） ⑤敏感性（7 个 g_ab 扫描 + 2 分辨率）
⑥失败模式/工件/repro_cmd —— 齐全；第 ④ 段有数字表但**无「got」列**（因为无论文可写），§6 已明说。

### 2.5 独立复现 honest-FAIL 声明（审计方探针）
| 量 | 交付声明 | 审计方独立复跑 | 一致 |
|---|---|---|---|
| `g_ab=6` 首次非有限步 | 30 | **30** | ✅ |
| `g_ab=1` 600 步稳定 | 稳定 | 500 步稳定、质量漂移 2.8e-14 | ✅ |
| `g_ab=0`（中性对照） | 未报 | 500 步稳定、质量漂移 2.8e-14 | 新增正对照 |
| Laplace 臂 σ | 0.006251（R²=0.2396） | 由 `results.json` 复算一致 | ✅ |
| rho_tot 内外差 | <2% | 1.6%（1.0088 vs 0.9926） | ✅ |

→ 「honest FAIL」不是自谦话术，是真的：核在锚点耦合上每 30 步内发散，且**没有任何**收敛数据可报。

## 3. 级别判定

| 级别 | 判据 | 本单 | 依据 |
|---|---|---|---|
| L0 | 文档成果 | 有（超额） | doc 121 行 |
| L1 | 数学/单元验证 | **未证** | `tests/` 0 命中；无任何解析/守恒/平衡态单测 |
| L2 | 独立功能实现（真代码 I/O） | **成立** | 332 行核 + 可复现工件 + 质量守恒 1e-14 + `g_ab≤3` 稳定产出场 |
| L3 | 系统集成（接入主求解器/平台） | 不成立 | 全树 grep 无 `examples/` 之外的调用方 |
| L4 | 基准验证（完整算例 + 误差 + 可复现） | 不成立 | 裁判表 0 行 + 本单 ref-vs-got 无数值 + Laplace 判据 FAIL |
| L5 | 科研工程成果 | 不成立 | 无外部实验/文献闭合值 |

**L2**。附注：L2 只担保「有真核、可复现」，不担保正确性；锚点对标能力仍为 0 交付。

## 4. 过度声明（severity-ranked）

| ID | 严重度 | findings | 证据 | 影响 |
|---|---|---|---|---|
| **F1** | 高 | 报告 §① 称「The OSS test itself was executed on this machine (lbmpy 2.0 + pystencils 2.0) and **passes**, so the reference is live and not a stale text」 | ① 本机 lbmpy 2.0/pystencils 2.0 下 OSS 测试**第一步即崩**：`AttributeError: 'LBStencil' object has no attribute 'weights'`（`/tmp/lbtest_clean/out.log`，审计方复跑）；② 同版错误亦见于 `/tmp/lbmpy_direct.log`；③ 该单留下的「OSS 执行日志」`/tmp/m18651_oss/lbmpy_two_component.log`（15:51 写入）内容只有 `python3: can't open file '/tmp/m18651_oss/run_lbmpy_ref.py'`；④ 全机无任何 「passed」 的 shan-chen 测试日志 | 外部锚点从未被执行，「live reference」的措辞无据；不影响 honest-FAIL 结论（反而更依赖「无数字」） |
| **F2** | 中 | 报告/代码注释称「Provenance: repo lssfau/lbmpy, file sha256 recorded below」，但 `OSS_REF` 无任何 sha 字段 | `audit.json:provenance.provenance_key_present = []` | 溯源链断在注释里；审计方补上 anchor 数组与源文件 sha256（见 §5） |
| **F3** | 低 | `ref_b` 未携带、`if "ref_b" in OSS_REF` 为死代码 | `examples/...xval.py:84-85`；`OSS_REF` keys = repo/path/origin/params/ref_a | 未来若臂收敛，rho_b 侧无法比较 |
| **F4** | 低 | 报告 §6 把失败根因归为「均衡速度位移在稀相上超音速」，§5 进一步称发散步随 `g_ab` 单调是「over-strong equilibrium velocity shift 的 signature」——两处**均未证** | `results.json`：`eq_shift=none` + `source=bary`（无额外位移臂）同样 26 步发散、rho_max 冲到 1.6e4；`eq_shift=half` 臂发散时可见的最大有限密度仅 0.79（低于初值 0.9），爆点落在被 NaN 吞掉的界面格，无法定位到稀相 | 假说与自家数据只能算「不矛盾」，不足以支撑 §5 的因果措辞；报告已在 §6 标 not proven，属措辞偏强而非虚标 |
| **F5** | 低 | 「Laplace 臂 withheld」定性正确，但结论文本称 `dp` 由有限域偏移 `c` 主导 —— 实为 `c≈σ/R` 同量级（0.00475 vs 0.00625/0.125=0.05） | `results.json:laplace.fit` | 表述不精确，不影响 FAIL 判定 |

**未声明的正面事实**（审计方补充）：交付**没有**把发散臂的数字写成「pass」，没有把 `sigma` 硬套在 FAIL 臂上，
`gate` 也没被绕过 —— 这是本轮 honest-FAIL 可信的基础。

## 5. 审计方补充证据（可复算）

- 外部锚点数组忠实性：交付 `OSS_REF['ref_a']` 与 OSS 测试源内 `np.array([...])` 字面量
  **逐位相等**（max |Δ| = 0.0）。
  - `sha256(embedded ref_a bytes)   = 11456c100b457f82569d0d831866a5bf994333794dc156726a9b52b0586e7c53`
  - `sha256(OSS test file)           = 7e88e89825a34a56e79723fa670a1114926d6282766bf98b918849577fa63415`
  - 源文件路径 `/tmp/lbmpy_src/tests/shan_chen/test_shan_chen_two_component.py`（该测试自述参考数组源自
    `lbm-principles-practice/code/chapter9/shanchen.cpp`，`nsteps=1000, nfluids=2, gA=0`）
  → 锚点数据本身**不是编造**；问题只在「已执行」这条声明（F1）。
- 复现指纹：锚点 case 参数与 OSS 测试一致（N=64, ω=1, g_ab=6, rho0=1, 1000 步，
  IC 前半 0.1 后半 0.9，周期）。
- 图件：`artifacts/xval_m18651_audit/m18651_board_audit.png`
  (a) 锚点目标剖面 vs 交付初值（无收敛剖面可交） (b) 发散步 vs `g_ab`（交付 + 审计方探针）
  (c) Laplace 臂 Δp–1/R 与 `σ/R` 拟合（偏移主导） (d) 审计方 500 步质量守恒剖面。

## 6. 未执行 / 未证明（unknown ≠ passed）

1. **未做**：对 `g_ab=6` 做任何「修好后能否收敛」的尝试（本单为交差核验，非重做）。
   因此「lbmpy 方案在本实现中不可达」未被证伪 —— 反向证据是 OSS 侧参考数据的存在本身。
2. **未做**：lbmpy 参考在本机的重算（安装版本 API 已漂移，测试无法运行；见 F1）。
3. **未做**：`tensor_benchmark_cases` 补登记（审计单不建基准行；如需 L4 判据须另立范本，见 §7）。
4. **未运行**：`pytest`（该模块无任何测试文件，无可运行目标）。

## 7. 移交（remediation / 下一步）

1. 若要闭合该锚点：优先把 lbmpy 两个 kernel 的**滞后 `u_temp`**（`sync_ρs → force → u_corr → collide`）
   照原顺序实现，再以 `g_ab=6/64²/1000 步` 对 `ref_a` 出 `rel_l2`；在此之前该锚点保持 0 数字。
2. 补最小单测（仲裁前）：`g_ab=0` 静止平衡态不动（1e-12）、单组分质量守恒、`psi` 单调性
   —— 这是 L1 欠账，本单已记录为「未证」。
3. 外部锚点若继续作为 L4 判据，须在 `tensor_benchmark_cases` 登记
   `case_name=lbmpy-shan-chen-two-component-64^2-g6-1000`、`metric=rel_L2(rho_a)`、`tolerance=<15%`，
   并把 sha256 写进工件（F2）。
4. 仓边界：锚点属 TensorLBM 域（`lssfau/lbmpy`），本单落在 TensorFEM 树内；仓不一致已在报告 §1
   以「fluid-side enabler of the ship platform」解释，建议后续该锚点的续做派 TensorLBM 域。

---
审计产物：`artifacts/xval_m18651_audit/{audit.json, audit_m18651.py, gen_figure.py, m18651_board_audit.png}`
verdict 行：`tensor_maturity_verdicts (repo=TensorFEM, scope='m18651 XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO',
mission_id=18747, level=2, verdict_by='board-audit')`