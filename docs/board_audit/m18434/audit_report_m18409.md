# 交差审计 m18409: XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO (TensorFEM)

审计单: m18434 (board-audit) · 被审计单: m18409 (status=succeeded, approved) · 审计日: 2026-11-13

## 1. 案例介绍（被审计交付物）

m18409 挂锚 `XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO`（LBM 域 shan-chen MCMP
barycentric 锚，外部参照 lssfau/lbmpy），但实际交付为 **TensorFEM 证书图件欠账闭合**：
新增 `tools/anchor_figure_rebuild.py`（399 行），从 3 个 verified 锚的 committed
`result.json` 零求解重绘 10 张图到 `docs/assets/verified/<anchor>/`，sha256 manifest
逐图绑定；证书改指新路径；修掉 `fem_inverse_identification/run.py` 的证书回滚隐患。
commit `bf90086d`（agent/m18409，25 files，+1305/−94）。

**关键事实：锚点与交付物不匹配。** 锚点属 TensorLBM 域，交付物是 TensorFEM 验证
基础设施。agent 在报告 §5 诚实申报了该偏差（「仓不一致，按诚实申报改做本仓验证
欠账；建议另派 TensorLBM 单」）。

## 2. 审计方法

- psql 直查裁判表 `tensor_benchmark_cases`（runuser -u postgres psql -d tensorplatform）
- 工作树取证 `/root/tp-agent/work/m18409`（git log/show/status/diff）
- 独立复算 sha256 manifest（10 图）
- 全链复跑 repro_cmd（工具 3 件 + pytest 5 文件）
- 报告六段核对（tensor_reports id=17216）
- 全树 grep `shan`/`barycentric` 验证锚点痕迹

## 3. 验收三件套核验（逐项）

| 件 | 判定 | 证据 |
|---|---|---|
| run_artifacts 文件 | ✅ 真存在 | 10 PNG（53952/30615-40970/43652-74524 B）+ 3 份 figures_manifest.json；审计方独立重算 sha256 **ok=10 bad=0** |
| benchmark ref vs got | ⚠️ **裁判表 0 行** | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO'` → **0 rows**；无任何物理量 ref-vs-got 比较；报告 §6 自述「本单不产生新物理量」 |
| repro_cmd 可执行 | ✅ 可执行 | 审计方实跑全链：rebuild --write（3 锚 10 图）→ audit --write（certified=31 contradicted=1，gates 349/368）→ renderables --check（violations=0）→ pytest **44 passed in 17.79s**；确定性：仅 `generated_at` 时间戳漂移，gate 数值零变化 |

报告六段：✅ 齐全（案例介绍/怎么做/计算结果/结果图/注意点/误差判定，id=17216 published）。

## 4. 基准裁判（唯一裁判）

- 裁判表该锚 **0 行** → 无 case 可逐项核容差 → **L4 基准验证不成立（未证明）**。
- 报告声称的「误差」全部为审计计数（certified 28→31、contradicted 4→1、
  E_ARTIFACT_MISSING 3→0、renderables 1→0、gates 349/368 不变）；审计方复算**后态
  全部吻合**（certified=31、contradicted=1 仅 fem_structural_acoustics 的
  E_SCHEMA/E_PASS_WITHOUT_CRITERION/E_GATE_VALUE_NOT_FINITE、349/368、violations=0）。
- 「前」基线数（certified=28 等）未独立复算（需回退工作树），仅后态经复算验证。
- 锚点自身验证欠账**未闭合**：TensorFEM 树内无 shan-chen-mcmp barycentric 任何
  运行/日志/SHA（grep 0 命中，barycentric 命中均为无关 FEM 接触力学）。

## 5. 判级：L2

- **L2 成立**：真代码（输入 committed result.json → 确定性重绘 → 输出 PNG+sha256
  manifest）+ 44 测试绿 + 可复现 + 接入证书/审计子系统（certificates/audit/
  renderables 消费其产物）。
- **L3 不成立**：未接入主求解器；集成面限于验证子系统。
- **L4 不成立**：裁判表 0 行，无完整算例+误差分析+可复现的基准验证；锚点验证欠账
  未闭合（属 TensorLBM 单）。
- 级别=功能实质，非报告篇幅。对照先例：m18323=L2（真代码+测试过但基准不过/未集成）、
  m18326=L1（调用壳+零 pytest）。本单代码量与测试面均高于 m18323，判 L2 不判 L1；
  无基准行不判 L4。

## 6. 结论与处置

- **verdict: L2**，落表 `tensor_maturity_verdicts`（verdict_by=board-audit）。
- 偏差如实记录：锚点≠交付物（已诚实申报，不加重）；锚点欠账建议另派 TensorLBM 单。
- 无造假/虚标发现：所有可复算数字与交付一致；「前」基线数未独立复算（已声明）。
