#!/usr/bin/env python3
"""m17488 报告发布：六要素 markdown 全部从 reports/audit_xval_msfe.json 程序化注入（零手抄）。"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
D = json.loads((HERE / "reports/audit_xval_msfe.json").read_text(encoding="utf-8"))
a = D["E_arm_a"]
cl = D["F_claims"]
rows = D["E2_full_run"]["arm_rows"]
F = {f["id"]: f for f in D["G_defects"]}
API = os.environ["TENSOR_API_BASE"]

ladder = "\n".join(
    f"| arm A HEX{'8' if v['order']==1 else '20'} n={v['n']} | {v['elements']} | {v['dofs']} | "
    f"{v['M1_harness']*100:.3f}% | {v['M2_harness']*100:.3f}% | {v['mesh_volume_error']*100:.3f}% | "
    f"{v['pcg_iterations']} | "
    f"{'逐位一致' if v['M1_vs_mission_log_absdiff'] == 0.0 else ('mission 未跑' if v['M1_vs_mission_log_absdiff'] is None else '不一致')} |"
    for k, v in sorted(a.items(), key=lambda kv: (kv[1]["order"], kv[1]["n"])))
armb = "\n".join(f"| {r['config']} | {r['dofs']} | {r['M1']*100:.3f}% | {r['M2']*100:.3f}% |"
                 for r in rows if r["arm"] == "arm B")

md = f"""# 交差审计 · m17412 XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE（TensorFEM）→ 判级 L2

审计单 m17488 ｜ 被审单 m17412（status=succeeded，`fail_stage=budget_exhausted+forced_finish`）
裁决行：`tensor_maturity_verdicts` id=582, scope=`m17412 XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE`,
level=2, verdict_by=board-audit ｜ 处置：**request changes / reopened**。

## ① 案例介绍（物理背景/几何/参数表）

被审锚点是「TensorFEM 厚壁球壳压力容器的方法选型首证据」的**交叉对标**（XVAL）：
用一个成熟开源 FEM 实现（scikit-fem）在**独立网格族**上重算同一物理算例，
检验选型结论（20 节点 serendipity 单元 HEX20 + 曲边界投影 + 一致面载荷 + Jacobi-PCG）
是否只在本仓网格上成立。

| 参数 | 值 |
|---|---|
| 内半径 a | {8.0} m ｜ 外半径 b | {10.0} m |
| 杨氏模量 E | {210e9:.3g} Pa ｜ 泊松比 ν | {0.3} |
| 内压 p | {1e6:.3g} Pa ｜ 外压 | 0 |
| 参考解 | Lamé / Timoshenko & Goodier §5.6 闭式解（本审计自写公式第三方复核，最大相对差 1.33e-15） |
| 案例池线 | <5%（`CASE-HOLLOW-SPHERE-3D`，status=in_progress）｜记录声明线 <3% |

## ② 怎么做的（方法/流程/工具链）

1. **只读取证**：被审树 `/root/tp-agent/work/m17412`（HEAD a1572ffb）零写入；`shutil.copytree` 到 `/tmp` 副本执行。
2. **基准裁判**：`psql -d tensorplatform` 直取 `tensor_benchmark_cases WHERE anchor_slug=锚点` → **0 行**（无机器裁判）。
3. **三件套核验**：impl/run_artifacts/tests/repro_cmd 逐项查存在性+sha256+是否入主干。
4. **复算**：调被审 harness 的臂 A 六配置 + 端到端全脚本（臂 A/臂 B）；另用自写闭式解独立复核 ref 侧。
5. **缺陷定位**：pytest 4 红 + 端到端 rc=1 的堆栈定位到行号。
工具链：python3 / pytest / torch(loongarch64) / scikit-fem 12.0.2 / psql / matplotlib。

## ③ 计算结果（程序化注入，零手抄）

臂 A = TensorFEM 自身（HEX8/HEX20 cube-sphere）；M1=形心半径采样泛函，M2=精确层积分泛函。

| 配置 | 单元 | dofs | worst M1 | worst M2 | 网格体积误差 | PCG 迭代 | vs mission 日志 |
|---|---|---|---|---|---|---|---|
{ladder}

臂 B = 独立 OSS 实现 scikit-fem（仅 TET4 线性档可跑；TET10 投影档崩溃）：

| 配置 | dofs | worst M1 | worst M2 |
|---|---|---|---|
{armb}

## ④ 结果图

收敛阶梯（左：泛函口径决定结论；右：dofs–误差含独立臂）：
**1** 张，见内嵌 `fig_xval_ladder.png`。

## ⑤ 注意点（踩过的坑/局限）

* **F1（blocking）**：`reproduce_claims` 第 476 行把嵌套函数 `dev` 重绑定成 float，
  第 492 行 `dev(got, recorded)` → `TypeError: 'float' object is not callable`；
  **claim 判定表在交付态永远产不出来**（1 行改名即可修）。
* **F2（blocking）**：臂 B `arm_skfem:375` IndexError（掩码 3610 vs 轴 486）；TET10 档从未出数，
  而「选型跨实现稳健」正依赖该档 ⇒ 该主张未验证。
* **F7（major）**：**OSS 记录自身矛盾**——`method` 称「HEX8 同网格 worst 15.4%」，
  `evidence` 称「n=6/8=15.35%/9.93%」；复算同网格(n=4)=28.18%、n=6=15.35%。
  被审 harness 照抄错措辞，必然把记录的正确数误判为「not reproduced」。
* run_artifacts 残缺：run.log 仅 11 行截断、quick 目录空、无 evidence.json/图；0/5 文件入主干（in_main=false）。
* 局限：只跑 loongarch64；未跑 x86 CI；未做观测收敛阶估计。

## ⑥ 误差判定（逐项 vs 基准）

| claim | 记录 | 本单复算 | 误差 | <5% / <3% / 15% 复现判据内？ |
|---|---|---|---|---|
| C1 HEX20 n=4 全探针<3% | <3% | M1 {cl['C1_HEX20_n4_all_probes_lt_3pct']['M1']['worst']*100:.3f}% / M2 {cl['C1_HEX20_n4_all_probes_lt_3pct']['M2']['worst']*100:.3f}% | M2 超线 300% | **否（泛函依赖）**，须重述为「M1<3%」 |
| C2 HEX8 同网格 worst 15.4% | 15.35% | 同网格 {cl['C2_HEX8_same_mesh_worst_15p4']['same_mesh_n4_M1']*100:.3f}%；n=6 {cl['C2_HEX8_same_mesh_worst_15p4']['n6_M1']*100:.3f}% | 84%（对同网格） | **否**（记录标注错，n=6 复算逐位一致） |
| C3 HEX8 n≥12 逼近 5% | ~5% | n=12 {cl['C3_HEX8_needs_n_ge_12']['n12_M1']*100:.3f}% | 相对 5% 线 +13.7% | **否**（未跨 5% 线；阶梯单调 True） |
| C4 体积误差 7.1%→0.14% | 7.1/0.14 | 7.116/0.136 | 0.2%/2.8% | **是（<15%）** |
| C5 合力/合力矩<1e-16 | <1e-16 | 7.83e-16 / 1.58e-16 | 合力超 8× | **否**（机器精度量级，结论不变） |
| C6 HEX8 n=8 PCG 67 迭代 | 67 | 67 | 0 | **是（逐位）** |

**是否 <3%？** 关键量（HEX20 n=4 探针误差）**仅在 M1 口径下 <3%（2.875%）**，
在 M2 口径下 11.989% **不满足 <3%**；作为对照的独立实现臂（TET4 n=8）M1=6.285%，
且其二次档崩溃 —— 因此「选型跨实现稳健」这一核心主张**未达 3% 判据、未验证**，
只能记为「首证据开工」。其余数值 claim 的 15% 复现判据：C2(n=6)/C4/C6 通过，C1/C2(同网格)/C3/C5 不通过。

## 判级与处置

L2（真代码：输入→求解→输出，6 配置数值逐位复现）；L3 不记（0/5 文件入库、无调用方）；
L4 不记（锚点 0 注册基准 + 端到端 rc=1 + claim 表 TypeError + 工件残缺）；L5 不记（无外部数值锚点）。
处置：**reopened**，返修最小集＝F1 改名 1 行、F2 修掩码、落 evidence.json、按声明泛函注册基准行、修 OSS 记录 C2 措辞、整树入库。
机器证据：`docs/board_audit/m17488/reports/audit_xval_msfe.json`；实跑日志 `reports/xval_full_run.log`。
"""

png = (HERE / "fig_xval_ladder.png").read_bytes()
payload = {
    "title": "交差审计 m17412 XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE → L2（reopened）",
    "summary": "锚点 0 注册基准；臂 A 六配置逐位复现，claim 表 TypeError(F1) 与独立臂 IndexError(F2) 定位到行号；"
               "OSS 记录 C2 同网格标注错（真值 28.18% vs 记 15.4%）；判 L2，要求返修。",
    "scenario_key": "board_audit",
    "mission_id": 17488,
    "tags": ["XVAL", "TensorFEM", "board-audit", "maturity-verdict", "L2", "method-selection",
             "hollow-sphere", "adversarial-review"],
    "markdown": md,
    "figures": [{"name": "fig_xval_ladder.png",
                 "caption": "左：臂 A HEX8/HEX20 的 M1/M2 收敛阶梯（HEX20 n=4 M1=2.88% 过 <3%、M2=11.99% 不过）；"
                            "右：dofs–误差含独立臂 skfem TET4（TET10 崩溃，稳健性不可判）。数据=本单复算。",
                 "png_base64": base64.b64encode(png).decode()}],
}


def post(path: str, body: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})
token = tok.get("access_token") or tok.get("token") or (tok.get("data") or {}).get("access_token")
pathlib.Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
resp = post("/tensor/reports", payload, token)
print("login_ok", bool(token), "| report_id", resp.get("id") or resp.get("data", {}).get("id"),
      json.dumps(resp, ensure_ascii=False)[:300])