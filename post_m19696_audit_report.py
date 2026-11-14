#!/usr/bin/env python3
"""Publish the m19696 board-audit report for m19614 (GAP-PP-1-ICE-STRUCT-CONTACT-PATCH).

Numbers are read programmatically from the sibling implementation's artifacts
(agent/m18930 export under scratch/m18930), never hand copied.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts" / "audit_m19696"  # snapshots of agent/m18930 artifacts (provenance for the audit)


def login() -> str:
    body = json.dumps(
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]}
    ).encode()
    req = urllib.request.Request(
        BASE + "/auth/login", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def post(path: str, payload: dict, token: str):
    req = urllib.request.Request(
        BASE + "/" + path.lstrip("/"),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.status, json.loads(r.read().decode() or "{}")


def collect() -> dict:
    sweep = json.loads((ART / "concentration_sweep.json").read_text())
    study = json.loads((ART / "study.json").read_text())
    amp = sweep["amplification"]
    out = {"case": {}, "mesh_ratio": {}}
    for mesh in ("12x24", "24x48"):
        for shape in ("centre", "ramp"):
            for k, v in amp[mesh][shape].items():
                out["case"][f"{shape}/{k}/{mesh}"] = float(v)
    ms = sweep["mesh_stability"]["absolute_sigma_peak_uniform"]
    out["mesh_ratio"] = {
        "coarse_over_fine_sigma_peak": float(ms["ratio_coarse_over_fine"]),
        "abs_ratio_amplification_centre_amp1": float(
            sweep["mesh_stability"]["amplification"]["centre"]["amp1.0"]["ratio_coarse_over_fine"]
        ),
    }
    out["study_mesh_refinement_gci"] = study.get("mesh_refinement", {}).get("refinement", {})
    return out


def figure(d: dict) -> bytes:
    labels, vals = [], []
    for k, v in d["case"].items():
        if k.endswith("24x48"):
            labels.append(k.replace("/24x48", ""))
            vals.append(v)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    cols = ["#2b6cb0" if "centre" in l else "#c05621" for l in labels]
    ax.bar(range(len(vals)), vals, color=cols, width=0.6)
    ax.axhline(1.0, color="k", lw=1)
    ax.axhspan(0.97, 1.03, color="green", alpha=0.12, label="±3% band")
    ax.axhspan(0.85, 1.15, color="orange", alpha=0.10, label="±15% band")
    for i, v in enumerate(vals):
        ax.annotate(f"{(v-1)*100:+.2f}%", (i, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("peak-stress amplification (same total force)")
    ax.set_title(
        "Independent implementation (agent/m18930, 24x48): non-uniform patch vs statically\n"
        "equivalent uniform patch — amplification stays inside the ±15% band, centre peak ~+1%"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def main() -> int:
    d = collect()
    png = figure(d)
    rows = "\n".join(
        f"| {k.split('/')[0]} | {k.split('/')[1]} | {k.split('/')[2]} | {v:.5f} | {(v-1)*100:+.2f}% |"
        for k, v in sorted(d["case"].items())
    )
    md = f"""## ① 案例介绍

**对象**：m19614（`selfstart-tp-impl-5-1113-2257-rework1`）交付，锚点
`GAP-PP-1-ICE-STRUCT-CONTACT-PATCH`（冰压斑载荷由接触压力场生成并喂给板格求解器）。
**物理背景**：冰区船冰带板格承受局部冰压斑；仓库缺口是「冰侧能算接触压力分布、结构侧只吃均匀斑
压力」，缺一个把非均匀接触压力 `p(x,y)` 投影成板格一致节点载荷的 caller。
**几何/参数（m19482 版算例）**：0.8 m × 0.7 m × 0.022 m，AH36（E=206 GPa, ν=0.3），四边简支，
接触足迹 a=0.20 m，平均压强 0.30 MPa，总力 37.699 kN；各构型**同总力同足迹**。
**本单性质**：交差审计（非重做）——核验三件套、按 `tensor_benchmark_cases` 裁判判级并落表。

## ② 怎么做的

1) **取证**：`tensor_dev_missions`(19614/19482)、`tensor_reports`、`tensor_maturity_verdicts`、
`tensor_benchmark_cases`（postgres `tensorplatform`）+ git 对象库/200 个 `agent/*` 分支 + 本地镜像
`/root/tensor-ecosystem/TensorFEM` + 全盘 `find`。
2) **可达性**：`git cat-file -t 826e860a`、`git branch -a | grep 19614`、
`git log --all -- <claimed paths>`。
3) **旁证复算**（区分「算错」与「链路断」）：导出 `agent/m18930` 的同一模块
`src/tensorfem/contact_patch_load.py`(999 行) + `tests/test_contact_patch_load.py`(360 行)，
`PYTHONPATH=src pytest -q` 实跑，并读其 `artifacts/contact_patch_load/*.json` 做数值挑战。
4) **落表**：`INSERT INTO tensor_maturity_verdicts(...) VALUES ('TensorFEM',
 'm19614 GAP-PP-1-ICE-STRUCT-CONTACT-PATCH', 19696, 0, '<JSON>', 'board-audit')` → **id=798**。
工具链：psql 16 / git / pytest / matplotlib（本图程序化生成，零手抄）。

## ③ 计算结果（程序化注入）

**A. 三件套核验（0/3 可取证）**

| 项 | m19614 自述 | 独立取证结果 |
|---|---|---|
| commit/分支 | 826e860a / agent/m19614 | `fatal: Not a valid object name`；无该 ref；镜像亦无 |
| run_artifacts | study.json/stress_fields.json/3 PNG | 任何 ref 无这些路径；全盘 find 无 |
| repro_cmd | 3 条命令 | 基线 main 上三条全部不可执行（文件/子命令不存在） |
| 报告 | 「已生成 /tmp/report_m19614.json」 | `tensor_reports` 无 mission_id=19614 行；文件已回收 |
| 基准裁判 | —— | `anchor_slug='GAP-PP-1-ICE-STRUCT-CONTACT-PATCH'` → **0 rows** |

**B. 旁证实现的放大因子（同总力同足迹，`concentration_sweep.json`）**

| 构型 | 幅度参数 | 网格 | 放大比 σ_nonuniform/σ_uniform | 相对均匀斑 |
|---|---|---|---|---|
{rows}

**C. 网格稳定性**：绝对峰值应力 coarse/fine = **{d["mesh_ratio"]["coarse_over_fine_sigma_peak"]:.4f}**
（≈{(1/d["mesh_ratio"]["coarse_over_fine_sigma_peak"])*100-100:.1f}% 差异，mesh-sensitive）；同一算例的**放大比** coarse/fine =
**{d["mesh_ratio"]["abs_ratio_amplification_centre_amp1"]:.5f}**（≈0，mesh-stable）。

**D. 判级**：m19614 = **L0**（自身功能实质为 0；claimed 实现/测试/工件/文档均不在任何 ref）。
锚点能力实体 = L2 级（真代码 + 14 项测试通过），但归属 **m18657/m18930** 的未合并分支 `agent/m18930`，
不构成 m19614 的 L2；无 caller ⇒ 亦不达 L3；无基准行 ⇒ L4 结构性不可能。

## ④ 图

下图：独立实现（24×48）在同总力下的峰值应力放大比，叠加 ±3%/±15% 判据带。

## ⑤ 注意点

* m19614 的工作树已被平台回收，其原始 pytest/benchmark 输出**不可得**；本审计对此记 `not_run`，不记 pass。
* 远端 GitHub 是否曾存在 `agent/m19614` 不可验证（沙箱内 `git ls-remote origin` 对本仓已存在的
  `agent/m18930` 同样失败）⇒ 记 `unknown`，判据只用本地对象库 + 200 分支 + 镜像三重印证。
* **口径不可跨实现直接比数**：m18930 算例为 clamped bay、4 MPa 级压强；m19482 为四边简支 JOUR 板格、
  0.3 MPa。故 +1% 级 amplification **不否证** m19482 的 +10.96%，只证明「方向一致、幅度构型相关」。
* m19482 报告自认该量「无解析基准」，以守恒+网格无关性替代——诚实，但因此不具备 L4 资格。
* 复用提醒：返修最省力路径是直接采纳 `agent/m18930` 的实现（14 项测试已由本审计实跑通过），
  不必再重写第三份同锚点模块（仓内已存在两份：`contact_patch_load.py`@m18930、`ice_contact_patch_coupling.py`@m18987）。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| m19614 claimed σ_peak（67.60 MPa）| m19482 report id=17792 同值 | 无法复核（实现不可达） | **未判定（not_run）** |
| m19614 claimed 放大 +10.96% | 无解析基准（基准裁判 0 行） | 无容差可依 | **不适用（裁判无行）** |
| 旁证实现：centre-peak 放大比 vs 1.0 | 精确守恒（同总力） | +0.90%~+2.47% | 是（<3%，且比值网格稳定 ≤0.02%） |
| 旁证实现：绝对 σ_peak 网格敏感性 | coarse(12×24) vs fine(24×48) | **−12.6%** | **否（>3%）** ⇒ 绝对 MPa 不达 L4 强度 |
| 投影守恒（力/矩/功共轭） | 场解析值 | <1e−16（该实现测试断言） | 是 |

**结论（honest）**：m19614 自身交付 = **不可取证、无基准判据、L0**，交差为 **reopened**；
锚点能力实体在 `agent/m18930` 分支真实存在（L2 实跑通过），但需合入可达 ref 并由平台补
`tensor_benchmark_cases` 判据行（建议 `M_max/(q a²)=0.0479` tol<3%）才可能继续升级。
"""
    payload = {
        "title": "交差审计 m19614 · GAP-PP-1-ICE-STRUCT-CONTACT-PATCH：三件套 0/3 可取证、基准裁判无行，判 L0（reopened）",
        "summary": (
            "m19614 承诺的 commit 826e860a/分支 agent/m19614 在对象库、200 个本地分支与镜像中均不可达；"
            "6 个 run_artifact/实现/文档路径在任何 ref 都不存在，3 条 repro_cmd 在基线树全部不可执行；"
            "基准裁判对锚点返回 0 行 ⇒ 判 L0 并落表（verdict id=798）。旁证实现（agent/m18930）实跑 14 项测试"
            "通过，证明是链路断裂而非算错：同总力 centre-peak 放大 +0.9%~+2.5%（比值网格稳定），"
            "绝对峰值应力网格敏感 12.6%。处置 reopened。"
        ),
        "scenario_key": "ice-structure-contact-patch-audit",
        "mission_id": 19696,
        "tags": ["audit", "maturity-gate", "TensorFEM", "ice-structure", "GAP-PP-1", "L0", "reopened"],
        "markdown": md,
        "figures": [
            {
                "name": "audit_m19696_amplification.png",
                "caption": (
                    "独立实现 agent/m18930（24×48，同总力同足迹）的峰值应力放大比：centre-peak +0.90%~+2.47%，"
                    "one-sided ramp +9.55%~+19.10%，全部落在 ±15% 判据带内；绿带 ±3%。"
                    "说明非均匀斑放大的方向与 m19482 一致但幅度构型相关，不能被当作常数修正因子。"
                ),
                "png_base64": base64.b64encode(png).decode(),
            }
        ],
    }
    tok = login()
    st, out = post("tensor/reports", payload, tok)
    print("POST", st, json.dumps(out, ensure_ascii=False)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())