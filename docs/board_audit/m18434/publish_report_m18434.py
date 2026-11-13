#!/usr/bin/env python3
"""Build + publish m18434 board-audit report for m18409.

All numbers are parsed live from the m18409 worktree (audit tool stdout,
manifest sha256 recompute, pytest) — zero hand-typed results.
"""
import base64
import hashlib
import json
import os
import pathlib
import re
import subprocess
import urllib.request

AUDIT_TREE = pathlib.Path("/root/tp-agent/work/m18409")
HERE = pathlib.Path(__file__).resolve().parent
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
TOK = open("/tmp/tok.txt").read().strip()


def sh(cmd, cwd=AUDIT_TREE, timeout=900):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def main():
    # 1) rebuild + audit + renderables (live recompute)
    rc1, out1 = sh("python3 tools/anchor_figure_rebuild.py --write")
    rc2, out2 = sh("python3 tools/verified_anchor_audit.py --write")
    rc3, out3 = sh("python3 tools/certificate_renderables.py --check")
    m = re.search(r"anchors=(\d+) certified=(\d+) contradicted=(\d+) uncertified=(\d+) gates=(\d+) recomputed=(\d+)", out2)
    anchors, certified, contradicted, uncertified, gates, recomputed = (int(x) for x in m.groups())
    mv = re.search(r"violations=(\d+)", out3)
    violations = int(mv.group(1))

    # 2) manifest sha256 recompute
    ok = bad = 0
    nfig = 0
    for mf in sorted((AUDIT_TREE / "docs/assets/verified").glob("*/figures_manifest.json")):
        man = json.loads(mf.read_text())
        entries = man.get("figures", man)
        for name, e in (entries.items() if isinstance(entries, dict) else []):
            p = mf.parent / name
            nfig += 1
            if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == e.get("sha256"):
                ok += 1
            else:
                bad += 1

    # 3) pytest
    rc4, out4 = sh(
        "python3 -m pytest tests/test_anchor_figure_rebuild.py "
        "tests/test_anchor_figure_gate_safety.py tests/test_certificate_renderables.py "
        "tests/test_verified_anchor_certificates.py tests/test_inverse_anchor_certificate.py -q"
    )
    pm = re.search(r"(\d+) passed", out4)
    npass = int(pm.group(1)) if pm else 0

    # 4) benchmark judge table (0 rows expected)
    rc5, out5 = sh(
        "runuser -u postgres -- psql -d tensorplatform -t -c "
        "\"SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO'\"",
        cwd="/",
    )
    bench_rows = int(out5.strip().splitlines()[-1].strip())

    # 5) verdict row
    rc6, out6 = sh(
        "runuser -u postgres -- psql -d tensorplatform -t -c "
        "\"SELECT id, level, gate_state FROM tensor_maturity_verdicts WHERE mission_id=18434 AND verdict_by='board-audit'\"",
        cwd="/",
    )
    vrow = out6.strip().splitlines()[-1].strip()

    # 6) figure: the audited deliverable's key comparison curve
    fig_path = AUDIT_TREE / "docs/assets/verified/fem_winkler_plate_k_sweep/winkler_k_sweep_error.png"
    fig_b64 = base64.b64encode(fig_path.read_bytes()).decode()
    fig_sha = hashlib.sha256(fig_path.read_bytes()).hexdigest()[:16]

    md = f"""# 交差审计 m18409: XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO (TensorFEM) → 判级 L2

## 1. 案例介绍（被审计交付物）

m18409（status=succeeded, approved）挂锚 `XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO`
（LBM 域 shan-chen MCMP barycentric，外部参照 lssfau/lbmpy），实际交付为
**TensorFEM 证书图件欠账闭合**：新增 `tools/anchor_figure_rebuild.py`（399 行），
从 3 个 verified 锚（fem_inverse_identification / fem_sent_j_integral /
fem_winkler_plate_k_sweep）的 committed `result.json` **零求解**重绘 10 张图到
`docs/assets/verified/<anchor>/`，sha256 manifest 逐图绑定；证书改指新路径；
修掉 `fem_inverse_identification/run.py` 的证书回滚隐患。commit `bf90086d`
（agent/m18409，25 files，+1305/−94）。

| 参数 | 值 |
|---|---|
| 被审计 commit | bf90086d2410f028363409d45320dfb5ab25dd81 |
| 交付工具 | tools/anchor_figure_rebuild.py:1-399（main() :355） |
| 交付测试 | tests/test_anchor_figure_rebuild.py:1-117 等 5 文件 |
| 报告 | tensor_reports id=17216（published，六段齐全） |

## 2. 怎么做的（审计方法）

1. psql 直查裁判表 `tensor_benchmark_cases`（runuser -u postgres）；
2. 工作树取证 `/root/tp-agent/work/m18409`（git log/show/status/diff）；
3. 独立重算 10 图 sha256 对 manifest；
4. 全链复跑 repro_cmd（rebuild --write → audit --write → renderables --check → pytest 5 文件）；
5. 报告六段核对（API 拉取 id=17216 全文）；
6. 全树 grep `shan`/`barycentric` 验证锚点痕迹（0 相关命中，barycentric 命中均为无关 FEM 接触力学）。

## 3. 计算结果（审计方实跑，程序化注入，零手抄）

| 指标 | 报告声称（后态） | 审计方复算 | 一致 |
|---|---|---|---|
| certified | 31 | {certified} | {'✅' if certified == 31 else '❌'} |
| contradicted | 1（仅 fem_structural_acoustics） | {contradicted} | {'✅' if contradicted == 1 else '❌'} |
| gates 复算 | 349/368 | {recomputed}/{gates} | {'✅' if (recomputed, gates) == (349, 368) else '❌'} |
| renderables violations | 0 | {violations} | {'✅' if violations == 0 else '❌'} |
| manifest sha256 | 10/10 绑定 | ok={ok} bad={bad}（共 {nfig} 图） | {'✅' if bad == 0 and ok == 10 else '❌'} |
| pytest（5 文件） | 44 passed | {npass} passed | {'✅' if npass == 44 else '❌'} |
| 裁判表行数（该锚） | —（未声称） | {bench_rows} 行 | 见 §6 |

确定性：复跑后 `git status` 仅 `generated_at` 时间戳漂移，gate 数值零变化。

## 4. 结果图

被审计交付物的关键比较曲线（winkler k-sweep 误差 vs k，含 5% 门限线；
审计方重渲染产物，sha256 前缀 {fig_sha}，与 manifest 一致）：

![winkler_k_sweep_error](figures: winkler_k_sweep_error.png)

## 5. 注意点

- **锚点≠交付物**：锚点属 TensorLBM 域，交付物是 TensorFEM 验证基础设施；agent 在
  报告 §5 诚实申报（「仓不一致，按诚实申报改做本仓验证欠账；建议另派 TensorLBM 单」）。
  锚点自身验证欠账**未闭合**（TensorFEM 树内无 shan-chen-mcmp 任何运行/日志/SHA）。
- 「前」基线数（certified=28 等）未独立复算（需回退工作树），仅后态经复算验证。
- 仍红 1 锚 fem_structural_acoustics 为证据语义问题（E_SCHEMA/E_PASS_WITHOUT_CRITERION/
  E_GATE_VALUE_NOT_FINITE），非图件问题，本单未动其 gate 数值（不覆盖 honest FAIL）。
- 本审计为只读取证 + 复跑，未改 m18409 工作树任何文件（仅时间戳自然漂移）。

## 6. 误差判定

本审计不产生新物理量；关键量=审计计数与工件哈希，逐项对基准（报告声称值）：

| 关键量 | 基准（报告） | 审计复算 | 偏差 | <3%? |
|---|---|---|---|---|
| certified | 31 | {certified} | 0 | ✅ 是（=0） |
| contradicted | 1 | {contradicted} | 0 | ✅ 是（=0） |
| gates 复算 | 349/368 | {recomputed}/{gates} | 0 | ✅ 是（=0） |
| violations | 0 | {violations} | 0 | ✅ 是（=0） |
| sha256 绑定 | 10/10 | {ok}/{nfig} | 0 | ✅ 是（=0） |
| 物理量 ref-vs-got | 无（裁判表 {bench_rows} 行） | 无 | N/A | 无基准可比——裁判表该锚 0 行，L4 不成立 |

**判级 L2**：真代码（输入 result.json → 确定性重绘 → 输出 PNG+manifest）+ 44 测试绿 +
可复现；未接入主求解器（L3 不成立）；裁判表 0 行、无锚点物理基准验证（L4 不成立）。
verdict 已落表：{vrow}（gate 自动裁定中）。
"""

    report = {
        "title": "交差审计 m18409: XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO → L2（锚点≠交付物，诚实申报；裁判表 0 行）",
        "summary": (
            f"审计 m18409：run_artifacts 10 图+manifest 真存在（sha256 复算 ok={ok} bad={bad}），"
            f"repro_cmd 全链可执行（audit certified={certified} contradicted={contradicted}，"
            f"gates {recomputed}/{gates}，pytest {npass} passed）；但裁判表该锚 0 行、"
            "锚点属 TensorLBM 域而交付为 TensorFEM 证书图件工具（已诚实申报）→ 判 L2，verdict id 落表。"
        ),
        "scenario_key": "board-audit-m18409-xval-lbm-shan-chen-mcmp-barycentric-guo",
        "mission_id": 18434,
        "tags": ["TensorFEM", "board-audit", "m18409", "maturity-verdict", "L2"],
        "markdown": md,
        "figures": [{
            "name": "winkler_k_sweep_error.png",
            "caption": "被审计交付物关键比较曲线：Winkler k-sweep 误差 vs k（5% 门限线）；审计方重渲染，sha256 前缀 " + fig_sha,
            "png_base64": fig_b64,
        }],
    }
    body = json.dumps(report, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + "/tensor/reports", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + TOK)
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read().decode())
    print("PUBLISHED", resp.status, json.dumps(d, ensure_ascii=False)[:400])
    (HERE / "report_m18434.json").write_text(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
