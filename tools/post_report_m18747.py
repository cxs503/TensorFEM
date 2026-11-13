#!/usr/bin/env python3
"""Publish the m18747 board-audit report (six-section, figures embedded as base64)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

HERE = Path(__file__).resolve().parents[1]
AUDIT = HERE / "artifacts" / "xval_m18651_audit"
FIG = AUDIT / "m18651_board_audit.png"

MD = Path(HERE / "docs" / "audit_report_m18747.md").read_text()


SIX = """## ① 案例介绍（物理背景/几何/参数表）

被审对象是 m18651 在 TensorFEM 树内交付的两组分（MCMP）Shan-Chen D2Q9 格子 Boltzmann 核，用于平台气体减阻/空化/气液混合的流体侧使能。锚点 = `lssfau/lbmpy` `tests/shan_chen/test_shan_chen_two_component.py` 的 64x64 周期域、`omega_a=omega_b=1`、`g_ab=g_ba=6`、`rho0=1`、`psi=rho0(1-e^{-rho/rho0})`、初值 rho_a=0.1(左半)/0.9(右半)、1000 步参考解（该测试自述参考数组引自 `lbm-principles-practice/code/chapter9/shanchen.cpp`）。次级臂 = Laplace 律：96x96、`g_ab=1`、R∈{8,12,16}、3000 步。

| 量 | 值 |
|---|---|
| 网格/边界 | 64x64（锚点臂）、96x96（Laplace 臂）、周期 |
| tau_a=tau_b | 1.0 |
| g_ab=g_ba | 6.0（锚点）、1.0（Laplace） |
| psi | 指数型，rho0=1 |
| 步数 | 1000（锚点）、3000（Laplace）；稳定性扫描 600 |
| 判据 | rel L2(rho_a) < 0.15（owner 精度标准；<3% 为生态自设目标） |

## ② 怎么做的（方法/流程/工具链）

交差核验，非重做。流程：①读 m18651 工作树与 `tensor_reports id=17387`、`tensor_dev_missions.result`；②`psql` 直查裁判表 `tensor_benchmark_cases`；③把交付 `src/`+`examples/` 复制到 `/tmp/audit18651` 原样重跑 repro_cmd，与交付 `results.json` 做递归逐字段比对；④`artifacts/xval_m18651_audit/audit_m18651.py` 做四类独立检查：外部锚点数组忠实性（对 OSS 源码字面量逐位比对 + sha256）、doc 数字回溯 JSON（30 个打印值）、六段齐全性、以及**独立探针**（`g_ab=0/1` 500 步质量守恒、`g_ab=6` 120 步复现发散步）；⑤`grep` 全树查测试与调用方；⑥`gen_figure.py` 出四联图；⑦verdict 落表 `tensor_maturity_verdicts`。

## ③ 计算结果（数字表）

| 检查 | 交付声明 | 审计方结果 | 一致 |
|---|---|---|---|
| repro 工件 | results.json | 重跑后逐字段 **0 差异** | ✅ |
| g_ab=6 首次非有限步 | 30 | **30** | ✅ |
| g_ab=1（600 步） | 稳定 | 500 步稳定，质量漂移 2.8e-14 | ✅ |
| g_ab=0（审计方新增对照） | 未报 | 500 步稳定，质量漂移 2.8e-14 | 新增 |
| Laplace σ / c / R² | 0.006251 / 0.004754 / 0.2396 | 由 JSON 复算一致 | ✅ |
| rho_tot 内外差 | <2% | 1.6% | ✅ |
| 裁判表行数 | —— | **0 行** | L4 无判据 |
| ref-vs-got 数值 | 无（"—"） | 无（三臂 30/26/14 步发散，rel_l2 分支从未执行） | ✅（诚实） |
| 单测 | 自述「未做」 | `tests/` 0 命中 | ✅（但 L1 未证） |
| 调用方（L3） | —— | examples/ 之外 0 命中 | 不成立 |

## ④ 结果图

四联图见 figures（matplotlib 由 `artifacts/xval_m18651_audit/gen_figure.py` 生成，全部曲线数据来自交付 `results.json` 与审计方探针）：(a) 锚点 ref_a vs 交付初值；(b) 发散步 vs g_ab（交付 + 审计方探针）；(c) Laplace Δp–1/R 与 σ/R 拟合；(d) 审计方 500 步质量守恒剖面。**没有画出「与解析解/参考解的比较曲线」——因为交付侧不存在任何 got 剖面**，这本身就是核心审计发现。

## ⑤ 注意点（坑/局限/适用范围）

- 外部锚点数组忠实（与 OSS 测试字面量逐位相等，max|Δ|=0.0），但**「OSS 测试在本机执行并通过」不成立**：该单留下的 `/tmp/m18651_oss/lbmpy_two_component.log` 只有 `can't open file run_lbmpy_ref.py`；本机 lbmpy 2.0/pystencils 2.0 下复跑该测试第 30 行即 `AttributeError: 'LBStencil' object has no attribute 'weights'`（`/tmp/lbtest_clean/out.log`），`/tmp/lbmpy_direct.log` 同错。
- 代码注释称「file sha256 recorded below」，实际 `OSS_REF` 无 sha 字段；`ref_b` 未携带，fallback 为死代码。
- 报告根因叙述（稀相均衡速度位移超音速）与其自家数据不矛盾但不足以支撑：`eq_shift=none` 臂同样 26 步发散。
- 局限：本单不做修复，故「lbmpy 方案在本实现中不可达」未被证伪；未在裁判表补登记基准行。
- 仓边界：锚点属 TensorLBM 域，交付落在 TensorFEM 树内（报告 §1 已申报）。

## ⑥ 误差判定（逐项 + 是否 <3%）

| 关键量 | 基准 | 交付 got | 误差 | <3%？ |
|---|---|---|---|---|
| rho_a 剖面 rel L2（锚点 g_ab=6） | lbmpy ref_a（64 值） | **无数值（30 步发散）** | 不可计算 | **否（无基准可判，honest FAIL）** |
| Laplace σ | σ/R 线性律（R²=0.2396） | 0.006251 | 逐半径相对偏差 5.91% / 9.9% / 11.48% | **否** |
| 审计复现一致性 | 交付 results.json | 同 | 0.0%（逐字段相等） | 是 |
| 质量守恒 | 解析（周期域应守恒） | 漂移 2.8e-14 | ~1e-14 | 是 |

明确回答：**锚点判据（rel L2 < 0.15，owner 标准）与生态自设 <3% 均未被满足，且不是「差一点」而是「没有数字」**——三臂全部在 1000 步之前发散。honest-FAIL 如实记录，审计方独立复跑复现同一发散步，不做任何粉饰。

"""
MD = SIX + MD

TITLE = "m18747 交差审计：m18651 XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO（L2 确认，honest-FAIL 属实，外部锚点「已执行」一条不成立）"


def login() -> str:
    payload = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=payload,
                                 headers={"Content-Type": "application/json"})
    body = json.loads(urllib.request.urlopen(req).read())
    return body.get("token") or body.get("access_token")


def main() -> None:
    token = login()
    fig = base64.b64encode(FIG.read_bytes()).decode()
    body = {
        "title": TITLE,
        "summary": (
            "对 m18651 交付做独立交差核验：真代码/工件/复现三件套成立（审计方重跑 results.json 逐字段 0 差异、"
            "g_ab=6 发散步=30 与交付一致、500 步质量漂移 2.8e-14），但裁判表该锚点 0 行、ref-vs-got 数值从未产生、"
            "无任何单测与调用方；报告 §①「OSS 测试在本机执行并通过」被本机日志与复跑证伪。级别判 L2。"
        ),
        "scenario_key": "xval-lbm-shan-chen-mcmp-barycentric-guo",
        "mission_id": 18747,
        "tags": ["board-audit", "m18651", "shan-chen", "MCMP", "lbm", "TensorFEM", "maturity-L2",
                 "honest-FAIL", "overstated-claim"],
        "markdown": MD,
        "figures": [{
            "name": "m18651_board_audit.png",
            "caption": ("m18651 审计四联图：(a) OSS 锚点 ref_a 剖面 vs 交付初值（无收敛剖面可交）；"
                        "(b) 交付 stability_boundary 与审计方独立探针的首次非有限步 vs g_ab；"
                        "(c) Laplace 臂 Δp–1/R 与 σ/R 拟合（偏移项 c 与 σ/R 同量级 → FAIL）；"
                        "(d) 审计方 500 步探针剖面与 1e-14 量级质量漂移（核是真的）。"),
            "png_base64": fig,
        }],
    }
    req = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer " + token})
    resp = json.loads(urllib.request.urlopen(req).read())
    print(json.dumps({k: resp.get(k) for k in ("id", "title", "created_at")}, ensure_ascii=False))


if __name__ == "__main__":
    main()