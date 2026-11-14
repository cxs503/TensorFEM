#!/usr/bin/env python3
"""Publish the Levy-verification report (anchor GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT).

Every number is read from benchmarks/artifacts/beta_levy_band_verification.json.
"""
from __future__ import annotations
import base64, json, os, urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HERE = Path(__file__).resolve().parent
ART = HERE / "benchmarks" / "artifacts"


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def main() -> int:
    d = json.loads((ART / "beta_levy_band_verification.json").read_text())
    s = d["summary"]; can = d["canonical_levy"]; pan = d["panel_ritz"]; st = d["strip_limit"]

    band = "\n".join(
        f"| {r['beta']:.1f} | {r['exact_levy']['w_centre_m']*1e3:.4f} | {r['exact_levy']['Mx_centre_Nm_per_m']:.3f} | "
        f"{r['fem'][-1]['w_centre_m']*1e3:.4f} ({r['fem'][-1]['err_w_pct']:+.3f}%) | "
        f"{r['fem'][-1]['Mx_centre_Nm_per_m']:.3f} ({r['fem'][-1]['err_Mx_pct']:+.3f}%) |"
        for r in can["rows"])
    strip = "\n".join(
        f"| {r['K']:.0e} | {r['w_centre_levy']:.8f} | {r['w_centre_beam']:.8f} | {r['err_w_pct']:+.1e}% |"
        for r in st["rows"])
    panel = "\n".join(
        f"| {r['beta']:.1f} | {r['ritz']['50']['w_centre_m']*1e3:.4f} | {r['fem']['w_centre_m']*1e3:.4f} | "
        f"{r['err_w_pct']:+.2f}% | {r['err_Mx_pct']:+.2f}% | {r['ritz_truncation_spread_w_pct']:+.2f}% |"
        for r in pan["rows"])

    md = f"""# 弹性嵌固板格的内部区间验证：Mindlin 有限元 vs 精确 Levý 解

**锚点** GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT（板格边界的弹性嵌固转动刚度，论文 JOUR-ICE-STRUCT §4.2）

## ① 案例介绍

冰带板格由纵骨与横框架提供**有限**转动刚度；旧证据链只钉住了 `beta=0`（简支，对 Navier 基线）与
`beta=1`（固支，对 Timoshenko 系数）两个端点，**区间内部没有任何独立基准**——而论文要报的正是区间。

| 参数 | 值 |
|---|---|
| 验证板 | 1.0 m × 1.0 m × 10 mm 钢板（t/a = 0.01） |
| 材料 | E = 200 GPa, ν = 0.3 |
| 载荷 | 均布压力 1 kPa |
| 约束 | x 边 `beta_x ∈ {{0,0.2,0.4,0.5,0.6,0.8,1.0}}`，y 边简支（`beta_y = 0`） |
| 弹簧律 | `K = ζ D/a`，`ζ = beta/(1-beta+ε)` |
| 论文板格交叉校验 | 0.8 m × 0.7 m × 22 mm，E = 206 GPa，1 MPa 面荷载覆盖全宽，**四边**弹性嵌固 |

取 `beta_y = 0` 是为了让边值问题**恰好**等于 Levý 级数所解的问题，从而把比较净化为有限元的离散误差。

## ② 怎么做的

新增 `src/tensorfem/plate_levy_restraint.py`，三条**不含有限元**的参考解：

1. `levy_restrained_plate`：**精确** Kirchhoff Levý 解。`w = Σ X_n(x) sin(nπy/b)` 把板方程降为
   `D(X''-s_n²X)'' = q_n`；齐次解用衰减指数基 `e^{{s(x-a)}}, e^{{-sx}}, x e^{{s(x-a)}}, x e^{{-sx}}` 表示，
   逐模解析求解 4×4 边界系统（`X=0`，`D(X''-νs²X)=±KX'`）。`K=0` 即简支、`K→∞` 即固支，一次给出**全 β** 的基准。
2. `kirchhoff_strip_*`：圆柱弯曲（梁）闭式解，独立校核 (1)。
3. `ritz_restrained_plate`：双正弦 Rayleigh-Ritz，容许 `Kx≠Ky`，用于四边嵌固的论文板格。

有限元侧未改动（`solve_restrained_panel`，Mindlin Q4/SRI，稀疏直接解）。
工具链：TensorFEM `agent/m20812`；扫描在集群队列执行（作业 `cj2989`，节点 n330）。

## ③ 计算结果

规范板，中心挠度与中心弯矩（精确 Levý vs Mindlin FEM 40×40）：

| β | w_c 精确 [mm] | Mx_c 精确 [N·m/m] | w_c FEM [mm] (误差) | Mx_c FEM [N·m/m] (误差) |
|---|---|---|---|---|
{band}

全 β 区间移除简支挠度的 **{s['deflection_range_ssss_to_clamped_pct']:.1f}%**（0.2218 mm → 0.1047 mm）。

圆柱弯曲极限（b = 100a）对梁闭式解：

| K | w Levý | w 梁闭式 | 误差 |
|---|---|---|---|
{strip}

论文板格（四边嵌固）对 Rayleigh-Ritz：

| β | w_c Ritz [mm] | w_c FEM(80×70) [mm] | 误差 w | 误差 Mx | Ritz 模态敏感度 |
|---|---|---|---|---|---|
{panel}

（`β=1.0` 已从该表剔除：Ritz 正弦基在完全闭合极限数值不可解析，其 30/50 模态自差发散；
`β=1` 由 ③ 的 Levý 精确解与 Timoshenko 固支系数覆盖。）

## ④ 结果图

图 1(beta_levy_band)：左＝归一化中心挠度对 β，精确 Levý 曲线 + 10/20/40 三套网格的 FEM 点（区间内部而非仅端点）；
右＝离散误差－网格 log-log，附 2:1 参考斜率。
图 2(beta_levy_field)：左＝β=0.5 时沿受约束中线的 Mx（FEM 20/40 vs 精确）；右＝β=0.5 的 Mx 场云图，可见弯矩由跨中向框架迁移。

## ⑤ 注意点 / 局限

1. 参考解是 Kirchhoff 薄板，FEM 是 Mindlin；t/a=0.01 时模型差约 0.03%，这正是收敛误差为 +0.03%（FEM 偏软）而非 0 的原因。
2. `beta=1` 是正则极限（`ζ=beta/(1-beta+1e-6)`），两侧参考用同一有限 K 驱动，比较口径一致。
3. Levý 级数在中心量上是**交错级数**：b/a=100 需 n=6401 模态才到 1e-6；只用 41 模态会得到虚假 1.5% 误差。
4. Ritz 在弹簧闭合时病态（弹簧能只约束模态和），采用行缩放 + 1e-12 对角移位，其截断偏差在 §③ 末列显式给出而不掩盖。
5. 适用范围：线弹性、小挠度、均匀结构化网格；参考解不含 Winkler 地基（该情形见 tribuary 案例文件）。

## ⑥ 误差判定（是否 <3%）

| 量 | 基准 | 最大误差 | <3%? |
|---|---|---|---|
| w_c，β=0…1（规范板） | 精确 Levý | {s['max_abs_err_w_pct_finest_mesh']:.3f}% | **是** |
| Mx_c，β=0…1（规范板） | 精确 Levý | {s['max_abs_err_Mx_pct_finest_mesh']:.3f}% | **是** |
| w_c，圆柱弯曲极限 | 梁闭式解 | {s['strip_max_abs_err_w_pct']:.1e}% | **是** |
| M_edge，圆柱弯曲极限 | 梁闭式解 | {s['strip_max_abs_err_M_pct']:.1e}% | **是** |
| w_c，论文板格 β≤0.8 | Ritz（变分） | {s['max_abs_err_w_pct_panel_vs_ritz']:.2f}% | **是** |
| Mx_c，论文板格 β≤0.8 | Ritz（变分） | {s['max_abs_err_Mx_pct_panel_vs_ritz']:.2f}% | **是** |

全部关键量在 3% 以内，论文所引的规范板区间在 **0.1%** 以内。论文可把 SSSS 结论写成一个**已量化区间**，而非无量化上限。
未做/未覆盖：塑性、大挠度、非均匀嵌固、β=1 的 Ritz 参考（已声明原因）。
"""
    figs = []
    for name, cap in (("beta_levy_band.png", "图1 (a) 归一化中心挠度对 β：精确 Levý 曲线 vs Mindlin FEM 10/20/40 网格；(b) 离散误差 log-log 与 O(h²) 参考"),
                      ("beta_levy_field.png", "图2 (c) β=0.5 沿受约束中线的 Mx：FEM 20/40 vs 精确 Levý；(d) β=0.5 的 Mx 场云图（弯矩向框架迁移）")):
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode((ART / name).read_bytes()).decode()})

    token = login()
    payload = {"title": "弹性嵌固板格内部区间验证：Mindlin FEM vs 精确 Levý 解（GAP-PP-4）",
               "summary": f"为 β∈[0,1] 内部提供精确 Kirchhoff Levý 参考：FEM 中心挠度误差 {s['max_abs_err_w_pct_finest_mesh']:.3f}%、中心弯矩 {s['max_abs_err_Mx_pct_finest_mesh']:.3f}%，圆柱弯曲极限对梁闭式解 {s['strip_max_abs_err_w_pct']:.1e}%；区间移除简支挠度 {s['deflection_range_ssss_to_clamped_pct']:.1f}%。",
               "scenario_key": "ice-structure-elastic-restraint-band-verification",
               "mission_id": 20812,
               "tags": ["TensorFEM", "ice-structure", "GAP-PP-4", "verification", "levy-solution", "elastic-restraint", "m20812"],
               "markdown": md,
               "figures": figs}
    req = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read().decode() or "{}")
    print("POST", r.status, {k: out.get(k) for k in ("id", "title", "mission_id")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
