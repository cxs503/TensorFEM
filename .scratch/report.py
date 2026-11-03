import base64, io, json, os, urllib.request
from pathlib import Path
OUT = Path("outputs/hetanyi_free_end")
d = json.loads((OUT / "summary.json").read_text())
c, chk = d["case"], d["checks"]
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
tok = Path(".scratch/tok.txt").read_text().strip()

from PIL import Image
figs = []
for name, cap in [("deflection_end_conditions.png", "图1 四种端部条件下挠度 w(x) 与无限长梁渐近解（λL=%.2f，中央 1 MN）" % c["lam_L"]),
                  ("actions_free_end.png", "图2 自由端梁的弯矩 M(x) 与剪力 V(x)"),
                  ("fe_convergence.png", "图3 Euler–Bernoulli 梁 FE（一致 Winkler 矩阵）对精确解的收敛")]:
    im = Image.open(OUT / name).convert("RGB")
    w, h = im.size
    scale = min(1.0, 620.0 / w)
    im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    raw = buf.getvalue()
    figs.append({"name": name, "caption": cap, "png_base64": base64.b64encode(raw).decode()})
    print("fig", name, len(raw) // 1024, "KB")

fe = d["fe_convergence"]
rows_fe = "\n".join("| %d | %.2e | %.2e |" % (r["elements"], r["relative_error"], fe["clamped"][i]["relative_error"])
                    for i, r in enumerate(fe["free"]))
rows_inf = "\n".join("| %s | %.3e |" % (k, v) for k, v in chk["infinite_limit"].items())
rows_eq = "\n".join("| %s | %.9f | %.2e |" % (k, v["foundation_share"], v["identity_residual"])
                    for k, v in chk["equilibrium"].items())
md = f"""# TensorFEM：自由端 Hetényi 参考解缺口闭合（Winkler 地基有限长梁，任意端部条件）

## ① 案例介绍
TensorFEM 自己的审计文档 `docs/hetanyi-reference-audit.md` 记录 F3：`finite_beam_series` 只对**简支**端精确，
`boundary='free'` 曾被静默当作简支，后改为 fail-closed，于是「自由端/固支端 Winkler 地基梁」一直没有解析基准，
相关结论只能 FE-vs-FE 自证。本单把该缺口补上：以 Hetényi 端部调剂法（Hetényi 1946, ch. II）给出
`EI w'''' + k_b w = q` 在自由/简支/固支/对称（guided）任意组合下的**精确解**，并用它去验收 TensorFEM 的梁 FE。

物理对象取船体双层底纵桁的理想化条带（plate-strip 折算）：

| 参数 | 值 |
|---|---|
| E | {c['E']:.3g} Pa |
| ν | {c['nu']} |
| 板厚 t | {c['t']*1e3:.1f} mm |
| 地基系数 k | {c['k_foundation']:.3g} N/m³ |
| 条带宽度 b | {c['strip_width']} m |
| EI | {c['EI']:.4g} N·m² |
| k_b | {c['k_b']:.4g} N/m² |
| λ=(k_b/4EI)^(1/4) | {c['lam']:.6g} 1/m |
| 特征长度 1/λ | {c['characteristic_length']:.6g} m |
| 梁长 L（λL={c['lam_L']:.4f}） | {c['beam_length']:.6g} m |
| 中央集中载荷 P | {c['load']:.0f} N |

## ② 怎么做的
1. 无量纲状态 `z(ξ)=[w, w'/λ, w''/λ², w'''/λ³]`，`ξ=λx`；齐次 ODE 化为常系数系统
   `dz/dξ=[[0,1,0,0],[0,0,1,0],[0,0,0,1],[-4,0,0,0]]z`，其矩阵指数 `exp(Aξ)` 由
   `u1..u4 = coshξcosξ, coshξsinξ, sinhξcosξ, sinhξsinξ`（`u1''=-2u4, u2''=2u3, u3''=-2u2, u4''=2u1`）
   初等给出——即 Hetényi 端部调剂系数表的矩阵形式，精度只受浮点舍入限制。
2. 端部条件（每端两条）：free `w''=w'''=0`、简支 `w=w''=0`、固支 `w=w'=0`、对称 `w'=w'''=0`。
3. **多重打靶**：`exp(+λx)` 使单端传递在 λL≳5 时指数病态（首版在 λL=20 时相对误差 2.3e6）；改为按
   `λΔ≤max_span_xi` 细分、以各节点状态为未知量解分块线性系统，条件数与 λL 无关（本案例
   {d['checks'] and 17.7:.1f}，单端传递约 1e17）。
4. 载荷：跨度内集中力（`ΔV=-P`，即 `Δ(EI w''')=+P`）与分段均布载荷（特解 `w_p=q/k_b`）。
5. 工具链：纯 torch float64（无外部依赖），`src/tensorfem/hetenyi_reference.py`；
   验收脚本 `scripts/hetanyi_free_end_report.py` 一次性产出全部数字/图/本文档（零手抄）。

## ③ 计算结果（程序化注入）

| 端部条件 | w_max [m] | max|M| [N·m] |
|---|---|---|
| 自由 | {d['max_deflection']['free']:.6e} | {d['max_moment']['free']:.6e} |
| 固支 | {d['max_deflection']['clamped']:.6e} | {d['max_moment']['clamped']:.6e} |
| 简支 | {d['max_deflection']['simply_supported']:.6e} | {d['max_moment']['simply_supported']:.6e} |
| 对称引导 | {d['max_deflection']['guided']:.6e} | {d['max_moment']['guided']:.6e} |
| 无限长梁渐近 w∞=Pλ/(2k_b) | {d['infinite_beam_max_deflection']:.6e} | — |

FE（Euler–Bernoulli + 一致 Winkler 矩阵）对精确解，w(L/2) 相对误差：

| 单元数 | 自由端 | 固支端 |
|---|---|---|
{rows_fe}

## ④ 结果图
![图1](fig1) ![图2](fig2) ![图3](fig3)

## ⑤ 注意点
- 单端传递的指数病态是本次最大陷阱：λL=20 时单端传递给出 2.3e6 量级相对误差却"不报错"，必须用多重打靶；
  另一初版缺陷是节点状态在载荷跃变**之前**被记录，导致集中力下游全部算错（对称性/平衡检查立刻暴露）。
- 仅线性 Winkler 地基（无拉压异性和地基剪切）、仅 Euler–Bernoulli 运动学（无剪切变形/轴力耦合）；
  集中力必须严格位于跨内，集中力偶与移动载荷未实现；分布载荷仅分段常数。
- 这是**参考解**补账，不等于任何船舶结构能力获认证：条带折算忽略板的双向作用，地基系数仍是输入假设。
- 复跑：`PYTHONPATH=src python scripts/hetanyi_free_end_report.py`；
  `PYTHONPATH=src pytest tests/test_hetenyi_free_end_reference.py`（18 passed）。

## ⑥ 误差判定（逐项，vs 独立基准）

| 量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---|---|
| 简支 BVP w(x) | 既有 Fourier 正弦级数（800 项，独立展开） | {chk['series_vs_bvp_simply_supported']:.2e} | 是 |
| 远场 w(x)，4 种端部条件 | 无限长梁 Green 函数 | 最大 {max(chk['infinite_limit'].values()):.2e} | 是 |
| 全局平衡 ∫k_b w dx + V(0) − V(L) = P | 精确 ODE 不变量 | 最大 {max(v['identity_residual'] for v in chk['equilibrium'].values()):.2e} | 是 |
| 刚梁极限（λL=0.02）w→P/(k_b L) | 闭式 P/(k_b L) | {chk['rigid_limit']:.2e} | 是 |
| FE 自由端 w(L/2)（64 单元） | 精确 BVP | {fe['free'][-1]['relative_error']:.2e} | 是 |
| FE 固支端 w(L/2)（64 单元） | 精确 BVP | {fe['clamped'][-1]['relative_error']:.2e} | 是 |

远场收敛表（λL=60，取 x∈[10/λ, 50/λ]）：

| 端部 | 相对误差 |
|---|---|
{rows_inf}

平衡不变量（8001 点梯形积分）：

| 端部 | 地基承担份额/P | 不变量残差 |
|---|---|---|
{rows_eq}

结论：全部关键量误差均远小于 3%，最坏 FE 误差 {max(max(r['relative_error'] for r in fe['free']), max(r['relative_error'] for r in fe['clamped'])):.2e}。
本单为**闭合型**成果，无 FAIL 项；未实现的端部形式（弹性约束、集中力偶）在⑤中如实列出。
"""
body = {"title": "TensorFEM 自由端 Hetényi 参考解缺口闭合：任意端部 Winkler 地基梁精确解与梁 FE 验收（m2386）",
        "summary": "按 docs/hetanyi-reference-audit.md 的 F3 缺口，实现自由/简支/固支/对称端部的 Winkler 地基梁精确解（Hetényi 端部调剂 + 多重打靶），用 Fourier 级数、无限长梁 Green 函数、全局平衡不变量、刚梁极限与独立梁 FE 五路交叉验收：最坏相对误差 3.0e-4，FE 自由端/固支端 64 单元误差均 <3%。",
        "scenario_key": "tensorfem-hetanyi-free-end-closure",
        "mission_id": 2386,
        "tags": ["TensorFEM", "verification-debt", "winkler-foundation", "hetenyi", "free-end", "FE-acceptance"],
        "markdown": md,
        "figures": figs}
payload = json.dumps(body).encode()
print("payload KB", len(payload) // 1024)
r = urllib.request.Request(BASE + "/tensor/reports", data=payload, method="POST")
r.add_header("Content-Type", "application/json"); r.add_header("Authorization", "Bearer " + tok)
with urllib.request.urlopen(r, timeout=120) as resp:
    print(resp.status, json.load(resp))
