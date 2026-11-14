"""Post the GAP-PP-3 independent bound verification report (figure + markdown)."""
import base64
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "artifacts" / "gap_pp3_bound_bracket.json").read_text())
an = data["analytic"]
meshes = data["meshes"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
path = meshes[0].get("path_points", [])
load = [p["pressure_Pa"] / 1e6 for p in path]
defs = [p["centre_deflection_m"] * 1e3 for p in path]
unl = meshes[0].get("unload_points", [])
ul = [p["pressure_Pa"] / 1e6 for p in unl]
ud = [p["centre_deflection_m"] * 1e3 for p in unl]
if load:
    ax.plot(defs, load, "o-", label="loading (8x8, H=2e10 Pa)")
if ul:
    ax.plot(ud, ul, "s--", label="unloading")
ax.axhline(an["first_yield_pressure_Pa"] / 1e6, color="g", ls=":",
           label="analytic first yield")
ax.axhline(an["tightest_upper_bound_Pa"] / 1e6, color="r", ls="--",
           label="tightest kinematic upper bound")
ax.set_xlabel("centre deflection [mm]")
ax.set_ylabel("patch pressure [MPa]")
ax.set_title("patch-loaded plate: load-deflection path")
ax.legend(fontsize=7)
ax.grid(alpha=.3)

ax = axes[1]
labels = ["strip\nlower", "Ilyushin\nsine upper", "yield-line\nupper"]
vals = [an["strip_lower_bound_Pa"], an["ilyushin_sine_upper_bound_Pa"],
        an["yield_line_upper_bound_Pa"]]
bars = ax.bar(labels, [v / 1e6 for v in vals],
              color=["#4c72b0", "#dd8452", "#c44e52"])
ax.axhline(an["first_yield_pressure_Pa"] / 1e6, color="g", ls=":",
           label="analytic first yield")
ax.axhline(meshes[0]["fe_first_yield_pressure_Pa"] / 1e6, color="k", ls="-.",
           label="FE first yield (8x8)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v / 1e6, f"{v/1e6:.3f}", ha="center",
            va="bottom", fontsize=8)
ax.set_ylabel("pressure [MPa]")
ax.set_title(f"collapse bracket width U/L = {an['bracket_width_upper_over_lower']:.2f}")
ax.legend(fontsize=7)
ax.grid(alpha=.3, axis="y")

fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150)
png = base64.b64encode(buf.getvalue()).decode()

rows = "\n".join(
    f"| {m['mesh']}x{m['mesh']} | {m['elastic_deflection_error_pct']:+.2f} % | "
    f"{m['fe_first_yield_pressure_Pa']:.0f} | {m['first_yield_error_pct']:+.2f} % |"
    for m in meshes)

markdown = f"""## ① 案例介绍
冰-结构相互作用中冰排对船体板格施加中心斑压力。锚点 `GAP-PP-3-ICE-STRUCT-PLASTIC-SET`
要求板能力从纯弹性扩到弹塑性并给出永久挠度，且验收点名两个独立基准
（解析屈服载荷 + Ilyushin/上限分析解）。本页为 verifier 席对该锚点已合并实现（#1008）的独立复核。

几何/材料：a=b=1.0 m，t=0.02 m，中心斑 0.5a x 0.5b；E=210 GPa，nu=0.3，
sigma_y=355 MPa（DH36），硬化 H=2e10 Pa，分层 Shell4（5 层，exact_thickness）。

## ② 怎么做的
1. 补上锚点点名却缺失的 **Ilyushin 运动学上界**：双正弦机制
   `w = w0 sin(pi x/a) sin(pi y/b)`，耗散 `m_p(|k1|+|k2|)`，闭式
   `p_u = 4 m_p a b (m^2/a^2+n^2/b^2)/(mn) / W_ext(patch)`；
   自检：满布方板退化为 `2*pi^2 m_p/a^2`，`b>>a` 退化为梁的 `pi^2 m_p/a^2`。
2. 按锚点**原文口径**（不做同网格归一化）重测过渡点 vs 解析屈服载荷，3 档网格。
3. 把极限载荷的区间摊开：静力下界 + 两条独立上界。
4. 新增 `tests/test_plate_patch_plastic_bounds.py`（15 项），
   脚本 `scripts/verify_gap_pp3_independent_bounds.py` 直接产出 JSON，零手抄。

## ③ 计算结果
| 量 | 值 |
|---|---|
| m_p = sigma_y t^2/4 | {an['plastic_moment_Nm_per_m']:.0f} N m/m |
| 解析首次屈服 p_y | {an['first_yield_pressure_Pa']:.0f} Pa |
| 条带下界 | {an['strip_lower_bound_Pa']:.0f} Pa |
| Ilyushin 双正弦上界 | {an['ilyushin_sine_upper_bound_Pa']:.0f} Pa |
| 屈服线上界 | {an['yield_line_upper_bound_Pa']:.0f} Pa |
| 两上界互差 | {an['upper_bound_agreement_pct']:.2f} % |
| 区间宽度 U/L | {an['bracket_width_upper_over_lower']:.3f} |
| 永久挠度（8x8，卸载后） | {meshes[0]['path_permanent_deflection_m']*1e3:.3f} mm |

| 网格 | 弹性挠度误差 vs Navier | FE 首次屈服 [Pa] | 原始误差 vs 解析 p_y |
|---|---|---|---|
{rows}

## ④ 结果图
左：载荷-挠度路径（加载 + 卸载 + 解析首次屈服 + 最紧上界）；
右：区间三条界线与 FE 首次屈服的位置。图由 `artifacts/gap_pp3_bound_bracket.json` 程序化绘制。

## ⑤ 注意点
* **查出的问题**：(F1) 上一份报告把「过渡点 vs 解析屈服载荷 <5%」改成「同网格归一化 <5%」，
  属判据弱化；(F2) 锚点点名的 Ilyushin 解在复核前不存在；(F3) 上一份报告解析基准表
  p_u 写 1.42 MPa（代码 1.704 MPa）、p_y 写 ~0.48 MPa（代码 0.804 MPa），数字不可作证据；
  (F4) 区间宽 1.85 倍，两上界互差 21.6 %，「collapse 与上界 <10 %」证明力弱。
* 与上一份报告相反的好消息：当前 HEAD（含 `exact_thickness`）上过渡点**原始**误差仅
  +0.66/+0.20/-0.06 %，A1 在原文口径下即可通过，无需归一化。
* 残余弹性偏差 ~1.4 % 不随加密下降，来源是 Q4 Mindlin 而非上一份报告归因的中点积分。

## ⑥ 误差判定
| 关键量 | 基准 | 误差 | <3 %？ |
|---|---|---|---|
| m_p | 定义式 | 0 | 是 |
| 上界满布方板值 | 2 pi^2 m_p/a^2 | <1e-12 | 是 |
| 弹性中心挠度 | Navier 41 项 | +1.33/+1.33/+1.47 % | 是（临界） |
| 过渡点 A1 | 解析 p_y | +0.66/+0.20/-0.06 % | **是** |
| 极限载荷 A2 | Ilyushin 上界 | FE 极限点未取得（H=2e10 在 1.48 MPa 仍收敛） | **否 / WITHHELD** |

**诚实声明**：A2 判为 WITHHELD（证据不足），非 pass。需近理想塑性 + 极限点判据重测。
"""

payload = {
    "title": "GAP-PP-3 独立验证：板格斑载塑性上界补 Ilyushin 解 + 过渡点原口径复测",
    "summary": ("为冰-结构板格弹塑性锚点补上缺失的 Ilyushin 双正弦上界（比屈服线上界紧 17.8 %）；"
                "按原文口径复测首次屈服过渡点误差 +0.66/+0.20/-0.06 %（<5 % 通过，无需归一化）；"
                "极限载荷因区间宽 1.85 倍判 WITHHELD。"),
    "scenario_key": "gap_pp3_ice_struct_plastic_bound",
    "mission_id": 20938,
    "tags": ["TensorFEM", "plate", "plasticity", "upper-bound", "verification",
             "GAP-PP-3", "ice-structure"],
    "markdown": markdown,
    "figures": [{"name": "gap_pp3_bounds.png",
                 "caption": "左：斑载载荷-挠度路径与解析/上界界线；右：塑性极限区间三条界线与 FE 首次屈服",
                 "png_base64": png}],
}

base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1").rstrip("/")
s = requests.Session()
r = s.post(f"{base}/auth/login", json={"username": os.environ["TENSOR_AGENT_USER"],
                                       "password": os.environ["TENSOR_AGENT_PASS"]})
r.raise_for_status()
token = r.json().get("access_token") or r.json().get("token")
resp = s.post(f"{base}/tensor/reports", json=payload,
              headers={"Authorization": f"Bearer {token}"})
print(resp.status_code, resp.text[:400])