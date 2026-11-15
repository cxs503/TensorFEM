"""Build the platform report payload for XVAL v11 from the benchmark artifacts."""
import base64
import json
import os

ART = "/root/tp-agent/work/m21657/benchmarks/xval_aitken_rate_law_artifacts"
d = json.load(open(os.path.join(ART, "xval_aitken_rate_law.json")))
S, B = d["scalar"], d["real"]


def fig(name, caption):
    with open(os.path.join(ART, name), "rb") as fh:
        return {"name": name, "caption": caption,
                "png_base64": base64.b64encode(fh.read()).decode()}


rows_a = [r for r in S["rows"] if r["test"] == "A1_A2_aitken_exact"]
rows_q = [r for r in S["rows"] if r["test"] == "A3_fixed_rate"]
worst_q = max(r["q_rel_err"] for r in rows_q)
hits = sum(1 for r in rows_q if r["count_agrees"])
stair_ok = sum(1 for r in B["staircase"]
               if (r["converged"] and isinstance(r["n_iter_predicted"], int)
                   and r["n_iter_measured"] == r["n_iter_predicted"])
               or ((not r["converged"]) and not isinstance(r["n_iter_predicted"], int)))
bnd = {r["mass_ratio_param"]: r for r in B["boundary"]}

def tbl_a():
    out = ["| R | omega* closed form | omega* measured | rel err | sub-iterations |",
           "|---|---|---|---|---|"]
    for r in rows_a:
        out.append(f"| {r['R']:g} | {r['omega_star_closed_form']:.6f} | "
                   f"{r['omega_star_measured']:.6f} | {r['rel_err']:.1e} | "
                   f"{r['n_iter_measured']} (pred 3) |")
    return "\n".join(out)


def tbl_b():
    out = ["| m_a/m_s | R measured | gain spread | omega* closed form | omega* measured "
           "| rel err | sub-iters |", "|---|---|---|---|---|---|---|"]
    inv5 = [r for r in B["invariance"] if r["n_warm"] == 5]
    for g, a, i in zip(B["gain"], B["aitken"], inv5):
        out.append(f"| {g['mass_ratio_param']:g} | {g['R_measured']:.4f} | "
                   f"{g['ratio_spread_rel']:.1e} | {a['omega_star_closed_form']:.6f} | "
                   f"{a['omega_star_measured']:.6f} | {a['rel_err']:.1e} | "
                   f"{a['n_iter_measured']} |")
    return "\n".join(out)


def tbl_bnd():
    out = ["| m_a/m_s | R measured | omega_crit closed form | measured | rel err "
           "| bias-corrected rel err |", "|---|---|---|---|---|---|"]
    for r in B["boundary"]:
        out.append(f"| {r['mass_ratio_param']:g} | {r['R_measured']:.4f} | "
                   f"{r['omega_crit_closed_form']:.5f} | {r['omega_crit_bisected']:.5f} | "
                   f"{r['rel_err']:.2e} | {r['rel_err_bias_corrected']:.2e} |")
    return "\n".join(out)


md = f"""# XVAL v11 — implicit IBM + Aitken：从「3 it vs divergence」到闭式成本与稳定律

**锚点** `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`（joconnor22/LIFE implicit IBM + Aitken Δ²）
**仓** TensorFEM · mission m21657 · 验收：对标误差 < 15%（owner 标准）

## 1 案例背景

v1–v10 只验证了一件事：周期 D2Q9 箱中簧板耦合频率对 Lamb (1932) 薄膜附加质量的误差
（3.2 %–12.7 %，全部 < 15 %）。锚点标题的另一半——**松弛本身**——从未量化。v6 报告原话：

> Aitken convergence vs fixed `omega = 1` | qualitative (3 it vs divergence) | n/a

「它能收敛而另一个不收敛」不是把实现对标到参考值，而是证明「确实有事情在发生」。
一个没有参考值的陈述无法适用 < 15 % 判据。**本轮补上参考值**：由求解器实际迭代的映射
推出的闭式结果，先写下来、再测量。

## 2 方法

一个时间步内流体冻结（`IBMImplicitFSI.step` 只在步初取一次 `macroscopic(include_force=False)`），
直接力 IBM 对试探位移精确线性（已由 `test_interface_force_is_linear_in_trial_displacement`
钉住，二阶差分 1e-12），Newmark 求解也是仿射的。故单自由度接口映射为

    y_new = G(y) = g·y + b,   r_k = G(y_k) − y_k,   r_(k+1) = q·r_k,
    q = 1 + ω(g−1) = 1 − ω(1+R),   R := −g

由此四条**精确**（非渐近）律：

| 律 | 陈述 | 出处 |
|---|---|---|
| Aitken 最优 | `ω* = 1/(1+R)`，且从**任意**基准因子都收敛到它 | Irons–Tuck 1969 / Küttler–Wall 2008 |
| 有限终止 | `q = 0` ⇒ **两次松弛 = 三次子迭代**精确命中不动点，对任意 `R` | 本轮 |
| 稳定边界 | `ω < ω_crit = 2/(1+R)` ⇔ `R_crit(ω) = 2/ω − 1`（`ω=1` 即 Causin 等 2005 的 `R_crit=1`） | Causin–Gerbeau–Nobile 2005（`ω=1`） |
| 成本 | `n(tol) = 1 + ⌈ln(tol/|r_1|)/ln|q|⌉` | 本轮 |

测量手段（**不侵入源码**）：`sub_tol=0, max_sub_iter=m` 使 `step()` 报出第 m 个残差范数，
每个 m 用新求解器复现同一确定性前缀，即从公开 API 取回整条残差序列。
配置取锚点本身：`m_a = ρπc²`（Lamb，`c = 8Δ`）、`m_s = m_a/mr`，`m_a/m_s ∈ {{1,2,4,12,30}}`
（12 为锚点强耦合工况）、`k = (m_s+m_a)(2π/T)²`、`T = 400Δt`、box 64²、`dt = Δ = 1`。

## 3 计算结果

### 3.1 标量模型（生产类 `ibm_aitken_implicit.AitkenRelaxation`）

{tbl_a()}

第二次松弛估计即 `1/(1+R)`，误差 ≤ 2.9e-16；迭代精确落到不动点（机器零）。这就是 v6「3 it」的解析内容。

### 3.2 固定松弛：几何率与整数成本律

`|q|` 最大误差 **{worst_q:.1e}**；成本/发散律在 **{hits}/{len(rows_q)}** 个 `(ω,R)` 对上一致。

### 3.3 临界比 `R_crit(ω) = 2/ω − 1`

ω=0.5 → +3.000000（误差 0）；ω=1 → +1.000000（2.2e-16）；ω=1.5 → +0.333333（1.1e-16）；
ω≥2 → 无任何 `R ≥ 0` 可收敛（在 `R=1e-6` 与 `R=50` 两端实测确认）。

### 3.4 真实耦合求解器

{tbl_b()}

同时确立三件事：增益 `R` 在子迭代间**恒定**（散布 ≤ 2.3e-10，这是标量律成立的前提）；
五个耦合步之后**不变**（0.1775→0.1775、2.1284→2.1284）；求解器实际施加的 Aitken 因子等于
`1/(1+R)`，误差 ≤ 1.7e-11。成本对 `R` 从 0.18 到 5.32（30 倍）**恒为 3 次子迭代**。

固定松弛成本律在 `tol ∈ [1e-4,1e-8]` 的 **{stair_ok}/{len(B["staircase"])}** 个点上**整数精确**，
含全部 12 个预测发散点。

### 3.5 稳定边界（二分测量）

{tbl_bnd()}

未校正偏差对五个构型**完全相同**（2.13–2.19e-4），且等于衰减判据可解析的分辨率下限
`1 − (1+q_floor)/2 = 2.15e-4`（`q_floor = 0.95^(1/120) = 0.99957`）。按该下限校正后剩 5e-7…1.5e-6。

## 4 误差判定（owner 标准：< 3 %，否则如实 fail）

| 关键量 | 参考 | 实测 | 误差 | < 3 %？ |
|---|---|---|---|---|
| `ω*` 标量 | `1/(1+R)` | 7 点 | ≤ 2.9e-16 | **是** |
| `ω*` 真实求解器 | `1/(1+R)` | R=0.18…5.32 | ≤ 1.7e-11 | **是** |
| 子迭代次数（Aitken） | 3 | 5 个构型全 3 | 0 | **是** |
| `|q|` 标量 | `|1−ω(1+R)|` | 35 点 | ≤ 1.1e-15 | **是** |
| 子迭代次数（固定） | 整数律 | 50 点 | 0（整数精确） | **是** |
| `R_crit(ω)` 标量 | `2/ω−1` | 3 条正分支 | ≤ 2.2e-16 | **是** |
| `ω_crit` 真实求解器 | `2/(1+R)` | 5 构型 | 2.1e-4 / 校正后 ≤1.5e-6 | **是** |

**本轮无 honest-FAIL 项**：所有量都有参考值，且余量以数量级计。

## 5 注意点

1. 律的适用范围是**仿射映射**（步内流体冻结）——正与 `step()` 的写法一致；跨步 `g` 会变，
   本配置的五步不变性是工况性质、不是定理。
2. `R` 是**接口增益**而非附加质量比：锚点构型 `R = 2.1284` 而解析 `m_a/m_s = 12`，
   差值来自 IBM 抹平界面（v8/v9 记录的 1.18–2.79× 同一离散伪影），故律以 `R = −g` 陈述。
3. **踩过的坑（可复用知识）**：首版用「`max_sub_iter=80` 内是否达到 `sub_tol=1e-6`」定位边界，
   得到偏低 0.3–2.0 % 且随 `R` 增大的误差——不是物理：近 `|q|=1` 所需次数发散，有限
   `max_sub_iter` 在真边界**之前**截断，偏差还依赖初始残差。改用衰减判据并扣除分辨率下限后
   趋势消失。**发散边界不能用容差判据定位。**
4. LIFE 本体仍无法在本镜像构建，全部参考值为闭式（Irons–Tuck / Küttler–Wall / Causin 等），
   配对 LIFE 运行仍是本锚点唯一未了项。
5. 二维、周期箱、静止流体；只针对耦合算法，不覆盖流动求解器。`ω ≥ 2` 对本类问题不可用，
   因此生产的 `relax_max = 1.0` 永不截断最优值 `1/(1+R) ∈ (0,1]`。

## 6 复现

```
python benchmarks/xval_aitken_rate_law.py           # 29 s 单核，表 + JSON + 3 PNG
python -m pytest tests/test_aitken_rate_law.py -q   # 26 passed, 5 s
```
"""

payload = {
    "title": "XVAL v11 — implicit IBM + Aitken：闭式成本与稳定律（ω*=1/(1+R)，3 次子迭代，R_crit=2/ω−1）",
    "summary": ("把锚点 v6 遗留的「Aitken 定性 n/a」一行换成可预测的闭式律：Aitken 最优因子 "
                "ω*=1/(1+R) 精确命中（标量 2.9e-16、真实耦合求解器 ≤1.7e-11），终止次数恒为 3 "
                "（m_a/m_s 1→30 全为 3），固定松弛成本律 50/50 整数精确，稳定边界 ω_crit=2/(1+R) "
                "五个构型一致（偏差 2.1e-4，可解析校正至 ≤1.5e-6）。"),
    "scenario_key": "fem_xval_implicit_ibm_aitken_relaxation",
    "mission_id": 21657,
    "tags": ["TensorFEM", "IBM", "Aitken", "relaxation", "FSI", "added-mass",
             "verification", "closed-form"],
    "markdown": md,
    "figures": [
        fig("xval_aitken_v11_rate.png",
            "固定松弛成本坍缩到闭式 ln(tol)/ln|q|：横轴 |q|=|1-ω(1+R)|，纵轴到达 1e-8 的子迭代数。"),
        fig("xval_aitken_v11_boundary.png",
            "稳定边界：五个附加质量比下二分实测的 (ω_crit, R) 落在闭式 R=2/ω−1 上。"),
        fig("xval_aitken_v11_cost.png",
            "每步耦合成本：Aitken 对任意 R 恒为 3 次子迭代，固定松弛在 R→R_crit(ω) 处发散。"),
    ],
}

with open("/tmp/report.json", "w") as fh:
    json.dump(payload, fh, ensure_ascii=False)
print("report bytes:", os.path.getsize("/tmp/report.json"))
print("figures:", [f["name"] for f in payload["figures"]])

def post_report():
    import urllib.request
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        token = json.loads(r.read())["access_token"]
    req = urllib.request.Request(base + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    print("report id:", out.get("id"), "title:", str(out.get("title"))[:60])


if __name__ == "__main__":
    post_report()
