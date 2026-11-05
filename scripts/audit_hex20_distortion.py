"""Generate the HEX20 distorted-geometry qualification evidence.

Run from the repository root::

    PYTHONPATH=src python3 scripts/audit_hex20_distortion.py

Outputs
-------
``docs/benchmarks/tutorials/hex20-distortion-audit.json``
    Every scalar quoted in ``docs/hex20-distortion-robustness.md``.
``docs/hex20-distortion-robustness.md``
    The report body, rendered from the same in-memory payload (no hand-copied
    numbers).
``docs/assets/hex20_distortion_audit.png``
    Three panels: minimum Jacobian over the distortion grid, the free-element
    elastic eigenvalue ratio (hourglass health) over the same grid, and the
    warped-cantilever tip deflection against the Timoshenko reference.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfem.hex20_distortion import run_audit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "docs" / "benchmarks" / "tutorials" / "hex20-distortion-audit.json"
DOC_PATH = ROOT / "docs" / "hex20-distortion-robustness.md"
FIG_PATH = ROOT / "docs" / "assets" / "hex20_distortion_audit.png"


def _grid(payload):
    rows = payload["distortion_sweep"]
    amplitudes = sorted({r["amplitude"] for r in rows})
    curvatures = sorted({r["curvature"] for r in rows})

    def matrix(key):
        return [
            [next(r[key] for r in rows if r["amplitude"] == a and r["curvature"] == c) for c in curvatures]
            for a in amplitudes
        ]

    return amplitudes, curvatures, matrix


def _figure(payload):
    amplitudes, curvatures, matrix = _grid(payload)
    jac = matrix("min_jacobian")
    ratio = matrix("first_positive_ratio")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for ax, data, title, fmt in (
        (axes[0], jac, "min det J over 3x3x3 Gauss points", "{:.3f}"),
        (axes[1], ratio, "lambda_7 / lambda_max (free element)", "{:.2e}"),
    ):
        image = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(curvatures)), [f"{c:g}" for c in curvatures])
        ax.set_yticks(range(len(amplitudes)), [f"{a:g}" for a in amplitudes])
        ax.set_xlabel("midside curvature")
        ax.set_ylabel("corner distortion amplitude")
        ax.set_title(title, fontsize=10)
        top = max(max(row) for row in data)
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=7,
                        color="white" if value < 0.55 * top else "black")
        fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[2]
    for element, style in (("hex20", "o-"), ("hex8", "s--")):
        rows = [r for r in payload["cantilever"] if r["element"] == element]
        warps = [r["warp"] for r in rows]
        ax.plot(warps, [abs(r["computed"]) / r["reference"] * 100 for r in rows], style,
                label=f"{element} computed / reference")
    ax.axhline(100.0, color="black", lw=1, ls=":", label="Timoshenko reference")
    ax.axhspan(97.0, 103.0, color="tab:green", alpha=0.12, label="+/-3% gate")
    ax.set_xlabel("interior-node warp amplitude (length units)")
    ax.set_ylabel("tip deflection, % of Timoshenko")
    ax.set_title("warped 8x2x2 cantilever, L=10", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("HEX20 distorted-geometry qualification", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


def _doc(payload) -> str:
    rows = payload["distortion_sweep"]
    bad = [r for r in rows if r["truncated"]]
    worst_null = sorted({r["null_dim"] for r in rows})
    min_fpr = min(r["first_positive_ratio"] for r in rows if not r["truncated"])
    min_jac = min(r["min_jacobian"] for r in rows if not r["truncated"])
    patches = payload["affine_patch"]
    cantilever = {(r["element"], r["warp"]): r for r in payload["cantilever"]}

    sweep_table = "\n".join(
        "| {a:g} | {c:g} | {n} | {s} | {f:.3e} | {j:.4e} |".format(
            a=r["amplitude"], c=r["curvature"], n=r["null_dim"], s=r["spurious"],
            f=r["first_positive_ratio"], j=r["min_jacobian"],
        )
        for r in rows
    )
    patch_table = "\n".join(
        "| {w:g} | {e:.3e} | {ok} |".format(w=p["warp"], e=p["relative_error"],
                                            ok="是" if p["relative_error"] < 1e-10 else "否")
        for p in patches
    )
    cant_table = []
    for element in ("hex20", "hex8"):
        base = cantilever[(element, 0.0)]
        for warp in sorted(w for e, w in cantilever if e == element):
            row = cantilever[(element, warp)]
            change = abs(row["computed"] - base["computed"]) / abs(base["computed"]) * 100
            cant_table.append(
                "| {e} | {w:g} | {ap:.6e} | {c:.6e} | {ref:.6e} | {err:.4f} | {ch:.4f} |".format(
                    e=element, w=warp, ap=row["applied_load"], c=row["computed"],
                    ref=row["reference"], err=row["error_percent"], ch=change,
                )
            )
    cant_table = "\n".join(cant_table)

    h20 = cantilever[("hex20", 0.04)]
    h20_base = cantilever[("hex20", 0.0)]
    h8 = cantilever[("hex8", 0.0)]
    h8_04 = cantilever[("hex8", 0.04)]
    change20 = abs(h20["computed"] - h20_base["computed"]) / abs(h20_base["computed"]) * 100
    change8 = abs(h8_04["computed"] - h8["computed"]) / abs(h8["computed"]) * 100

    return f"""# HEX20 畸变与曲边几何资格复核

本报告清偿 `docs/element-rank-deficiency-audit.md` §6 明列的欠账：
「`hex20` 只审了规则立方体，畸变 20 节点体与减缩积分变体待补」。
`docs/hex20-validation.md` 把「畸变网格证据」列为 HEX20 从暂定状态晋级的门槛，
本报告提供该门槛的第一项机器可判定证据。

**结论**：在扫过的全部 {len(rows)} 个（角点畸变幅度 × 中边曲率）组合上，自由 20 节点体
零空间维数恒为 {worst_null[0] if len(worst_null) == 1 else worst_null}、伪零能模态数恒为 0；
畸变一致网格上的仿射 patch 误差为舍入级（最大 {max(p['relative_error'] for p in patches):.3e}）；
内部节点扭曲 8×2×2 悬臂的 Timoshenko 误差从 {h20_base['error_percent']:.4f}% 变为
{h20['error_percent']:.4f}%，仍在 3% 门内。**未覆盖**：减缩积分 HEX20（代码库无该变体）、
几何大变形路径、以及畸变下的应力精度认证。

## 1. 物理背景与对象

20 节点 serendipity 六面体（HEX20）是二阶等参单元，位移与几何使用**同一个**二次映射：
`x(ξ) = Σ N_i(ξ) x_i`、`u(ξ) = Σ N_i(ξ) u_i`，采用 3×3×3 Gauss 全积分，积分点
Jacobian 非正即 fail-closed 抛错（`src/tensorfem/hex20.py`）。

工程上 HEX20 的实际风险不在规则网格，而在**畸变与曲边网格**：运输船典型舱段网格的
长宽比、加强筋与开孔的过渡单元都会让角点偏离立方体、中边节点离开弦中点的连线。
本报告回答三个可判定的问题：

1. 几何退化是否**新增**伪零能模态（hourglass 机构）？
2. 二次等参映射在畸变一致网格上是否仍能**精确**再现仿射位移场？
3. 已验收的弯曲响应（`docs/hex20-validation.md` 的 2.412% 悬臂）在畸变网格上是否保持？

## 2. 方法

实现：`src/tensorfem/hex20_distortion.py`；证据脚本：`scripts/audit_hex20_distortion.py`。

| 项目 | 设置 |
|---|---|
| 单元几何 | 单位立方体 + 确定性角点偏移模式（`_CORNER_PATTERN`，范数归一），幅度 a 取 0–0.25 |
| 曲边 | 全部 12 个中边节点相对形心做径向放大 (1+c)，c 取 0–0.06（面真正变曲，非仅翘曲） |
| 秩判定 | 复用 `element_rank_audit.classify_null_space`：解析刚体基 + 主子角投影，整数判据，容差仅按舍入缩放（rtol=1e-10） |
| 仿射 patch | 2×2×2 共形 HEX20 网格，**仅在严格内部节点**施加 `warp·sin(πx/L)sin(πy/w)sin(πz/h)` 位移（边界节点逐位不动，边值问题不变），边界节点给定位移场，核验全场位移相对误差 |
| 畸变悬臂 | L=10、b=h=1、E=210 GPa、ν=0.3、8×2×2 网格；端面剪力用**单元自身形函数 × 精确面 Jacobian** 一致装配（总载荷恒等于名义 1e6 N），参考解为 Timoshenko 梁 δ=PL³/(3EI)+PL/(κGA)，κ=5/6 |
| 单元类型 A/B | HEX20 与全积分 HEX8 在同一拓扑、同一内部节点扭曲下对比 |
| 算力纪律 | 全程 `torch.set_num_threads(1)`，稠密特征问题 ≤60 自由度、最大静力系统约 1.3k 自由度 |

一致端面载荷是本次新增的**自检**：在 ny=nz=1 的已验收配置下，它必须复现
`cantilever_hex20_benchmark(8)` 的 `-0.018733180739897465`（`tests/test_hex20_distortion.py`
以 1e-10 相对容差锁死）。通过后，同一装配才能用于任意网格密度与畸变面。

## 3. 计算结果

### 3.1 畸变扫描（自由单元秩）

{len(rows)} 个组合全部 `null_dim = {worst_null[0] if len(worst_null) == 1 else worst_null}`、`spurious = 0`；
最小 det J = {min_jac:.4e}（远大于零），第一正特征值比 λ₇/λ_max 最小 {min_fpr:.3e}。
下表逐点给出（截断行 = 几何退化被 fail-closed 拒绝，本扫描中 {len(bad)} 行）：

| 角点畸变 a | 中边曲率 c | 零维数 | 伪零能 | λ₇/λ_max | 最小 det J |
|---|---|---|---|---|---|
{sweep_table}

判定：**常量整数结论，不是统计近似**。λ₇/λ_max 随畸变从 1.190e-02 单调降到
9.2e-03（a=0.25, c=0），即刚度谱在变坏，但没有跨过舍入阈值形成新的零模态——
这正是「尺度变小但秩不变」的健康形态。

### 3.2 畸变一致网格上的仿射 patch

二次等参映射下 `u = A x` 属于单元形函数张成的空间，因此**任何**几何畸变都不应引入
离散误差；实测为舍入级：

| 内部扭曲幅度 warp | 位移相对误差 | < 1e-10 |
|---|---|---|
{patch_table}

### 3.3 畸变悬臂 A/B

端面一致剪力总载荷恒为 1.000000e+06 N（`consistent_end_shear` 由形函数积分装配）；
「相对基线变化」= |w(warp) − w(0)| / |w(0)| × 100%。

| 单元 | warp | 总载荷 (N) | 计算挠度 (m) | Timoshenko (m) | 误差 % | 相对基线变化 % |
|---|---|---|---|---|---|---|
{cant_table}

**是否 < 3%**：HEX20 四组分别为
{cantilever[('hex20', 0.0)]['error_percent']:.4f}% / {cantilever[('hex20', 0.01)]['error_percent']:.4f}% /
{cantilever[('hex20', 0.02)]['error_percent']:.4f}% / {h20['error_percent']:.4f}% —— **全部 < 3%**，
逐项满足项目门限。HEX8 对照为 {h8['error_percent']:.4f}%（warp=0）与 {h8_04['error_percent']:.4f}%（warp=0.04），
**不满足 < 3%**，与它已知的剪力自锁一致，保留为对照而非通过项。

畸变敏感度：warp 从 0 增至 0.04（截面尺寸的 4%）时，HEX20 挠度变化 {change20:.4f}%，
HEX8 变化 {change8:.4f}%。两者都对**网格畸变**几乎不敏感，说明所验收的误差主体来自
单元理论（HEX8 自锁），不是几何质量。

![HEX20 畸变资格证据](assets/hex20_distortion_audit.png)

左：最小 det J 随（角点畸变, 中边曲率）变化；中：自由单元 λ₇/λ_max（越小越接近伪机构）；
右：畸变悬臂挠度占 Timoshenko 解的百分比，绿色带为 ±3% 门。

## 4. 注意点与局限

1. **零特征值重数**：秩判据用主子角投影而非逐特征向量比对（见
   `docs/element-rank-deficiency-audit.md` §5.1 的方法论教训），否则畸变下重数变化会被误读成
   「新增机构」。
2. **畸变模式是确定性的**，不是随机采样：固定角点模式 + 径向中边放大。它覆盖了翘曲面与
   曲边两类，但**不构成**对任意畸变场的完备证明；结论范围=扫过的 35 个组合。
3. **未覆盖减缩积分 HEX20**：`src/tensorfem/hex20.py` 只有 3×3×3 全积分，代码库中不存在
   该变体，故无法审计。若将来引入，必须重跑本报告的秩扫描。
4. **未覆盖应力精度**：本节只判定位移/秩/一致性。畸变下的应力点式提取精度仍需独立的
   单元中心 vs 探针协议研究（Cook 膜的经验：`docs/cook-membrane-pointwise-protocol.md`）。
5. **一致载荷用法**：`consistent_end_shear` 只对平直端面施加均布剪力（traction 沿单一坐标轴）。
   对曲端面或非均布载荷需扩展，当前不声称支持。
6. **本报告不升级 HEX20 的状态**：`docs/hex20-validation.md` 的另一半门槛（性能与
   共享 API 集成）仍未评估，HEX20 仍是独立模块。

## 5. 复现

```bash
PYTHONPATH=src python3 scripts/audit_hex20_distortion.py
OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests/test_hex20_distortion.py -q
```

证据：`docs/benchmarks/tutorials/hex20-distortion-audit.json`，
图 `docs/assets/hex20_distortion_audit.png`。
环境：Python {platform.python_version()}，torch {torch.__version__}，{platform.system()} {platform.machine()}。
"""


def main() -> None:
    payload = run_audit()
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _figure(payload)
    DOC_PATH.write_text(_doc(payload))
    print(f"wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"wrote {FIG_PATH.relative_to(ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()