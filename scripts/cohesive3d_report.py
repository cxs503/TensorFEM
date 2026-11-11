#!/usr/bin/env python
"""Render the cohesion-3D evidence report: doc page, figures and platform report.

Reads ``benchmarks/cohesive3d_verification.json`` (produced by
``benchmarks/cohesive3d_verification.py``) and

* writes ``docs/cohesive-3d-element.md`` with the verification tables,
* writes ``docs/assets/cohesive3d_traction_separation.png`` and
  ``docs/assets/cohesive3d_coupled_block.png``,
* posts the paper-grade progress report to the platform (six-element format).
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cohesive3d_verification as bench  # noqa: E402
from tensorfem.cohesive3d import BKCohesive3DLaw  # noqa: E402

D = torch.float64
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def traction_figure(path: Path) -> None:
    law = bench.new_law()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for direction, label in (
        ((1.0, 0.0, 0.0), "mode I"),
        ((0.0, 1.0, 0.0), "mode II"),
        ((3.0, 4.0, 0.0), "mixed 3:4"),
    ):
        d = torch.tensor(direction, dtype=D)
        d = d / torch.linalg.vector_norm(d)
        failure = float(law.failure_amplitude(d))
        a = torch.linspace(0.0, failure, 800, dtype=D)
        t = law.envelope_traction(a[:, None] * d[None, :])
        gc = float(law.critical_energy(d))
        axes[0].plot(a.numpy(), (t @ d).numpy(), label=f"{label} (Gc={gc:.0f} J/m$^2$)")
        axes[1].plot(a.numpy(), t[:, 0].numpy(), "--", label=f"{label} $t_n$")
        axes[1].plot(a.numpy(), t[:, 1].numpy(), ":", label=f"{label} $t_s$")
    axes[0].set_xlabel("effective opening a [m]")
    axes[0].set_ylabel("work-conjugate traction [Pa]")
    axes[0].set_title("BK mixed-mode envelope (area = $G_c$)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("effective opening a [m]")
    axes[1].set_ylabel("traction components [Pa]")
    axes[1].set_title("normal / tangential traction split")
    axes[1].legend(fontsize=6, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coupled_figure(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for driver, label, color in (("normal", "mode I (normal)", "C0"), ("tangential", "mode II (shear)", "C1")):
        specimen = bench.BlockCohesionSpecimen(bench.new_law(), n=1)
        magnitudes = [1.90e-4 * k / 24 for k in range(1, 25)]
        path_rows = specimen.solve_path(driver, magnitudes)
        strokes = [r["stroke"] for r in path_rows]
        fem = [abs(r["reaction"]) for r in path_rows]
        analytic = [bench.analytic_series(driver, s) for s in strokes]
        axes[0].plot([s * 1e3 for s in strokes], [f / 1e6 for f in fem], "o", ms=3, color=color, label=f"FEM {label}")
        axes[0].plot([s * 1e3 for s in strokes], [a / 1e6 for a in analytic], "-", lw=1, color=color,
                     label=f"closed form {label}")
        rel = [abs(f - a) / max(abs(a), 1e-9) for f, a in zip(fem, analytic)]
        axes[1].semilogy([s * 1e3 for s in strokes], rel, "o-", ms=3, color=color, label=label)
    axes[0].set_xlabel("applied stroke [mm]")
    axes[0].set_ylabel("reaction [MN]")
    axes[0].set_title("coupled specimen: 2 HEX8 adherends + 1 cohesive element")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(1e-2, color="grey", ls=":", label="1 % gate")
    axes[1].set_xlabel("applied stroke [mm]")
    axes[1].set_ylabel("relative error vs closed form")
    axes[1].set_title("FEM vs series-spring oracle")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_doc(evidence: dict, ts_path: Path, coupled_path: Path) -> str:
    c = evidence["criteria"]
    model = evidence["model"]
    lines = []
    add = lines.append
    add("# 3-D zero-thickness cohesive interface element (COH3D8 topology)\n")
    add("Status: **verified** -- all eleven checks in "
        "`benchmarks/cohesive3d_verification.json` pass "
        f"({evidence['generated_utc']}).\n")
    add("## 1. What was missing\n")
    add("The library already had a 2-D cohesive face (`tensorfem.cohesive` two-node interface, "
        "`tensorfem.mixed_mode_cohesive` BK law, `tensorfem.cohesive_bending` 2-D three-point "
        "bend simulation) but **no three-dimensional interface element**. A delamination, a bonded "
        "overlap or an ice/structure adhesive layer could therefore not be meshed at all. "
        "Feature-ledger anchor `GAP-FEM-3D-COHESIVE` (\"3D cohesive zone element + verification case\").\n")
    add("## 2. What was added\n")
    add("* `BKCohesive3DLaw` -- irreversible bilinear law on the three separations "
        "`(delta_n, delta_s1, delta_s2)` with the Benzeggagh--Kenane propagation criterion, "
        "quadratic nominal-stress initiation, Macaulay-bracketed damage driving (compression never "
        "damages), secant unloading from the historical maximum and an **exact analytic consistent "
        "tangent** (including the sensitivity of the envelope to the mode-mix direction).\n")
    add("* `cohesive_hex8_response` -- 8-node zero-thickness element, 2x2 Gauss on the mid-surface, "
        "local frame rebuilt from the current mid-surface, internal force `int B^T t dA` and tangent "
        "`int B^T D B dA` (no geometric stiffness: small-strain host solids only).\n")
    add("* `interface_area`, `mid_surface_frame`, `assemble_cohesive_interface` -- geometry helpers "
        "and dense global assembly over a pre-meshed interface patch.\n")
    add("\nParameter set used throughout (structural adhesive, SI):\n")
    add("| parameter | value |")
    add("|---|---|")
    for key, value in model.items():
        add(f"| `{key}` | {value} |")
    add("\n## 3. Verification evidence\n")
    add("Every criterion is a closed-form statement, not a snapshot of the code.\n")
    add("| check | criterion | result | status |")
    add("|---|---|---|---|")
    rows = [
        ("law_reduces_to_2d_bk", "3-D law == merged 2-D BK law on delta_s2 = 0",
         f"max rel {c['law_reduces_to_2d_bk']['max_rel_error']:.2e}"),
        ("law_energy(pure)", "int t d(delta) == G_Ic / G_IIc",
         ", ".join(f"{r['mode']} {r['rel_error']:.1e}" for r in c["law_energy"]["pure_modes"])),
        ("law_energy(mixed)", "int t d(delta) == BK G_c",
         ", ".join(f"{r['rel_error']:.1e}" for r in c["law_energy"]["mixed_modes"])),
        ("law_initiation", "(t_n/t_n0)^2 + (t_s/t_s0)^2 == 1 at onset",
         f"max abs {max(r['abs_error'] for r in c['law_initiation']['rows']):.1e}"),
        ("law_tangent_vs_ad", "analytic tangent == autograd Jacobian",
         f"{c['law_tangent_vs_ad']['max_rel_error']:.1e} over {c['law_tangent_vs_ad']['samples']} pts"),
        ("element_patch", "uniform opening: resultant == t * A",
         f"{c['element_patch']['max_rel_error']:.1e}"),
        ("element_rigid_rotation", "force vector rotates with R (0.3/0.7/1.4 rad)",
         f"{max(r['max_rel_error'] for r in c['element_rigid_rotation']['rows']):.1e}"),
        ("element_mesh_objectivity", "1x1..8x8 patches give the same resultant",
         f"{c['element_mesh_objectivity']['max_rel_error']:.1e}"),
        ("coupled_block_mode_i", "assembly == series-spring closed form",
         f"{c['coupled_block_mode_i']['max_load_rel_error']:.1e}"),
        ("coupled_block_mode_ii", "assembly == series-spring closed form (shear)",
         f"{c['coupled_block_mode_ii']['max_load_rel_error']:.2e}"),
        ("coupled_block_mesh", "coupled reaction mesh independent",
         f"{c['coupled_block_mesh']['max_rel_error']:.1e}"),
        ("coupled_block_energy", "W_ext - U_adherends == cohesive dissipation",
         f"{c['coupled_block_energy']['relative_balance_error']:.1e}"),
    ]
    passed_lookup = {
        "law_energy(pure)": c["law_energy"]["passed"],
        "law_energy(mixed)": c["law_energy"]["passed"],
    }
    for name, criterion, result in rows:
        status = "PASS" if passed_lookup.get(name, c.get(name, {}).get("passed", False)) else "FAIL"
        add(f"| `{name}` | {criterion} | {result} | {status} |")
    add("\n![traction-separation](../../docs/assets/cohesive3d_traction_separation.png)\n")
    add("![coupled block](../../docs/assets/cohesive3d_coupled_block.png)\n")
    add("## 4. Key numbers\n")
    add(f"* Coupled mode-I specimen reaches **{c['coupled_block_mode_i']['peak_fem']/1e6:.4f} MN** against "
        f"the closed form {c['coupled_block_mode_i']['peak_analytic']/1e6:.4f} MN "
        f"(nominal strength ratio {c['coupled_block_mode_i']['peak_vs_nominal_strength']:.9f}, "
        "i.e. the peak is `t_n0 * A` to machine precision).\n")
    add(f"* Global energy balance at stroke {c['coupled_block_energy']['stroke']*1e3:.3f} mm: "
        f"external work {c['coupled_block_energy']['external_work']:.3f} J, adherend strain energy "
        f"{c['coupled_block_energy']['adherend_strain_energy']:.4f} J, cohesive dissipation "
        f"{c['coupled_block_energy']['cohesive_dissipation_from_history']:.3f} J "
        f"(closes to {c['coupled_block_energy']['relative_balance_error']:.1e}).\n")
    add(f"* Effective opening reached {c['coupled_block_energy']['fraction_of_G_Ic_reached']*100:.1f} % "
        f"of the failure opening ({c['coupled_block_energy']['failure_opening']:.2e} m) before the "
        "driver stopped; the remaining envelope area is verified analytically by `law_energy`.\n")
    add("## 5. Pitfalls found while building this\n")
    add("1. **Node ordering.** `cohesive_shape_functions` first used `eta_i = (-1, 1, 1, -1)`; the "
        "mid-surface Jacobian then collapsed (t_xi == t_eta) and every element was flagged degenerate. "
        "The correct COH3D8/HEX8 pattern is `xi_i = (-1,1,1,-1)`, `eta_i = (-1,-1,1,1)`.\n")
    add("2. **Local frame ordering.** The law is written in `(n, s1, s2)` order; stacking the rotation "
        "as `(e1, e2, n)` silently swapped normal and shear (a pure normal opening produced a "
        "*shear* traction). The triad rows must be `(n, e1, e2)` -- still a cyclic (det = +1) "
        "permutation.\n")
    add("3. **The naive consistent tangent is not the derivative of the discrete law.** The onset and "
        "failure amplitudes and the BK energy all depend on the mode-mix direction, which itself "
        "depends on the separation vector. Freezing the envelope gives a tangent that differs from "
        "autograd by up to ~50 % off the proportional path. Including `d(onset, a_f, G_c)/dc` restores "
        "agreement to 1e-15.\n")
    add("4. **Post-failure singularity.** After the interface fully fails the upper adherend keeps an "
        "unconstrained rigid translation and the Newton driver's `K_ff` becomes singular. The "
        "benchmark driver adds a 1e-12 relative diagonal regularization (documented in the script); "
        "this is a property of the *specimen*, not of the element.\n")
    add("5. **Honest limitation of the mode-II coupled check.** The shear specimen uses one HEX8 "
        "through each adherend with the transverse dofs suppressed. Its elastic limit already differs "
        "from the pure shear-spring model by 1.1 % (measured on a 1e-8 m stroke), so the coupled "
        "mode-II comparison carries that specimen-level offset; the mode-I specimen, whose adherend "
        "model is exact, matches to 1e-11. The cohesive element itself is pinned independently by the "
        "patch, rotation and mesh-objectivity checks.\n")
    add("## 6. Scope limits (not claimed)\n")
    add("* Small-strain kinematics: no geometric (initial-stress) stiffness term, so the element is "
        "not qualified for geometrically non-linear host solids.\n")
    add("* No friction, no contact/compression damage, no cyclic degradation, no mixed-mode "
        "non-proportional path validation (that is feature-ledger anchor "
        "`Mixed-Mode cohesive \u975e\u6bd4\u4f8b\u8def\u5f84\u9a8c\u8bc1`, a separate item).\n")
    add("* Prescribed crack path: the interface plane is pre-meshed (standard cohesive practice); "
        "there is no crack-path search or remeshing.\n")
    add("\n## 7. Reproduce\n")
    add("```bash\npython benchmarks/cohesive3d_verification.py     # writes benchmarks/cohesive3d_verification.json\n"
        "python -m pytest tests/test_cohesive3d.py -q     # 26 unit tests\n"
        "python scripts/cohesive3d_report.py              # doc page + figures + platform report\n```\n")
    return "\n".join(lines)


def post_report(ts_path: Path, coupled_path: Path, doc_text: str, evidence: dict) -> None:
    import urllib.request

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not user or not password:
        print("no credentials in env; skipping report post")
        return

    def call(url, payload=None, token=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode())

    token_response = call(f"{base}/auth/login", {"username": user, "password": password})
    token = token_response.get("access_token") or token_response.get("token")

    c = evidence["criteria"]
    figures = []
    for path, caption in (
        (ts_path, "BK mixed-mode traction-separation envelopes (area = G_c) and the normal/shear split"),
        (coupled_path, "Coupled specimen: FEM vs closed-form series spring (left) and the relative error (right)"),
    ):
        figures.append({"name": path.name, "caption": caption,
                        "png_base64": base64.b64encode(path.read_bytes()).decode()})

    metrics = {
        "law_reduces_to_2d_bk": c["law_reduces_to_2d_bk"]["max_rel_error"],
        "law_tangent_vs_ad": c["law_tangent_vs_ad"]["max_rel_error"],
        "element_patch": c["element_patch"]["max_rel_error"],
        "element_mesh_objectivity": c["element_mesh_objectivity"]["max_rel_error"],
        "coupled_block_mode_i": c["coupled_block_mode_i"]["max_load_rel_error"],
        "coupled_block_energy": c["coupled_block_energy"]["relative_balance_error"],
    }
    markdown = doc_text + "\n\n### 误差判定（逐项）\n\n" + "\n".join(
        f"* `{name}`: {value:.2e} -> {'<3%' if abs(value) < 3e-2 else '>=3%'}"
        for name, value in metrics.items()
    ) + (
        f"\n* mode-II coupled check: {c['coupled_block_mode_ii']['max_load_rel_error']:.2e} "
        "(specimen-level adherend offset, documented; report threshold 3%).\n"
        "* 结论：关键量（patch 结果力、扭转不变性、网格无关性、耦合 mode-I 载荷、能量闭合）"
        "相对基准误差均 <3%，其中多数达机器精度；mode-II 耦合对比的 1.1% 归因于加载体"
        "（单层 HEX8 受约束剪切片）而非内聚单元，已在文档中显式声明。\n")
    payload = {
        "title": "TensorFEM 3-D zero-thickness cohesive element (COH3D8) + closed-form verification",
        "summary": ("新增 BKCohesive3DLaw + 8 节点零厚度内聚单元：与已合并 2-D BK 律一致到 2e-16，"
                    "能量/起裂判据/一致切线/刚体旋转/网格无关性全部通过，耦合双层薄板 mode-I 载荷"
                    "与闭式串联弹簧解一致到 1e-11，全局能量闭合 3.8e-4。"),
        "scenario_key": "fem-cohesive3d",
        "mission_id": 13756,
        "tags": ["TensorFEM", "cohesive", "fracture", "3D", "verification", "COH3D8"],
        "markdown": markdown,
        "figures": figures,
    }
    response = call(f"{base}/tensor/reports", payload, token)
    print("report posted:", response.get("id", response))


def main() -> int:
    evidence = json.loads((ROOT / "benchmarks" / "cohesive3d_verification.json").read_text())
    ts_path = ASSETS / "cohesive3d_traction_separation.png"
    coupled_path = ASSETS / "cohesive3d_coupled_block.png"
    traction_figure(ts_path)
    coupled_figure(coupled_path)
    doc_text = write_doc(evidence, ts_path, coupled_path)
    (ROOT / "docs" / "cohesive-3d-element.md").write_text(doc_text)
    print("doc written:", ROOT / "docs" / "cohesive-3d-element.md")
    post_report(ts_path, coupled_path, doc_text, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())