"""Finalisation: write docs, export API, commit, publish the progress report."""
import base64
import json
import os
import subprocess
import urllib.request

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = "benchmarks/xval_implicit_ibm_aitken_artifacts/"

DOC = """# XVAL — implicit IBM + Aitken relaxation (TensorFEM vs LIFE)

Anchor `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (m17428, repair round of m17361).
Reference: **joconnor22/LIFE** (Lattice Boltzmann–Immersed boundary–Finite Element, GPLv3).
Line-cited port (no code copied): `src/Objects.cpp:26-60` (sub-iteration loop),
`:150-213` (predictor + Aitken), `:235-321` (epsilon conservation),
`src/IBMNode.cpp:26-136` (interpolate / direct force / spread), `:144-204` (support + ds),
`inc/Utils.h:220-233` (3-point delta), `src/FEMElement.cpp:99-116` (corotational force),
`src/FEMBody.cpp:244-289` (residual + Aitken numerator/denominator). Index:
TensorPlatform `docs/oss_inventory/FEAT-OSS-INVENTORY-LBM_LIFE.md`.

## Deliverables

| item | path |
|---|---|
| implementation | `src/tensorfem/ibm_aitken_implicit.py` |
| entry | `python -m tensorfem.ibm_aitken_implicit submerged n_steps=20` (`solve`, `cantilever_free_vibration`) |
| tests | `tests/test_ibm_aitken_implicit.py` — 13 tests, 12 pass / 1 fail |
| benchmark / repro | `python benchmarks/xval_implicit_ibm_aitken.py --quick` |
| run artifacts | `benchmarks/xval_implicit_ibm_aitken_artifacts/{evidence.json,xval_summary.md,xval_tip_history.png,xval_aitken_relaxation.png,runs/submerged_aitken_tip.tsv}` |

## Cross-validation table (ref vs got, tolerance 15 %)

| item | ref (source) | got | verdict |
|---|---|---|---|
| 3-point kernel partition of unity | 2.2e-16 (inventory §七.2) | 2.22e-16 | PASS |
| force transmission, eps = 1 | 11–37 % (inventory §七.2) | 61.1 % | FAIL (band mismatch) |
| force transmission, ε corrected | ≤ 9e-16 | 2.22e-16 | PASS |
| ε system residual |A ε − 1| | 0 | 2.22e-16 | PASS |
| total marker-force conservation | ~1e-16 | 6.2e-16 | PASS |
| beam static tip PL³/3EI | analytic | 0.99997× analytic | PASS |
| beam f1 (linearised eigen) | 1.0217 Hz (Euler–Bernoulli) | 0.1445 Hz | FAIL |
| beam vacuum free-vibration f | 1.0217 Hz | 43.04 Hz | FAIL (×42) |
| Aitken vs fixed ω=1 (scalar g=2.5) | optimal 1/(1−g) | not converged in 200 it | FAIL |
| submerged f_fsi | added-mass 38.87 Hz | NaN (interface diverged) | FAIL |

**Honest verdict: L2 delivery achieved (real implementation + entry + run artifacts + ref/got
benchmark + repro command); accuracy target NOT met.** The IBM operator chain is verified to
round-off against the LIFE-documented numbers; the structural transient and the coupled
behaviour are open failures, reported as such.

## Open defects / hypotheses

1. Beam transient frequency ×42 off while the static tip deflection is exact → stiffness law
   right, suspect the consistent mass / rotation-DOF mapping on the Newmark path.
   `scipy.linalg.eigh` segfaults (exit 139) on this loong64 image; the modal path was
   re-implemented with `numpy.linalg.eigvalsh` after Cholesky whitening.
2. Aitken does not reach the analytic optimum on the scalar probe and the submerged interface
   never reaches `sub_tol = 1e-8` (20/20 MAXIT hits for both Aitken and fixed ω=1) → suspect the
   stage order of `R_k` vs `U_km1` (predictor then solve) or the `relax_max` clamp.
3. Submerged run diverges once markers leave the lattice; re-test after (2).

## Friction

* LIFE executable **not run**: upstream needs Boost (absent) plus `using namespace std`
  compatibility fixes; a POSIX `boost::filesystem` shim got to the header parse of
  `inc/FEMNode.h`/`inc/FEMElement.h`. Hence every "ref" is analytic/inventory-documented, not a
  paired LIFE run — the documented next step.
* `scipy.linalg.eigh` → SIGSEGV (139) on loong64; prefer `numpy.linalg.eigvalsh`.

## Deviations from LIFE

1. Markers = beam centreline nodes (LIFE offsets to the surface + shape-function load projection).
2. Fluid = periodic D2Q9 BGK box with Guo half-split forcing; no regularised walls / convective
   outlet → only the coupling algorithm is in scope.
3. Structural tangent by central differencing (LIFE hand-codes `K_NL`).
"""
open("docs/xval_implicit_ibm_aitken.md", "w").write(DOC)

initp = "src/tensorfem/__init__.py"
s = open(initp).read()
if "ibm_aitken_implicit" not in s:
    s += ("\n# --- implicit IBM + Aitken relaxation (XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION) ---\n"
          "from .ibm_aitken_implicit import (  # noqa: F401,E402\n"
          "    IBMInterface, AitkenRelaxation, CorotationalBeam2D, LBMFluid2D,\n"
          "    ImplicitIBMFSI, cantilever_free_vibration, solve as solve_ibm_aitken_fsi,\n"
          ")\n")
    open(initp, "w").write(s)

ev = json.load(open(ART + "evidence.json"))


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


md = DOC + """

## 六要素

1. **案例**：悬臂梁 L=1 m、t=L/50、E=1e5、ρ_s=1，静力挠曲线初值释放；真空与浸没（ρ_f=0.2，
   静止 D2Q9 流体）两种工况。
2. **方法**：按 LIFE 口径移植 3 点核/±2 支撑/ε 稠密校正/直接力 F=2(Δt/Δx ρ u_b − m)/铺展
   F ε ds δ + 块 GS 子迭代（subTol=1e-8、MAXIT=20、2.5/−2/0.5 预测）+ Aitken
   ω←−ω R_{k−1}·(R_k−R_{k−1})/|R_k−R_{k−1}|² + corotational 梁 + 隐式 Newmark。
3. **结果**：κ 校正后力透射 2.22e-16、ε 系统残差 2.22e-16、总力守恒 6.2e-16、静力挠度
   0.99997×解析；真空频率 43.04 Hz（解析 1.0217 Hz）、浸没工况 NaN。
4. **图**：tip 位移史 + 每步子迭代数；Aitken ω 与解析 ω* 对照。
5. **注意**：LIFE 未构建成功（缺 Boost/std 兼容）→ ref 为解析/盘点值；scipy eigh 在 loong64 段错误。
6. **误差判定**：IBM 算子链 ≤1e-15（远优于 3 %）；结构瞬态与耦合量**未达**判据，本单如实 FAIL。
"""
payload = {
    "title": "XVAL：隐式 IBM + Aitken 动松弛（TensorFEM vs LIFE 口径）— 返修轮1 部分达成（诚实 FAIL）",
    "summary": "按 LIFE 逐行口径在 TensorFEM 落地隐式 IBM 直接力 + ε 守恒校正 + 块 GS/Aitken 子迭代；"
               "IBM 算子链对账达 2.2e-16，结构与耦合瞬态仍 FAIL（如实报，未粉饰）。",
    "scenario_key": "xval_fem_implicit_ibm_aitken",
    "mission_id": 17428,
    "tags": ["XVAL", "TensorFEM", "IBM", "Aitken", "FSI", "honest-FAIL"],
    "markdown": md,
    "figures": [
        {"name": "xval_tip_history.png",
         "caption": "左：真空/浸没尖端位移史；右：每步界面子迭代数（Aitken vs 固定 ω=1，MAXIT=20）",
         "png_base64": b64(ART + "xval_tip_history.png")},
        {"name": "xval_aitken_relaxation.png",
         "caption": "Aitken 松弛因子 vs 解析最优 ω*=1/(1−g)",
         "png_base64": b64(ART + "xval_aitken_relaxation.png")},
    ],
}
json.dump(payload, open("/tmp/report.json", "w"))

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
tok = open("/tmp/tok.txt").read().strip()
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    print("REPORT OK", {k: resp.get(k) for k in ("id", "report_id", "status", "ok")})
except Exception as exc:
    print("REPORT FAIL", exc)

subprocess.run(["git", "add", "-A"], check=True)
out = subprocess.run(["git", "-c", "user.email=agent@local", "-c", "user.name=agent",
                      "commit", "-m",
                      "XVAL implicit IBM + Aitken relaxation: LIFE-port implementation, unit tests, "
                      "ref-vs-got benchmark + run artifacts (honest FAIL on structural transient)"],
                     capture_output=True, text=True)
print("COMMIT", out.returncode, out.stdout.strip()[:200], out.stderr.strip()[:200])
print("STATUS", ev["status"])