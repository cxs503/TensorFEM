"""Publish the mission report to the platform API (token via env, never argv)."""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
ROOT = pathlib.Path("/root/tp-agent/work/m5254")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(BASE + path, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def get(path: str, token: str | None = None) -> dict:
    request = urllib.request.Request(BASE + path, method="GET")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode())


def token() -> str:
    for path in ("/auth/login", "/auth/token", "/login"):
        try:
            payload = post(path, {"username": USER, "password": PASSWORD})
        except urllib.error.HTTPError as error:
            print("login", path, error.code, error.read()[:200])
            continue
        for key in ("access_token", "token", "accessToken"):
            if payload.get(key):
                print("login ok via", path)
                return payload[key]
    raise SystemExit("could not obtain a token")


def shrink(path: pathlib.Path, width: int) -> str:
    """Downscale a PNG so the whole report stays under the 250 KB payload limit."""
    from PIL import Image

    image = Image.open(path)
    if image.width > width:
        height = int(round(image.height * width / image.width))
        image = image.resize((width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    # palette quantisation keeps the payload small enough for the report API
    # without visibly damaging a contour/curve figure
    image.convert("RGB").quantize(colors=96, method=Image.MEDIANCUT).save(
        buffer, format="PNG", optimize=True
    )
    encoded = base64.b64encode(buffer.getvalue()).decode()
    print(path.name, len(encoded) // 1024, "KB base64")
    return encoded


def main() -> None:
    tok = token()
    markdown = (ROOT / "docs" / "structural-verification-face.md").read_text()
    evidence = json.loads(
        (ROOT / "benchmarks" / "structural_verification_face_evidence.json").read_text()
    )
    cases = {case["case"]: case for case in evidence["cases"]}

    figures = []
    for name, caption, width in (
        (
            "docs/assets/structural_verification_face_cantilever.png",
            "悬臂梁：最细网格 σ_x 单元中心场（左）与端部挠度误差收敛曲线（右，1 % 判据虚线）。",
            860,
        ),
        (
            "docs/assets/structural_verification_face_cylinder.png",
            "厚壁圆筒：Gauss 点 σ_θ / σ_r 与 Lamé 曲线（左）与最坏点环向应力误差收敛（右，2 % 判据）。",
            860,
        ),
        (
            "docs/assets/structural_verification_face_plate.png",
            "固支方板：最细网格挠度云图（左）与中剖面 FEM / Rayleigh–Ritz 对比（右）。",
            860,
        ),
    ):
        figures.append(
            {
                "name": pathlib.Path(name).stem,
                "caption": caption,
                "png_base64": shrink(ROOT / name, width),
            }
        )

    summary = (
        "TensorFEM 结构力学验证面（悬臂梁 / 厚壁圆筒 / 薄板）建立单一门禁契约："
        "Q8 平面应力悬臂梁 vs Euler–Bernoulli 最细网格误差 "
        f"{100 * cases['cantilever_plane_stress_q8']['levels'][-1]['error_vs_euler_bernoulli']:.3f} %"
        f"（判据 <1 %，观测阶 {cases['cantilever_plane_stress_q8']['metrics']['observed_order']:.2f}）；"
        "Q8 平面应变厚壁圆筒 vs Lamé 最坏点环向应力误差 "
        f"{100 * cases['thick_cylinder_plane_strain_q8']['metrics']['gate_value']:.4f} %"
        f"（判据 <2 %，阶 {cases['thick_cylinder_plane_strain_q8']['metrics']['observed_order']:.2f}）；"
        "固支方板 Q4 Mindlin/SRI vs Kirchhoff α 误差 "
        f"{100 * cases['clamped_plate_uniform_pressure_q4_mindlin']['metrics']['gate_value']:.4f} %"
        f"（判据 <3 %，阶 {cases['clamped_plate_uniform_pressure_q4_mindlin']['metrics']['observed_order']:.2f}）。"
        "三案例均做三层网格收敛（细化比 2）与参数扫描（L/h、r_o/r_i、a/t）；"
        "a/t 扫描定量给出 Kirchhoff 基准适用下限 a/t ≥ 50。"
    )

    notes = """
## 要点速览

- **代码**：`src/tensorfem/structural_verification_face.py`（门禁契约 + 三臂驱动 + 解析基准），
  驱动既有生产路径 `quadratic_plane.solve_q8_plane`（Q8）与
  `plate_uniform_load.solve_uniform_square_plate`（Q4 Mindlin/SRI）。
- **验证利器**：`benchmarks/structural_verification_face.py`（一键复跑证据/图/报告），
  `tests/test_structural_verification_face.py`（20 个用例，22 s；其中一条把已提交的证据 JSON
  与代码里的 `GATE_*` / `MIN_ORDER` 对账，防止判据静默漂移或证据被改成手抄）。
- **一处真 bug（已修，已留证）**：按极角重排内壁节点链时链首取到中点节点，二次边被平移半条边，
  角点/中点 1/6–2/3 权重互换 → 环向应力误差 24 %、周向位移起伏 12 %；总合力仍自平衡，
  只看位移会漏判。改成从单元连接取边 + 逐边合力审计（`2 p r sin(Δθ/2)`）后降到 0.27 % → 0.018 %。
- **诚实边界**：二维连续体极限比 EB 高 0.101 %、比 Timoshenko 低 0.094 %，故 "vs EB" 误差序列
  会过零，判据因此写入"最细层 + 连续极限"两道；固支板误差在最细层回弹（0.103 % → 0.113 %），
  已声明该单元不宜追求优于 0.1 % 的板挠度；`a/t < 50` 的行明确标注为**基准失效区**、不参与判据。
- **与本任务书表述的差异**：任务书写"填补 FEM 零验证的结构性空白"，但本仓已有单案例验证文档
  （`docs/q4-cantilever-airy-verification.md`、`docs/lame-cylinder-plane-strain.md`、
  `docs/plate-uniform-load-verification.md` 等）。本文档第 7 节逐条列出本节点**新增**的部分
  （统一门禁契约、Q8 连续体+端剪、精确圆弧曲边+压力载荷+壁厚比扫描、观测阶门禁+a/t 适用边界），
  不重复声明既有案例。
- **顺带发现（非本单引入）**：`tests/test_public_api_contract.py` 在 HEAD 上已失败——
  `tensorfem.__all__` 比 `api/public-api-v1.json` 多 7 项（如 `J2Material`），
  属其它分支合并遗留，本单未触碰 API manifest，未修改。
"""
    body = {
        "title": "TensorFEM 结构力学验证面：悬臂梁 / 厚壁圆筒 / 薄板弯曲解析对拍（3 案例 × 3 层网格 + 参数扫描）",
        "summary": summary,
        "scenario_key": "structural_verification_face",
        "mission_id": 5254,
        "tags": ["tensorfem", "structural", "verification", "convergence", "lame", "plate", "cantilever"],
        "markdown": markdown + notes,
        "figures": figures,
    }
    payload = json.dumps(body)
    print("payload", len(payload) // 1024, "KB")
    if len(payload) > 240 * 1024:
        raise SystemExit("payload too large")
    result = post("/tensor/reports", body, tok)
    print(json.dumps(result)[:400])
    report_id = result.get("id") or result.get("report_id")
    if report_id:
        check = get(f"/tensor/reports/{report_id}", tok)
        print("readback title:", str(check.get("title"))[:60], "| figures:", len(check.get("figures") or []))


if __name__ == "__main__":
    main()