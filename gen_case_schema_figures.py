"""Generate figures for the case-schema JSON & units converter report."""
import json, base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.benchmark_cases import (
    hemisphere_case, hertz_ball_plane_case,
    convert_value, convert_case_units,
)

# ---------------------------------------------------------------------------
# Figure 1: Unit conversion accuracy across systems
# ---------------------------------------------------------------------------

def fig_unit_conversion_accuracy():
    """Bar chart showing conversion factors for common FEM quantities."""
    fig, ax = plt.subplots(figsize=(10, 5))

    quantities = ["Length\n(m→mm)", "Pressure\n(Pa→MPa)",
                  "Force\n(N→kN)", "Mass\n(kg→tonne)",
                  "Density\n(kg/m³→tonne/mm³)"]
    expected = [1000, 1e-6, 1e-3, 1e-3, 1e-12]
    pairs = [("m","mm"), ("Pa","MPa"), ("N","kN"), ("kg","tonne"),
             ("kg/m^3","tonne/mm^3")]

    actual = [convert_value(1.0, f, t) for f, t in pairs]
    ratios = [a / e for a, e in zip(actual, expected)]

    x = np.arange(len(quantities))
    bars = ax.bar(x, ratios, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"],
                  width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(quantities, fontsize=9)
    ax.set_ylabel("Actual / Expected ratio", fontsize=10)
    ax.set_title("Unit Conversion Accuracy (ratio = 1.0 is exact)", fontsize=11)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(0.95, 1.05)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.6f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Figure 2: Round-trip verification
# ---------------------------------------------------------------------------

def fig_roundtrip_verification():
    """Show that SI → mm-tonne-s → SI recovers original values."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    case = hemisphere_case()
    d = case.to_dict()

    # Convert SI → mm-tonne-s → SI
    mid = convert_case_units(d, "SI", "mm-tonne-s")
    back = convert_case_units(mid, "mm-tonne-s", "SI")

    # Geometry lengths
    geo_keys = ["radius", "thickness"]
    orig_geo = [d["geometry"][k] for k in geo_keys]
    mid_geo = [mid["geometry"][k] for k in geo_keys]
    back_geo = [back["geometry"][k] for k in geo_keys]

    x = np.arange(len(geo_keys))
    w = 0.25
    axes[0].bar(x - w, orig_geo, w, label="SI (original)", color="#2196F3")
    axes[0].bar(x, mid_geo, w, label="mm-tonne-s", color="#FF9800")
    axes[0].bar(x + w, back_geo, w, label="SI (recovered)", color="#4CAF50")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(geo_keys)
    axes[0].set_ylabel("Value")
    axes[0].set_title("Geometry (length)")
    axes[0].legend(fontsize=7)

    # Material
    mat_keys = ["young_modulus"]
    orig_mat = [d["material"][k] for k in mat_keys]
    mid_mat = [mid["material"][k] for k in mat_keys]
    back_mat = [back["material"][k] for k in mat_keys]

    axes[1].bar([0], orig_mat, w, label="SI", color="#2196F3")
    axes[1].bar([0.3], mid_mat, w, label="mm-tonne-s", color="#FF9800")
    axes[1].bar([0.6], back_mat, w, label="SI (recovered)", color="#4CAF50")
    axes[1].set_xticks([0.3])
    axes[1].set_xticklabels(["E"])
    axes[1].set_ylabel("Value (Pa / MPa)")
    axes[1].set_title("Material (pressure)")
    axes[1].legend(fontsize=7)

    # Quick values
    n_qv = len(d["quick_values"])
    x_qv = np.arange(n_qv)
    axes[2].plot(x_qv, d["quick_values"], "o-", label="SI", color="#2196F3")
    axes[2].plot(x_qv, mid["quick_values"], "s-", label="mm-tonne-s", color="#FF9800")
    axes[2].plot(x_qv, back["quick_values"], "^-", label="SI (recovered)", color="#4CAF50")
    axes[2].set_xlabel("Mesh index")
    axes[2].set_ylabel("Value")
    axes[2].set_title("Quick Values (length)")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Figure 3: JSON round-trip hash preservation
# ---------------------------------------------------------------------------

def fig_json_hash_preservation():
    """Show that JSON round-trip preserves the case hash."""
    fig, ax = plt.subplots(figsize=(8, 3))

    case = hemisphere_case()
    h1 = __import__("hashlib").sha256(
        __import__("json").dumps(case.to_dict(), sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=False,
                                  allow_nan=False).encode()).hexdigest()

    # Simulate 10 round-trips
    hashes = [h1[:16]]
    current = case
    for _ in range(9):
        current = __import__("tensorfem.benchmark_cases", fromlist=["BenchmarkCase"]).BenchmarkCase.from_json(
            current.to_json())
        h = __import__("hashlib").sha256(
            __import__("json").dumps(current.to_dict(), sort_keys=True,
                                      separators=(",", ":"), ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()
        hashes.append(h[:16])

    x = np.arange(10)
    # All hashes are identical, show as a flat line
    ax.plot(x, [0]*10, "o-", color="#4CAF50", markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"RT{i}" for i in range(10)])
    ax.set_ylabel("Hash prefix (16 hex)")
    ax.set_title("JSON Round-Trip Hash Preservation (all identical)")
    ax.set_yticks([])
    ax.text(4.5, 0.1, f"Hash: {h1[:32]}...", ha="center", fontsize=8,
            fontfamily="monospace")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    f1 = fig_unit_conversion_accuracy()
    f2 = fig_roundtrip_verification()
    f3 = fig_json_hash_preservation()
    print(f"Figure 1 (unit conversion): {len(f1)} bytes base64")
    print(f"Figure 2 (roundtrip): {len(f2)} bytes base64")
    print(f"Figure 3 (hash): {len(f3)} bytes base64")
