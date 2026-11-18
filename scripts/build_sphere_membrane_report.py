"""Recompute the closed-sphere Shell4 mesh ladder and local stress gates."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorfem.sphere_membrane_benchmark import PROBLEM, membrane_reference, solve_sphere_membrane

ASSETS = Path("docs/assets/sphere-membrane")
METRICS = (
    "principal_min_max_rel_error",
    "principal_max_max_rel_error",
    "hoop_max_rel_error",
    "meridional_max_rel_error",
    "radial_displacement_error",
    "uniform_expansion_relative_residual",
)


def main():
    rows = []
    for n in (2, 4, 6, 8, 12, 16, 24):
        result = solve_sphere_membrane(n)
        row = dict(result.radius_ratios, facets_per_face=n)
        row["passed"] = (
            all(row[k] < 0.03 for k in METRICS)
            and row["applied_force_resultant_n"] < 1e-9
            and row["load_moment_norm_Nm"] < 1e-8
            and row["reaction_norm_n"] / row["applied_force_total_n"] < 1e-6
        )
        rows.append(row)
    ASSETS.mkdir(exist_ok=True, parents=True)
    (ASSETS / "sweep.json").write_text(
        json.dumps(
            dict(
                problem=PROBLEM,
                reference=membrane_reference(),
                tolerance=0.03,
                gated_metrics=METRICS,
                rows=rows,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key in (
        "hoop_max_rel_error",
        "meridional_max_rel_error",
        "radial_displacement_error",
        "uniform_expansion_relative_residual",
    ):
        ax.loglog(
            [r["facet_count"] for r in rows],
            [100 * r[key] for r in rows],
            "o-",
            label=key.replace("_", " "),
        )
    ax.axhline(3, ls="--", color="k", label="3%")
    ax.set(xlabel="Shell4 facets", ylabel="Error (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence.png")
    plt.close(fig)
    xyz = result.nodes[result.elements].mean(1)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    dots = ax.scatter(*xyz.T, c=(100 * (result.element_resultants["hoop"] / membrane_reference()["resultant_n_per_m"] - 1)), s=5)
    fig.colorbar(dots, ax=ax, label="Hoop resultant error (%)")
    ax.set_box_aspect((1, 1, 1))
    fig.savefig(ASSETS / "fields.png")
    plt.close(fig)
    lines = [
        "# Shell4 闭合球壳内压验证",
        "",
        "沿用原四节点面元和解析解。半径 10 m，厚度 0.04 m，内压 1 Pa。"
        "参考膜力为 5 N/m，应力为 125 Pa。表中应力指标按各面元中心取最大相对误差。",
        "",
        "| 面元 | 平均膜力误差 | 环向最大误差 | 子午向最大误差 | 径向位移误差 | 非均匀位移残差 | 综合 3% |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        keys = (
            "resultant_error_vs_exact",
            "hoop_max_rel_error",
            "meridional_max_rel_error",
            "radial_displacement_error",
            "uniform_expansion_relative_residual",
        )
        lines.append(
            "| "
            + str(r["facet_count"])
            + " | "
            + " | ".join(f"{100 * r[k]:.3f}%" for k in keys)
            + " | "
            + ("通过" if r["passed"] else "未通过")
            + " |"
        )
    lines += [
        "",
        "修正前 216 面元平均膜力误差为 1005.6%。旧装配打乱节点与分量顺序，"
        "旧压力散布在重复索引处丢失贡献，位移拟合矩阵也打乱了坐标分量。"
        "现对双线性面元做 2×2 积分，累加共享节点载荷，使用三点平移约束去除刚体自由度。",
        "",
        "综合门限同时检查两项主膜力、环向及子午向膜力、径向位移和扣除刚体运动后的位移残差。"
        "验证适用于本闭合薄球壳、均匀内压与面元中心恢复量。",
        "",
        "![误差随细化变化](assets/sphere-membrane/convergence.png)",
        "",
        "![最细网格环向膜力误差](assets/sphere-membrane/fields.png)",
        "",
        "复算命令为 `PYTHONPATH=src python scripts/build_sphere_membrane_report.py`。",
    ]
    Path("docs/sphere-membrane-validation.md").write_text("\n".join(lines) + "\n")
    if not rows[-1]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
