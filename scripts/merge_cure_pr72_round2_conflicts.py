"""One-shot resolver for the two PR #72 (agent/m3327) conflicts.

Both conflicts are "each side appended its own entry" -> resolution is the
verbatim union of the two sides, no side dropped.  Run from the worktree root.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def split_conflict(text, path):
    start = text.index("<<<<<<< HEAD\n")
    mid = text.index("=======\n", start)
    end = text.index(">>>>>>> origin/main\n", mid)
    ours = text[start + len("<<<<<<< HEAD\n"):mid]
    theirs = text[mid + len("=======\n"):end]
    head = text[:start]
    tail = text[end + len(">>>>>>> origin/main\n"):]
    if "<<<<<<<" in tail or ">>>>>>>" in tail:
        sys.exit(f"{path}: more than one conflict hunk, handle manually")
    return head, ours, theirs, tail


def resolve_changelog():
    path = ROOT / "CHANGELOG.md"
    head, ours, theirs, tail = split_conflict(path.read_text(), path)
    # ours   = the branch's 2026-11-03 in-plane linear-variation buckling section
    # theirs = main's single prestressed-modal bullet, which main keeps at the top
    merged = head + theirs + "\n" + ours + tail
    path.write_text(merged)


def resolve_bqs():
    path = ROOT / "docs/benchmark-qualification-status.md"
    head, ours, theirs, tail = split_conflict(path.read_text(), path)
    # Sequence of the union paragraph:
    #  1. Cook membrane attribution + Gauss-recovery numbers, fail-closed flag  (ours & theirs share this)
    #  2. hemisphere stress oracle: main closed the gap with a NEGATIVE verdict  (theirs only)
    #  3. local plate buckling: uniaxial closed, shear / combined closed       (shared)
    #  4. in-plane linearly varying pre-buckling stress now closed            (ours only)
    #  5. remaining gaps = union of both lists, minus what item 4 closed
    sentence_oracle = (
        "开孔半球壳的独立应力参照**已补并给出否定结论**："
        "[三维 HEX20 实体 oracle](hemisphere-hex20-stress-oracle.md) 逐站位复算后，"
        "内部区域表面 von Mises 相对误差平均 5.09%、最大 9.00%（孔边 6.87%、点载附近 7.56%），"
        "不满足 3% 门限，`stress_accuracy_qualified` 维持 `False`；"
        "oracle 自身先在 Lamé 圆环（最大 8.6e-7）与薄板弯曲（0.26%）上过标；"
    )
    sentence_linear = (
        "**沿板宽线性变化的预屈曲压缩**（含均匀、三角形、纯面内弯曲）"
        "已按经典系数 k=4.0/7.81/23.9/25.6 闭环，最大误差 0.28%，"
        "并由独立差分条路线二阶收敛旁证，见 "
        "[marine-plate-linear-stress-buckling-validation.md]"
        "(marine-plate-linear-stress-buckling-validation.md)；"
    )
    sentence_gaps = (
        "仍缺：面内应力沿 x 变化或非线性的预屈曲分布、壳单元几何刚度、"
        "Hertz 接触及其他经典壳体案例的完整场验证与相同质量的报告，"
        "不能列为完整报告通过。"
    )
    common_prefix = (
        "Cook 膜已有独立 Q8 应力参考（m2430），其点式 5.02% 残差的归因由 "
        "[cook-probe-protocol-audit.md](cook-probe-protocol-audit.md) 定谳："
        "全部 5 个规范采样点落在 Q4 单元边界上，取值不唯一且该协议逐点不收敛；"
        "同一位移解经 Gauss 点+节点恢复后 n=64 为 0.120%、二阶收敛，"
        "但恢复场尚未进入生产 `cook_fields`，故 `stress_accuracy_qualified` 保持 False。"
    )
    common_mid = (
        "局部板屈曲的**单轴压缩**已闭环，面内**剪切与压-剪组合**亦已按文献锚点"
        "（k_s=9.34@a/b=1、5.34 长板渐近，误差 0.17%/2.6%）闭环，见 "
        "[marine-plate-shear-buckling-validation.md]"
        "(marine-plate-shear-buckling-validation.md)；"
    )
    # Guards: every claim-bearing fragment must really exist on its original side.
    guards = [
        ("hemisphere-hex20-stress-oracle.md", theirs),
        ("5.09%", theirs),
        ("6.87%", theirs),
        ("Lamé", theirs),
        ("stress_accuracy_qualified` 维持 `False`", theirs),
        ("stress_accuracy_qualified` 保持 False", ours),
        ("5.02%", ours),
        ("cook_fields", ours),
        ("k_s=9.34", ours),
        ("k_s=9.34", theirs),
        ("marine-plate-linear-stress-buckling-validation.md", ours),
        ("23.9", ours),
        ("Hertz", theirs),
        ("Hertz", ours),
    ]
    for frag, side in guards:
        if frag not in side:
            sys.exit(f"guard failed: {frag!r} not on expected side")
    merged_para = (
        common_prefix + sentence_oracle + common_mid + sentence_linear + sentence_gaps
    )
    path.write_text(head + merged_para + tail)


if __name__ == "__main__":
    resolve_changelog()
    resolve_bqs()
    print("resolved CHANGELOG.md and docs/benchmark-qualification-status.md")