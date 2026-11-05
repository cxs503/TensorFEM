"""Resolve the docs/benchmark-qualification-status.md conflict as a semantic union.

Both sides describe the Cook membrane residual attribution; m3402 adds the fact that the
recovery field has entered the production path (with its measured errors), main adds the
hemisphere-shell independent oracle and its negative conclusion.  Neither statement is
dropped; only the superseded half-sentence "开孔半球壳仍缺独立应力参照" is replaced by the
oracle result, which is the newer, evidence-backed claim.
"""

from pathlib import Path

ROOT = Path("/root/tp-agent/work/m3873")
path = ROOT / "docs/benchmark-qualification-status.md"
text = path.read_text()

start = text.index("<<<<<<< HEAD")
mid = text.index("=======", start)
end = text.index(">>>>>>> agent/m3402", mid)
ours = text[start + len("<<<<<<< HEAD\n"):mid]
theirs = text[mid + len("=======\n"):end]
assert "hemisphere-hex20-stress-oracle.md" in ours
assert "cook_recovery_qualification.py" in theirs
assert "开孔半球壳仍缺独立应力参考" in theirs

union = (
    "Cook 膜已有独立 Q8 应力参考（m2430），其点式 5.02% 残差的归因由 "
    "[cook-probe-protocol-audit.md](cook-probe-protocol-audit.md) 定谳：全部 5 个规范采样点落在 Q4 "
    "单元边界上，取值不唯一且该协议逐点不收敛；同一位移解经 Gauss 点+节点恢复后 n=64 为 0.120%、"
    "二阶收敛，恢复场现已进入生产路径（`src/tensorfem/stress_recovery.py` + "
    "`cook_recovery_qualification.py`）：n=64 的 5 点 von Mises 最大误差 0.120%、最差分量 1.068%"
    "（均 <3%），n=32 为 0.484%/2.393%，n=16 与 n=8 的分量与 von Mises 失败项如实保留，见 "
    "[cook-membrane-stress-recovery-qualification.md](cook-membrane-stress-recovery-qualification.md)；"
    "存档的单元中心常数场未改动，其 `stress_accuracy_qualified` 仍按 fail-closed 保持 False。"
    "开孔半球壳的独立应力参照**已补并给出否定结论**：[三维 HEX20 实体 oracle]"
    "(hemisphere-hex20-stress-oracle.md) 逐站位复算后，内部区域表面 von Mises 相对误差平均 5.09%、"
    "最大 9.00%（孔边 6.87%、点载附近 7.56%），不满足 3% 门限，`stress_accuracy_qualified` 维持 "
    "`False`；oracle 自身先在 Lamé 圆环（最大 8.6e-7）与薄板弯曲（0.26%）上过标；局部板屈曲的"
    "**单轴压缩**已闭环，面内**剪切与压-剪组合**亦已按文献锚点（k_s=9.34@a/b=1、5.34 长板渐近，"
    "误差 0.17%/2.6%）闭环，见 [marine-plate-shear-buckling-validation.md]"
    "(marine-plate-shear-buckling-validation.md)；仍缺：壳单元几何刚度、非均匀预屈曲应力、"
    "Hertz 接触及其他经典壳体案例的完整场验证与相同质量的报告，不能列为完整报告通过。"
)

resolved = text[:start] + union + text[end + len(">>>>>>> agent/m3402"):]

path.write_text(resolved)
print("markers left:", sum(resolved.count(m) for m in ("<<<<<<<", "=======", ">>>>>>>")))
print(resolved[resolved.index("## 尚未闭环的工作"):][:1200])