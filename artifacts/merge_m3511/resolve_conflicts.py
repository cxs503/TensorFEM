"""Resolve the two textual conflicts of merge agent/m3511 <- main (lossless union)."""

import re
import sys
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m4018")

DOC = ROOT / "docs/element-rank-deficiency-audit.md"
TEST = ROOT / "tests/test_element_rank_audit.py"

BLOCK = re.compile(
    r"<<<<<<< [^\n]*\n(?P<ours>.*?)=======\n(?P<theirs>.*?)>>>>>>> [^\n]*\n",
    re.DOTALL,
)

OURS_A = (
    "（3×3×3 全积分；表格与记录此前写作「2×2×2」是标注错误，已更正，见\n"
    "[HEX20 畸变与曲率鉴定](hex20-distortion-qualification.md)）；\n"
)
THEIRS_A = "（3×3×3 精确阶积分）、`tet10`（五点 Keast）；\n"
UNION_A = (
    "（3×3×3 精确阶积分；表格与记录此前写作「2×2×2」是标注错误，已更正，见\n"
    "[HEX20 畸变与曲率鉴定](hex20-distortion-qualification.md)）、`tet10`（五点 Keast）；\n"
)

OURS_B = "| hex20（3×3×3 全积分） | 60 | 6 | 6 | 0 | 1.190e−02 |\n"
THEIRS_B = "| hex20（3×3×3 精确阶积分） | 60 | 6 | 6 | 0 | 1.190e−02 |\n"
UNION_B = "| hex20（3×3×3 精确阶积分，即 3×3×3 全积分） | 60 | 6 | 6 | 0 | 1.190e−02 |\n"

OURS_C = (
    "   刚度；`hex20` 原先只审了规则立方体；畸变/曲面 20 节点体与 2×2×2 变体已由\n"
    "   [HEX20 畸变与曲率鉴定](hex20-distortion-qualification.md) 补齐（3×3×3：6 零能/0 伪零能；\n"
    "   2×2×2：12 零能/6 伪零能，故减缩积分变体不可采用，生产核不得改为 2×2×2）。\n"
)
THEIRS_C = (
    "   刚度。二次实体单元的欠账（畸变/曲边 `hex20`、`tet10`、减缩积分变体）已在\n"
    "   [二次实体单元的零能审计](quadratic-solid-rank-audit.md) 中补齐：生产 27 点规则三种几何\n"
    "   伪零能均为 0，8 点规则为 6 且刚度偏软 7.65%–8.84%，装配后机构不可传递；同时订正了本表\n"
    "   原先把生产规则误标为 `2×2×2` 的错误。TET10 的积分精度损失（非秩问题）仍待办。\n"
)
UNION_C = (
    "   刚度。二次实体单元的欠账（畸变/曲边 `hex20`、`tet10`、减缩积分变体）已在\n"
    "   [二次实体单元的零能审计](quadratic-solid-rank-audit.md) 中补齐：生产 27 点规则三种几何\n"
    "   伪零能均为 0，8 点规则为 6 且刚度偏软 7.65%–8.84%，装配后机构不可传递；同时订正了本表\n"
    "   原先把生产规则误标为 `2×2×2` 的错误。TET10 的积分精度损失（非秩问题）仍待办。\n"
    "   `hex20` 原先只审了规则立方体；畸变/曲面 20 节点体与 2×2×2 变体已由\n"
    "   [HEX20 畸变与曲率鉴定](hex20-distortion-qualification.md) 补齐（3×3×3：6 零能/0 伪零能；\n"
    "   2×2×2：12 零能/6 伪零能，故减缩积分变体不可采用，生产核不得改为 2×2×2）。\n"
)

UNIONS = {
    "A": (OURS_A, THEIRS_A, UNION_A),
    "B": (OURS_B, THEIRS_B, UNION_B),
    "C": (OURS_C, THEIRS_C, UNION_C),
}

TEST_UNION = '        "tet10 (3-D, straight edges)",\n'  # ours empty, theirs adds the line


def resolve_doc() -> None:
    text = DOC.read_text(encoding="utf-8")
    blocks = list(BLOCK.finditer(text))
    if len(blocks) != 3:
        raise SystemExit(f"expected 3 doc conflict blocks, found {len(blocks)}")
    out = []
    last = 0
    for key, m in zip("ABC", blocks):
        ours, theirs, union = UNIONS[key]
        if m.group("ours") != ours:
            raise SystemExit(f"doc block {key}: ours mismatch:\n{m.group('ours')!r}")
        if m.group("theirs") != theirs:
            raise SystemExit(f"doc block {key}: theirs mismatch:\n{m.group('theirs')!r}")
        out.append(text[last : m.start()])
        out.append(union)
        last = m.end()
    out.append(text[last:])
    DOC.write_text("".join(out), encoding="utf-8")


def resolve_test() -> None:
    text = TEST.read_text(encoding="utf-8")
    blocks = list(BLOCK.finditer(text))
    if len(blocks) != 1:
        raise SystemExit(f"expected 1 test conflict block, found {len(blocks)}")
    m = blocks[0]
    if m.group("ours") != "" or m.group("theirs") != TEST_UNION:
        raise SystemExit(f"test block mismatch: ours={m.group('ours')!r} theirs={m.group('theirs')!r}")
    TEST.write_text(text[: m.start()] + TEST_UNION + text[m.end() :], encoding="utf-8")


resolve_doc()
resolve_test()
for path in (DOC, TEST):
    if BLOCK.search(path.read_text(encoding="utf-8")):
        raise SystemExit(f"leftover conflict markers in {path}")
    if "<<<<<<<" in path.read_text(encoding="utf-8") or ">>>>>>>" in path.read_text(encoding="utf-8"):
        raise SystemExit(f"leftover raw markers in {path}")
print("resolved:", DOC.name, TEST.name)
sys.exit(0)