"""Fail-closed keyword gate and content fidelity of the bounded ``.inp`` reader.

The reader is the only path from a ship structural deck into ``ModelDB``, so two
properties are gated here:

* **nothing model-bearing may be dropped silently** -- an unknown keyword, an
  unresolved ``*INCLUDE`` or an unsupported section must raise, naming the
  keyword and the line;
* **what is read is exactly what the deck says** -- node coordinates are
  bit-preserved, sets and section data are compared against values derived by
  hand from the deck text, and the write->read round trip is bit exact.
"""
from __future__ import annotations

import pytest

from tensorfem.abaqus_io import (NON_MODEL_KEYWORDS, SUPPORTED_KEYWORDS,
                                 read_inp, write_inp)

# Side-shell strake between frames: one CPS4 panel, thickness 12 mm, edge load.
DECK = """*Heading
side shell strake
**
*Node, nset=NALL
10, 0.0, 0.0
20, 0.4, 0.0
30, 0.4, 0.3
40, 0.0, 0.3
*Element, type=CPS4, elset=STRAKE
7, 10, 20, 30, 40
*Nset, nset=FIX
10, 40
*Nset, nset=EDGE, generate
20, 30, 10
*Material, name=STEEL
*Elastic
210.E9, 0.3
*Solid Section, elset=STRAKE, material=STEEL
0.012
*Boundary
FIX, 1, 2
30, 1, 1, 0.0
*Cload
20, 2, -2500.0
"""


def _read(tmp_path, text, name="deck.inp"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return read_inp(path)


# ----------------------------------------------------------------------
# Content fidelity (values derived by hand from the deck above)
# ----------------------------------------------------------------------
def test_deck_reads_to_the_expected_model(tmp_path):
    db = _read(tmp_path, DECK)
    assert db.nodes == {10: (0.0, 0.0), 20: (0.4, 0.0),
                        30: (0.4, 0.3), 40: (0.0, 0.3)}
    assert len(db.elements) == 1
    block = db.elements[0]
    assert (block.element_type, block.ids, block.connectivity, block.name) == (
        "CPS4", [7], [[10, 20, 30, 40]], "STRAKE")
    # *NODE,NSET= defines the whole-node set; *NSET,GENERATE is applied verbatim.
    assert db.node_sets == {"NALL": {10, 20, 30, 40}, "FIX": {10, 40},
                            "EDGE": {20, 30}}
    assert db.element_sets == {"STRAKE": {7}}
    assert db.materials["STEEL"].elastic == (210.0e9, 0.3)
    section = db.sections[0]
    assert (section.elset, section.material, section.kind) == ("STRAKE", "STEEL", "solid")
    assert section.properties == (0.012,)
    assert [(bc.target, bc.dof_start, bc.dof_end, bc.value) for bc in db.boundaries] == [
        ("FIX", 1, 2, 0.0), (30, 1, 1, 0.0)]
    assert [(load.target, load.dof, load.value) for load in db.loads] == [(20, 2, -2500.0)]


def test_write_read_round_trip_is_bit_exact(tmp_path):
    first = _read(tmp_path, DECK)
    target = tmp_path / "roundtrip.inp"
    write_inp(target, first)
    second = read_inp(target)
    assert second.nodes == first.nodes            # exact float equality
    assert second.elements == first.elements
    assert second.node_sets == first.node_sets
    assert second.element_sets == first.element_sets
    assert second.materials == first.materials
    assert second.sections == first.sections
    assert second.boundaries == first.boundaries
    assert second.loads == first.loads


def test_round_trip_preserves_awkward_coordinates(tmp_path):
    """``.17g`` is required: 6 digits would lose 1e-7 of the ship's geometry."""
    values = [0.1 + 1.0 / 3.0, 1.0 / 7.0, 1234.5678901234567, -0.30000000000000004]
    lines = ["*Node"] + [f"{i + 1}, {v:.17g}, 0.0" for i, v in enumerate(values)]
    db = _read(tmp_path, "\n".join(lines) + "\n", "coords.inp")
    captured = [db.nodes[i + 1][0] for i in range(len(values))]
    assert captured == values
    target = tmp_path / "coords_out.inp"
    write_inp(target, db)
    reread = read_inp(target)
    assert [reread.nodes[i + 1][0] for i in range(len(values))] == values
    # Negative control: the same experiment at 6 significant digits must fail.
    short = _read(tmp_path, "\n".join(["*Node"] + [f"{i + 1}, {v:.6g}, 0.0"
                                                   for i, v in enumerate(values)]) + "\n",
                  "coords6.inp")
    assert [short.nodes[i + 1][0] for i in range(len(values))] != values


def test_keywords_and_set_names_are_case_insensitive(tmp_path):
    upper = _read(tmp_path, DECK, "upper.inp")
    lower = _read(tmp_path, DECK.lower().replace("210.e9", "210.E9"), "lower.inp")
    assert lower.nodes == upper.nodes
    assert lower.element_sets == upper.element_sets
    assert lower.materials["STEEL"].elastic == upper.materials["STEEL"].elastic


# ----------------------------------------------------------------------
# Fail-closed gate: model-bearing keywords may not be dropped silently
# ----------------------------------------------------------------------
def test_shell_section_is_rejected_instead_of_losing_thickness(tmp_path):
    text = """*Node
1,0,0
2,1,0
3,1,1
4,0,1
*Element, type=CPS4, elset=PLATE
1,1,2,3,4
*Shell Section, elset=PLATE, material=STEEL
0.02,
"""
    with pytest.raises(ValueError, match=r"SHELL SECTION.*line 8"):
        _read(tmp_path, text, "shell.inp")


def test_include_is_rejected(tmp_path):
    text = "*Heading\nplate\n*Include, input=part2.inp\n" + DECK
    with pytest.raises(ValueError, match=r"INCLUDE.*line 3.*expand the deck"):
        _read(tmp_path, text, "include.inp")


def test_unknown_keyword_is_rejected_with_its_line(tmp_path):
    text = "*Node\n1,0,0\n2,1,0\n*Distortion Control\n1.0\n*Element, type=T2D2\n1,1,2\n"
    with pytest.raises(ValueError, match=r"unsupported Abaqus keyword \*DISTORTION CONTROL"
                                         r" on line 4"):
        _read(tmp_path, text, "unknown.inp")


@pytest.mark.parametrize("prefix, suffix, match", [
    ("*Step, name=LOAD\n*Static\n1., 1.\n", "*End Step\n", "STEP"),
    ("*Output, field, variable=preselect\n", "", "OUTPUT"),
    ("*Node Print, nset=FIX\nU,\n", "", "NODE PRINT"),
])
def test_non_model_keywords_are_accepted_and_change_nothing(tmp_path, prefix, suffix, match):
    reference = _read(tmp_path, DECK, "plain.inp")
    decorated = _read(tmp_path, prefix + DECK + suffix, f"deco_{match.replace(' ', '_')}.inp")
    for field in ("nodes", "elements", "node_sets", "element_sets", "materials",
                  "sections", "boundaries", "loads"):
        assert getattr(decorated, field) == getattr(reference, field)


def test_unsupported_element_type_is_rejected(tmp_path):
    text = "*Node\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n*Element, type=S4R\n1,1,2,3,4\n"
    with pytest.raises(ValueError, match=r"unsupported Abaqus element type 'S4R' on line 6"):
        _read(tmp_path, text, "s4r.inp")


def test_missing_required_option_is_rejected(tmp_path):
    text = "*Node\n1,0,0\n2,1,0\n*Element, type=T2D2, elset=BARS\n1,1,2\n*Material, name=M\n*Elastic\n1.,0.3\n*Truss Section, material=M\n3.0\n"
    with pytest.raises(ValueError, match=r"\*TRUSS SECTION on line 9 requires ELSET="):
        _read(tmp_path, text, "noset.inp")


def test_elastic_before_material_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"\*ELASTIC on line 1 must follow \*MATERIAL"):
        _read(tmp_path, "*Elastic\n1.,0.3\n", "lonely.inp")


def test_keyword_sets_are_disjoint_and_cover_the_documented_deck(tmp_path):
    assert not (SUPPORTED_KEYWORDS & NON_MODEL_KEYWORDS)
    used = {line.split(",")[0].lstrip("*").strip().upper()
            for line in DECK.splitlines() if line.startswith("*") and not line.startswith("**")}
    assert used <= (SUPPORTED_KEYWORDS | NON_MODEL_KEYWORDS)
    assert "INCLUDE" not in (SUPPORTED_KEYWORDS | NON_MODEL_KEYWORDS)