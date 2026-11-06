"""Executable guard for ``docs/tutorials/typical-static-workflow.md``.

The tutorial is a *runnable* document, not a screenshot of one. This module

* executes every ``python`` block of the tutorial in one shared namespace, so the
  assertions the document makes are real assertions about the current code;
* compares every number in the document's machine-checked fact table with the
  number the code computes (relative tolerance ``RELATIVE_TOLERANCE``), so a
  documented result cannot drift away from the solver output;
* proves that the guard has teeth: a mutated document number, a silently relaxed
  acceptance tolerance and a removed README link each have to turn it red.

The mutation cases are permanent negative controls, not a one-off demonstration.
"""

import re
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_RELPATH = "docs/tutorials/typical-static-workflow.md"
TUTORIAL = ROOT / TUTORIAL_RELPATH
README = ROOT / "README.md"

FACTS_BEGIN = "<!-- tutorial-facts:begin -->"
FACTS_END = "<!-- tutorial-facts:end -->"
RELATIVE_TOLERANCE = 1e-6
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
FACT_ROW = re.compile(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|\s*([-+0-9.eE]+)\s*\|\s*$", re.M)


def python_blocks(text):
    """Return the tutorial's ``python`` code blocks in document order."""
    return BLOCK.findall(text)


def documented_facts(text):
    """Parse the fact table between the check markers."""
    start = text.find(FACTS_BEGIN)
    end = text.find(FACTS_END)
    if start < 0 or end < 0 or end < start:
        raise AssertionError("the tutorial is missing its machine-checked fact table")
    facts = {key: float(value) for key, value in FACT_ROW.findall(text[start:end])}
    if not facts:
        raise AssertionError("the tutorial's fact table is empty")
    return facts


def execute_tutorial(text):
    """Execute the tutorial's python blocks in document order; return the namespace."""
    blocks = python_blocks(text)
    if len(blocks) < 5:
        raise AssertionError(f"expected at least 5 python blocks, found {len(blocks)}")
    namespace = {"__name__": "__tutorial__"}
    for index, block in enumerate(blocks, start=1):
        exec(compile(block, f"<tutorial block {index}>", "exec"), namespace)
    if not isinstance(namespace.get("FACTS"), dict):
        raise AssertionError("the tutorial did not produce a FACTS dict")
    return namespace


def compare_documented(documented, computed):
    """Fail on a missing key or on a documented value that drifted."""
    missing = sorted(set(documented) - set(computed))
    if missing:
        raise AssertionError(f"fact table keys with no computed value: {missing}")
    drifted = []
    for key, value in documented.items():
        expected = float(computed[key])
        if abs(value - expected) > RELATIVE_TOLERANCE * abs(expected) + 1e-12:
            drifted.append(f"{key}: document={value!r} code={expected!r}")
    if drifted:
        raise AssertionError("documented numbers drifted from the code:\n" + "\n".join(drifted))


def with_fact(text, key, value):
    """Return ``text`` with one documented fact replaced by ``value``."""
    pattern = re.compile(r"(^\|\s*`" + re.escape(key) + r"`\s*\|\s*)([-+0-9.eE]+)", re.M)
    mutated, count = pattern.subn(lambda match: match.group(1) + repr(value), text)
    assert count == 1, f"expected exactly one row for {key}, patched {count}"
    return mutated


def readme_problems(text):
    """Problems with the README's link to the tutorial (empty list means healthy)."""
    return [] if TUTORIAL_RELPATH in text else [f"README does not link {TUTORIAL_RELPATH}"]


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    """Pin torch to one thread while this module runs.

    The tutorial case is milliseconds of work; on a shared many-core host the
    BLAS pool costs seconds of scheduling per solve and the timings stop being
    about the solver.  The previous setting is restored afterwards.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tutorial_text():
    assert TUTORIAL.is_file(), f"missing tutorial {TUTORIAL_RELPATH}"
    return TUTORIAL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tutorial_run(tutorial_text):
    return execute_tutorial(tutorial_text)


def test_readme_links_the_tutorial():
    problems = readme_problems(README.read_text(encoding="utf-8"))
    assert not problems, problems
    # The link gate is only useful if it fires on a README without the link.
    assert readme_problems("# TensorFEM\n\nNo links here.\n"), "README link gate is vacuous"


def test_tutorial_blocks_execute(tutorial_text, tutorial_run):
    assert len(python_blocks(tutorial_text)) >= 5
    assert {"CASES", "MODAL", "FACTS", "LIMITS"} <= set(tutorial_run)
    facts = tutorial_run["FACTS"]
    assert facts["mesh_40x8_nodes"] == 369.0
    assert facts["mesh_40x8_elements"] == 320.0


def test_documented_numbers_match_computed_output(tutorial_text, tutorial_run):
    documented = documented_facts(tutorial_text)
    assert len(documented) >= 15, f"fact table too small: {sorted(documented)}"
    compare_documented(documented, tutorial_run["FACTS"])


def test_documented_table_cannot_drift(tutorial_text, tutorial_run):
    """A 5 % error in one documented number must turn the guard red."""
    documented = documented_facts(tutorial_text)
    key = "tip_40x8_mm"
    mutated = with_fact(tutorial_text, key, documented[key] * 1.05)
    with pytest.raises(AssertionError, match=key):
        compare_documented(documented_facts(mutated), tutorial_run["FACTS"])


def test_acceptance_gate_cannot_be_silently_relaxed(tutorial_text):
    """Relaxing the declared 3 % tolerance makes the tutorial reject its own case."""
    mutated = tutorial_text.replace("3e-2", "1e-9")
    assert mutated != tutorial_text, "tutorial no longer declares its acceptance tolerances"
    with pytest.raises(AssertionError, match="finest mesh must meet the declared gate"):
        execute_tutorial(mutated)


def test_missing_fact_table_is_rejected(tutorial_text):
    truncated = tutorial_text.replace(FACTS_BEGIN, "<!-- no facts -->")
    with pytest.raises(AssertionError, match="machine-checked fact table"):
        documented_facts(truncated)