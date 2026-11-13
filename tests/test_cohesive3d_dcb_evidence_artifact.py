"""Gates on the specimen-level DCB evidence (``benchmarks/cohesive3d_dcb_3d.json``).

Background
----------
The specimen-level evidence used to be a free-floating claim: nothing tied its
numbers to the code that produced them, so it could (and did) go stale -- the
committed JSON was generated in quick mode *before* the displacement-control and
external-work fixes landed, and still reported ``all_passed: False`` with large
errors the current code no longer exhibits.  The element-level artifact
(``cohesive3d_verification.json``) already carries a provenance + digest gate;
this module brings the specimen-level artifact to the same standard.

What is gated (fast, no coupled-specimen solve)
------------------------------------------------
* **machinery** -- the benchmark's ``source_provenance`` / ``criteria_digest``
  helpers bind the *current* checkout: the recorded SHA-256 of every source
  module must match the files on disk, and the digest must be a pure function of
  the criteria (any hand edit changes it);
* **schema** -- every criterion is present, each carries its closed-form
  statement, and ``all_passed`` really is the conjunction of the ``passed``
  flags (no silent partial pass);
* **tamper gate** -- when the artifact carries ``checks_digest`` it must hash the
  stored criteria;
* **freshness** -- when the artifact carries ``provenance`` its source digests
  must still match the checkout (the artifact cannot outlive the code);
* **no host leakage** -- the artifact is repo-relative (no ``/root/...`` paths).

The expensive coupled-specimen half is covered by re-running
``python benchmarks/cohesive3d_dcb_3d.py`` (full mode); this module deliberately
stays under a few seconds so it can run in the default suite.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "cohesive3d_dcb_3d.py"
ARTIFACT = REPO / "benchmarks" / "cohesive3d_dcb_3d.json"

HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_CRITERIA = (
    "compliance_bracket",
    "peak_load_vs_lefm",
    "energy_balance",
    "interface_work_vs_GIc",
    "peak_ratio_two_cracks",
    "width_objectivity",
)

SOURCE_FILES = (
    "src/tensorfem/cohesive3d.py",
    "src/tensorfem/cohesive3d_dcb.py",
    "src/tensorfem/solid3d.py",
)


def _artifact() -> dict:
    assert ARTIFACT.exists(), f"missing artifact: run {SCRIPT.name} first"
    return json.loads(ARTIFACT.read_text())


def _bench_module():
    """Import the benchmark script (it pins the torch thread pool at import).

    The thread pool is restored at teardown so this gate cannot change the
    numerics or throughput of any other test module in the same session.
    """
    prior_threads = torch.get_num_threads()
    spec = importlib.util.spec_from_file_location("cohesive3d_dcb_3d", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._test_prior_threads = prior_threads
    return module


def test_provenance_machinery_binds_the_checkout():
    m = _bench_module()
    try:
        prov = m.source_provenance()
        assert prov["module"] == "tensorfem.cohesive3d_dcb"
        assert set(prov["source_files"]) == set(SOURCE_FILES)
        for path, digest in prov["source_files"].items():
            assert HEX64.match(digest), path
            current = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            assert digest == current, f"provenance does not bind {path}"
        assert prov["git_head"] is None or re.match(r"^[0-9a-f]{40}$", prov["git_head"])
    finally:
        torch.set_num_threads(m._test_prior_threads)


def test_criteria_digest_is_a_pure_function_of_the_criteria():
    m = _bench_module()
    try:
        a = {"x": {"v": 1.0, "passed": True}}
        b = {"x": {"v": 1.0, "passed": True}}
        c = {"x": {"v": 1.0 + 1e-9, "passed": True}}
        assert m.criteria_digest(a) == m.criteria_digest(b)
        assert m.criteria_digest(a) != m.criteria_digest(c)
        assert HEX64.match(m.criteria_digest(a))
    finally:
        torch.set_num_threads(m._test_prior_threads)


def test_every_criterion_is_present_and_stated():
    art = _artifact()
    criteria = art["criteria"]
    assert tuple(criteria) == EXPECTED_CRITERIA
    assert art["schema"] == "tensorfem.cohesive3d-dcb-3d/1.0"
    assert art["generator"] == "benchmarks/cohesive3d_dcb_3d.py"
    for name, check in criteria.items():
        assert isinstance(check.get("criterion"), str) and check["criterion"], name
        assert isinstance(check.get("passed"), bool), name


def test_all_passed_is_the_conjunction_of_the_checks():
    art = _artifact()
    flags = {name: check["passed"] for name, check in art["criteria"].items()}
    assert art["all_passed"] is all(flags.values())


def test_artifact_provenance_is_fresh_when_present():
    art = _artifact()
    prov = art.get("provenance")
    if prov is None:
        return  # pre-provenance artifact: the machinery gate above still applies
    files = prov["source_files"]
    assert set(files) == set(SOURCE_FILES)
    for path, digest in files.items():
        assert HEX64.match(digest), path
        current = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        assert digest == current, (
            f"stale artifact: {path} changed since {art['generated_utc']}; "
            f"re-run benchmarks/cohesive3d_dcb_3d.py"
        )


def test_artifact_checks_digest_when_present():
    art = _artifact()
    if "checks_digest" not in art:
        return  # pre-digest artifact
    canonical = json.dumps(art["criteria"], sort_keys=True, separators=(",", ":"))
    assert art["checks_digest"] == hashlib.sha256(canonical.encode()).hexdigest()


def test_artifact_does_not_leak_host_paths():
    text = json.dumps(_artifact())
    for marker in ("/root/", "/home/", "\\\\"):
        assert marker not in text, f"artifact leaks a host path via {marker!r}"
