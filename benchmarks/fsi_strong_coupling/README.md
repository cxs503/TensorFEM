# FSI strong-coupling validation drivers

* `run_validation.py` — the four cases (cylinder 2D-2 anchor, structure,
  interface conservation, coupled FSI-2). Writes numbers-only JSON into
  `artifacts/fsi_strong_coupling/`.
* `run_chunk.py` — checkpointed FSI-2 runner for time-boxed shell calls.
* `make_report.py` — matplotlib figures from the JSON artefacts.
* `make_acceptance.py` — assembles the acceptance JSON and the Markdown report
  in `docs/reports/`.

The drivers are deterministic (fixed initial perturbation, no RNG in the
physics) so the artefacts can be regenerated.
