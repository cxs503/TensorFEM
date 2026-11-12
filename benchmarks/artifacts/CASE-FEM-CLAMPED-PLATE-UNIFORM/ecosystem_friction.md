# Ecosystem friction observed while building CASE-FEM-CLAMPED-PLATE-UNIFORM

Recorded by the case worker (mission 16181), 2026-11-12. Nothing here was worked
around by forking the capability -- it is reported so the platform/ecosystem
owners can fix it at the source.

1. **The installed `tensorfem` is not the worktree you are standing in.**
   A bare `python` in `work/m16181` imported `tensorfem` from
   `/root/tp-agent/work/m14687/src/tensorfem/...` (another agent's worktree) and
   silently ran the wrong code. Every command in this case therefore uses
   `PYTHONPATH=src`. Any case-pool runner that forgets this will publish numbers
   produced by someone else's checkout. Suggested fix: make the package refuse to
   import (or warn) when `__file__` is outside the current repository, or have the
   harness pin `PYTHONPATH=src` by default.

2. **The 开工三查 tables are not readable by the agent role.**
   `knowledge_base`, `negative_knowledge_ledger` and `oss_solutions` all return
   `permission denied for table ...` for the worker DB role (`TP_PG_DB`). The
   three-check protocol cannot be executed from SQL as specified.

3. **The knowledge search engine is offline.**
   `GET /tensor/knowledge/search` returns
   `{"available": false, "reason": "engine_offline"}` (backend
   `http://127.0.0.1:8021` refuses connections); the fallback route
   `GET /knowledge/search` answers `total: 0` for "plate". So the only usable
   prior art was in-repo (`benchmarks/plate_uniform_load_evidence.json`,
   `scripts/verify_plate_uniform_load.py`, `tests/test_plate_uniform_load.py`).

4. **`CLAMPED_BEAM_EIGENVALUES` is capped at 12 roots.**
   `clamped_uniform_ritz(..., n_modes=13)` raises a bare `IndexError: tuple index
   out of range` from `_clamped_mode`. The reference is therefore hard-capped at
   N = 12 with no explicit error, and a caller asking for a finer reference gets a
   crash instead of "N <= 12". The measured truncation band (0.22 % deflection,
   1.71 % centre moment) is reported in the case artifact so the cap is visible.

5. **`nx = 17` has no centre node.** `centre_node_deflection` raises
   `ValueError: mesh must contain exactly one node at the plate centre`, which is
   correct but easy to trip: odd `nx` (used elsewhere in the repo for
   element-centre stress recovery) is not usable for centre values. Worth a note
   in the docstring of `solve_uniform_square_plate`.

6. **Cluster admission, not compute, is the bottleneck for small cases.**
   Two submissions sat `queued` for 7-9 min with ~430 jobs ahead, while the whole
   case (9 meshes up to 32x32 plus two sensitivity sweeps, 94 s single core)
   finishes locally well inside the shell limit. Scheduling a 15 s job the same
   way as a 2 h job starves the case pool; a small-job lane would help.