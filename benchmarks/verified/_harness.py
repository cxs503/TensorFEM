"""Shared plumbing for the ``benchmarks/verified`` anchor drivers.

The three IN36-N01 anchors (cantilever bending, Hertz contact, plate modal) are
each a *self-contained* verification: an independent closed-form oracle, a
refinement ladder, graded gates and machine-readable evidence.  Only the
book-keeping is shared here -- JSON writing, Richardson/GCI arithmetic, the
gate manifest schema, and the matplotlib boilerplate.  No physics lives in this
file, so a reader can audit each anchor driver without reading the others.

Evidence schema
---------------
``gates_verified.json`` is the machine contract of an anchor::

    {"schema": "tensorfem.verified-gates.v1",
     "anchor": "fem_cantilever",
     "criteria": {...},          # thresholds exactly as the campaign asked
     "gates": [{"id": ..., "quantity": ..., "reference": ..., "value": ...,
                "threshold": ..., "operator": "<=", "passed": bool}, ...],
     "passed": bool,
     "limitations": [...]}

A gate never re-states the model's own output as its reference: ``reference``
names the closed-form oracle (formula string) the value is graded against.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

GATE_SCHEMA = "tensorfem.verified-gates.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> Path:
    """Write ``payload`` with a stable key order and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Gate:
    """One graded acceptance criterion."""

    id: str
    quantity: str
    reference: str
    value: float
    threshold: float
    operator: str = "<="  # "<=" or ">="
    note: str = ""

    @property
    def passed(self) -> bool:
        if self.operator == "<=":
            return bool(self.value <= self.threshold)
        if self.operator == ">=":
            return bool(self.value >= self.threshold)
        raise ValueError(f"unsupported operator {self.operator!r}")

    def to_dict(self) -> dict:
        payload = {
            "id": self.id,
            "quantity": self.quantity,
            "reference": self.reference,
            "value": self.value,
            "threshold": self.threshold,
            "operator": self.operator,
            "passed": self.passed,
        }
        if self.note:
            payload["note"] = self.note
        return payload


def gate_block(gates: list[Gate]) -> dict:
    """Serialise gates plus the single overall verdict."""
    return {
        "gates": [g.to_dict() for g in gates],
        "passed": all(g.passed for g in gates),
        "failed_gates": [g.id for g in gates if not g.passed],
    }


def gate_manifest(
    *,
    anchor: str,
    title: str,
    criteria: dict,
    gates: list[Gate],
    artifacts: dict,
    limitations: list[str],
    mission: str = "IN36-N01",
) -> dict:
    """Assemble the ``gates_verified.json`` document.

    ``criteria`` must carry the numeric thresholds as declared by the campaign
    so a reader can check the report against the mandate without trusting this
    code.
    """
    block = gate_block(gates)
    return {
        "schema": GATE_SCHEMA,
        "anchor": anchor,
        "title": title,
        "mission": mission,
        "generated_at": utc_now(),
        "criteria": criteria,
        "gates": block["gates"],
        "passed": block["passed"],
        "failed_gates": block["failed_gates"],
        "undefined_terms": [g.id for g in gates if not math.isfinite(g.value)],
        "artifacts": artifacts,
        "limitations": limitations,
    }


def richardson(
    coarse: float,
    medium: float,
    fine: float,
    refinement_ratio: float = 2.0,
) -> dict:
    """Observed order and GCI from three successive approximations.

    ``p = ln((w_3-w_2)/(w_2-w_1)) / ln(r)`` (Richardson extrapolation) and
    ``GCI_fine = Fs |(w_3-w_2)/w_3| / (r^p - 1)`` with the standard safety
    factor ``Fs = 1.25`` (Roache 1994).  The routine is purely arithmetic: if
    the sequence is not monotone it returns ``p = nan`` and the caller must
    report that honestly instead of inventing an order.
    """
    d21 = medium - coarse
    d32 = fine - medium
    if d21 == 0.0 or d32 == 0.0 or d21 / d32 <= 0.0:
        order = float("nan")
        gci = float("nan")
    else:
        order = math.log(abs(d21 / d32)) / math.log(refinement_ratio)
        if not math.isfinite(order) or order <= 0.0:
            gci = float("nan")
        else:
            gci = 1.25 * abs(d32 / fine) / (refinement_ratio**order - 1.0)
    extrapolated = (
        fine + (fine - medium) / (refinement_ratio**order - 1.0)
        if math.isfinite(order) and order > 0.0
        else float("nan")
    )
    return {
        "observed_order": order,
        "gci_fine": gci,
        "richardson_limit": extrapolated,
        "safety_factor": 1.25,
        "refinement_ratio": refinement_ratio,
    }


def pyplot():
    """Return a headless ``matplotlib.pyplot`` module (Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
        }
    )
    return plt


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
