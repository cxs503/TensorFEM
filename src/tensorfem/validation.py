"""Fail-closed benchmark evidence for TensorFEM capabilities."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Reference:
    """Traceable reference value for one benchmark quantity."""

    benchmark: str
    quantity: str
    value: float
    unit: str
    source: str
    tolerance: float = 0.03

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("benchmark reference must be finite")
        if not self.source.strip():
            raise ValueError("benchmark reference source is required")
        if not 0 < self.tolerance <= 0.03:
            raise ValueError("formal benchmark tolerance must be in (0, 3%]")


@dataclass(frozen=True)
class ValidationResult:
    reference: Reference
    computed: float
    relative_error: float
    passed: bool

    def to_dict(self) -> dict:
        value = asdict(self)
        value["reference"] = asdict(self.reference)
        return value


def validate(reference: Reference, computed: float) -> ValidationResult:
    """Compare a computed scalar to a nonzero standard reference value."""
    if reference.value == 0:
        raise ValueError("relative benchmark reference must be nonzero")
    if not math.isfinite(float(computed)):
        raise ValueError("computed benchmark value must be finite")
    error = abs(float(computed) - reference.value) / abs(reference.value)
    return ValidationResult(reference, float(computed), error, error < reference.tolerance)


def require_all(results: Iterable[ValidationResult]) -> tuple[ValidationResult, ...]:
    """Fail when any advertised benchmark does not meet its declared tolerance."""
    materialized = tuple(results)
    if not materialized:
        raise AssertionError("benchmark accuracy gate requires nonempty evidence")
    failed = [item for item in materialized if not item.passed or
              item != validate(item.reference, item.computed)]
    if failed:
        summary = ", ".join(
            f"{item.reference.benchmark}/{item.reference.quantity}={item.relative_error:.3%}"
            for item in failed
        )
        raise AssertionError(f"benchmark accuracy gate failed: {summary}")
    return materialized


def write_evidence(results: Iterable[ValidationResult], path: str | Path) -> None:
    checked = require_all(results)
    payload = {"policy": "relative_error < declared tolerance <= 3%",
               "results": [item.to_dict() for item in checked]}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
