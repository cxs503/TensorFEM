"""Provenance layer for calibration factors applied to ice-structure design loads.

Why this module exists
----------------------
The live ice-structure chain reports its answer as ``total = raw * cf`` where
``cf`` is a **constant per ship** (one ship: ``cf = 0.753028``) while the fleet
value of ``cf`` scatters over ``0.670077 .. 0.901199`` (34.5 % spread) and the
response carries no provenance.  Under those conditions *any* accuracy statement
about a calibrated total carries an unauditable bias: the reader cannot tell
whether ``cf`` is a measured ratio, a code ratio, a fitted residual, or a
number someone typed in.

That is not a numerical problem, it is a **bookkeeping** problem, and it is
solved the same way the rest of this repository solves evidence problems: make
the missing evidence a first-class, machine-checkable object and let a
fail-closed gate refuse the claim while it is absent.

What this module provides
-------------------------
1. :class:`ProvenanceRecord` -- the evidence bundle of a single number: which
   kind of source (code rule / experiment / regression dataset / judgement),
   the citation (document + clause, or dataset + estimator), the SHA-256 of the
   document or dataset, the estimator, the sample size, a declared relative
   uncertainty with its basis, and the validity envelope of the number.
   :meth:`ProvenanceRecord.audit` returns an :class:`AuditVerdict` under
   **fail-closed** rules: an incomplete record is *not* auditable, an unknown
   source kind is *not* auditable, and a ``judgement`` value is never auditable
   for a quantitative claim.
2. :func:`combine_provenance` -- the **weakest-link rule** for derived numbers:
   a ratio of two quantities can never be more auditable than its weaker
   operand.  This is the rule that stops "code ratio" claims from being built on
   one transcribed standard and one typed-in coefficient.
3. :class:`CalibrationFactor` -- ``cf`` with its identity checked: if both the
   raw and the reference value are supplied, ``cf`` must equal their ratio to
   within a tolerance, otherwise construction fails.  A calibration factor that
   does not reproduce its own definition is a defect, not a parameter.
4. :class:`CalibratedTotal` / :func:`calibrated_total` -- the response object
   that *carries* the provenance, exposes the unaudited bias bound of the
   scaled total, and can be asked to refuse to exist at all
   (:class:`UnauditableCalibration`) when the caller demands an auditable
   number.
5. :func:`fleet_dispersion` / :func:`constant_penalty` -- the audit of a single
   constant applied across a fleet: the spread of the underlying values and the
   worst-case relative error that a constant inflicts on an individual member,
   compared against a declared tolerance.

Nothing here contains a normative number: every value enters as an argument, and
every audit verdict is derived from the arguments.  That is deliberate -- the
module's job is to make it impossible to *silently* use a number whose origin is
unknown, not to supply the number.

References for the bookkeeping concepts
---------------------------------------
* JCGM 100:2008 (GUM), Sec. 5 -- combination of standard uncertainties of
  derived quantities (used for the quadrature combination in
  :func:`combine_provenance`).
* ASME V&V 20-2009, Sec. 2--3 -- validation metrics require the uncertainty of
  both the simulation and the reference to be stated; a comparison without them
  is a verification, not a validation.
* IACS UR I2 Rev.5, Pt.1 Ch.1 -- rule values are traceable to a clause of a
  controlled document, which is why :attr:`SourceKind.CODE_RULE` demands a
  clause locator plus the SHA-256 of the document.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

__all__ = [
    "SourceKind",
    "MIN_REGRESSION_SAMPLES",
    "AUDITABLE",
    "PARTIAL",
    "UNAUDITABLE",
    "ProvenanceRecord",
    "AuditVerdict",
    "combine_provenance",
    "CalibrationFactor",
    "CalibratedTotal",
    "UnauditableCalibration",
    "calibrated_total",
    "fleet_dispersion",
    "constant_penalty",
    "best_single_constant",
]

#: verdict levels, worst-last ordering is *not* used; they are compared through
#: :data:`_VERDICT_RANK` so that "worse" is unambiguous.
AUDITABLE = "AUDITABLE"
PARTIAL = "PARTIAL"
UNAUDITABLE = "UNAUDITABLE"

_VERDICT_RANK = {AUDITABLE: 2, PARTIAL: 1, UNAUDITABLE: 0}

#: a regression-derived factor needs at least this many samples to be auditable
MIN_REGRESSION_SAMPLES = 5


class SourceKind:
    """Allowed provenances of a number (a plain namespace, not an Enum, so that
    the strings survive JSON round-trips unchanged)."""

    CODE_RULE = "code_rule"
    EXPERIMENT = "experiment"
    REGRESSION_DATASET = "regression_dataset"
    JUDGEMENT = "judgement"
    ALL = (CODE_RULE, EXPERIMENT, REGRESSION_DATASET, JUDGEMENT)


def _is_sha256_hex(value: str) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in value)


def _finite(name: str, value: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class AuditVerdict:
    """Outcome of :meth:`ProvenanceRecord.audit`."""

    verdict: str
    reasons: tuple = ()
    missing: tuple = ()

    @property
    def auditable(self) -> bool:
        return self.verdict == AUDITABLE

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "auditable": self.auditable,
            "reasons": list(self.reasons),
            "missing": list(self.missing),
        }


def _worst(a: "AuditVerdict", b: "AuditVerdict") -> str:
    return a.verdict if _VERDICT_RANK[a.verdict] <= _VERDICT_RANK[b.verdict] else b.verdict


@dataclass(frozen=True)
class ProvenanceRecord:
    """Evidence bundle for one number.

    Parameters
    ----------
    quantity:
        Human-readable name of the number this record describes.
    source_kind:
        One of :data:`SourceKind.ALL`.  Unknown kinds are rejected by
        :meth:`audit` (fail-closed) rather than silently accepted.
    citation:
        Free-text locator that a reader can follow: a standard and clause, a
        test report id, a dataset name and version.  Mandatory.
    document_sha256:
        SHA-256 of the controlled document the number was transcribed from
        (required for :attr:`SourceKind.CODE_RULE`).
    clause:
        Clause / table / page locator inside that document (required for
        :attr:`SourceKind.CODE_RULE`).
    dataset_sha256:
        SHA-256 of the dataset the number was fitted to (required for
        :attr:`SourceKind.REGRESSION_DATASET`).
    estimator:
        The estimation procedure, e.g. ``"log-log OLS of ln p on ln A"``
        (required for :attr:`SourceKind.REGRESSION_DATASET`).
    sample_size:
        Number of independent samples behind the number (regression only).
    relative_uncertainty:
        Declared relative standard uncertainty of the number (dimensionless,
        e.g. ``0.05`` for 5 %).  ``None`` means "not stated".
    uncertainty_basis:
        Where the uncertainty comes from ("replicate scatter, n=6").
    validity:
        Mapping of the envelope in which the number may be used, e.g.
        ``{"area_m2": [1.0, 10.0], "thickness_m": [0.5, 3.0]}``.
    note:
        Anything a human auditor must see (declared substitutions, known gaps).
    """

    quantity: str
    source_kind: str
    citation: str = ""
    document_sha256: str = ""
    clause: str = ""
    dataset_sha256: str = ""
    estimator: str = ""
    sample_size: int = 0
    relative_uncertainty: float | None = None
    uncertainty_basis: str = ""
    validity: Mapping = field(default_factory=dict)
    note: str = ""

    # -- construction ---------------------------------------------------- #
    def __post_init__(self) -> None:
        if not str(self.quantity).strip():
            raise ValueError("quantity must be a non-empty label")
        if not isinstance(self.source_kind, str) or not self.source_kind.strip():
            raise ValueError("source_kind must be a non-empty string")
        if self.relative_uncertainty is not None:
            u = _finite("relative_uncertainty", self.relative_uncertainty)
            if u < 0.0:
                raise ValueError("relative_uncertainty must be >= 0")
            object.__setattr__(self, "relative_uncertainty", u)
        n = int(self.sample_size)
        if n < 0:
            raise ValueError("sample_size must be >= 0")
        object.__setattr__(self, "sample_size", n)
        if not isinstance(self.validity, Mapping):
            raise ValueError("validity must be a mapping")
        object.__setattr__(self, "validity", {str(k): v for k, v in self.validity.items()})

    # -- integrity ------------------------------------------------------- #
    def canonical(self) -> dict:
        """Deterministic, timestamp-free representation (hash input)."""

        def _norm(value):
            if isinstance(value, Mapping):
                return {str(k): _norm(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
            if isinstance(value, (list, tuple)):
                return [_norm(v) for v in value]
            return value

        return {
            "quantity": str(self.quantity),
            "source_kind": str(self.source_kind),
            "citation": str(self.citation),
            "document_sha256": str(self.document_sha256).lower(),
            "clause": str(self.clause),
            "dataset_sha256": str(self.dataset_sha256).lower(),
            "estimator": str(self.estimator),
            "sample_size": int(self.sample_size),
            "relative_uncertainty": self.relative_uncertainty,
            "uncertainty_basis": str(self.uncertainty_basis),
            "validity": _norm(dict(self.validity)),
            "note": str(self.note),
        }

    def sha256(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- audit ----------------------------------------------------------- #
    def audit(self) -> AuditVerdict:
        """Fail-closed auditability verdict.

        A record that lacks any prerequisite is reported as ``UNAUDITABLE``
        with the missing fields listed; nothing is ever inferred.
        """
        missing: list[str] = []
        reasons: list[str] = []
        kind = str(self.source_kind)

        if not str(self.citation).strip():
            missing.append("citation")

        if kind not in SourceKind.ALL:
            reasons.append(
                f"unknown source_kind {kind!r} (allowed: {', '.join(SourceKind.ALL)})"
            )
            return AuditVerdict(UNAUDITABLE, tuple(reasons), tuple(missing))

        if kind == SourceKind.JUDGEMENT:
            reasons.append(
                "a judgement value carries no external evidence and cannot support a "
                "quantitative claim; replace it with a code rule, an experiment or a "
                "regression dataset before quoting a calibrated total"
            )
            return AuditVerdict(UNAUDITABLE, tuple(reasons), tuple(missing))

        if kind == SourceKind.CODE_RULE:
            if not _is_sha256_hex(self.document_sha256):
                missing.append("document_sha256")
            if not str(self.clause).strip():
                missing.append("clause")

        if kind == SourceKind.EXPERIMENT:
            if self.relative_uncertainty is None:
                missing.append("relative_uncertainty")
            if not str(self.uncertainty_basis).strip():
                missing.append("uncertainty_basis")

        if kind == SourceKind.REGRESSION_DATASET:
            if not _is_sha256_hex(self.dataset_sha256):
                missing.append("dataset_sha256")
            if not str(self.estimator).strip():
                missing.append("estimator")
            if int(self.sample_size) < MIN_REGRESSION_SAMPLES:
                missing.append(f"sample_size>={MIN_REGRESSION_SAMPLES}")
            if self.relative_uncertainty is None:
                missing.append("relative_uncertainty")

        if missing:
            reasons.append(
                f"{kind} provenance is incomplete: missing {', '.join(missing)}"
            )
            return AuditVerdict(UNAUDITABLE, tuple(reasons), tuple(missing))

        if not dict(self.validity):
            reasons.append(
                "no validity envelope declared: the number is quoted outside the "
                "range it was established for"
            )
            return AuditVerdict(PARTIAL, tuple(reasons), tuple(missing))

        return AuditVerdict(AUDITABLE, tuple(reasons), tuple(missing))

    # -- presentation ---------------------------------------------------- #
    def as_dict(self) -> dict:
        out = self.canonical()
        out["record_sha256"] = self.sha256()
        out["audit"] = self.audit().as_dict()
        return out

    def with_uncertainty(self, u: float, basis: str) -> "ProvenanceRecord":
        return replace(
            self,
            relative_uncertainty=_finite("relative_uncertainty", u),
            uncertainty_basis=str(basis),
        )


def combine_provenance(
    left: ProvenanceRecord,
    right: ProvenanceRecord,
    *,
    quantity: str,
) -> ProvenanceRecord:
    """Provenance of a quantity *derived* from ``left`` and ``right``.

    Weakest-link rule
    -----------------
    The derived record is auditable only if **both** operands are auditable; a
    verdict of ``PARTIAL`` or ``UNAUDITABLE`` on either side propagates.  The
    declared uncertainties are combined in quadrature per GUM Sec. 5.  This is
    the rule that prevents a ratio from laundering a typed-in coefficient into a
    "code ratio".

    The returned record keeps the weaker source kind and concatenates both
    citations, so the audit trail of the derivation stays readable.
    """
    if not isinstance(left, ProvenanceRecord) or not isinstance(right, ProvenanceRecord):
        raise TypeError("combine_provenance expects two ProvenanceRecord instances")
    v_left, v_right = left.audit(), right.audit()
    verdict = _worst(v_left, v_right)
    weaker = left if _VERDICT_RANK[v_left.verdict] <= _VERDICT_RANK[v_right.verdict] else right
    stronger = right if weaker is left else left

    uncertainties = [p.relative_uncertainty for p in (left, right) if p.relative_uncertainty is not None]
    combined = math.sqrt(sum(u * u for u in uncertainties)) if uncertainties else None

    validity: dict = {}
    for record in (left, right):
        for key, value in dict(record.validity).items():
            if key in validity and validity[key] != value:
                # an envelope that cannot be intersected is declared, never guessed
                note_key = f"{key}__conflict"
                validity[note_key] = [validity.pop(key), value]
            else:
                validity[key] = value

    reasons = []
    if verdict != AUDITABLE:
        reasons.append(
            "derived quantity inherits the weaker operand's verdict "
            f"({left.quantity}: {v_left.verdict}, {right.quantity}: {v_right.verdict})"
        )
    note_bits = [str(weaker.note), str(stronger.note)]
    note_bits.append(
        "derived: weakest-link provenance of "
        f"{left.quantity} and {right.quantity}"
    )
    return ProvenanceRecord(
        quantity=str(quantity),
        source_kind=weaker.source_kind,
        citation=" | ".join(c for c in (stronger.citation, weaker.citation) if str(c).strip()),
        document_sha256=weaker.document_sha256 or stronger.document_sha256,
        clause=weaker.clause or stronger.clause,
        dataset_sha256=weaker.dataset_sha256 or stronger.dataset_sha256,
        estimator=weaker.estimator or stronger.estimator,
        sample_size=min(left.sample_size, right.sample_size) if min(left.sample_size, right.sample_size) > 0 else 0,
        relative_uncertainty=combined,
        uncertainty_basis=(
            "quadrature combination of declared uncertainties (GUM 5.1.2): "
            + ", ".join(f"{p.quantity}={p.relative_uncertainty}" for p in (left, right)
                        if p.relative_uncertainty is not None)
        ) if uncertainties else "",
        validity=validity,
        note="; ".join(b for b in note_bits if b) + (" | " + " | ".join(reasons) if reasons else ""),
    )


class UnauditableCalibration(Exception):
    """Raised when a calibrated total is demanded without auditable provenance."""

    def __init__(self, message: str, verdict: AuditVerdict):
        super().__init__(message)
        self.verdict = verdict


@dataclass(frozen=True)
class CalibrationFactor:
    """A multiplicative calibration factor together with its provenance.

    ``value`` is checked against its own definition: when both ``raw_value`` and
    ``reference_value`` are supplied, ``value`` must equal
    ``reference_value / raw_value`` to within ``identity_tolerance`` (relative).
    """

    value: float
    raw_quantity: str
    reference_quantity: str
    provenance: ProvenanceRecord
    raw_value: float | None = None
    reference_value: float | None = None
    identity_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        v = _finite("value", self.value)
        if v <= 0.0:
            raise ValueError("a calibration factor must be strictly positive")
        object.__setattr__(self, "value", v)
        if not isinstance(self.provenance, ProvenanceRecord):
            raise TypeError("provenance must be a ProvenanceRecord")
        if self.raw_value is not None and self.reference_value is not None:
            raw = _finite("raw_value", self.raw_value)
            ref = _finite("reference_value", self.reference_value)
            if raw <= 0.0:
                raise ValueError("raw_value must be strictly positive")
            implied = ref / raw
            tol = _finite("identity_tolerance", self.identity_tolerance)
            if tol < 0.0:
                raise ValueError("identity_tolerance must be >= 0")
            if abs(v - implied) > tol * max(abs(implied), 1e-30):
                raise ValueError(
                    f"calibration factor {v!r} does not reproduce its own definition "
                    f"{self.reference_quantity}/{self.raw_quantity} = {implied!r} "
                    f"(relative deviation {abs(v - implied) / abs(implied):.3e} > {tol:g})"
                )

    @property
    def audit(self) -> AuditVerdict:
        return self.provenance.audit()

    @property
    def relative_uncertainty(self) -> float:
        u = self.provenance.relative_uncertainty
        return 0.0 if u is None else float(u)

    def apply(self, raw_total: float, *, require_auditable: bool = False) -> "CalibratedTotal":
        return calibrated_total(
            raw_total, self, require_auditable=require_auditable
        )

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "raw_quantity": self.raw_quantity,
            "reference_quantity": self.reference_quantity,
            "raw_value": self.raw_value,
            "reference_value": self.reference_value,
            "provenance": self.provenance.as_dict(),
        }


@dataclass(frozen=True)
class CalibratedTotal:
    """``raw * cf`` **with** the provenance and the unaudited bias of ``cf``."""

    value: float
    raw_total: float
    calibration: CalibrationFactor
    verdict: AuditVerdict
    unaudited_bias_rel: float

    @property
    def auditable(self) -> bool:
        return self.verdict.auditable

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "raw_total": self.raw_total,
            "calibration_factor": self.calibration.value,
            "auditable": self.auditable,
            "verdict": self.verdict.as_dict(),
            "unaudited_bias_rel": self.unaudited_bias_rel,
            "provenance_sha256": self.calibration.provenance.sha256(),
        }


def calibrated_total(
    raw_total: float,
    calibration: CalibrationFactor,
    *,
    require_auditable: bool = False,
) -> CalibratedTotal:
    """Scale ``raw_total`` by ``calibration`` and carry the provenance along.

    The returned object always states the relative bias that the calibration
    factor may carry without evidence (``unaudited_bias_rel``).  With
    ``require_auditable=True`` the call raises :class:`UnauditableCalibration`
    instead of returning a number that cannot be defended.
    """
    if not isinstance(calibration, CalibrationFactor):
        raise TypeError("calibration must be a CalibrationFactor")
    raw = _finite("raw_total", raw_total)
    verdict = calibration.audit
    if require_auditable and not verdict.auditable:
        raise UnauditableCalibration(
            f"refusing to calibrate a total with a {verdict.verdict} calibration "
            f"factor: {'; '.join(verdict.reasons) or 'no reasons recorded'}",
            verdict,
        )
    return CalibratedTotal(
        value=raw * calibration.value,
        raw_total=raw,
        calibration=calibration,
        verdict=verdict,
        unaudited_bias_rel=calibration.relative_uncertainty,
    )


# --------------------------------------------------------------------------- #
# fleet-level audit of a single constant
# --------------------------------------------------------------------------- #
def _as_values(values) -> list:
    if isinstance(values, Mapping):
        return [_finite(f"values[{k}]", v) for k, v in values.items()]
    return [_finite("values[i]", v) for v in values]


def fleet_dispersion(values) -> dict:
    """Spread of the per-member calibration factors.

    ``dispersion_rel`` is ``(max - min) / min`` -- the definition under which the
    reported fleet spread of 34.5 % is reproduced by 0.670077 .. 0.901199.
    """
    data = _as_values(values)
    if not data:
        raise ValueError("fleet_dispersion requires at least one value")
    if any(v <= 0.0 for v in data):
        raise ValueError("calibration factors must be strictly positive")
    lo, hi = min(data), max(data)
    mean = sum(data) / len(data)
    var = sum((v - mean) ** 2 for v in data) / len(data)
    return {
        "n": len(data),
        "min": lo,
        "max": hi,
        "mean": mean,
        "sd": math.sqrt(var),
        "cov_rel": (math.sqrt(var) / mean) if mean else float("nan"),
        "dispersion_rel": (hi - lo) / lo,
    }


def constant_penalty(values, constant: float, *, tolerance: float = 0.15) -> dict:
    """Worst-case relative error that one constant inflicts on a member.

    The live chain reports ``total = raw * cf`` with a *single* ``cf`` per ship.
    If the underlying (per-member) factor scatters, every member's calibrated
    total is wrong by ``cf_used / cf_member - 1``.  This function reports the
    two-sided worst case and compares ``max |error|`` against ``tolerance`` --
    the platform's 15 % acceptance standard.
    """
    data = _as_values(values)
    c = _finite("constant", constant)
    if c <= 0.0:
        raise ValueError("constant must be strictly positive")
    tol = _finite("tolerance", tolerance)
    if tol <= 0.0:
        raise ValueError("tolerance must be strictly positive")
    errors = [c / v - 1.0 for v in data]
    worst = max(errors, key=abs)
    return {
        "constant": c,
        "n": len(data),
        "error_rel_min": min(errors),
        "error_rel_max": max(errors),
        "worst_abs_error_rel": abs(worst),
        "worst_member_factor": data[errors.index(worst)],
        "tolerance": tol,
        "verdict": "PASS" if abs(worst) <= tol else "FAIL",
    }


def best_single_constant(values) -> dict:
    """Best single constant for a set of per-member factors (minimax).

    Returns the centre of the range and the minimax relative error achieved by
    the best geometric mid-point, which is the theoretical floor of any
    single-constant representation (the relative error ``c/v - 1`` is monotone on
    each side of ``c``, so the minimax point is the harmonic midpoint of the
    range).  Used to answer "can a constant ever meet the tolerance for this
    spread?" without fitting anything to data.
    """
    data = _as_values(values)
    if not data:
        raise ValueError("best_single_constant requires at least one value")
    lo, hi = min(data), max(data)
    if hi == lo:
        return {"constant": lo, "minimax_abs_error_rel": 0.0, "uniform": True}
    # relative error cf/v - 1 is monotone decreasing in v, |err| minimised where
    # cf/v - 1 = -(cf/lo... ) -> solve c/hi - 1 = 1 - c/lo  =>  2c = hi*lo*(1/hi+1/lo)
    c = 2.0 / (1.0 / lo + 1.0 / hi)
    err = max(abs(c / lo - 1.0), abs(c / hi - 1.0))
    return {
        "constant": c,
        "minimax_abs_error_rel": err,
        "uniform": False,
        "at_min": c / lo - 1.0,
        "at_max": c / hi - 1.0,
    }