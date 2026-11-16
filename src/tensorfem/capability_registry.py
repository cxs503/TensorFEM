"""TensorFEM capability registry — self-registering, dumpable, triad-classified.

Why this module exists
----------------------
The ``tensorfem`` CLI ``capabilities`` command used to print a hand-maintained
list of "stable" and "experimental" features.  That list drifted from the code:
a feature could be listed while its kernel was removed, or a real kernel could
be missing from the list.  This module replaces the hand-maintained dump with a
*self-registering* capability registry (the TensorLBM capability-registry pilot
pattern, applied to TensorFEM as the first solver-repo of the list):

* ``@capability`` — a function/class declares itself a capability with an id,
  a human title, a physics domain, a maturity triad and a callable contract
  (signature, params, returns).
* ``CapabilityRegistry`` — collects every registered capability in one place.
* ``dump_capabilities`` — emits the registry in two modes:
    * ``import`` mode: import the module and introspect the live registry;
    * ``ast`` mode: parse the source file *without* importing it, so a
      capability is discoverable even when its module cannot be imported
      (broken dependency, missing optional backend).
* ``classify_triad`` — maps each capability to 成熟/开发中/未开发
  (mature / developing / undeveloped) from machine-visible evidence
  (has a registered benchmark, has a test, is exported in the public API).

The *declarations* live in :mod:`tensorfem.capability_catalog` — every
``@capability`` there is applied with *literal* arguments so the ``ast`` dump
mode can read them without importing the module.

Design principles
-----------------
* Zero hand-copying: the registry is the single source of truth; the CLI dump
  is derived from it.
* Fail-closed: an ``@capability`` with a missing id or contract is an error.
* Deterministic: sorted output, no timestamps in canonical form.
* Two-mode agreement: ``ast`` mode must agree with ``import`` mode on the *set*
  of capability ids (contract fields may differ because ``ast`` mode cannot
  evaluate default values).
"""
from __future__ import annotations

import ast
import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "Capability",
    "CapabilityContract",
    "capability",
    "CapabilityRegistry",
    "dump_capabilities",
    "classify_triad",
    "TRIAD_ORDER",
    "registry",
    "assert_two_mode_agreement",
]

# --------------------------------------------------------------------------- #
# Triad vocabulary
# --------------------------------------------------------------------------- #

TRIAD_ORDER = ("mature", "developing", "undeveloped")
"""成熟 / 开发中 / 未开发 — machine-visible maturity ladder."""

_TRIAD_ZH = {
    "mature": "成熟",
    "developing": "开发中",
    "undeveloped": "未开发",
}


# --------------------------------------------------------------------------- #
# Contract & capability dataclasses
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CapabilityContract:
    """Callable endpoint contract: signature, params, returns."""

    callable_name: str
    signature: str
    params: list[str] = field(default_factory=list)
    returns: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Capability:
    """A self-registered solver capability."""

    id: str
    title: str
    domain: str
    maturity: str
    contract: CapabilityContract
    evidence: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.title or not self.domain:
            raise ValueError("capability requires non-empty id, title, domain")
        if self.maturity not in TRIAD_ORDER:
            raise ValueError(
                f"maturity must be one of {TRIAD_ORDER}, got {self.maturity!r}"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["maturity_zh"] = _TRIAD_ZH.get(self.maturity, self.maturity)
        return d


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

class CapabilityRegistry:
    """Collects every ``@capability``-registered capability."""

    def __init__(self) -> None:
        self._by_id: dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        if cap.id in self._by_id:
            raise ValueError(f"duplicate capability id {cap.id!r}")
        self._by_id[cap.id] = cap

    def get(self, cap_id: str) -> Capability | None:
        return self._by_id.get(cap_id)

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def all(self) -> list[Capability]:
        return [self._by_id[i] for i in self.ids()]

    def __len__(self) -> int:
        return len(self._by_id)


registry = CapabilityRegistry()
"""The process-wide default registry."""


# --------------------------------------------------------------------------- #
# Signature / contract helpers
# --------------------------------------------------------------------------- #

def _param_names(fn: Callable) -> list[str]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    return [p.name for p in sig.parameters.values()]


def _return_hint(fn: Callable) -> str:
    try:
        ann = inspect.signature(fn).return_annotation
    except (TypeError, ValueError):
        return ""
    if ann is inspect.Signature.empty:
        return ""
    if isinstance(ann, str):
        return ann
    return getattr(ann, "__name__", str(ann))


def _signature_of(fn: Callable) -> str:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return f"{getattr(fn, '__name__', '?')}(...)"
    ret = _return_hint(fn)
    return f"{getattr(fn, '__name__', '?')}{sig}" + (f" -> {ret}" if ret else "")


def _build_contract(fn: Callable, *, callable_name: str | None = None) -> CapabilityContract:
    name = callable_name or getattr(fn, "__name__", "?")
    return CapabilityContract(
        callable_name=name,
        signature=_signature_of(fn),
        params=_param_names(fn),
        returns=_return_hint(fn),
    )


# --------------------------------------------------------------------------- #
# The @capability decorator
# --------------------------------------------------------------------------- #

def capability(
    id: str,
    *,
    title: str,
    domain: str,
    maturity: str = "developing",
    contract: CapabilityContract | None = None,
    **evidence: Any,
) -> Callable[[Callable], Callable]:
    """Self-registering decorator.

    Declares the decorated function/class as a capability in the process-wide
    ``registry``.  All metadata is passed as *literal* keyword arguments so the
    declaration is also discoverable by :func:`dump_capabilities` in ``ast``
    mode (which never imports the module).

    Raises
    ------
    ValueError
        if ``id``/``title``/``domain`` are empty, ``maturity`` is not in
        :data:`TRIAD_ORDER`, or ``id`` is already registered.
    """

    def _wrap(fn: Callable) -> Callable:
        if not id or not title or not domain:
            raise ValueError("capability requires non-empty id, title, domain")
        if maturity not in TRIAD_ORDER:
            raise ValueError(f"maturity must be one of {TRIAD_ORDER}, got {maturity!r}")
        c = contract or _build_contract(fn)
        cap = Capability(
            id=id,
            title=title,
            domain=domain,
            maturity=maturity,
            contract=c,
            evidence=dict(evidence),
        )
        registry.register(cap)
        return fn

    return _wrap


# --------------------------------------------------------------------------- #
# Triad classification from machine-visible evidence
# --------------------------------------------------------------------------- #

def classify_triad(
    *,
    has_benchmark: bool = False,
    has_test: bool = False,
    in_public_api: bool = False,
) -> str:
    """Map evidence to a maturity triad.

    * ``mature``      — has a registered benchmark *and* a test (or is in the
      public API): the capability is numerically verified and exercised.
    * ``developing``  — has a test or is in the public API, but no registered
      benchmark: exercised but not yet numerically anchored.
    * ``undeveloped`` — no test, no benchmark, not in the public API: declared
      but not yet exercised.
    """
    if has_benchmark and (has_test or in_public_api):
        return "mature"
    if has_test or in_public_api:
        return "developing"
    return "undeveloped"


# --------------------------------------------------------------------------- #
# Two-mode dump
# --------------------------------------------------------------------------- #

def _kw_from_call(call: ast.Call) -> dict:
    """Read literal keyword args from a ``capability(...)`` call."""
    kw: dict[str, Any] = {}
    for k in call.keywords:
        if k.arg is None:
            continue
        if isinstance(k.value, ast.Constant):
            kw[k.arg] = k.value.value
        else:
            kw[k.arg] = None
    if call.args:
        a0 = call.args[0]
        if isinstance(a0, ast.Constant):
            kw.setdefault("id", a0.value)
    return kw


def _ast_extract(source: str) -> list[dict]:
    """Extract ``@capability`` declarations from *source* without importing.

    Detects two equivalent declaration forms:

    * ``@capability("id", title=..., ...)`` applied directly to a function or
      class definition;
    * ``capability("id", title=..., ...)(fn)`` — an explicit registration call.

    Returns a list of dicts with keys ``id``, ``title``, ``domain``,
    ``maturity``, ``callable_name`` and ``signature``.  Only literal keyword
    arguments are read; anything non-literal is reported as ``None`` so the
    caller can flag it as not-ast-discoverable.
    """
    tree = ast.parse(source)
    out: list[dict] = []

    # 1) @capability(...) decorators on function/class definitions
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for dec in node.decorator_list:
            if not (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "capability"
            ):
                continue
            kw = _kw_from_call(dec)
            out.append(
                {
                    "id": kw.get("id"),
                    "title": kw.get("title"),
                    "domain": kw.get("domain"),
                    "maturity": kw.get("maturity"),
                    "callable_name": node.name,
                    "signature": _ast_signature(node),
                }
            )

    # 2) capability(...)(fn) registration calls
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name) \
                and node.func.func.id == "capability":
            kw = _kw_from_call(node.func)
            name = None
            if node.args and isinstance(node.args[0], ast.Name):
                name = node.args[0].id
            out.append(
                {
                    "id": kw.get("id"),
                    "title": kw.get("title"),
                    "domain": kw.get("domain"),
                    "maturity": kw.get("maturity"),
                    "callable_name": name or "?",
                    "signature": f"{name or '?'}(...)",
                }
            )
    return out


def _ast_signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    if isinstance(node, ast.ClassDef):
        return f"{node.name}(...)  # class"
    args = node.args
    pos = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
    if args.vararg:
        pos.append(f"*{args.vararg.arg}")
    if args.kwonlyargs:
        pos.append("*")
        pos.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg:
        pos.append(f"**{args.kwarg.arg}")
    return f"{node.name}({', '.join(pos)})"


def _find_capability_sources() -> list[Path]:
    """Locate the source files that carry ``@capability`` declarations."""
    here = Path(__file__).resolve().parent
    return sorted(here.glob("*.py"))


def dump_capabilities(mode: str = "import") -> dict:
    """Dump the capability registry.

    Parameters
    ----------
    mode
        ``"import"`` — introspect the live ``registry`` (requires the module to
        be importable).  ``"ast"`` — parse the source files without importing,
        so capabilities are discoverable even when a module cannot be imported.

    Returns
    -------
    dict
        ``{"mode": ..., "count": ..., "capabilities": [...]}`` where each
        capability carries ``id``/``title``/``domain``/``maturity``/``contract``
        and (in ``import`` mode) ``evidence``.
    """
    if mode not in ("import", "ast"):
        raise ValueError(f"mode must be 'import' or 'ast', got {mode!r}")
    if mode == "import":
        caps = [c.to_dict() for c in registry.all()]
        return {"mode": mode, "count": len(caps), "capabilities": caps}
    # ast mode
    caps: list[dict] = []
    for path in _find_capability_sources():
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for decl in _ast_extract(src):
            if decl["id"] is None:
                continue
            caps.append(
                {
                    "id": decl["id"],
                    "title": decl["title"],
                    "domain": decl["domain"],
                    "maturity": decl["maturity"] or "developing",
                    "contract": {
                        "callable_name": decl["callable_name"],
                        "signature": decl["signature"],
                        "params": [],
                        "returns": "",
                    },
                    "source": str(path.name),
                }
            )
    caps.sort(key=lambda c: c["id"])
    return {"mode": mode, "count": len(caps), "capabilities": caps}


def _ast_ids() -> set[str]:
    return {c["id"] for c in dump_capabilities("ast")["capabilities"]}


def _import_ids() -> set[str]:
    return set(registry.ids())


def assert_two_mode_agreement() -> None:
    """Fail-closed check that ``ast`` and ``import`` modes agree on the id set."""
    ast_ids = _ast_ids()
    import_ids = _import_ids()
    missing_in_ast = import_ids - ast_ids
    if missing_in_ast:
        raise AssertionError(
            "capabilities registered in import mode but not found by AST dump: "
            + ", ".join(sorted(missing_in_ast))
        )
