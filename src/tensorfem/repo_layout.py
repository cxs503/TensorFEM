"""Locate sibling Tensor repository checkouts without hard-coded absolute paths.

Several scripts and cross-repository exchanges need files that live in a
sibling checkout (TensorFVM wall-load artifacts, TensorLBM CAD modules).  Those
locations used to be written as absolute paths into a single developer's
machine, so every consumer failed anywhere else.  Resolution order:

  1. an explicit environment variable (``TENSORFVM_ROOT`` / ``TENSORLBM_ROOT``)
  2. a sibling checkout next to this repository
  3. a loud error -- never a silent fallback to a wrong path
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

__all__ = ["ROOT", "sibling_repo", "sibling_file"]


def sibling_repo(name: str) -> Path:
    """Return the checkout of the sibling repository *name*.

    Raises FileNotFoundError listing every searched location when the
    checkout cannot be found, so callers fail loudly instead of reading
    stale or unrelated data.
    """
    env = os.environ.get(f"{name.upper()}_ROOT")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"{name.upper()}_ROOT={env!r} is not a directory")
    searched = []
    base = ROOT
    for _ in range(3):
        base = base.parent
        candidate = base / name
        searched.append(str(candidate))
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"cannot locate the {name} checkout: set {name.upper()}_ROOT or place it "
        f"next to this repository. Searched: {', '.join(searched)}"
    )


def sibling_file(name: str, relative: str) -> Path:
    """Return ``relative`` inside the sibling repository *name*."""
    return sibling_repo(name) / relative
