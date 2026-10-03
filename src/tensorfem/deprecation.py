"""Small public deprecation primitives for staged API compatibility."""
from __future__ import annotations
from functools import wraps
import warnings


class TensorFEMDeprecationWarning(FutureWarning):
    """Warning category for public APIs scheduled for removal."""


def warn_deprecated(name: str,*,since: str,remove: str,replacement: str | None=None,stacklevel=2):
    if not name or not since or not remove:raise ValueError("deprecation metadata is required")
    message=f"{name} is deprecated since TensorFEM {since} and is planned for removal in {remove}."
    if replacement:message+=f" Use {replacement} instead."
    warnings.warn(message,TensorFEMDeprecationWarning,stacklevel=stacklevel)


def deprecated(*,since: str,remove: str,replacement: str | None=None):
    """Decorate a function without changing its signature metadata."""
    def decorate(function):
        @wraps(function)
        def wrapper(*args,**kwargs):
            warn_deprecated(function.__qualname__,since=since,remove=remove,
                            replacement=replacement,stacklevel=3)
            return function(*args,**kwargs)
        wrapper.__tensorfem_deprecation__={"since":since,"remove":remove,"replacement":replacement}
        return wrapper
    return decorate
