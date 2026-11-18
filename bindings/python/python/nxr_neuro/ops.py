"""Deprecated: `nxr_neuro.ops` is now `nxr_neuro.compute` (the maintainers' naming, 2026-11-17). This alias stays for one release."""

import warnings as _warnings

from .compute import operators, signal, stats  # noqa: F401

_warnings.warn("nxr_neuro.ops is deprecated: use nxr_neuro.compute (the same functions)", DeprecationWarning, stacklevel=2)
