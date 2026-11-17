"""The library's errors: `nxr_neuro.Error` and one subclass per error code."""

from __future__ import annotations


class Error(Exception):
    """Base of every error the library raises. `.code` is the library's error code."""

    code: str = ""


class FormatError(Error):
    """The bytes are not the format they claim to be (a bad magic, a truncated header, a field out of range)."""


class UnsupportedError(Error, NotImplementedError):
    """A valid file or request the library refuses, with the reason (an epoched recording, an ASCII BrainVision file)."""


class InvalidArgumentError(Error, ValueError):
    """An argument outside what the call accepts (a block of samples or channels the recording does not have)."""


class IOError_(Error, OSError):
    """The file cannot be opened, read or written."""


class NotConvergedError(Error):
    """An eigensolve did not converge."""


class CancelledError(Error):
    """The call was stopped on request."""


_BY_CODE: dict[str, type[Error]] = {
    "format_error": FormatError,
    "unsupported": UnsupportedError,
    "invalid_argument": InvalidArgumentError,
    "io": IOError_,
    "EIGENSOLVE_NOT_CONVERGED": NotConvergedError,
    "cancelled": CancelledError,
    "CANCELLED": CancelledError,
}


def make(code: str, message: str) -> Error:
    """The exception for a library error code (called by the compiled core)."""
    exc = _BY_CODE.get(code, Error)(message)
    exc.code = code
    return exc
