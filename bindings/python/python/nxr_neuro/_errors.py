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


class WriterLockHeld(Error):
    """Another live writer holds the dataset's writer lock (one writer a dataset, across processes)."""


class NotWritable(Error, PermissionError):
    """The dataset (its folder or its database) cannot be written by this process."""


class SchemaVersionError(Error):
    """A database at a schema version this library does not read (the message names the remedy)."""


class NotADatabaseError(Error):
    """The folder or file is not a dataset this library reads."""


class NoSuchRow(Error, LookupError):
    """Nothing stored of that name or id (a subject, a node, a labelling)."""


class Ambiguous(Error, LookupError):
    """Several stored things match the name: give the id."""


class AlreadyExists(Error, FileExistsError):
    """A dataset, subject or node of that name is already stored (a node: give replace=True to move the old one aside)."""


class NotFound(Error, FileNotFoundError):
    """A file or folder the call names does not exist."""


_BY_CODE: dict[str, type[Error]] = {
    "format_error": FormatError,
    "unsupported": UnsupportedError,
    "invalid_argument": InvalidArgumentError,
    "io": IOError_,
    "EIGENSOLVE_NOT_CONVERGED": NotConvergedError,
    "cancelled": CancelledError,
    "CANCELLED": CancelledError,
    "writer_lock_held": WriterLockHeld,
    "not_writable": NotWritable,
    "schema_version": SchemaVersionError,
    "not_a_database": NotADatabaseError,
    "no_such_row": NoSuchRow,
    "ambiguous": Ambiguous,
    "already_exists": AlreadyExists,
    "not_found": NotFound,
}


def make(code: str, message: str) -> Error:
    """The exception for a library error code (called by the compiled core)."""
    exc = _BY_CODE.get(code, Error)(message)
    exc.code = code
    return exc
