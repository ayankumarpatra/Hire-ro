"""Upload validation for HireLens AI.

Responsibility: decide whether an uploaded file is *acceptable to look at*
(right extension, not empty, not too big, starts with the PDF signature) and
whether a whole request is acceptable (at least one file, not too many).

It does NOT open or parse PDFs - that is resume_parser.py's job.

Nothing here ever uses a user-supplied filename as a filesystem path. Names are
only cleaned up so they can be *displayed* safely.
"""

import math
import os
import unicodedata
from dataclasses import dataclass

BYTES_PER_MB = 1024 * 1024
PDF_SIGNATURE = b"%PDF-"  # every real PDF starts with these 5 bytes


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class UploadError(Exception):
    """A problem with an upload that is safe to show to the user.

    `message` is written for the recruiter. `code` is a short machine-readable
    label used in logs and tests (so logs never need the file's contents).
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class UploadValidationError(UploadError):
    """Raised by this module when a file or request fails a check."""


class ConfigError(ValueError):
    """Raised at start-up when an environment variable has a bad value."""


# --------------------------------------------------------------------------
# Configurable limits
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UploadLimits:
    max_file_bytes: int = 5 * BYTES_PER_MB  # 5 MB per file
    max_files: int = 10                      # files per request
    max_pages: int = 15                      # pages per PDF


def _read_positive_number(environ, name: str, default, whole_number: bool):
    """Read one positive number from `environ`, or return `default` if unset."""
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    kind = "whole number" if whole_number else "number"
    try:
        value = int(raw) if whole_number else float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a positive {kind}, but got {raw!r}.") from None
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} must be a positive {kind}, but got {raw!r}.")
    return value


def load_limits(environ=None) -> UploadLimits:
    """Build UploadLimits from MAX_FILE_MB, MAX_FILES and MAX_PAGES.

    Unset variables fall back to the safe defaults (5 MB, 10 files, 15 pages).
    A bad value stops start-up with a clear message instead of being ignored.
    """
    environ = os.environ if environ is None else environ
    max_mb = _read_positive_number(environ, "MAX_FILE_MB", 5, whole_number=False)
    return UploadLimits(
        max_file_bytes=int(max_mb * BYTES_PER_MB),
        max_files=_read_positive_number(environ, "MAX_FILES", 10, whole_number=True),
        max_pages=_read_positive_number(environ, "MAX_PAGES", 15, whole_number=True),
    )


def format_size(num_bytes: int) -> str:
    """Human-friendly size for messages, e.g. 5242880 -> '5 MB'."""
    if num_bytes >= BYTES_PER_MB:
        return f"{num_bytes / BYTES_PER_MB:g} MB"
    return f"{num_bytes / 1024:g} KB"


# --------------------------------------------------------------------------
# Filenames (display only - never used as a path)
# --------------------------------------------------------------------------
def _strip_path_and_control_chars(raw_name) -> str:
    """Keep only the last path component and drop invisible/control characters.

    Dropping 'format' characters (category Cf) also removes the right-to-left
    override trick that can make 'resumefdp.exe' look like 'resumeexe.pdf'.
    """
    name = (raw_name or "").replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch) not in ("Cc", "Cf"))
    return name.strip()


def clean_filename(raw_name, max_length: int = 80) -> str:
    """Return a short, safe-to-display version of an uploaded filename.

    Templates still HTML-escape it; this just removes path parts, control
    characters and absurd lengths.
    """
    name = _strip_path_and_control_chars(raw_name) or "unnamed file"
    if len(name) > max_length:
        name = name[: max_length - 3] + "..."
    return name


# --------------------------------------------------------------------------
# Per-file checks
# --------------------------------------------------------------------------
def validate_extension(raw_name) -> None:
    """Reject anything whose name does not end in .pdf (case-insensitive).

    This is only a first, cheap filter. The browser-supplied name and content
    type can be faked, so validate_pdf_bytes() checks the real content too.
    """
    name = _strip_path_and_control_chars(raw_name).lower()
    if not name.endswith(".pdf"):
        raise UploadValidationError(
            "bad_extension",
            "Only PDF files are accepted (the file name must end in .pdf).",
        )


def read_upload_bytes(file_storage, max_bytes: int) -> bytes:
    """Read an upload into memory, but never more than max_bytes + 1 bytes.

    Reading one byte past the limit is enough to know the file is too big,
    without loading a huge file into memory.
    """
    return file_storage.stream.read(max_bytes + 1)


def validate_pdf_bytes(data: bytes, limits: UploadLimits) -> None:
    """Check size and PDF signature of an uploaded file's bytes."""
    if len(data) == 0:
        raise UploadValidationError("empty_file", "The file is empty (0 bytes).")
    if len(data) > limits.max_file_bytes:
        raise UploadValidationError(
            "file_too_large",
            f"The file is larger than the {format_size(limits.max_file_bytes)} limit.",
        )
    if not data.startswith(PDF_SIGNATURE):
        raise UploadValidationError(
            "not_a_pdf",
            "This does not look like a real PDF - its contents do not start with the "
            "PDF signature. It may be another file type that was renamed to .pdf.",
        )


# --------------------------------------------------------------------------
# Whole-request checks
# --------------------------------------------------------------------------
def select_uploaded_files(files, limits: UploadLimits) -> list:
    """Drop empty form slots, then enforce 'at least one' and 'not too many'.

    Browsers send a blank entry (empty filename) when no file was chosen, so
    those are ignored rather than counted.
    """
    chosen = [f for f in files if f is not None and f.filename]
    if not chosen:
        raise UploadValidationError("no_files", "Please choose at least one PDF resume to upload.")
    if len(chosen) > limits.max_files:
        raise UploadValidationError(
            "too_many_files",
            f"You selected {len(chosen)} files, but the limit is {limits.max_files} "
            "files per upload. Please upload fewer files at once.",
        )
    return chosen