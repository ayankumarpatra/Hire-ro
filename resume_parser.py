"""PDF text extraction and lightweight resume information extraction for HireLens AI.

Responsibility: take the bytes of an already size/signature-checked upload,
return clean text, a cautious resume-likeness assessment, and a small
structured summary that can later be used by the matching layer.

Safety notes
- Works fully in memory: PyMuPDF opens the PDF from bytes, nothing is written
  to disk, and the original filename is never needed here.
- Only text is extracted. Pages are never rendered, and no links, scripts or
  embedded files in the PDF are followed or executed.
- Extracted resume text is never logged.
- Extraction is deliberately heuristic. Missing data means "not detected",
  not that the candidate does not have that information.
"""

import logging
import re
import threading
from dataclasses import dataclass, field

import pymupdf

from validators import UploadError, UploadLimits

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 100
MAX_TEXT_CHARS = 100_000
MAX_FIELD_ITEMS = 30
MAX_FIELD_CHARS = 4_000

_PYMUPDF_LOCK = threading.Lock()


class ResumeParseError(UploadError):
    """The PDF is structurally unusable (damaged, encrypted, too long, no text...)."""


@dataclass
class ResumeAssessment:
    status: str
    indicators_found: list = field(default_factory=list)
    indicators_total: int = 0
    non_resume_cues: int = 0

    @property
    def message(self):
        if self.status == "uncertain":
            return ("This document only partly resembles a typical resume. "
                    "Please review it manually before relying on its score.")
        if self.status == "not_resume_like":
            return ("This document does not look like a typical resume. It was kept "
                    "so you can check it, but any score for it may be meaningless.")
        return None


@dataclass
class ResumeInfo:
    """Small structured summary extracted with deterministic heuristics."""

    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    education: list = field(default_factory=list)
    skills: list = field(default_factory=list)
    experience: list = field(default_factory=list)
    projects: list = field(default_factory=list)


@dataclass
class ParsedResume:
    text: str
    page_count: int
    truncated: bool
    assessment: ResumeAssessment
    info: ResumeInfo = field(default_factory=ResumeInfo)

    @property
    def char_count(self) -> int:
        return len(self.text)


def _read_document(doc, max_pages: int):
    if not doc.is_pdf:
        raise ResumeParseError("malformed", "This file could not be read as a PDF document.")

    if doc.needs_pass or doc.is_encrypted:
        raise ResumeParseError(
            "encrypted",
            "This PDF is password-protected or encrypted. Please save an unprotected "
            "copy and upload that instead.",
        )

    page_count = doc.page_count
    if page_count < 1:
        raise ResumeParseError(
            "malformed",
            "This PDF could not be read - it appears to contain no readable pages.",
        )
    if page_count > max_pages:
        raise ResumeParseError(
            "too_many_pages",
            f"This PDF has {page_count} pages, but the limit is {max_pages}. "
            "Resumes are normally much shorter - please upload just the resume.",
        )

    chunks, total_chars, truncated = [], 0, False
    try:
        for page_number in range(page_count):
            page_text = doc.load_page(page_number).get_text("text")
            chunks.append(page_text)
            total_chars += len(page_text)
            if total_chars > MAX_TEXT_CHARS:
                truncated = True
                break
    except Exception as exc:
        logger.warning("Text extraction failed (%s).", type(exc).__name__)
        raise ResumeParseError(
            "unreadable",
            "Text could not be extracted from this PDF. The file may be damaged.",
        ) from exc

    return "\n".join(chunks), page_count, truncated


def extract_pdf_text(data: bytes, max_pages: int):
    with _PYMUPDF_LOCK:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            logger.warning("PDF could not be opened (%s).", type(exc).__name__)
            raise ResumeParseError(
                "malformed",
                "This PDF could not be opened. The file may be damaged or incomplete.",
            ) from exc
        try:
            return _read_document(doc, max_pages)
        finally:
            doc.close()


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(raw: str) -> str:
    text = _CONTROL_CHARS.sub("", raw)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_SECTION_HEADINGS = {
    "education": ["education", "academic background", "academic qualifications", "qualifications"],
    "skills": ["skills", "technical skills", "core competencies", "technologies", "tech stack"],
    "projects": ["projects", "project", "academic projects", "personal projects"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "work history", "internship", "internships"],
    "certifications": ["certifications", "certification", "certificates", "licenses",
                       "licences", "courses"],
}
_HEADING_PATTERNS = {
    name: re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b")
    for name, phrases in _SECTION_HEADINGS.items()
}

_DEGREE_WORDS = re.compile(
    r"\b(?:bachelors?|bachelor's|master's|master of|b\.?\s?tech|m\.?\s?tech|b\.?\s?sc|"
    r"m\.?\s?sc|mba|ph\.?\s?d|diploma|cgpa)\b",
    re.IGNORECASE,
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()\-.]{8,}\d)(?!\d)")
_PROFILE_LINK = re.compile(r"(?:https?://)?(?:www\.)?(linkedin\.com/[^\s|,;]+|github\.com/[^\s|,;]+)", re.IGNORECASE)
_DATE_RANGE = re.compile(
    r"\b(?:19|20)\d{2}\s*(?:-|\u2013|\u2014|to)\s*(?:(?:19|20)\d{2}|present|current|now|ongoing)\b",
    re.IGNORECASE,
)

INDICATORS_TOTAL = len(_HEADING_PATTERNS) + 2

_NON_RESUME_CUES = [
    "job description", "we are looking for", "about the company", "apply now",
    "invoice no", "bill to", "terms and conditions", "all rights reserved",
    "table of contents", "payment terms",
]


def _short_heading_lines(text: str):
    for line in text.splitlines():
        if len(line) <= 60:
            words = re.sub(r"[^a-z& ]+", " ", line.lower()).split()
            if 1 <= len(words) <= 5:
                yield " ".join(words)


def assess_resume_likeness(text: str) -> ResumeAssessment:
    found = []
    headings = list(_short_heading_lines(text))
    for name, pattern in _HEADING_PATTERNS.items():
        if any(pattern.search(h) for h in headings):
            found.append(name)
    if "education" not in found and _DEGREE_WORDS.search(text):
        found.append("education")
    if _EMAIL.search(text) or _PROFILE_LINK.search(text):
        found.append("contact details")
    if _DATE_RANGE.search(text):
        found.append("dated roles/education")

    levels = ["not_resume_like", "uncertain", "resume_like"]
    level = 2 if len(found) >= 4 else 1 if len(found) >= 2 else 0

    lowered = text.lower()
    cues = sum(1 for cue in _NON_RESUME_CUES if cue in lowered)
    if cues >= 2:
        level = max(0, level - 1)

    return ResumeAssessment(
        status=levels[level],
        indicators_found=found,
        indicators_total=INDICATORS_TOTAL,
        non_resume_cues=cues,
    )


def _normalise_item(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" •|\t")


def _unique_items(items, max_items=MAX_FIELD_ITEMS, max_chars=MAX_FIELD_CHARS):
    result = []
    seen = set()
    for item in items:
        item = _normalise_item(item)
        if not item or len(item) > max_chars:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= max_items:
            break
    return result


def _section_ranges(text: str):
    """Return (section, body) pairs using recognizable one-line headings."""
    lines = text.splitlines()
    sections = []
    current_name = ""
    current_lines = []

    def flush():
        if current_name and current_lines:
            sections.append((current_name, "\n".join(current_lines).strip()))

    for line in lines:
        candidate = _normalise_item(line).casefold().rstrip(":")
        matched = None
        for name, phrases in _SECTION_HEADINGS.items():
            if candidate in {p.casefold() for p in phrases}:
                matched = name
                break
        if matched:
            flush()
            current_name = matched
            current_lines = []
        elif current_name:
            current_lines.append(line)

    flush()
    return sections


def _extract_name(text: str, email: str, profile_link: str) -> str:
    """Use the first plausible heading-like line near the top of the resume."""
    for line in text.splitlines()[:12]:
        value = _normalise_item(line)
        if not value or len(value) > 80:
            continue
        if email and email.casefold() in value.casefold():
            continue
        if profile_link and "linkedin.com" in value.casefold() or "github.com" in value.casefold():
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'`.-]*", value)
        if 2 <= len(words) <= 5 and not _HEADING_PATTERNS["skills"].search(value) \
                and not any(_HEADING_PATTERNS[s].search(value) for s in _SECTION_HEADINGS):
            if not any(ch.isdigit() for ch in value):
                return value
    return ""


def _extract_contact(text: str):
    email_match = _EMAIL.search(text)
    email = email_match.group(0) if email_match else ""

    phone = ""
    for match in _PHONE.finditer(text):
        candidate = re.sub(r"\s+", " ", match.group(0)).strip()
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 15:
            phone = candidate
            break

    linkedin = ""
    github = ""
    for match in _PROFILE_LINK.finditer(text):
        link = match.group(0).rstrip(".,;)")
        if "linkedin.com" in link.casefold() and not linkedin:
            linkedin = link
        elif "github.com" in link.casefold() and not github:
            github = link

    return email, phone, linkedin, github


def _extract_section_items(sections, target):
    for name, body in sections:
        if name == target:
            lines = [line for line in body.splitlines() if _normalise_item(line)]
            return _unique_items(lines)
    return []


def _extract_skills(sections):
    items = _extract_section_items(sections, "skills")
    if not items:
        return []
    expanded = []
    for item in items:
        parts = re.split(r"[,;|•]", item)
        expanded.extend(parts)
    return _unique_items(expanded)


def extract_resume_info(text: str) -> ResumeInfo:
    """Extract common resume fields without making claims about missing data."""
    email, phone, linkedin, github = _extract_contact(text)
    sections = _section_ranges(text)
    return ResumeInfo(
        name=_extract_name(text, email, linkedin or github),
        email=email,
        phone=phone,
        linkedin=linkedin,
        github=github,
        education=_extract_section_items(sections, "education"),
        skills=_extract_skills(sections),
        experience=_extract_section_items(sections, "experience"),
        projects=_extract_section_items(sections, "projects"),
    )


def parse_resume_pdf(data: bytes, limits: UploadLimits) -> ParsedResume:
    raw_text, page_count, truncated = extract_pdf_text(data, limits.max_pages)
    text = clean_text(raw_text)

    visible_chars = len(re.sub(r"\s", "", text))
    if visible_chars < MIN_TEXT_CHARS:
        raise ResumeParseError(
            "no_text",
            "No readable text was found in this PDF. It may be a scanned image or a "
            "photo of a resume - OCR is not supported in this version. Please upload "
            "a text-based PDF.",
        )

    if len(text) > MAX_TEXT_CHARS:
        text, truncated = text[:MAX_TEXT_CHARS], True

    assessment = assess_resume_likeness(text)
    info = extract_resume_info(text)
    logger.info("PDF parsed: pages=%d chars=%d truncated=%s status=%s fields=%d",
                page_count, len(text), truncated, assessment.status,
                sum(bool(getattr(info, name)) for name in ("name", "email", "phone", "linkedin", "github")))
    return ParsedResume(
        text=text,
        page_count=page_count,
        truncated=truncated,
        assessment=assessment,
        info=info,
    )
