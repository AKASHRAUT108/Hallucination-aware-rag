"""Load raw documents (.pdf, .txt, .md) into page-level records."""
import re
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".pdf", ".txt", ".md"}


def clean_text(text: str) -> str:
    """Normalise whitespace and fix hyphenated line breaks."""
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(\w)", r"\1", text)   # "informa-\ntion" -> "information"
    text = re.sub(r"\s*\n\s*", " ", text)    # newlines -> spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _load_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"source": path.name, "page": i, "text": text})
    return pages


def _load_text(path: Path) -> list[dict]:
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [{"source": path.name, "page": 1, "text": text}] if text else []


def load_documents(raw_dir: Path) -> list[dict]:
    """Return a list of {source, page, text} for every supported file in raw_dir."""
    raw_dir = Path(raw_dir)
    records: list[dict] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        if path.suffix.lower() == ".pdf":
            records.extend(_load_pdf(path))
        else:
            records.extend(_load_text(path))
    return records
