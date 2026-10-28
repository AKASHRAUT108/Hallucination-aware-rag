"""Sentence splitting that keeps trailing [n] citations attached to their sentence."""
import re

from src.generation.citations import strip_citations

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: list[str] = []
    for part in (p.strip() for p in _SPLIT.split(text)):
        # Models often write "Sentence. [1]". The split leaves "[1]" alone, so glue it back.
        if out and part and not strip_citations(part):
            out[-1] = f"{out[-1]} {part}"
        elif part:
            out.append(part)
    return out


def passage_windows(text: str, max_sentences: int = 2) -> list[str]:
    """Sliding windows of 1..max_sentences consecutive sentences.
    NLI models judge short premises far better than 150-word chunks."""
    sents = split_sentences(text)
    windows: list[str] = []
    for size in range(1, max_sentences + 1):
        for i in range(0, max(len(sents) - size + 1, 0)):
            windows.append(" ".join(sents[i:i + size]))
    return windows
