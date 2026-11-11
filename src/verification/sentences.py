"""Sentence splitting that keeps trailing [n] citations attached to their sentence."""
import re

from src.generation.citations import strip_citations

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
# PDF bullets arrive as private-use glyphs (e.g. U+F0B7) or "•". They are not whitespace, so without this the
# items of a bullet list fuse into one huge "sentence" that NLI models cannot judge.
_BULLETS = re.compile("[\ue000-\uf8ff\u2022\u2023\u25aa\u25cf\u25e6]+")


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for segment in _BULLETS.split(text):                  # every bullet item starts a new sentence
        segment = re.sub(r"\s+", " ", segment).strip()
        if segment:
            parts.extend(p.strip() for p in _SPLIT.split(segment))
    out: list[str] = []
    for part in parts:
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
