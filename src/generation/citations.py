"""Helpers for the [n] citation markers the generator writes."""
import re

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(text: str) -> list[int]:
    """Citation ranks in order of first appearance: 'A [1]. B [2, 1].' -> [1, 2]."""
    seen: list[int] = []
    for m in _CITE.finditer(text):
        for n in re.split(r"\s*,\s*", m.group(1)):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def strip_citations(text: str) -> str:
    return re.sub(r"\s+", " ", _CITE.sub("", text)).strip()


def is_refusal(text: str, phrases: list[str]) -> bool:
    t = text.lower().strip()
    return (not t) or any(p in t for p in phrases)


def resolve_citations(answer: str, passages: list[dict]) -> list[dict]:
    """Map cited ranks to the passages that were actually shown. Unknown ranks (e.g. [9]) are dropped."""
    by_rank = {p["rank"]: p for p in passages}
    return [{"rank": r, "chunk_id": by_rank[r]["chunk_id"]}
            for r in parse_citations(answer) if r in by_rank]
