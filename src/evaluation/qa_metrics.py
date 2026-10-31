"""Exact Match, token F1 and a lenient 'contains the gold answer' check (SQuAD-style normalisation)."""
import re
import string
from collections import Counter

_KEEP = set("+%")   # keep these so a grade like "A+" is not reduced to the article "a"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(?<=\d)\.(?=\d)", "\x00", text)          # protect decimals such as 5.0
    text = "".join(ch for ch in text if ch not in string.punctuation or ch in _KEEP or ch == "\x00")
    text = text.replace("\x00", ".")
    text = re.sub(r"\b(a|an|the)\b(?!\+)", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = normalize(pred)
    return float(any(p == normalize(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_score(pred: str, golds: list[str]) -> float:
    return max((_f1(pred, g) for g in golds), default=0.0)


def contains_answer(pred: str, golds: list[str]) -> float:
    """Generators are wordy, so EM/F1 undercount. This asks: does the answer contain a gold answer?"""
    p = f" {normalize(pred)} "
    return float(any(normalize(g) and f" {normalize(g)} " in p for g in golds))
