"""Sentence-level support and citation checks against the passages the generator saw."""
import re

import numpy as np

from src.generation.citations import parse_citations, strip_citations
from src.verification.schemas import SentenceCheck
from src.verification.sentences import passage_windows, split_sentences


_LEAD_IN = re.compile(
    r"^\s*(?:(?:according to|based on|as per)\s+(?:the\s+)?(?:given\s+)?passages?(?:\s*\[[\d,\s]+\])?"
    r"|therefore,?\s+the\s+(?:correct\s+)?answer\s+is(?:\s+that)?)\s*,?\s*", re.I)
_LEAD_OUT = re.compile(
    r"[,;]?\s*(?:according to|based on|as per)\s+(?:the\s+)?(?:given\s+)?passages?\s*(?:\[[\d,\s]+\])?\s*(?=[.!?]?\s*$)", re.I)
_LEADING_ZEROS = re.compile(r"\b0+(?=\d)")
_META = re.compile(r"^\s*(?:this|that|the)\s+(?:information|answer|fact|statement)\b.*\b(?:passage|stated|provided|mentioned)\b", re.I)


def normalize_numbers(text: str) -> str:
    """'04 students' -> '4 students'. Documents often zero-pad numbers; NLI models do not treat 04 and 4 alike."""
    return _LEADING_ZEROS.sub("", text)


def clean_claim(sentence: str) -> str:
    """Remove 'According to the passage [1],' at the start and ', according to the passage' at the end,
    so NLI judges the claim itself, and normalise zero-padded numbers."""
    text = _LEAD_OUT.sub("", _LEAD_IN.sub("", sentence))
    text = normalize_numbers(strip_citations(text).strip())
    return text[:1].upper() + text[1:] if text else text


def is_meta(sentence: str) -> bool:
    """'This information is provided in Passage [1], ...' talks about the source and makes no factual claim."""
    return bool(_META.match(strip_citations(sentence)))


def check_answer(raw_answer: str, passages: list[dict], nli, window_sentences: int = 2) -> list[SentenceCheck]:
    """For each answer sentence:
      support        best P(entail) over all passages (best short window within each passage)
      cited_support  best P(entail) over the passages the model cited
    If the model cited nothing, the best-supporting passage is attributed instead (citation_inferred=True).
    """
    sentences = split_sentences(raw_answer)
    clean = [clean_claim(s) for s in sentences]
    keep = [i for i, c in enumerate(clean) if c and not is_meta(sentences[i])]
    if not keep:                                    # never drop everything: fall back to all non-empty sentences
        keep = [i for i, c in enumerate(clean) if c]
    if not keep or not passages:
        return []

    # Score each answer sentence against short windows of every passage, then keep the best window
    # per passage. (A 150-word premise makes small NLI models say "neutral" even for copied text.)
    windows: list[str] = []
    owner: list[int] = []                       # which passage each window came from
    for j, p in enumerate(passages):
        w = [normalize_numbers(x) for x in (passage_windows(p["text"], window_sentences) or [p["text"]])]
        windows.extend(w)
        owner.extend([j] * len(w))
    owner_arr = np.array(owner)
    raw = nli.entailment_probs([(w, clean[i]) for i in keep for w in windows]).reshape(len(keep), len(windows))
    probs = np.stack([raw[:, owner_arr == j].max(axis=1) for j in range(len(passages))], axis=1)
    col_of_rank = {p["rank"]: j for j, p in enumerate(passages)}

    checks: list[SentenceCheck] = []
    for row, i in enumerate(keep):
        cited = [r for r in parse_citations(sentences[i]) if r in col_of_rank]
        best = int(np.argmax(probs[row]))
        chk = SentenceCheck(text=clean[i], cited_ranks=cited, support=float(probs[row, best]),
                            best_rank=passages[best]["rank"])
        if cited:
            chk.cited_support = float(max(probs[row, col_of_rank[r]] for r in cited))
        else:
            chk.cited_support, chk.citation_inferred = chk.support, True
        checks.append(chk)
    return checks


def citation_score(checks: list[SentenceCheck], threshold: float = 0.5) -> float:
    """Share of sentences whose (stated or inferred) citation really entails them."""
    if not checks:
        return 0.0
    return float(np.mean([c.cited_support >= threshold for c in checks]))
