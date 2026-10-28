"""Sentence-level support and citation checks against the passages the generator saw."""
import numpy as np

from src.generation.citations import parse_citations, strip_citations
from src.verification.schemas import SentenceCheck
from src.verification.sentences import passage_windows, split_sentences


def check_answer(raw_answer: str, passages: list[dict], nli, window_sentences: int = 2) -> list[SentenceCheck]:
    """For each answer sentence:
      support        best P(entail) over all passages (best short window within each passage)
      cited_support  best P(entail) over the passages the model cited
    If the model cited nothing, the best-supporting passage is attributed instead (citation_inferred=True).
    """
    sentences = split_sentences(raw_answer)
    clean = [strip_citations(s) for s in sentences]
    keep = [i for i, c in enumerate(clean) if c]
    if not keep or not passages:
        return []

    # Score each answer sentence against short windows of every passage, then keep the best window
    # per passage. (A 150-word premise makes small NLI models say "neutral" even for copied text.)
    windows: list[str] = []
    owner: list[int] = []                       # which passage each window came from
    for j, p in enumerate(passages):
        w = passage_windows(p["text"], window_sentences) or [p["text"]]
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
