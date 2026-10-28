"""Combine retrieval, entailment and citation signals into one confidence score in [0, 1]."""
import numpy as np

from src.verification.citation_checker import citation_score
from src.verification.schemas import SentenceCheck, VerificationResult


def retrieval_signal(passages: list[dict]) -> float:
    """Relevance of the top passage. Uses the reranker probability when present, else dense similarity."""
    if not passages:
        return 0.0
    top = passages[0]
    if top.get("rerank_prob") is not None:
        return float(top["rerank_prob"])
    return float(min(1.0, max(0.0, top.get("dense_score", 0.0))))


def entailment_signal(checks: list[SentenceCheck], aggregation: str = "mean") -> float:
    if not checks:
        return 0.0
    vals = [c.support for c in checks]
    return float(min(vals) if aggregation == "min" else np.mean(vals))


def combine(retrieval: float, entailment: float, citation: float, weights: dict) -> float:
    total = sum(weights.values()) or 1.0
    score = (weights["retrieval"] * retrieval + weights["entailment"] * entailment
             + weights["citation"] * citation) / total
    return float(min(1.0, max(0.0, score)))


def build_verification(checks: list[SentenceCheck], passages: list[dict], weights: dict,
                       aggregation: str = "mean", citation_threshold: float = 0.5) -> VerificationResult:
    r = retrieval_signal(passages)
    e = entailment_signal(checks, aggregation)
    c = citation_score(checks, citation_threshold)
    return VerificationResult(sentences=checks, retrieval_score=r, entailment_score=e,
                              citation_score=c, confidence=combine(r, e, c, weights))
