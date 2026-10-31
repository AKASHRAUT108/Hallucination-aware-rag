"""Hallucination metrics for one answer, judged by an NLI model that is NOT the verifier's model."""
from src.verification.citation_checker import check_answer


def score_answer(raw_answer: str, passages: list[dict], nli, threshold: float = 0.5,
                 window_sentences: int = 2) -> dict:
    """faithfulness            mean over answer sentences of the best P(entail) against the passages
       unsupported_claim_rate  share of sentences whose best P(entail) is below `threshold`
       citation_accuracy       share of sentences with an explicit [n] whose cited passage entails them
                               (None if the answer has no explicit citations)"""
    checks = check_answer(raw_answer, passages, nli, window_sentences)
    if not checks:
        return {"faithfulness": 0.0, "unsupported_claim_rate": 1.0, "citation_accuracy": None}
    explicit = [c.cited_support >= threshold for c in checks if not c.citation_inferred]
    return {
        "faithfulness": sum(c.support for c in checks) / len(checks),
        "unsupported_claim_rate": sum(c.support < threshold for c in checks) / len(checks),
        "citation_accuracy": (sum(explicit) / len(explicit)) if explicit else None,
    }
