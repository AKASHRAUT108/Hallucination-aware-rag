"""Cross-encoder reranker: re-scores (query, passage) pairs jointly, which is
more accurate than the bi-encoder similarity used for first-stage retrieval."""
import math
from typing import Callable, Optional


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 scorer: Optional[Callable[[list[tuple[str, str]]], list[float]]] = None):
        """`scorer` lets tests inject a fake scoring function (no model download)."""
        self.model_name = model_name
        self._scorer = scorer
        self._model = None

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._scorer is not None:
            return list(self._scorer(pairs))
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return [float(s) for s in self._model.predict(pairs)]

    def rerank(self, query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
        if not hits:
            return []
        scores = self._score([(query, h["text"]) for h in hits])
        out = []
        for h, s in zip(hits, scores):
            h = dict(h)
            h["dense_score"] = h.pop("score", None)   # keep first-stage score
            h["rerank_score"] = s                      # raw logit
            h["rerank_prob"] = _sigmoid(s)             # 0-1, used later for confidence
            out.append(h)
        out.sort(key=lambda h: h["rerank_score"], reverse=True)
        for rank, h in enumerate(out[:top_n], start=1):
            h["rank"] = rank
        return out[:top_n]
