"""NLI scoring: how likely is it that a passage (premise) entails an answer sentence (hypothesis)?"""
import numpy as np

from src.utils.helpers import load_config


class NLIScorer:
    def __init__(self, model_name: str, batch_size: int = 16):
        self.model_name, self.batch_size = model_name, batch_size
        self._model = None
        self._entail_idx = 1       # cross-encoder/nli-* order: contradiction, entailment, neutral

    @classmethod
    def from_config(cls) -> "NLIScorer":
        v = load_config()["verification"]
        return cls(v["nli_model"], v.get("batch_size", 16))

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            inner = getattr(self._model, "model", self._model)
            id2label = getattr(getattr(inner, "config", None), "id2label", None) or {}
            for i, label in id2label.items():
                if "entail" in str(label).lower():
                    self._entail_idx = int(i)
        return self._model

    def entailment_probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """pairs = [(premise, hypothesis), ...] -> P(entailment) for each pair."""
        if not pairs:
            return np.zeros(0)
        out = np.asarray(self._load().predict(pairs, batch_size=self.batch_size), dtype="float64")
        if out.ndim == 1:                                   # single-logit model
            return 1.0 / (1.0 + np.exp(-out))
        already_probs = out.min() >= 0 and np.allclose(out.sum(axis=1), 1.0, atol=1e-3)
        if not already_probs:                               # logits -> softmax
            e = np.exp(out - out.max(axis=1, keepdims=True))
            out = e / e.sum(axis=1, keepdims=True)
        return out[:, self._entail_idx]
