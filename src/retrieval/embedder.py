"""Dense text embeddings via sentence-transformers."""
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # loaded on first use

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape (n, dim).
        Normalised vectors make inner product equal to cosine similarity."""
        vecs = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)
