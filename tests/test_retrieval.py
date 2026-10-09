"""Retrieval tests use a tiny fake embedder so they run offline and fast."""
import hashlib

import numpy as np

from src.retrieval.vector_store import VectorStore


class FakeEmbedder:
    """Hashed bag-of-words, L2-normalised. Similar words -> similar vectors."""
    dim = 256

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in t.lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
                out[r, h] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


CHUNKS = [
    {"chunk_id": "a::p1::c0", "source": "a.txt", "page": 1,
     "text": "attendance policy requires seventy five percent presence in classes"},
    {"chunk_id": "b::p1::c0", "source": "b.txt", "page": 1,
     "text": "hostel fees must be paid before the start of every semester"},
    {"chunk_id": "c::p1::c0", "source": "c.txt", "page": 1,
     "text": "the library remains open until midnight during examination weeks"},
]


def test_top_hit_is_relevant():
    store = VectorStore.build(CHUNKS, FakeEmbedder())
    hits = store.search_text("what is the attendance policy", FakeEmbedder(), k=2)
    assert hits[0]["chunk_id"] == "a::p1::c0"
    assert hits[0]["rank"] == 1 and len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]


def test_k_larger_than_corpus():
    store = VectorStore.build(CHUNKS, FakeEmbedder())
    assert len(store.search_text("fees", FakeEmbedder(), k=10)) == 3


def test_save_and_load_roundtrip(tmp_path):
    from src.utils.helpers import save_jsonl
    store = VectorStore.build(CHUNKS, FakeEmbedder())
    store.save(tmp_path / "i.faiss")
    save_jsonl(CHUNKS, tmp_path / "c.jsonl")
    loaded = VectorStore.load(tmp_path / "i.faiss", tmp_path / "c.jsonl")
    hits = loaded.search_text("hostel fees semester", FakeEmbedder(), k=1)
    assert hits[0]["chunk_id"] == "b::p1::c0"
