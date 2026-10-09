from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from tests.test_retrieval import CHUNKS, FakeEmbedder


def overlap_scorer(pairs):
    """Fake cross-encoder: counts shared words between query and passage."""
    return [float(len(set(q.lower().split()) & set(p.lower().split()))) - 1 for q, p in pairs]


def test_rerank_orders_by_score_and_keeps_fields():
    hits = [
        {"chunk_id": "x", "text": "nothing relevant here", "score": 0.9, "rank": 1},
        {"chunk_id": "y", "text": "hostel fees due each semester", "score": 0.1, "rank": 2},
    ]
    out = Reranker(scorer=overlap_scorer).rerank("hostel fees semester", hits, top_n=2)
    assert out[0]["chunk_id"] == "y" and out[0]["rank"] == 1
    assert out[0]["dense_score"] == 0.1
    assert 0.0 < out[0]["rerank_prob"] < 1.0


def test_top_n_and_empty():
    r = Reranker(scorer=overlap_scorer)
    assert r.rerank("q", [], 3) == []
    hits = [{"chunk_id": str(i), "text": "a b c", "score": 0.5} for i in range(5)]
    assert len(r.rerank("a", hits, top_n=2)) == 2


def test_retriever_end_to_end():
    store = VectorStore.build(CHUNKS, FakeEmbedder())
    ret = Retriever(store, FakeEmbedder(), Reranker(scorer=overlap_scorer), fetch_k=3, top_n=2)
    out = ret.retrieve("library open examination weeks")
    assert out[0]["chunk_id"] == "c::p1::c0" and len(out) == 2
