"""Full retrieval step: dense search (fetch_k) -> rerank -> top_n passages.

Run:  python -m src.retrieval.retriever --query "your question"
"""
import argparse

from src.retrieval.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore
from src.utils.helpers import ROOT, load_config


class Retriever:
    def __init__(self, store: VectorStore, embedder, reranker: Reranker,
                 fetch_k: int = 20, top_n: int = 5):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.fetch_k, self.top_n = fetch_k, top_n

    @classmethod
    def from_config(cls) -> "Retriever":
        cfg = load_config()
        rc, rr = cfg["retrieval"], cfg["reranking"]
        store = VectorStore.load(ROOT / cfg["paths"]["index_file"],
                                 ROOT / cfg["paths"]["chunks_file"])
        return cls(store, Embedder(rc["embedding_model"], rc["batch_size"]),
                   Reranker(rr["model"]), rr["fetch_k"], rr["top_n"])

    def retrieve(self, query: str) -> list[dict]:
        candidates = self.store.search_text(query, self.embedder, self.fetch_k)
        return self.reranker.rerank(query, candidates, self.top_n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    args = ap.parse_args()
    for h in Retriever.from_config().retrieve(args.query):
        print(f"[{h['rank']}] rerank={h['rerank_score']:.2f} (p={h['rerank_prob']:.2f}) "
              f"dense={h['dense_score']:.3f}  {h['chunk_id']}")
        print(f"    {h['text'][:160]}...")


if __name__ == "__main__":
    main()
