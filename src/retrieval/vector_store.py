"""FAISS vector store over chunks.jsonl.

Run:
    python -m src.retrieval.vector_store --build
    python -m src.retrieval.vector_store --query "your question"
"""
import argparse
from pathlib import Path

import faiss
import numpy as np

from src.utils.helpers import ROOT, load_config, load_jsonl


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[dict]):
        self.index = index
        self.chunks = chunks

    # ---- build / save / load ----
    @classmethod
    def build(cls, chunks: list[dict], embedder) -> "VectorStore":
        if not chunks:
            raise ValueError("No chunks to index. Run Stage 2 first.")
        vecs = embedder.encode([c["text"] for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])  # exact cosine search
        index.add(vecs)
        return cls(index, chunks)

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

    @classmethod
    def load(cls, index_path: Path, chunks_path: Path) -> "VectorStore":
        index = faiss.read_index(str(index_path))
        chunks = load_jsonl(chunks_path)
        if index.ntotal != len(chunks):
            raise ValueError("Index and chunks.jsonl are out of sync. Rebuild with --build.")
        return cls(index, chunks)

    # ---- search ----
    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vec.reshape(1, -1), k)
        results = []
        for rank, (score, i) in enumerate(zip(scores[0], ids[0]), start=1):
            if i == -1:
                continue
            hit = dict(self.chunks[i])
            hit["score"] = float(score)
            hit["rank"] = rank
            results.append(hit)
        return results

    def search_text(self, query: str, embedder, k: int = 5) -> list[dict]:
        return self.search(embedder.encode([query])[0], k)


def _paths(cfg: dict) -> tuple[Path, Path]:
    return ROOT / cfg["paths"]["index_file"], ROOT / cfg["paths"]["chunks_file"]


def main() -> None:
    from src.retrieval.embedder import Embedder

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="embed chunks and save index")
    ap.add_argument("--query", type=str, help="search the saved index")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    rc = cfg["retrieval"]
    index_path, chunks_path = _paths(cfg)
    embedder = Embedder(rc["embedding_model"], rc["batch_size"])

    if args.build:
        chunks = load_jsonl(chunks_path)
        store = VectorStore.build(chunks, embedder)
        store.save(index_path)
        print(f"Indexed {len(chunks)} chunks -> {index_path}")

    if args.query:
        store = VectorStore.load(index_path, chunks_path)
        for hit in store.search_text(args.query, embedder, args.k or rc["top_k"]):
            preview = hit["text"][:160].replace("\n", " ")
            print(f"[{hit['rank']}] {hit['score']:.3f}  {hit['chunk_id']}\n    {preview}...")

    if not args.build and not args.query:
        ap.print_help()


if __name__ == "__main__":
    main()
