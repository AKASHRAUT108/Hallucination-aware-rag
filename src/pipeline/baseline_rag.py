"""Baseline RAG: retrieve -> rerank -> generate with citations. No verification layer.

Run:  python -m src.pipeline.baseline_rag --query "your question"
"""
import argparse

from src.generation.citations import is_refusal, resolve_citations
from src.generation.generator import Generator
from src.pipeline.result import RAGResult
from src.utils.helpers import load_config

DEFAULT_REFUSALS = ["i don't know", "i do not know", "cannot answer", "can't answer", "not mentioned"]
DEFAULT_ABSTAIN = "I can't answer this reliably from the available documents."


class BaselineRAG:
    def __init__(self, retriever, generator: Generator, refusal_phrases=None, abstain_message=DEFAULT_ABSTAIN):
        self.retriever, self.generator = retriever, generator
        self.refusal_phrases = refusal_phrases or DEFAULT_REFUSALS
        self.abstain_message = abstain_message

    @classmethod
    def from_config(cls) -> "BaselineRAG":
        from src.retrieval.retriever import Retriever
        p = load_config().get("pipeline", {})
        return cls(Retriever.from_config(), Generator.from_config(),
                   p.get("refusal_phrases"), p.get("abstain_message", DEFAULT_ABSTAIN))

    def answer(self, question: str) -> RAGResult:
        passages = self.retriever.retrieve(question)[: self.generator.max_passages]
        raw = self.generator.generate(question, passages)
        # The baseline only abstains if the generator itself says it does not know.
        refused = is_refusal(raw, self.refusal_phrases)
        return RAGResult(question=question,
                         answer=self.abstain_message if refused else raw,
                         raw_answer=raw, passages=passages,
                         citations=[] if refused else resolve_citations(raw, passages),
                         abstained=refused)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    args = ap.parse_args()
    res = BaselineRAG.from_config().answer(args.query)
    print(res.answer)
    print("citations:", [f"[{c['rank']}] {c['chunk_id']}" for c in res.citations] or "none written by the model")
    for p in res.passages:
        print(f"  [{p['rank']}] {p['chunk_id']}")


if __name__ == "__main__":
    main()
