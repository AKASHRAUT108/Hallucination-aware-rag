"""Full framework: retrieve -> rerank -> generate with citations -> verify -> answer or abstain.

Run:  python -m src.pipeline.hallucination_aware_rag --query "your question"
"""
import argparse

from src.generation.citations import is_refusal, resolve_citations
from src.generation.generator import Generator
from src.pipeline.baseline_rag import DEFAULT_ABSTAIN, DEFAULT_REFUSALS
from src.pipeline.result import RAGResult
from src.utils.helpers import load_config
from src.verification.citation_checker import check_answer
from src.verification.confidence import build_verification
from src.verification.entailment import NLIScorer


class HallucinationAwareRAG:
    def __init__(self, retriever, generator: Generator, nli, settings: dict,
                 refusal_phrases=None, abstain_message=DEFAULT_ABSTAIN):
        self.retriever, self.generator, self.nli = retriever, generator, nli
        self.weights = settings["weights"]
        self.aggregation = settings.get("entailment_aggregation", "mean")
        self.citation_threshold = settings.get("citation_threshold", 0.5)
        self.threshold = settings["confidence_threshold"]
        self.window_sentences = settings.get("window_sentences", 2)
        self.refusal_phrases = refusal_phrases or DEFAULT_REFUSALS
        self.abstain_message = abstain_message

    @classmethod
    def from_config(cls) -> "HallucinationAwareRAG":
        from src.retrieval.retriever import Retriever
        cfg = load_config()
        p = cfg.get("pipeline", {})
        return cls(Retriever.from_config(), Generator.from_config(), NLIScorer.from_config(),
                   cfg["verification"], p.get("refusal_phrases"), p.get("abstain_message", DEFAULT_ABSTAIN))

    def diagnose(self, question: str) -> RAGResult:
        """Run everything and attach the confidence WITHOUT applying the threshold.
        Stage 7 uses this to sweep thresholds without regenerating answers."""
        passages = self.retriever.retrieve(question)[: self.generator.max_passages]
        raw = self.generator.generate(question, passages)
        if is_refusal(raw, self.refusal_phrases):
            return RAGResult(question, self.abstain_message, raw, passages, [], abstained=True, confidence=0.0)
        checks = check_answer(raw, passages, self.nli, self.window_sentences)
        ver = build_verification(checks, passages, self.weights, self.aggregation, self.citation_threshold)
        return RAGResult(question, raw, raw, passages, resolve_citations(raw, passages),
                         abstained=False, confidence=ver.confidence, verification=ver)

    def answer(self, question: str, threshold: float | None = None) -> RAGResult:
        res = self.diagnose(question)
        t = self.threshold if threshold is None else threshold
        if not res.abstained and res.confidence < t:
            res.abstained, res.answer = True, self.abstain_message
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="override verification.confidence_threshold")
    args = ap.parse_args()
    rag = HallucinationAwareRAG.from_config()
    res = rag.answer(args.query, args.threshold)
    t = rag.threshold if args.threshold is None else args.threshold
    print(res.answer)
    print(f"confidence: {res.confidence:.2f} (threshold {t})  ->  {'ABSTAINED' if res.abstained else 'answered'}")
    v = res.verification
    if v:
        print(f"signals: retrieval={v.retrieval_score:.2f} entailment={v.entailment_score:.2f} citation={v.citation_score:.2f}")
        for s in v.sentences:
            cite = s.cited_ranks or f"inferred [{s.best_rank}]"
            print(f"  support={s.support:.2f} cited_support={s.cited_support:.2f} cite={cite} | {s.text}")


if __name__ == "__main__":
    main()
