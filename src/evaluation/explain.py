"""Show exactly what the verifier saw for one question: for each answer sentence, the best-matching
window of every passage and its entailment probability. Use it to find out WHY an answer was withheld.

Run:  python -m src.evaluation.explain "your question" [--window N]
Uses the generation cache, so questions you already evaluated are instant.
"""
import argparse

from src.generation.citations import parse_citations
from src.verification.citation_checker import clean_claim, is_meta, normalize_numbers
from src.verification.sentences import passage_windows, split_sentences


def explain_answer(raw_answer: str, passages: list[dict], nli, window_sentences: int = 2) -> list[dict]:
    sentences = split_sentences(raw_answer)
    rows = []
    for sent in sentences:
        claim = clean_claim(sent)
        if not claim or is_meta(sent):
            continue
        per_passage = []
        for p in passages:
            windows = [normalize_numbers(w) for w in (passage_windows(p["text"], window_sentences) or [p["text"]])]
            probs = nli.entailment_probs([(w, claim) for w in windows])
            best = int(probs.argmax())
            per_passage.append({"rank": p["rank"], "chunk_id": p["chunk_id"], "p_entail": float(probs[best]),
                                "window": windows[best], "words": len(p["text"].split())})
        rows.append({"sentence": sent, "claim": claim, "cited": parse_citations(sent), "passages": per_passage})
    return rows


def main() -> None:
    from src.evaluation.run_eval import CachedGenerator
    from src.pipeline.hallucination_aware_rag import HallucinationAwareRAG
    from src.utils.helpers import ROOT, load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--window", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    rag = HallucinationAwareRAG.from_config()
    rag.generator = CachedGenerator(rag.generator, ROOT / cfg["evaluation"]["results_dir"] / "generation_cache.json")
    if args.window is not None:
        rag.window_sentences = args.window

    passages = rag.retriever.retrieve(args.question)[: rag.generator.max_passages]
    raw = rag.generator.generate(args.question, passages)
    print("ANSWER:", raw)
    print(f"\\nPASSAGES SHOWN TO THE MODEL ({len(passages)}):")
    for p in passages:
        print(f"  [{p['rank']}] {p['chunk_id']} rerank_p={p.get('rerank_prob', 0):.2f} words={len(p['text'].split())}")
        print("      " + p["text"][:260].replace("\\n", " ") + " ...")
    print(f"\\nVERIFIER (window={rag.window_sentences}):")
    for row in explain_answer(raw, passages, rag.nli, rag.window_sentences):
        print(f"  claim: {row['claim']}   (cited: {row['cited'] or 'nothing'})")
        for pp in row["passages"]:
            print(f"    [{pp['rank']}] P(entail)={pp['p_entail']:.2f}  best window: {pp['window'][:170]}")
    res = rag.diagnose(args.question)
    v = res.verification
    if v is None:
        print("\nThe generator refused to answer, so there is nothing to verify.")
        return
    print(f"\nsignals: retrieval={v.retrieval_score:.2f} entailment={v.entailment_score:.2f} "
          f"citation={v.citation_score:.2f} -> confidence {v.confidence:.2f} (threshold {rag.threshold})")


if __name__ == "__main__":
    main()
