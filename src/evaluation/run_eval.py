"""Baseline RAG vs the hallucination-aware pipeline on data/eval/eval.jsonl.

Each question is generated and verified ONCE. The confidence threshold is applied afterwards, so
baseline and proposed are compared on identical answers and the threshold sweep costs nothing extra.

Run:  python -m src.evaluation.run_eval            (add --limit 5 for a quick trial)
"""
import argparse
import csv
import json
import math

import numpy as np

from src.evaluation.hallucination_metrics import score_answer
from src.evaluation.qa_metrics import contains_answer, exact_match, f1_score
from src.generation.citations import strip_citations
from src.utils.helpers import ROOT, load_config


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def collect_rows(rag, eval_nli, data: list[dict], threshold: float = 0.5, window_sentences: int = 2,
                 verbose: bool = True) -> list[dict]:
    """Run the pipeline once per question and score every answer."""
    rows = []
    for i, item in enumerate(data, start=1):
        res = rag.diagnose(item["question"])
        golds = item.get("answers", [])
        pred = strip_citations(res.raw_answer)
        row = {"id": item["id"], "question": item["question"], "answerable": bool(item.get("answerable", True)),
               "answer": res.raw_answer, "generator_refused": res.abstained, "confidence": res.confidence or 0.0,
               "em": exact_match(pred, golds), "f1": f1_score(pred, golds), "contains": contains_answer(pred, golds),
               "faithfulness": None, "unsupported_claim_rate": None, "citation_accuracy": None}
        if not res.abstained:
            row.update(score_answer(res.raw_answer, res.passages, eval_nli, threshold, window_sentences))
        rows.append(row)
        if verbose:
            print(f"[{i}/{len(data)}] conf={row['confidence']:.2f} refused={row['generator_refused']} | {item['question'][:60]}")
    return rows


def answered_mask(rows: list[dict], tau: float | None) -> list[bool]:
    """tau=None -> baseline (answers unless the generator itself refused). Otherwise also require confidence >= tau."""
    return [(not r["generator_refused"]) and (tau is None or r["confidence"] >= tau) for r in rows]


def summarise(rows: list[dict], mask: list[bool]) -> dict:
    answered = [r for r, a in zip(rows, mask) if a]
    answerable = [(r, a) for r, a in zip(rows, mask) if r["answerable"]]
    unanswerable = [(r, a) for r, a in zip(rows, mask) if not r["answerable"]]
    return {
        "coverage": sum(mask) / len(rows) if rows else float("nan"),
        # abstaining on an answerable question scores 0
        "EM": mean([r["em"] if a else 0.0 for r, a in answerable]),
        "F1": mean([r["f1"] if a else 0.0 for r, a in answerable]),
        "contains_gold": mean([r["contains"] if a else 0.0 for r, a in answerable]),
        "faithfulness": mean([r["faithfulness"] for r in answered]),
        "unsupported_claim_rate": mean([r["unsupported_claim_rate"] for r in answered]),
        "citation_accuracy": mean([r["citation_accuracy"] for r in answered]),
        "answered_unanswerable_rate": mean([1.0 if a else 0.0 for _, a in unanswerable]),
        "false_abstention_rate": mean([0.0 if a else 1.0 for _, a in answerable]),
    }


def sweep(rows: list[dict], start: float, stop: float, step: float) -> list[dict]:
    out = []
    for t in np.arange(start, stop + 1e-9, step):
        out.append({"threshold": round(float(t), 3), **summarise(rows, answered_mask(rows, float(t)))})
    return out


def write_csv(path, dict_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        w.writeheader()
        w.writerows(dict_rows)


def main() -> None:
    from src.pipeline.hallucination_aware_rag import HallucinationAwareRAG
    from src.verification.entailment import NLIScorer

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N questions")
    args = ap.parse_args()
    cfg = load_config()
    ev = cfg["evaluation"]
    tau = cfg["verification"]["confidence_threshold"]
    window = cfg["verification"].get("window_sentences", 2)

    data = read_jsonl(ROOT / ev["eval_file"])
    if args.limit:
        data = data[: args.limit]
    rag = HallucinationAwareRAG.from_config()
    # A different NLI model than the verifier's, so the system is not graded by the model it optimises.
    eval_nli = NLIScorer(ev["nli_model"], cfg["verification"].get("batch_size", 16))

    rows = collect_rows(rag, eval_nli, data, ev["unsupported_threshold"], window)
    out = ROOT / ev["results_dir"]
    write_csv(out / "tables" / "per_question.csv", rows)

    base = summarise(rows, answered_mask(rows, None))
    prop = summarise(rows, answered_mask(rows, tau))
    write_csv(out / "tables" / "comparison.csv",
              [{"metric": m, "baseline": round(base[m], 4), f"proposed_tau_{tau}": round(prop[m], 4)} for m in base])
    sw = sweep(rows, ev["sweep_start"], ev["sweep_stop"], ev["sweep_step"])
    write_csv(out / "tables" / "threshold_sweep.csv", sw)

    print(f"\n=== Baseline vs proposed (tau={tau}, {len(rows)} questions) ===")
    print(f"{'metric':30s}{'baseline':>10s}{'proposed':>10s}")
    for m in base:
        print(f"{m:30s}{base[m]:10.3f}{prop[m]:10.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([s["coverage"] for s in sw], [s["unsupported_claim_rate"] for s in sw], marker="o", label="proposed (threshold sweep)")
        ax.scatter([base["coverage"]], [base["unsupported_claim_rate"]], color="red", zorder=3, label="baseline")
        ax.set_xlabel("Answer coverage")
        ax.set_ylabel("Unsupported-claim rate")
        ax.set_title("Coverage vs hallucination")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        (out / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "figures" / "coverage_vs_hallucination.png", dpi=200)
    except ImportError:
        print("matplotlib not installed; skipped the figure (pip install matplotlib)")
    print(f"\nSaved tables and figure under {out}")


if __name__ == "__main__":
    main()
