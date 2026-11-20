"""Turn evaluation CSVs into report-ready markdown: tables, confidence intervals and the exact questions
the verifier caught, lost or missed. Nothing is typed by hand, so the report cannot drift from the data.

Run:  python -m src.evaluation.make_report --test test --tune tune_v3 --out docs/results_galgotias.md
(reads results/tables/per_question_<tag>.csv; the threshold and model names come from config.yaml)
"""
import argparse
import csv
import math
from pathlib import Path

from src.evaluation.run_eval import answered_mask, summarise


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k/n. Honest about how little 10-30 questions can show."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _num(x):
    return None if x in (None, "", "nan") else float(x)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["id"], "question": r["question"], "answer": r["answer"],
                "answerable": r["answerable"] == "True", "generator_refused": r["generator_refused"] == "True",
                "confidence": float(r["confidence"] or 0.0),
                "em": _num(r["em"]) or 0.0, "f1": _num(r["f1"]) or 0.0, "contains": _num(r["contains"]) or 0.0,
                "faithfulness": _num(r.get("faithfulness")), "unsupported_claim_rate": _num(r.get("unsupported_claim_rate")),
                "citation_accuracy": _num(r.get("citation_accuracy")),
            })
    return rows


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.2f}"


def _ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} (95% CI {lo:.2f}-{hi:.2f})" if n else "n/a"


def _short(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def counts(rows: list[dict], tau: float) -> dict:
    base, prop = answered_mask(rows, None), answered_mask(rows, tau)
    unans = [i for i, r in enumerate(rows) if not r["answerable"]]
    ans = [i for i, r in enumerate(rows) if r["answerable"]]
    return {
        "n": len(rows), "n_ans": len(ans), "n_unans": len(unans),
        "base_leak": sum(base[i] for i in unans), "prop_leak": sum(prop[i] for i in unans),
        "base_lost": sum(not base[i] for i in ans), "prop_lost": sum(not prop[i] for i in ans),
        "caught": [rows[i] for i in unans if base[i] and not prop[i]],
        "leaked": [rows[i] for i in unans if prop[i]],
        "lost": [rows[i] for i in ans if base[i] and not prop[i]],
        "base": summarise(rows, base), "prop": summarise(rows, prop),
    }


def section(title: str, rows: list[dict], tau: float) -> str:
    c = counts(rows, tau)
    b, p = c["base"], c["prop"]
    out = [f"### {title}", "",
           f"{c['n']} questions: {c['n_ans']} answerable, {c['n_unans']} unanswerable. Threshold {tau}.", "",
           "| Measure | Baseline (no verifier) | With verifier |", "|---|---|---|",
           f"| Unanswerable questions answered | {_ci(c['base_leak'], c['n_unans'])} | {_ci(c['prop_leak'], c['n_unans'])} |",
           f"| Answerable questions not answered | {_ci(c['base_lost'], c['n_ans'])} | {_ci(c['prop_lost'], c['n_ans'])} |",
           f"| Coverage | {_pct(b['coverage'])} | {_pct(p['coverage'])} |",
           f"| Answer contains gold answer | {_pct(b['contains_gold'])} | {_pct(p['contains_gold'])} |",
           f"| Faithfulness (answered only) | {_pct(b['faithfulness'])} | {_pct(p['faithfulness'])} |",
           f"| Unsupported-claim rate (answered only) | {_pct(b['unsupported_claim_rate'])} | {_pct(p['unsupported_claim_rate'])} |",
           f"| Citation accuracy | {_pct(b['citation_accuracy'])} | {_pct(p['citation_accuracy'])} |", ""]
    for label, key in (("Unanswerable questions the verifier withheld", "caught"),
                       ("Unanswerable questions that still got through", "leaked"),
                       ("Correct answers the verifier withheld", "lost")):
        out.append(f"**{label} ({len(c[key])}):**")
        out.extend(f"- {r['id']} (conf {r['confidence']:.2f}): {_short(r['question'])}  ->  {_short(r['answer'], 70)}"
                   for r in c[key]) if c[key] else out.append("- none")
        out.append("")
    return "\n".join(out)


def drop_questions(rows: list[dict], ids: set[str]) -> list[dict]:
    return [r for r in rows if r["id"] not in ids]


def render_report(test_rows, tune_rows, tau: float, meta: dict, excluded=()) -> str:
    parts = ["# Results on real documents (Galgotias University, Part B: Examination Rules & Regulations)", "",
             f"Generator: `{meta.get('generator', '?')}`. Verifier: `{meta.get('verifier', '?')}`, "
             f"window {meta.get('window', '?')}, confidence threshold {tau}. "
             "Settings were chosen on the tuning half; the test half was run once.", ""]
    if excluded:
        parts += [f"**Excluded after review: {', '.join(sorted(excluded))}** (the question label was found to be wrong). "
                  "Rerun without `--exclude` for the unadjusted numbers.", ""]
    if test_rows:
        parts += [section("Test half (the reported result)", test_rows, tau), ""]
    if tune_rows:
        parts += [section("Tuning half (used to choose settings; not a result)", tune_rows, tau), ""]
    parts += ["## Limitations", "",
              "- One document in one domain, and a question set written by the author; intervals above show how little "
              "20-30 questions can establish.",
              "- The verifier checks that a sentence is supported by the retrieved passages, not that it answers the question "
              "asked, and it depends on the retriever ranking the right passage first.",
              "- NLI models make paraphrase errors (for example, 'minimum 7' versus '7 or more'), which withhold some "
              "correct answers.", ""]
    return "\n".join(parts)


def main() -> None:
    from src.utils.helpers import ROOT, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=None, help="tag of the test run, e.g. test")
    ap.add_argument("--tune", default=None, help="tag of the final tuning run, e.g. tune_v3")
    ap.add_argument("--out", default="docs/results_galgotias.md")
    ap.add_argument("--exclude", default="", help="comma-separated question ids found to be mislabeled, e.g. q44")
    args = ap.parse_args()
    cfg = load_config()
    tables = ROOT / cfg["evaluation"]["results_dir"] / "tables"
    test_rows = load_rows(tables / f"per_question_{args.test}.csv") if args.test else []
    tune_rows = load_rows(tables / f"per_question_{args.tune}.csv") if args.tune else []
    excluded = {x.strip() for x in args.exclude.split(",") if x.strip()}
    test_rows, tune_rows = drop_questions(test_rows, excluded), drop_questions(tune_rows, excluded)
    meta = {"generator": cfg["generation"]["model"], "verifier": cfg["verification"]["nli_model"],
            "window": cfg["verification"].get("window_sentences", 2)}
    text = render_report(test_rows, tune_rows, cfg["verification"]["confidence_threshold"], meta, excluded)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
