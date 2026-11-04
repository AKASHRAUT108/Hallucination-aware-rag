"""Validate an evaluation file before you trust any numbers from it.

Run:  python -m src.evaluation.check_eval [data/eval/eval.jsonl]

Checks the format, flags duplicate questions, verifies that every answerable question's gold answer actually
appears in the indexed documents, and warns when the set is too small or has too few unanswerable questions.
"""
import json
import sys
from pathlib import Path

from src.evaluation.qa_metrics import normalize


def check_rows(rows: list[dict], corpus_text: str | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    corpus = f" {normalize(corpus_text)} " if corpus_text else None

    for i, r in enumerate(rows, start=1):
        qid = str(r.get("id", f"row{i}"))
        if not r.get("id"):
            errors.append(f"row {i}: missing id")
        elif qid in seen_ids:
            errors.append(f"{qid}: duplicate id")
        seen_ids.add(qid)

        q = (r.get("question") or "").strip()
        if not q:
            errors.append(f"{qid}: empty question")
        elif normalize(q) in seen_questions:
            warnings.append(f"{qid}: duplicate question text")
        seen_questions.add(normalize(q))

        answerable = r.get("answerable")
        answers = r.get("answers")
        if not isinstance(answerable, bool):
            errors.append(f"{qid}: 'answerable' must be true or false")
            continue
        if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
            errors.append(f"{qid}: 'answers' must be a list of strings")
            continue
        if answerable and not [a for a in answers if a.strip()]:
            errors.append(f"{qid}: answerable question has no gold answer")
        if not answerable and answers:
            errors.append(f"{qid}: unanswerable question should have an empty answers list")
        if answerable and corpus is not None:
            found = [a for a in answers if a.strip() and f" {normalize(a)} " in corpus]
            if not found:
                warnings.append(f"{qid}: no gold answer appears verbatim in the documents "
                                f"({answers}); fine if the answer is a paraphrase, otherwise check for typos")

    n = len(rows)
    n_un = sum(1 for r in rows if r.get("answerable") is False)
    n_trap = sum(1 for r in rows if r.get("type") == "trap")
    if n < 50:
        warnings.append(f"only {n} questions; aim for 50 or more")
    if n_un < 20:
        warnings.append(f"only {n_un} unanswerable questions; aim for 20 or more")
    if n_un and n_trap < n_un // 2:
        warnings.append(f"only {n_trap} questions are marked \"type\": \"trap\"; "
                        "near-miss unanswerable questions are the ones that test the verifier")
    return errors, warnings


def corpus_text_from_chunks(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return " ".join(json.loads(line)["text"] for line in f if line.strip())


def main() -> None:
    from src.utils.helpers import ROOT, load_config
    cfg = load_config()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(cfg["evaluation"]["eval_file"])
    path = path if path.is_absolute() else ROOT / path
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    chunks = ROOT / cfg["paths"]["processed_dir"] / "chunks.jsonl"
    corpus = corpus_text_from_chunks(chunks)
    if corpus is None:
        print(f"note: {chunks} not found, so gold answers were not checked against the documents")
    errors, warnings = check_rows(rows, corpus)

    n_un = sum(1 for r in rows if r.get("answerable") is False)
    print(f"{path.name}: {len(rows)} questions ({len(rows) - n_un} answerable, {n_un} unanswerable, "
          f"{sum(1 for r in rows if r.get('type') == 'trap')} traps)")
    for w in warnings:
        print("  warning:", w)
    for e in errors:
        print("  ERROR:", e)
    print("OK" if not errors else f"{len(errors)} error(s): fix these before evaluating")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
