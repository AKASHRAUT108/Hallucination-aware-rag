import csv

from src.evaluation.make_report import counts, load_rows, render_report, wilson


def row(i, answerable, refused, conf, contains=1.0, answer="x"):
    return {"id": f"q{i}", "question": f"Question {i}?", "answer": answer, "answerable": answerable,
            "generator_refused": refused, "confidence": conf, "em": 0.0, "f1": 0.0, "contains": contains,
            "faithfulness": None if refused else 0.9, "unsupported_claim_rate": None if refused else 0.0,
            "citation_accuracy": None}


ROWS = [row(1, True, False, 0.95), row(2, True, False, 0.95), row(3, True, False, 0.30),     # one correct answer lost
        row(4, False, True, 0.0), row(5, False, False, 0.20), row(6, False, False, 0.90)]   # one caught, one leaked


def test_wilson_interval_matches_known_values():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.31
    lo, hi = wilson(5, 10)
    assert 0.23 < lo < 0.25 and 0.75 < hi < 0.77
    assert wilson(0, 0)[0] != wilson(0, 0)[0]                  # nan for n=0


def test_counts_separate_caught_leaked_and_lost():
    c = counts(ROWS, 0.6)
    assert (c["n"], c["n_ans"], c["n_unans"]) == (6, 3, 3)
    assert (c["base_leak"], c["prop_leak"]) == (2, 1)          # q5 and q6 answered by the baseline; q6 still passes
    assert (c["base_lost"], c["prop_lost"]) == (0, 1)
    assert [r["id"] for r in c["caught"]] == ["q5"]
    assert [r["id"] for r in c["leaked"]] == ["q6"]
    assert [r["id"] for r in c["lost"]] == ["q3"]


def test_report_contains_tables_lists_and_limitations():
    text = render_report(ROWS, ROWS[:3], 0.6, {"generator": "gen", "verifier": "ver", "window": 1})
    assert "Test half" in text and "Tuning half" in text and "## Limitations" in text
    assert "1/3 (95% CI" in text and "2/3 (95% CI" in text
    assert "q5 (conf 0.20)" in text and "q6 (conf 0.90)" in text and "q3 (conf 0.30)" in text
    assert "`ver`" in text and "window 1" in text


def test_load_rows_handles_empty_cells_and_booleans(tmp_path):
    path = tmp_path / "per_question_x.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "answerable", "answer", "generator_refused", "confidence",
                                          "em", "f1", "contains", "faithfulness", "unsupported_claim_rate", "citation_accuracy"])
        w.writeheader()
        w.writerow({"id": "q1", "question": "Q?", "answerable": "False", "answer": "I don't know.", "generator_refused": "True",
                    "confidence": "0.0", "em": "0.0", "f1": "0.0", "contains": "0.0", "faithfulness": "", "unsupported_claim_rate": "", "citation_accuracy": ""})
    r = load_rows(path)[0]
    assert r["answerable"] is False and r["generator_refused"] is True and r["faithfulness"] is None


def test_excluding_a_mislabeled_question_changes_counts_and_is_disclosed():
    from src.evaluation.make_report import drop_questions
    rows = drop_questions(ROWS, {"q6"})
    assert len(rows) == 5 and all(r["id"] != "q6" for r in rows)
    c = counts(rows, 0.6)
    assert (c["n_unans"], c["prop_leak"]) == (2, 0)
    text = render_report(rows, [], 0.6, {"generator": "g", "verifier": "v", "window": 1}, excluded={"q6"})
    assert "Excluded after review: q6" in text
