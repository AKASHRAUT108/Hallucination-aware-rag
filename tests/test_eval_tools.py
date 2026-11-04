import json
import sys

from src.evaluation.check_eval import check_rows, corpus_text_from_chunks
from src.evaluation.make_split import split_rows


def q(i, answerable=True, typ="", answers=None):
    return {"id": f"q{i:02d}", "question": f"Question number {i}?", "answerable": answerable,
            "answers": answers if answers is not None else (["a"] if answerable else []), "type": typ}


ROWS = [q(i) for i in range(1, 11)] + [q(i, False, "plain") for i in range(11, 16)] + \
       [q(i, False, "trap") for i in range(16, 26)]


def test_split_is_stratified_disjoint_complete_and_deterministic():
    tune, test = split_rows(ROWS, 0.5, seed=1)
    assert len(tune) + len(test) == len(ROWS)
    assert not ({r["id"] for r in tune} & {r["id"] for r in test})
    for part in (tune, test):
        assert {r.get("type", "") for r in part} == {"", "plain", "trap"}
        assert 4 <= sum(1 for r in part if r["type"] == "trap") <= 6
    assert split_rows(ROWS, 0.5, seed=1) == (tune, test)
    assert split_rows(ROWS, 0.5, seed=2)[1] != test            # a different seed gives a different split


def test_split_keeps_one_of_each_kind_in_both_halves_even_for_tiny_groups():
    rows = [q(1), q(2), q(3, False, "trap"), q(4, False, "trap")]
    tune, test = split_rows(rows, 0.5, seed=0)
    assert sum(1 for r in tune if r["type"] == "trap") == 1 and sum(1 for r in test if r["type"] == "trap") == 1


def test_check_rows_catches_format_errors():
    rows = [q(1), q(1), {"id": "q3", "question": "", "answerable": True, "answers": ["x"]},
            q(4, answers=[]), q(5, False, answers=["oops"]), {"id": "q6", "question": "Q6?", "answerable": "yes", "answers": []}]
    errors, _ = check_rows(rows)
    joined = " | ".join(errors)
    assert "duplicate id" in joined and "empty question" in joined and "no gold answer" in joined
    assert "should have an empty answers list" in joined and "must be true or false" in joined


def test_gold_answers_are_checked_against_the_documents():
    corpus = "Students must maintain at least 75 percent attendance. The CGPA must be at least 5.0 to progress."
    rows = [dict(q(1), answers=["75 percent"]), dict(q(2), answers=["5.0"]), dict(q(3), answers=["80 percent"])]
    _, warnings = check_rows(rows, corpus)
    assert any("q03" in w for w in warnings) and not any("q01" in w or "q02" in w for w in warnings)


def test_size_and_trap_warnings():
    _, warnings = check_rows(ROWS[:12])
    assert any("only 12 questions" in w for w in warnings) and any("unanswerable" in w for w in warnings)
    _, ok_warnings = check_rows(ROWS)
    assert not any("trap" in w for w in ok_warnings)


def test_corpus_loader(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps({"text": "alpha"}) + "\n" + json.dumps({"text": "beta"}) + "\n", encoding="utf-8")
    assert corpus_text_from_chunks(path) == "alpha beta" and corpus_text_from_chunks(tmp_path / "missing") is None


def test_run_eval_file_overrides_only_the_eval_file(monkeypatch):
    import src.evaluation.run_eval as run_eval
    from src.evaluation import run_eval_file
    seen = {}
    monkeypatch.setattr(run_eval, "main", lambda: seen.update(cfg=run_eval.load_config(), argv=list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["prog", "data/eval/tune.jsonl", "--tag", "tune"])
    run_eval_file.main()
    assert seen["cfg"]["evaluation"]["eval_file"] == "data/eval/tune.jsonl"
    assert seen["argv"] == ["prog", "--tag", "tune"]
