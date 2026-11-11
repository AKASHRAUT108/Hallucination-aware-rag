"""Stage 7 tests with fake pipeline/NLI, so nothing is downloaded."""
import math
import re

import numpy as np

from src.evaluation.hallucination_metrics import score_answer
from src.evaluation.qa_metrics import contains_answer, exact_match, f1_score, normalize
from src.evaluation.run_eval import CachedGenerator, answered_mask, collect_rows, summarise, sweep
from src.pipeline.result import RAGResult

WORDS = re.compile(r"[a-z0-9]+")
PASSAGES = [{"rank": 1, "chunk_id": "attendance::0", "text": "Condonation costs 500 rupees per course.", "rerank_prob": 0.9}]


class FakeNLI:
    def entailment_probs(self, pairs):
        out = []
        for prem, hyp in pairs:
            h, p = set(WORDS.findall(hyp.lower())), set(WORDS.findall(prem.lower()))
            out.append(len(h & p) / max(len(h), 1))
        return np.array(out)


class FakeRAG:
    """Returns canned (answer, confidence, generator_refused) per question text."""
    def __init__(self, table):
        self.table = table

    def diagnose(self, question):
        raw, conf, refused = self.table[question]
        return RAGResult(question, raw, raw, PASSAGES, [], abstained=refused, confidence=conf)


DATA = [
    {"id": "a", "question": "cost?", "answers": ["500 rupees"], "answerable": True},
    {"id": "b", "question": "lunch?", "answers": ["sandwich"], "answerable": True},
    {"id": "c", "question": "vc?", "answers": [], "answerable": False},
    {"id": "d", "question": "dress?", "answers": [], "answerable": False},
]
TABLE = {
    "cost?": ("Condonation costs 500 rupees per course [1].", 0.95, False),   # right and supported
    "lunch?": ("Pizza is served every Friday evening.", 0.2, False),            # wrong, unsupported, low confidence
    "vc?": ("The Vice Chancellor is Dr Rao.", 0.3, False),                      # made-up answer to unanswerable
    "dress?": ("I don't know.", 0.0, True),                                     # generator refuses
}


def test_qa_metrics():
    assert normalize("The A+ grade.") == "a+ grade" and normalize("CGPA of 5.0.") == "cgpa of 5.0"
    assert exact_match("A+", ["A+"]) == 1.0 and exact_match("The grade is A+", ["A+"]) == 0.0
    assert f1_score("about 75 percent", ["75 percent"]) > 0.7
    assert contains_answer("Attendance must be at least 75 percent.", ["75 percent"]) == 1.0
    assert contains_answer("Attendance is mandatory.", ["75 percent"]) == 0.0


def test_score_answer_supported_vs_unsupported():
    good = score_answer("Condonation costs 500 rupees per course [1].", PASSAGES, FakeNLI())
    assert good["faithfulness"] > 0.9 and good["unsupported_claim_rate"] == 0.0 and good["citation_accuracy"] == 1.0
    bad = score_answer("The cafeteria serves pizza every Friday evening.", PASSAGES, FakeNLI())
    assert bad["unsupported_claim_rate"] == 1.0 and bad["citation_accuracy"] is None


def test_collect_rows_and_summaries():
    rows = collect_rows(FakeRAG(TABLE), FakeNLI(), DATA, verbose=False)
    assert [r["generator_refused"] for r in rows] == [False, False, False, True]
    assert rows[0]["contains"] == 1.0 and rows[1]["contains"] == 0.0

    base = summarise(rows, answered_mask(rows, None))
    assert base["coverage"] == 0.75
    assert base["answered_unanswerable_rate"] == 0.5          # answered "vc?", refused "dress?"
    assert base["false_abstention_rate"] == 0.0

    prop = summarise(rows, answered_mask(rows, 0.6))
    assert prop["coverage"] == 0.25                           # only "cost?" survives
    assert prop["answered_unanswerable_rate"] == 0.0 and prop["false_abstention_rate"] == 0.5
    assert prop["unsupported_claim_rate"] < base["unsupported_claim_rate"]
    assert prop["F1"] <= base["F1"] + 1e-9                  # suppressing answers can only lower F1 here


def test_abstaining_on_answerable_scores_zero_and_sweep_is_monotone_in_coverage():
    rows = collect_rows(FakeRAG(TABLE), FakeNLI(), DATA, verbose=False)
    none_answered = summarise(rows, [False] * len(rows))
    assert none_answered["coverage"] == 0.0 and none_answered["F1"] == 0.0 and none_answered["false_abstention_rate"] == 1.0
    assert math.isnan(none_answered["faithfulness"])
    sw = sweep(rows, 0.0, 1.0, 0.25)
    cov = [s["coverage"] for s in sw]
    assert cov == sorted(cov, reverse=True) and cov[0] == 0.75 and cov[-1] == 0.0


class CountingGenerator:
    backend, model_name, max_passages = "fake", "fake-model", 3

    def __init__(self):
        self.calls = 0

    def generate(self, question, passages):
        self.calls += 1
        return f"answer to {question}"


def test_cached_generator_reuses_answers_and_invalidates_on_new_prompt(tmp_path):
    path = tmp_path / "cache.json"
    inner = CountingGenerator()
    cached = CachedGenerator(inner, path)
    assert cached.generate("q1", PASSAGES) == "answer to q1" and inner.calls == 1
    assert cached.generate("q1", PASSAGES) == "answer to q1" and inner.calls == 1     # served from cache
    assert cached.generate("q2", PASSAGES) == "answer to q2" and inner.calls == 2     # different prompt
    reloaded = CachedGenerator(CountingGenerator(), path)                             # survives a restart
    reloaded.generate("q1", PASSAGES)
    assert reloaded.inner.calls == 0


def test_rows_record_verifier_signals_when_available():
    from src.verification.schemas import VerificationResult

    class RAGWithSignals:
        def diagnose(self, question):
            ver = VerificationResult([], retrieval_score=0.59, entailment_score=1.0, citation_score=1.0, confidence=0.92)
            return RAGResult(question, "x", "Condonation costs 500 rupees per course [1].", PASSAGES, [],
                             abstained=False, confidence=0.92, verification=ver)

    row = collect_rows(RAGWithSignals(), FakeNLI(), DATA[:1], verbose=False)[0]
    assert (row["retrieval_score"], row["entailment_score"], row["citation_score"]) == (0.59, 1.0, 1.0)
    refused = collect_rows(FakeRAG(TABLE), FakeNLI(), DATA[3:], verbose=False)[0]
    assert refused["retrieval_score"] is None


def test_zero_padded_numbers_match_in_answer_metrics():
    assert exact_match("4", ["04"]) == 1.0
    assert contains_answer("The batch size is 4 students.", ["04 students"]) == 1.0
    assert normalize("5.0 and 0.5") == "5.0 and 0.5"
