"""Stage 6 tests. A fake NLI model (word overlap) stands in for the real one, so nothing is downloaded."""
import re

import numpy as np

from src.pipeline.hallucination_aware_rag import HallucinationAwareRAG
from src.verification.citation_checker import check_answer, citation_score
from src.verification.confidence import combine, entailment_signal, retrieval_signal
from src.verification.sentences import split_sentences

WORDS = re.compile(r"[a-z0-9]+")
PASSAGES = [
    {"rank": 1, "chunk_id": "attendance::0", "text": "Condonation costs 500 rupees per course.", "rerank_prob": 0.95},
    {"rank": 2, "chunk_id": "exams::0", "text": "Hall tickets are issued seven days before the first examination.", "rerank_prob": 0.4},
]
SETTINGS = {"weights": {"retrieval": 0.2, "entailment": 0.5, "citation": 0.3},
            "entailment_aggregation": "mean", "citation_threshold": 0.5, "confidence_threshold": 0.6}


class FakeNLI:
    """P(entail) = share of hypothesis words that appear in the premise."""
    def entailment_probs(self, pairs):
        out = []
        for prem, hyp in pairs:
            h, p = set(WORDS.findall(hyp.lower())), set(WORDS.findall(prem.lower()))
            out.append(len(h & p) / max(len(h), 1))
        return np.array(out)


class FakeRetriever:
    def retrieve(self, query):
        return [dict(p) for p in PASSAGES]


class FakeGenerator:
    max_passages = 2

    def __init__(self, reply):
        self.reply = reply

    def generate(self, question, passages):
        return self.reply


def rag(reply):
    return HallucinationAwareRAG(FakeRetriever(), FakeGenerator(reply), FakeNLI(), SETTINGS)


def test_split_keeps_trailing_citation_with_its_sentence():
    assert split_sentences("RAG combines a retriever with a model. [1]") == ["RAG combines a retriever with a model. [1]"]
    assert split_sentences("A is true [1]. B is true [2].") == ["A is true [1].", "B is true [2]."]
    assert split_sentences("") == []


def test_check_answer_stated_wrong_and_inferred_citations():
    stated = check_answer("Condonation costs 500 rupees [1].", PASSAGES, FakeNLI())
    assert stated[0].cited_ranks == [1] and not stated[0].citation_inferred and stated[0].support > 0.9
    wrong = check_answer("Condonation costs 500 rupees [2].", PASSAGES, FakeNLI())
    assert wrong[0].cited_support < wrong[0].support and citation_score(wrong) == 0.0
    inferred = check_answer("Condonation costs 500 rupees.", PASSAGES, FakeNLI())
    assert inferred[0].citation_inferred and inferred[0].best_rank == 1 and citation_score(inferred) == 1.0


def test_signals_and_combine():
    assert retrieval_signal(PASSAGES) == 0.95
    assert retrieval_signal([{"dense_score": 0.4}]) == 0.4 and retrieval_signal([]) == 0.0
    checks = check_answer("Condonation costs 500 rupees. Pizza is served on Fridays.", PASSAGES, FakeNLI())
    assert entailment_signal(checks, "min") < entailment_signal(checks, "mean")
    w = SETTINGS["weights"]
    assert combine(1, 1, 1, w) == 1.0 and combine(0, 0, 0, w) == 0.0 and abs(combine(1, 0, 0, w) - 0.2) < 1e-9


def test_supported_answer_is_returned_with_confidence():
    res = rag("Condonation costs 500 rupees per course [1].").answer("How much is condonation?")
    assert not res.abstained and res.confidence > 0.6
    assert res.citations == [{"rank": 1, "chunk_id": "attendance::0"}] and res.verification is not None


def test_unsupported_answer_is_suppressed():
    res = rag("The cafeteria serves pizza every Friday evening.").answer("How much is condonation?")
    assert res.abstained and res.answer.startswith("I can't answer") and res.confidence < 0.6


def test_threshold_override_and_generator_refusal():
    r = rag("Condonation costs 500 rupees per course [1].")
    assert r.answer("q", threshold=1.01).abstained
    assert not r.answer("q", threshold=0.0).abstained
    refused = rag("I don't know.").answer("q")
    assert refused.abstained and refused.confidence == 0.0


class LongPremiseNLI(FakeNLI):
    """Mimics the real model's weakness: premises over 30 words are judged 'neutral' (P(entail) ~ 0)."""
    def entailment_probs(self, pairs):
        base = super().entailment_probs(pairs)
        return np.array([0.0 if len(prem.split()) > 30 else b for (prem, _), b in zip(pairs, base)])


LONG_PASSAGE = [{"rank": 1, "chunk_id": "doc::0", "rerank_prob": 0.9, "text": (
    "Retrieval augmented generation combines a retriever with a language model. "
    "The retriever searches an external document collection for passages relevant to a question. "
    "The language model uses those passages as context when writing its answer. "
    "This grounds the output in real documents instead of memory alone.")}]


def test_windows_rescue_support_for_sentence_copied_from_long_chunk():
    claim = "Retrieval augmented generation combines a retriever with a language model. [1]"
    assert len(LONG_PASSAGE[0]["text"].split()) > 30
    whole_chunk_only = check_answer(claim, LONG_PASSAGE, LongPremiseNLI(), window_sentences=0)
    assert whole_chunk_only[0].support == 0.0           # old behaviour: the bug seen on real data
    with_windows = check_answer(claim, LONG_PASSAGE, LongPremiseNLI(), window_sentences=2)
    assert with_windows[0].support > 0.9 and citation_score(with_windows) == 1.0


def test_windows_do_not_support_a_made_up_claim():
    made_up = check_answer("The cafeteria serves pizza every Friday evening. [1]", LONG_PASSAGE,
                           LongPremiseNLI(), window_sentences=2)
    assert made_up[0].support < 0.3
