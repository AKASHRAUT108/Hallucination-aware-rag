import re

import numpy as np

from src.evaluation.explain import explain_answer

WORDS = re.compile(r"[a-z0-9]+")


class LongPremiseNLI:
    """Overlap score that gives up on premises longer than 12 words, like a small NLI model."""
    def entailment_probs(self, pairs):
        out = []
        for prem, hyp in pairs:
            h, p = set(WORDS.findall(hyp.lower())), set(WORDS.findall(prem.lower()))
            out.append(0.0 if len(prem.split()) > 12 else len(h & p) / max(len(h), 1))
        return np.array(out)


PASSAGES = [{"rank": 1, "chunk_id": "doc::0", "rerank_prob": 0.9,
             "text": "Interns follow the company dress code. Max 04 students per batch. "
                     "Term will not be granted if student fails to have 90% attendance."}]


def test_explain_finds_the_supporting_window_and_normalises_numbers():
    rows = explain_answer("Max 4 students per batch [1].", PASSAGES, LongPremiseNLI(), window_sentences=1)
    assert len(rows) == 1 and rows[0]["cited"] == [1]
    best = rows[0]["passages"][0]
    assert best["p_entail"] == 1.0 and best["window"] == "Max 4 students per batch."


def test_explain_skips_meta_sentences_and_handles_empty_answers():
    answer = "Max 4 students per batch. This information is provided in Passage [1], which states something."
    assert len(explain_answer(answer, PASSAGES, LongPremiseNLI(), 1)) == 1
    assert explain_answer("", PASSAGES, LongPremiseNLI(), 1) == []
