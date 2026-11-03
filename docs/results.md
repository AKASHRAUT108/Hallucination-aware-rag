# Evaluation results

**Setup.** 30 hand-written questions over 4 synthetic "Northfield University" policy documents:
17 answerable, 13 unanswerable (some are deliberate near-misses, e.g. a fee that exists for a *different*
topic). Generator: Qwen2.5-1.5B-Instruct (greedy). Grader NLI: `cross-encoder/nli-roberta-base`, a different
model from the verifier. Threshold 0.6. All rows use the same cached answers, so they are directly comparable.

| System | Coverage | contains_gold | F1 | Faithfulness | Unsupported claims | Citation acc. | Unanswerable answered (of 13) | Correct answers withheld (of 17) |
|---|---|---|---|---|---|---|---|---|
| Baseline (no verifier) | 0.800 | 0.941 | 0.424 | 0.851 | 0.167 | 0.826 | 7 (0.538) | 0 |
| deberta-small, window 2 | 0.633 | 0.824 | 0.299 | 0.985 | 0.000 | 1.000 | 5 (0.385) | 3 (0.176) |
| deberta-small, window 1 | 0.533 | 0.765 | 0.279 | 0.984 | 0.000 | 1.000 | 3 (0.231) | 4 (0.235) |
| deberta-base, window 2 | 0.700 | 0.824 | 0.306 | 0.919 | 0.095 | 0.905 | 6 (0.462) | 2 (0.118) |
| **deberta-base, window 1 (default)** | 0.567 | 0.765 | 0.286 | 0.932 | 0.059 | 0.941 | 3 (0.231) | 3 (0.176) |

Reproduce a row: `python -m src.evaluation.run_eval --tag NAME --window 1 --verifier-nli cross-encoder/nli-deberta-v3-base`.
Per-question outputs: `results/tables/per_question_*.csv`.

## How to read this
- The verifier roughly halves the unanswerable questions that receive an answer (7 to 3 of 13) and costs
  3 of 17 correct answers. That is a trade-off, not a free improvement.
- Faithfulness and unsupported-claim rate rise partly because the verifier keeps only answers its own NLI
  check already likes. "Unanswerable answered" and "correct answers withheld" are the fairer measures.
- EM is near zero because the model answers in full sentences; `contains_gold` is the practical accuracy measure.

## Known limitations
- **Transplanted or off-question facts pass.** Entailment checks whether a sentence is supported by the
  passages, not whether it answers the question. A true fact about a different topic (or a different unit,
  e.g. semesters instead of weeks) can score high.
- **Fragments are withheld.** Answers such as "A+ [1]" give NLI too little to judge.
- **Small, self-written evaluation.** 13 unanswerable questions means one question is about 8 points.
  Four settings were compared on the same questions, so treat the ranking as indicative only.
- **Synthetic corpus.** Re-run on real domain documents with a larger set (50+ questions, 20+ unanswerable),
  choosing the threshold on one half and reporting on the other.

## Verifier signals (deberta-base, 1-sentence windows)
Per-question values: `results/tables/per_question_signals.csv`.

| Question | Outcome | Retrieval | Entailment | Citation | Confidence |
|---|---|---|---|---|---|
| q25 make-up exam fee | unanswerable, answered | 0.59 | 1.00 | 1.00 | 0.92 |
| q27 weeks of suspension | unanswerable, answered (true fact, different question) | 0.97 | 0.99 | 1.00 | 0.99 |
| q28 third condonation fee | unanswerable, answered (true fact, different question) | 0.88 | 1.00 | 1.00 | 0.98 |
| q04 "A+ [1]" | correct answer withheld (fragment) | 0.99 | 0.07 | 0.00 | 0.23 |
| q08 "Ten days [2]" | right fact, but the cited passage does not support it | 0.98 | 0.43 | 0.00 | 0.41 |
| q11 late-fee cap | correct answer withheld (NLI misses the inference) | 0.95 | 0.00 | 0.00 | 0.19 |

- Entailment and citation scores are close to 0 or 1, so confidence barely changes between thresholds 0.25 and 0.9.
- q25 has the lowest retrieval score among answers that passed (0.59), but correct answers go as low as 0.66 (q10) and 0.77 (q05), and two other leaked answers score 0.88 and 0.97. A retrieval cut-off would catch at most one of three leaks and would be fitted to one example, so none was added.

## Verifier signals (deberta-base, 1-sentence windows)
Per-question values: `results/tables/per_question_signals.csv`.

| Question | Outcome | Retrieval | Entailment | Citation | Confidence |
|---|---|---|---|---|---|
| q25 make-up exam fee | unanswerable, answered | 0.59 | 1.00 | 1.00 | 0.92 |
| q27 weeks of suspension | unanswerable, answered (true fact, different question) | 0.97 | 0.99 | 1.00 | 0.99 |
| q28 third condonation fee | unanswerable, answered (true fact, different question) | 0.88 | 1.00 | 1.00 | 0.98 |
| q04 "A+ [1]" | correct answer withheld (fragment) | 0.99 | 0.07 | 0.00 | 0.23 |
| q08 "Ten days [2]" | right fact, but the cited passage does not support it | 0.98 | 0.43 | 0.00 | 0.41 |
| q11 late-fee cap | correct answer withheld (NLI misses the inference) | 0.95 | 0.00 | 0.00 | 0.19 |

- Entailment and citation scores are close to 0 or 1, so confidence barely changes between thresholds 0.25 and 0.9.
- q25 has the lowest retrieval score among answers that passed (0.59), but correct answers go as low as 0.66 (q10) and 0.77 (q05), and two other leaked answers score 0.88 and 0.97. A retrieval cut-off would catch at most one of three leaks and would be fitted to one example, so none was added.
