# Writing and using the evaluation set

## 1. Write the questions (aim for 50+, one JSON object per line in `data/eval/eval.jsonl`)
```json
{"id": "q01", "question": "…", "answers": ["short exact answer"], "answerable": true}
{"id": "q31", "question": "…", "answers": [], "answerable": false, "type": "plain"}
{"id": "q41", "question": "…", "answers": [], "answerable": false, "type": "trap"}
```
- **Answerable (about 30):** copy the gold answer from the document in a short form (a number, a name, a short
  phrase). Add alternative spellings to `answers`, e.g. `["twice", "two times"]`.
- **Unanswerable, plain (about 10):** the topic is simply not in the documents.
- **Unanswerable, trap (about 10):** a fact that exists for a *different* item, unit or condition, so a model is
  tempted to answer. Example: the documents give a re-evaluation fee, you ask for a make-up exam fee.
  Mark these with `"type": "trap"`. They are what tests the verifier.
- Before writing a question, search the documents for its key terms. An "unanswerable" question whose answer is
  actually in a document is the most common mistake.

## 2. Check, split, tune, test
```powershell
python -m src.evaluation.check_eval                      # format, duplicates, gold answers present in the documents
python -m src.evaluation.make_split                      # writes tune.jsonl and test.jsonl (stratified, seed 42)
python -m src.evaluation.run_eval_file data/eval/tune.jsonl --tag tune
# try settings on the tuning half only, e.g.:
python -m src.evaluation.run_eval_file data/eval/tune.jsonl --tag tune_w1 --window 1
# choose threshold / window / verifier from the tuning results, put them in config.yaml, then ONCE:
python -m src.evaluation.run_eval_file data/eval/test.jsonl --tag test
```
Report the **test** numbers. Do not change settings after looking at them, and do not re-split with a new seed.

## 3. What to report
Baseline vs proposed on the test half: unanswerable questions answered, correct answers withheld, coverage,
`contains_gold`, faithfulness. State the set size, that settings were chosen on the tuning half, and the
limitations in `docs/results.md`.
