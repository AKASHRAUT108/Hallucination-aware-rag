# Evaluation log (what was run, in order)

Facts only. Fill the bracketed items yourself before citing this in the report.

## Development set (synthetic)
- 30 questions over 4 synthetic "Northfield University" documents (17 answerable, 13 unanswerable, 7 near-miss traps).
- Used to design the verifier: window size, NLI model size, claim cleaning. Results are development results, not reported results.

## Real document 1: Galgotias University Part B (Examination Rules & Regulations)
- 50 questions drafted from the document text (30 answerable, 10 plain unanswerable, 10 near-miss traps).
  Gold answers are short phrases copied from the text. [Author verification of each gold answer: ______]
- Stratified split with a fixed seed (42): 25 tuning, 25 test. Git tag `eval-frozen` before the test run.
- Changes made using the tuning half only: removal of "according to the passage" lead-ins/lead-outs, zero-padded number
  normalisation (04 -> 4), splitting at PDF bullet glyphs, extra refusal phrases, and two alternate gold answers (q14, q18) added
  before the test run.
- Test half, run once (threshold 0.6, deberta-v3-base verifier, 1-sentence windows):

| | No verifier | With verifier |
|---|---|---|
| Unanswerable questions answered | 3/10 | 1/10 |
| Correct answers not answered | 0/15 | 5/15 |
| Coverage | 0.72 | 0.44 |

- Review after the run: q44 (mobile-phone penalty) may be mislabeled as unanswerable. [Result of checking the document: ______]

## Prompt experiment (tuning half only)
| | Old prompt, no verifier | Old prompt + verifier (tune_v3) | New prompt, no verifier | New prompt + verifier (tune_v4) |
|---|---|---|---|---|
| Unanswerable answered | 1/10 | 0/10 | 0/10 | 0/10 |
| Correct answers not answered | 0/15 | 2/15 | 0/15 | 6/15 |
| Answer contains gold | 0.73 | 0.60 | 0.87 | 0.47 |

The complete-sentence prompt raised raw accuracy but made the verifier withhold more correct answers. Both prompts are
therefore reported on the held-out document.

## Real document 2: Galgotias University Part A (Academic Regulations), held-out
- 50 questions drafted from the document (30 answerable, 10 plain, 10 near-miss traps). No tuning on them.
- First attempt discarded: the index still contained Part B, so the run did not measure the intended document. Its files are
  kept as `*_partA_wrongindex`. No code, setting or question was changed between this run and the corrected run.
- Corrected run, complete-sentence prompt (threshold 0.6):

| | No verifier | With verifier |
|---|---|---|
| Unanswerable questions answered | 9/20 | 5/20 |
| Correct answers not answered | 1/30 | 3/30 |
| Coverage | 0.76 | 0.64 |
| Answer contains gold | 0.67 | 0.67 |

- Short-prompt run on Part A: [pending]
- Leaked unanswerable questions q33, q41, q42, q46, q50: [pending review of the answer text against the document]

## Known limitations
- Two documents from the same university, questions written by the system builder, 10-30 questions per category:
  95% intervals overlap in every comparison, so results are indicative.
- The verifier checks support by the retrieved passages, not that the answer addresses the question; it depends on
  retrieval ranking the right passage first.
- Four verifier settings and two prompts were compared; only the settings fixed on the tuning half were applied to test data.
