# Results on real documents (Galgotias University, Part B: Examination Rules & Regulations)

Generator: `Qwen/Qwen2.5-1.5B-Instruct`. Verifier: `cross-encoder/nli-deberta-v3-base`, window 1, confidence threshold 0.6. Settings were chosen on the tuning half; the test half was run once.

### Test half (the reported result)

50 questions: 30 answerable, 20 unanswerable. Threshold 0.6.

| Measure | Baseline (no verifier) | With verifier |
|---|---|---|
| Unanswerable questions answered | 9/20 (95% CI 0.26-0.66) | 5/20 (95% CI 0.11-0.47) |
| Answerable questions not answered | 1/30 (95% CI 0.01-0.17) | 3/30 (95% CI 0.03-0.26) |
| Coverage | 0.76 | 0.64 |
| Answer contains gold answer | 0.67 | 0.67 |
| Faithfulness (answered only) | 0.86 | 0.96 |
| Unsupported-claim rate (answered only) | 0.13 | 0.00 |
| Citation accuracy | 0.76 | 0.89 |

**Unanswerable questions the verifier withheld (4):**
- p43 (conf 0.20): What is the fee for a University leaving certificate?  ->  The fee for a University leaving certificate is Rs 500/- [1].
- p45 (conf 0.20): What minimum attendance percentage is needed to receive On Duty credit?  ->  According to Passage [2], the minimum attendance percentage required …
- p48 (conf 0.06): What is the maximum number of years allowed to complete the UG programme?  ->  The maximum number of years allowed to complete the UG programme is 4…
- p49 (conf 0.20): What is the maximum number of credits a student may take for Honours courses?  ->  A student may take up to 30 credits for Honours courses in a term. [3]

**Unanswerable questions that still got through (5):**
- p33 (conf 0.83): Who is the current Vice Chancellor of Galgotias University?  ->  The payment shall be made through bank draft in favor of "Vice Chance…
- p41 (conf 0.99): What fee must a rejoining student pay on readmission after being detained?  ->  A candidate, who is detained or discontinued in the year/semester, wi…
- p42 (conf 0.96): What is the fee for issuing a bonafide certificate?  ->  The fee for issuing a bonafide certificate is Rs 500/- per document (…
- p46 (conf 1.00): What specific punishment applies to a student found guilty of ragging?  ->  If a student found guilty of Ragging, he/she will liable for punishme…
- p50 (conf 0.85): How many total credits does a student need to graduate?  ->  A student needs to accumulate credits by registering for the required…

**Correct answers the verifier withheld (2):**
- p10 (conf 0.20): With the Vice Chancellor's consent, what is the lowest attendance percentage that still q…  ->  With the Vice Chancellor's consent, a student can maintain a minimum …
- p12 (conf 0.30): For which examination is no minimum attendance percentage criterion followed?  ->  For the Examination Test (ETE), no minimum attendance percentage crit…


### Tuning half (used to choose settings; not a result)

25 questions: 15 answerable, 10 unanswerable. Threshold 0.6.

| Measure | Baseline (no verifier) | With verifier |
|---|---|---|
| Unanswerable questions answered | 0/10 (95% CI 0.00-0.28) | 0/10 (95% CI 0.00-0.28) |
| Answerable questions not answered | 0/15 (95% CI 0.00-0.20) | 6/15 (95% CI 0.20-0.64) |
| Coverage | 0.60 | 0.36 |
| Answer contains gold answer | 0.87 | 0.47 |
| Faithfulness (answered only) | 0.73 | 0.98 |
| Unsupported-claim rate (answered only) | 0.27 | 0.00 |
| Citation accuracy | 0.64 | 1.00 |

**Unanswerable questions the verifier withheld (0):**
- none

**Unanswerable questions that still got through (0):**
- none

**Correct answers the verifier withheld (6):**
- q08 (conf 0.52): With the Vice Chancellor's consent, what is the lowest attendance percentage that still q…  ->  With the Vice Chancellor's consent, a student can maintain a minimum …
- q09 (conf 0.20): For which examination is no minimum attendance percentage criterion followed?  ->  For the Common Admission Test (CAT), no minimum attendance percentage…
- q14 (conf 0.23): What CGPA must an undergraduate student have to credit a Post Graduate level course?  ->  An undergraduate student must have a minimum CGPA of 7 to credit a Po…
- q18 (conf 0.16): What is the maximum number of students per internship batch?  ->  The maximum number of students per internship batch is 4. [1]
- q19 (conf 0.13): During an internship, what attendance is needed for the term to be granted?  ->  During an internship, the term to be granted requires 90% attendance.…
- q23 (conf 0.20): To how many marks are CAT marks converted for the final total?  ->  To convert CAT marks to the final total, 30 marks from Assignment-2, …


## Limitations

- One document in one domain, and a question set written by the author; intervals above show how little 20-30 questions can establish.
- The verifier checks that a sentence is supported by the retrieved passages, not that it answers the question asked, and it depends on the retriever ranking the right passage first.
- NLI models make paraphrase errors (for example, 'minimum 7' versus '7 or more'), which withhold some correct answers.
