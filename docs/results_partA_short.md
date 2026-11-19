# Results on real documents (Galgotias University, Part B: Examination Rules & Regulations)

Generator: `Qwen/Qwen2.5-1.5B-Instruct`. Verifier: `cross-encoder/nli-deberta-v3-base`, window 1, confidence threshold 0.6. Settings were chosen on the tuning half; the test half was run once.

### Test half (the reported result)

50 questions: 30 answerable, 20 unanswerable. Threshold 0.6.

| Measure | Baseline (no verifier) | With verifier |
|---|---|---|
| Unanswerable questions answered | 12/20 (95% CI 0.39-0.78) | 6/20 (95% CI 0.15-0.52) |
| Answerable questions not answered | 0/30 (95% CI 0.00-0.11) | 6/30 (95% CI 0.10-0.37) |
| Coverage | 0.84 | 0.60 |
| Answer contains gold answer | 0.70 | 0.63 |
| Faithfulness (answered only) | 0.83 | 0.96 |
| Unsupported-claim rate (answered only) | 0.19 | 0.03 |
| Citation accuracy | 0.72 | 0.92 |

**Unanswerable questions the verifier withheld (6):**
- p43 (conf 0.20): What is the fee for a University leaving certificate?  ->  The fee for a University leaving certificate is Rs 500/- [1].
- p44 (conf 0.06): What cash amount is attached to the Silver Medal?  ->  The Silver Medal is awarded with a cash amount of Rs. 5000/- [1].
- p47 (conf 0.16): What minimum CGPA must a student keep to avoid termination from the program?  ->  The minimum CGPA a student must achieve to avoid termination from the…
- p48 (conf 0.08): What is the maximum number of years allowed to complete the UG programme?  ->  The maximum number of years allowed to complete the UG programme is 4…
- p49 (conf 0.20): What is the maximum number of credits a student may take for Honours courses?  ->  No relaxation in the maximum number of credits is provided; however, …
- p50 (conf 0.45): How many total credits does a student need to graduate?  ->  A student needs to accumulate credits by completing a minimum of five…

**Unanswerable questions that still got through (6):**
- p33 (conf 0.83): Who is the current Vice Chancellor of Galgotias University?  ->  The payment shall be made through bank draft in favor of "Vice Chance…
- p37 (conf 0.79): How many books can a student borrow from the library at a time?  ->  All books and belongings must be left at the desk, away from the lab …
- p41 (conf 0.99): What fee must a rejoining student pay on readmission after being detained?  ->  On readmission after being detained, a student must pay the prescribe…
- p42 (conf 0.97): What is the fee for issuing a bonafide certificate?  ->  Rs 500/- per document (Rs 100/- postal charges extra) [1]
- p45 (conf 0.99): What minimum attendance percentage is needed to receive On Duty credit?  ->  The minimum attendance percentage needed to receive On Duty (OD) cred…
- p46 (conf 1.00): What specific punishment applies to a student found guilty of ragging?  ->  If students found guilty of Ragging, they will be liable for punishme…

**Correct answers the verifier withheld (6):**
- p04 (conf 0.21): How many weeks of a semester are devoted to instruction and laboratory work?  ->  17 Weeks [1]
- p07 (conf 0.20): What minimum attendance may a student keep without producing any proof for absence?  ->  A student is expected to maintain full attendance in all courses, but…
- p10 (conf 0.26): With the Vice Chancellor's consent, what is the lowest attendance percentage that still q…  ->  The lowest attendance percentage that still qualifies a student for t…
- p12 (conf 0.26): For which examination is no minimum attendance percentage criterion followed?  ->  For ETE, no minimum attendance percentage criterion is followed. [1]
- p17 (conf 0.20): How many courses does a Post-Diploma for additional specialization consist of?  ->  [1]
- p21 (conf 0.21): What is the minimum donation for instituting a scholarship?  ->  The minimum donation for instituting a scholarship is Rs.25,000. [2]


### Tuning half (used to choose settings; not a result)

25 questions: 15 answerable, 10 unanswerable. Threshold 0.6.

| Measure | Baseline (no verifier) | With verifier |
|---|---|---|
| Unanswerable questions answered | 1/10 (95% CI 0.02-0.40) | 0/10 (95% CI 0.00-0.28) |
| Answerable questions not answered | 0/15 (95% CI 0.00-0.20) | 2/15 (95% CI 0.04-0.38) |
| Coverage | 0.64 | 0.52 |
| Answer contains gold answer | 0.73 | 0.60 |
| Faithfulness (answered only) | 0.92 | 0.98 |
| Unsupported-claim rate (answered only) | 0.06 | 0.00 |
| Citation accuracy | 0.60 | 0.75 |

**Unanswerable questions the verifier withheld (1):**
- q48 (conf 0.44): What cash amount is attached to the Gold Medal?  ->  Rs. 5000/-

**Unanswerable questions that still got through (0):**
- none

**Correct answers the verifier withheld (2):**
- q14 (conf 0.23): What CGPA must an undergraduate student have to credit a Post Graduate level course?  ->  An undergraduate student must have a minimum CGPA of 7 to credit a Po…
- q18 (conf 0.16): What is the maximum number of students per internship batch?  ->  The maximum number of students per internship batch is 4. [1]


## Limitations

- One document in one domain, and a question set written by the author; intervals above show how little 20-30 questions can establish.
- The verifier checks that a sentence is supported by the retrieved passages, not that it answers the question asked, and it depends on the retriever ranking the right passage first.
- NLI models make paraphrase errors (for example, 'minimum 7' versus '7 or more'), which withhold some correct answers.
