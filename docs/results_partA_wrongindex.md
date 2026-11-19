# Results on real documents (Galgotias University, Part B: Examination Rules & Regulations)

Generator: `Qwen/Qwen2.5-1.5B-Instruct`. Verifier: `cross-encoder/nli-deberta-v3-base`, window 1, confidence threshold 0.6. Settings were chosen on the tuning half; the test half was run once.

### Test half (the reported result)

50 questions: 30 answerable, 20 unanswerable. Threshold 0.6.

| Measure | Baseline (no verifier) | With verifier |
|---|---|---|
| Unanswerable questions answered | 9/20 (95% CI 0.26-0.66) | 6/20 (95% CI 0.15-0.52) |
| Answerable questions not answered | 3/30 (95% CI 0.03-0.26) | 12/30 (95% CI 0.25-0.58) |
| Coverage | 0.72 | 0.48 |
| Answer contains gold answer | 0.63 | 0.50 |
| Faithfulness (answered only) | 0.85 | 0.98 |
| Unsupported-claim rate (answered only) | 0.14 | 0.00 |
| Citation accuracy | 0.65 | 0.93 |

**Unanswerable questions the verifier withheld (3):**
- p47 (conf 0.18): What minimum CGPA must a student keep to avoid termination from the program?  ->  To avoid termination from the program, a student must maintain a mini…
- p49 (conf 0.20): What is the maximum number of credits a student may take for Honours courses?  ->  The maximum number of credits a student may take for Honours courses …
- p50 (conf 0.19): How many total credits does a student need to graduate?  ->  To graduate, a student needs a total sum of Credits equal to Total Cr…

**Unanswerable questions that still got through (6):**
- p40 (conf 0.73): What is the total fee for the B.Tech Computer Science programme?  ->  Rs 1000/- for first set of copy and Rs 300/- extra per set [3]
- p41 (conf 0.84): What fee must a rejoining student pay on readmission after being detained?  ->  A rejoining student who is detained must pay the prescribed fees deci…
- p42 (conf 0.98): What is the fee for issuing a bonafide certificate?  ->  Rs 500/- per document (Rs 100/- postal charges extra)
- p43 (conf 0.88): What is the fee for a University leaving certificate?  ->  Rs 500/- per document (Rs 100/- postal charges extra) for WEAS.
- p45 (conf 0.87): What minimum attendance percentage is needed to receive On Duty credit?  ->  To receive On Duty (OD) credit, students must maintain a minimum of 7…
- p46 (conf 1.00): What specific punishment applies to a student found guilty of ragging?  ->  A student found guilty of ragging is liable for punishment according …

**Correct answers the verifier withheld (9):**
- p06 (conf 0.50): During which months does the even term run?  ->  The even term runs from January to May. [2]
- p07 (conf 0.20): What minimum attendance may a student keep without producing any proof for absence?  ->  A student is expected to maintain full attendance in all courses, but…
- p10 (conf 0.52): With the Vice Chancellor's consent, what is the lowest attendance percentage that still q…  ->  With the Vice Chancellor's consent, a student can maintain a minimum …
- p12 (conf 0.20): For which examination is no minimum attendance percentage criterion followed?  ->  For the Common Admission Test (CAT), no minimum attendance percentage…
- p13 (conf 0.21): Within how many instructional days after dropping a course must another course be added?  ->  Within ten instructional days/two weeks after dropping a course, anot…
- p15 (conf 0.23): What CGPA must an undergraduate student have to credit a Post Graduate level course?  ->  An undergraduate student must have a minimum CGPA of 7 to credit a Po…
- p22 (conf 0.01): What attendance in Campus Recruitment training is mandatory to appear for the placement p…  ->  To appear for the placement process, students must maintain an attend…
- p24 (conf 0.22): What minimum average CGPA is needed in the Honours courses to receive the Honours award?  ->  To receive the Honours award, a student needs to accumulate credits b…
- p27 (conf 0.20): How many calendar days after receipt of the final decision must a student file a petition…  ->  A student must file a petition with the office of the Vice Chancellor…


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
