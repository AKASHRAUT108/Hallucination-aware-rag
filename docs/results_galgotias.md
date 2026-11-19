# Results on real documents (Galgotias University, Part B: Examination Rules & Regulations)

Generator: `Qwen/Qwen2.5-1.5B-Instruct`. Verifier: `cross-encoder/nli-deberta-v3-base`, window 1, confidence threshold 0.6. Settings were chosen on the tuning half; the test half was run once.

### Test half (the reported result)

25 questions: 15 answerable, 10 unanswerable. Threshold 0.6.

| Measure | Baseline (no verifier) | With verifier |
|---|---|---|
| Unanswerable questions answered | 3/10 (95% CI 0.11-0.60) | 1/10 (95% CI 0.02-0.40) |
| Answerable questions not answered | 0/15 (95% CI 0.00-0.20) | 5/15 (95% CI 0.15-0.58) |
| Coverage | 0.72 | 0.44 |
| Answer contains gold answer | 1.00 | 0.67 |
| Faithfulness (answered only) | 0.86 | 0.96 |
| Unsupported-claim rate (answered only) | 0.11 | 0.00 |
| Citation accuracy | 0.86 | 1.00 |

**Unanswerable questions the verifier withheld (2):**
- q43 (conf 0.18): What is the minimum passing SGPA for Ph.D programs?  ->  The minimum passing SGPA for Ph.D programs is 6.0. [1]
- q44 (conf 0.01): What penalty does a student receive for possessing a mobile phone for the first time?  ->  The student receives a recommendation for a NE grade in the subject w…

**Unanswerable questions that still got through (1):**
- q42 (conf 0.85): What fine is charged for dropping a course?  ->  Rs. 7000/-. [1]

**Correct answers the verifier withheld (5):**
- q05 (conf 0.20): What minimum attendance may a student keep without producing any proof for absence?  ->  A student is expected to maintain full attendance in all courses but …
- q16 (conf 0.21): What share of the total credits of Semesters I and II is needed to progress from the firs…  ->  60% of total credits from (Semester I and Semester II) [1]
- q20 (conf 0.20): What is the maximum number of courses a student may register for in the Summer Term?  ->  A student may register for a maximum of four courses in the Summer Te…
- q22 (conf 0.20): What is the maximum share of total credits that can come from NPTEL courses within the fi…  ->  The maximum share of total credits that can come from NPTEL courses w…
- q25 (conf 0.26): How many days before the start of the End Term Exams must the question paper be prepared …  ->  30 days [1]


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
