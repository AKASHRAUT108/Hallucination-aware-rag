# Stage 12 protocol: complete-sentence prompt, tested on a fresh document

1. **Develop** the prompt change on the Part B tuning half (already used for development, so no claim rests on it).
2. **Freeze** (git tag) before running the Part A questions.
3. **Test once** on `data/eval/partA_test.jsonl` with the index built from Part A only.
4. Report the Part A result as the held-out number. Part A and Part B overlap in topic, with different values in
   places (for example 17 vs 14 instructional weeks), so Part A is a new document, not a new domain.
