# Hallucination-Aware RAG Framework for Domain-Specific Question Answering

A Retrieval-Augmented Generation (RAG) pipeline that doesn't just answer questions, it checks whether its own answer is actually supported by the retrieved documents, and holds back answers it isn't confident about.

**Domain:** Natural Language Processing · Generative AI (RAG) · Deep Learning

---

## The Problem

RAG systems ground an LLM in external documents, but they still hallucinate when the retrieved context is incomplete, ambiguous, or only partly relevant. Worse, they state wrong answers with full confidence and give no signal about how far to trust them. In areas like academic advising, legal research, or healthcare information, a confident but unverified answer can mislead people.

## What This Project Does

1. **Retrieves** relevant passages from a domain-specific document corpus using dense embeddings.
2. **Reranks** the retrieved passages so the best evidence comes first.
3. **Generates** an answer from the top passages, with citations to the source chunks.
4. **Verifies** the answer through three checks:
   - **Citation verification:** does each cited source really support the claim?
   - **Entailment scoring:** does the retrieved context logically entail the answer?
   - **Confidence scoring:** a combined score from retrieval, entailment, and citation signals.
5. **Decides:** answers above the confidence threshold are returned; low-confidence ones are flagged or suppressed ("I can't answer this reliably from the available documents").
6. **Evaluates** everything against a baseline RAG pipeline with no verification layer.

## Evaluation

| Type | Metrics |
|---|---|
| QA quality | Exact Match (EM), F1 |
| Hallucination | Faithfulness score, citation accuracy, unsupported-claim rate |
| Trade-off | Answer coverage (how many questions still get answered) |

**Goal:** a measurable drop in hallucination rate while keeping answer coverage acceptable.

## Pipeline Overview

```
Question
   │
   ▼
Dense Retrieval ──► Reranking ──► Answer Generation (with citations)
                                          │
                                          ▼
                     ┌────────── Verification Layer ──────────┐
                     │  Citation check · Entailment · Score   │
                     └────────────────────┬───────────────────┘
                                          ▼
                         Confidence ≥ threshold?
                           │                │
                          Yes               No
                           ▼                ▼
                    Return answer     Flag / suppress
```

## Project Structure

```
hallucination-aware-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config/
│   └── config.yaml              # model names, thresholds, paths
├── data/
│   ├── raw/                     # original documents
│   ├── processed/               # cleaned + chunked text
│   └── eval/                    # question–answer evaluation set
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # load PDFs / text files
│   │   └── chunker.py           # split documents into chunks
│   ├── retrieval/
│   │   ├── embedder.py          # dense embeddings
│   │   ├── vector_store.py      # FAISS index build + search
│   │   └── reranker.py          # cross-encoder reranking
│   ├── generation/
│   │   └── generator.py         # LLM answer generation with citations
│   ├── verification/
│   │   ├── citation_checker.py  # source-attribution check
│   │   ├── entailment.py        # answer–context entailment (NLI)
│   │   └── confidence.py        # combined confidence score + threshold
│   ├── pipeline/
│   │   ├── baseline_rag.py      # RAG without verification
│   │   └── hallucination_aware_rag.py  # full framework
│   ├── evaluation/
│   │   ├── qa_metrics.py        # EM, F1
│   │   ├── hallucination_metrics.py    # faithfulness, citation acc., unsupported claims
│   │   └── run_eval.py          # baseline vs proposed comparison
│   └── utils/
│       └── helpers.py
├── app/
│   └── demo.py                  # Streamlit demo
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── tables/
│   └── figures/
├── tests/
│   └── test_pipeline.py
└── docs/
    └── architecture.md
```

## Tech Stack

- **Language:** Python 3.10+
- **Embeddings / reranking / NLI:** `sentence-transformers`, Hugging Face `transformers`
- **Vector search:** FAISS
- **LLM:** configurable (open-source via Hugging Face, or an API model)
- **Evaluation:** custom metrics + `scikit-learn`
- **Demo UI:** Streamlit

## Setup

```powershell
git clone https://github.com/AKASHRAUT108/hallucination-aware-rag.git
cd hallucination-aware-rag
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
# 1. Build the index from documents in data/raw
python -m src.retrieval.vector_store --build

# 2. Ask a question
python -m src.pipeline.hallucination_aware_rag --query "Your question here"

# 3. Run the full evaluation (baseline vs proposed)
python -m src.evaluation.run_eval

# 4. Launch the demo
streamlit run app/demo.py
```

## Development Stages

| Stage | What gets built | Status |
|---|---|---|
| 1 | Project setup, repo structure, README | ✅ |
| 2 | Data ingestion and chunking | ⬜ |
| 3 | Dense retrieval and vector store | ⬜ |
| 4 | Reranking | ⬜ |
| 5 | Answer generation with citations (baseline RAG) | ⬜ |
| 6 | Verification layer (citation, entailment, confidence) | ⬜ |
| 7 | Full hallucination-aware pipeline | ⬜ |
| 8 | Evaluation and baseline comparison | ⬜ |
| 9 | Demo app, results, final documentation | ⬜ |

## Expected Outcome

A reproducible mitigation technique that lowers hallucination rate compared with a standard RAG pipeline, while keeping answer coverage and quality acceptable, contributing to trustworthy generative AI research.

## Author

**Akash**, B.Tech CSE (AI/ML), Galgotias University
Supervisor: Sachin Saurabh
GitHub: [AKASHRAUT108](https://github.com/AKASHRAUT108)

## License

MIT
