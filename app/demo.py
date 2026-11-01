"""Streamlit demo: baseline RAG vs the hallucination-aware pipeline, side by side.

Run from the project root:  streamlit run app/demo.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.pipeline.hallucination_aware_rag import HallucinationAwareRAG  # noqa: E402
from src.pipeline.presentation import build_views  # noqa: E402
from src.utils.helpers import load_config  # noqa: E402

st.set_page_config(page_title="Hallucination-Aware RAG", layout="wide")


@st.cache_resource(show_spinner="Loading models (the first run takes a minute)...")
def load_rag() -> HallucinationAwareRAG:
    return HallucinationAwareRAG.from_config()


@st.cache_data(show_spinner="Retrieving, generating and verifying...")
def diagnose(question: str):
    return load_rag().diagnose(question)


def example_questions() -> list[str]:
    path = ROOT / load_config()["evaluation"]["eval_file"]
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["question"] for line in f if line.strip()]


cfg = load_config()
st.title("Hallucination-Aware RAG")
st.caption("The verification layer checks every answer sentence against the retrieved passages "
           "and withholds answers it cannot support.")

tau = st.sidebar.slider("Confidence threshold", 0.0, 1.0, float(cfg["verification"]["confidence_threshold"]), 0.05,
                        help="Answers scoring below this are withheld. Moving the slider does not regenerate answers.")
st.sidebar.markdown(f"Verifier: `{cfg['verification']['nli_model']}`  \nWindow: {cfg['verification'].get('window_sentences', 2)} sentence(s)")

example = st.selectbox("Example questions", [""] + example_questions())
question = st.text_input("Or type your own question", value=example).strip()

if question:
    rag = load_rag()
    res = diagnose(question)
    views = build_views(res, tau, rag.abstain_message)

    left, right = st.columns(2)
    with left:
        st.subheader("Baseline RAG")
        st.write(views["baseline"]["text"])
        st.caption("Answers unless the model itself says it does not know.")
    with right:
        st.subheader("Hallucination-aware RAG")
        aware = views["aware"]
        (st.warning if aware["abstained"] else st.success)(aware["text"])
        if res.verification:
            v = res.verification
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence", f"{aware['confidence']:.2f}", help=f"Threshold {tau:.2f}")
            c2.metric("Retrieval", f"{v.retrieval_score:.2f}")
            c3.metric("Entailment", f"{v.entailment_score:.2f}")
            c4.metric("Citation", f"{v.citation_score:.2f}")
            if aware["suppressed_by_verifier"]:
                st.info("The model gave an answer, but the verifier could not support it from the documents.")
            st.caption("Sentence-level checks")
            st.table([{"sentence": s.text, "support": round(s.support, 2),
                       "cited": ", ".join(f"[{r}]" for r in s.cited_ranks) or f"inferred [{s.best_rank}]"}
                      for s in v.sentences])

    with st.expander("Retrieved sources"):
        for p in res.passages:
            st.markdown(f"**[{p['rank']}] {p['chunk_id']}**  (rerank p={p.get('rerank_prob', 0):.2f})")
            st.write(p["text"])
