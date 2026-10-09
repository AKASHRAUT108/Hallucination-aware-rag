from src.ingestion.chunker import chunk_documents, chunk_text
from src.ingestion.loader import clean_text


def test_clean_text_joins_hyphenated_breaks():
    assert clean_text("informa-\ntion  retrieval\n\nworks") == "information retrieval works"


def test_chunk_overlap_and_size():
    text = " ".join(str(i) for i in range(500))
    chunks = chunk_text(text, chunk_size=100, overlap=20, min_words=10)
    assert all(len(c.split()) <= 100 for c in chunks)
    # neighbouring chunks share 20 words
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_chunk_metadata():
    recs = [{"source": "a.txt", "page": 2, "text": " ".join(["word"] * 250)}]
    out = chunk_documents(recs, 100, 20, 10)
    assert out[0]["chunk_id"] == "a.txt::p2::c0"
    assert out[0]["page"] == 2
