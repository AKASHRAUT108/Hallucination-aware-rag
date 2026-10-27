"""Stage 5 tests. Fake retriever/generator, so nothing is downloaded."""
from src.generation.citations import is_refusal, parse_citations, resolve_citations, strip_citations
from src.generation.generator import build_prompt
from src.pipeline.baseline_rag import BaselineRAG

PASSAGES = [
    {"rank": 1, "chunk_id": "attendance::0", "text": "Condonation costs 500 rupees per course."},
    {"rank": 2, "chunk_id": "exams::0", "text": "Hall tickets are issued seven days before the exam."},
    {"rank": 3, "chunk_id": "grading::0", "text": "A minimum CGPA of 5.0 is needed to progress."},
]


class FakeRetriever:
    def retrieve(self, query):
        return [dict(p) for p in PASSAGES]


class FakeGenerator:
    def __init__(self, reply, max_passages=3):
        self.reply, self.max_passages, self.seen = reply, max_passages, None

    def generate(self, question, passages):
        self.seen = passages
        return self.reply


def make(reply, max_passages=3):
    gen = FakeGenerator(reply, max_passages)
    return BaselineRAG(FakeRetriever(), gen), gen


def test_parse_and_strip_citations():
    assert parse_citations("A [1]. B [2, 1]. C [3].") == [1, 2, 3]
    assert parse_citations("No citations here.") == []
    assert strip_citations("It costs 500 rupees [1].") == "It costs 500 rupees ."


def test_refusal_detection():
    phrases = ["i don't know"]
    assert is_refusal("I don't know.", phrases)
    assert is_refusal("   ", phrases)
    assert not is_refusal("It costs 500 rupees [1].", phrases)


def test_resolve_citations_drops_unknown_ranks():
    cites = resolve_citations("A [1]. B [9].", PASSAGES)
    assert cites == [{"rank": 1, "chunk_id": "attendance::0"}]


def test_prompt_contains_numbered_passages_and_question():
    prompt = build_prompt("How much is condonation?", PASSAGES[:2])
    assert "[1] Condonation costs 500 rupees" in prompt and "[2] Hall tickets" in prompt
    assert "[3]" not in prompt and "How much is condonation?" in prompt


def test_baseline_returns_answer_with_citations():
    rag, _ = make("Condonation costs 500 rupees per course [1].")
    res = rag.answer("How much is condonation?")
    assert not res.abstained and res.answer.startswith("Condonation")
    assert res.citations == [{"rank": 1, "chunk_id": "attendance::0"}]


def test_baseline_answer_without_citations_has_empty_list():
    rag, _ = make("Condonation costs 500 rupees per course.")
    res = rag.answer("How much is condonation?")
    assert not res.abstained and res.citations == []


def test_baseline_abstains_only_when_generator_refuses():
    rag, _ = make("I don't know.")
    res = rag.answer("Who is the Vice Chancellor?")
    assert res.abstained and res.citations == []
    assert res.answer == "I can't answer this reliably from the available documents."


def test_baseline_passes_only_max_passages_to_generator():
    rag, gen = make("Condonation costs 500 rupees per course [1].", max_passages=2)
    res = rag.answer("How much is condonation?")
    assert [p["rank"] for p in gen.seen] == [1, 2] and len(res.passages) == 2
