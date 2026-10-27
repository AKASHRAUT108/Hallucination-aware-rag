"""Result object shared by the baseline and (later) the hallucination-aware pipeline."""
from dataclasses import dataclass, field


@dataclass
class RAGResult:
    question: str
    answer: str                    # text shown to the user (abstain message if the model refused)
    raw_answer: str                # exactly what the generator wrote
    passages: list[dict]           # passages the generator saw; their rank is the [n] citation id
    citations: list[dict] = field(default_factory=list)   # [{"rank": 1, "chunk_id": "..."}]
    abstained: bool = False
    confidence: float | None = None     # filled in from Stage 6
