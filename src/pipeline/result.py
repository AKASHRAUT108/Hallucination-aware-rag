"""Result object shared by the baseline and the hallucination-aware pipeline."""
from dataclasses import dataclass, field


@dataclass
class RAGResult:
    question: str
    answer: str                    # text shown to the user (abstain message if refused or suppressed)
    raw_answer: str                # exactly what the generator wrote
    passages: list[dict]           # passages the generator saw; their rank is the [n] citation id
    citations: list[dict] = field(default_factory=list)   # [{"rank": 1, "chunk_id": "..."}]
    abstained: bool = False
    confidence: float | None = None          # set by the hallucination-aware pipeline
    verification: object | None = None       # VerificationResult, set by the hallucination-aware pipeline
