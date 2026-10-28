"""Plain dataclasses for the verification layer."""
from dataclasses import dataclass, field


@dataclass
class SentenceCheck:
    text: str                                   # answer sentence without [n] markers
    cited_ranks: list[int] = field(default_factory=list)   # citations the model wrote
    support: float = 0.0                        # best P(entail) over ALL shown passages
    best_rank: int = 0                          # rank of the passage that supports it best
    cited_support: float = 0.0                  # best P(entail) over the cited passages
    citation_inferred: bool = False             # True if the model cited nothing and we attributed a passage


@dataclass
class VerificationResult:
    sentences: list[SentenceCheck]
    retrieval_score: float
    entailment_score: float
    citation_score: float
    confidence: float
