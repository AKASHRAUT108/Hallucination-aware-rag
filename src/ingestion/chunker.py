"""Split page-level records into overlapping word chunks, keeping source + page
so that later stages can cite and verify against exact locations."""


def chunk_text(text: str, chunk_size: int, overlap: int, min_words: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if len(piece) >= min_words or (start == 0 and piece):
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(records: list[dict], chunk_size: int = 200,
                    overlap: int = 40, min_words: int = 30) -> list[dict]:
    chunks = []
    for rec in records:
        pieces = chunk_text(rec["text"], chunk_size, overlap, min_words)
        for j, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{rec['source']}::p{rec['page']}::c{j}",
                "source": rec["source"],
                "page": rec["page"],
                "text": piece,
                "n_words": len(piece.split()),
            })
    return chunks
