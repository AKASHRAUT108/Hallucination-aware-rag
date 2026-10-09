"""Stage 2 entry point: raw documents -> data/processed/chunks.jsonl

Run:  python -m src.ingestion.build_corpus
"""
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.utils.helpers import ROOT, load_config, save_jsonl


def main() -> None:
    cfg = load_config()
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    out_file = ROOT / cfg["paths"]["chunks_file"]
    ch = cfg["chunking"]

    records = load_documents(raw_dir)
    if not records:
        print(f"No documents found in {raw_dir}. Add .pdf/.txt/.md files and rerun.")
        return

    chunks = chunk_documents(records, ch["chunk_size"], ch["overlap"], ch["min_chunk_words"])
    save_jsonl(chunks, out_file)

    sources = {r["source"] for r in records}
    avg = sum(c["n_words"] for c in chunks) / len(chunks)
    print(f"Documents : {len(sources)}")
    print(f"Pages     : {len(records)}")
    print(f"Chunks    : {len(chunks)}  (avg {avg:.0f} words)")
    print(f"Saved to  : {out_file}")


if __name__ == "__main__":
    main()
