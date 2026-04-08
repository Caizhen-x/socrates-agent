"""
retrieve.py — Given a user question, retrieve the most relevant dialogue passages.
Usage: python scripts/retrieve.py "What is courage?"
Returns formatted passages ready for injection into the agent prompt.
"""

import sys
import json
import yaml
import datetime
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.theme_classifier import classify, themes_to_books


@dataclass
class Passage:
    text: str
    book: str
    speaker: str
    chunk_index: int
    score: float
    chunk_id: str


def load_settings():
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def load_theme_map():
    with open(PROJECT_ROOT / "config" / "theme_book_map.yaml") as f:
        return yaml.safe_load(f)


def get_collection(settings):
    import chromadb
    vectorstore_path = PROJECT_ROOT / settings["paths"]["vectorstore_dir"]
    client = chromadb.PersistentClient(path=str(vectorstore_path))
    return client.get_collection(settings["collection_name"])


def embed_question(question: str, settings: dict):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings["embedding"]["model"])
    return model.encode([question])[0].tolist()


def query_collection(collection, embedding, top_k: int, book_filter: list[str] | None) -> list[Passage]:
    where = {"book": {"$in": book_filter}} if book_filter else None
    kwargs = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception:
        # If filter yields no results or error, skip filter
        kwargs.pop("where", None)
        results = collection.query(**kwargs)

    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append(Passage(
            text=doc,
            book=meta.get("book", ""),
            speaker=meta.get("speaker", ""),
            chunk_index=meta.get("chunk_index", 0),
            score=1.0 - dist,  # cosine distance → similarity
            chunk_id=results["ids"][0][len(passages)],
        ))
    return passages


def deduplicate(passages: list[Passage]) -> list[Passage]:
    seen = set()
    out = []
    for p in passages:
        if p.chunk_id not in seen:
            seen.add(p.chunk_id)
            out.append(p)
    return out


def get_passages(question: str) -> list[Passage]:
    settings = load_settings()
    theme_map = load_theme_map()

    # Step 1: Classify themes → determine book filter
    themes = classify(question)
    books = themes_to_books(themes, theme_map)

    # Step 2: Embed question
    embedding = embed_question(question, settings)

    # Step 3: Load collection
    collection = get_collection(settings)

    # Step 4: Filtered query (by theme → books)
    filtered_passages = []
    if books:
        filtered_passages = query_collection(
            collection, embedding,
            top_k=settings["retrieval"]["top_k_filtered"],
            book_filter=books,
        )

    # Step 5: Unfiltered fallback
    unfiltered_passages = query_collection(
        collection, embedding,
        top_k=settings["retrieval"]["top_k_unfiltered"],
        book_filter=None,
    )

    # Step 6: Merge, deduplicate, re-rank by score
    all_passages = filtered_passages + unfiltered_passages
    all_passages = deduplicate(all_passages)
    all_passages.sort(key=lambda p: p.score, reverse=True)

    final_k = settings["retrieval"]["final_top_k"]
    selected = all_passages[:final_k]

    # Step 7: Log retrieval
    log_retrieval(question, themes, books, selected, settings)

    return selected


def format_passages(passages: list[Passage]) -> str:
    lines = ["=== RETRIEVED PASSAGES ===\n"]
    for i, p in enumerate(passages, 1):
        lines.append(f"[Passage {i} | {p.book} | score: {p.score:.3f}]")
        lines.append(p.text)
        lines.append("")
    lines.append("=== END RETRIEVED PASSAGES ===")
    return "\n".join(lines)


def log_retrieval(question: str, themes: list, books: list, passages: list[Passage], settings: dict):
    logs_dir = PROJECT_ROOT / settings["paths"]["logs_dir"]
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / "retrieval_log.jsonl"
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "themes": themes,
        "book_filter": books,
        "passages": [
            {"book": p.book, "score": p.score, "chunk_index": p.chunk_index, "text_preview": p.text[:100]}
            for p in passages
        ],
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is courage?"
    print(f"Question: {question}\n")
    passages = get_passages(question)
    print(format_passages(passages))
