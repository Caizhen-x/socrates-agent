"""
ingest.py — Parse Platonic dialogues into chunks, embed, and store in ChromaDB.
Run once (or re-run to rebuild the index).
Usage: python scripts/ingest.py
"""

import os
import re
import sys
import json
import yaml
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Chunk:
    text: str
    book: str
    chunk_index: int
    speaker: Optional[str]
    section_hint: str
    is_overlap: bool = False
    char_start: int = 0

    def chunk_id(self) -> str:
        h = hashlib.md5(f"{self.book}_{self.chunk_index}_{self.text[:50]}".encode()).hexdigest()[:12]
        return f"{self.book}_{self.chunk_index}_{h}"


def load_settings():
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def load_speakers():
    with open(PROJECT_ROOT / "config" / "speakers.yaml") as f:
        return yaml.safe_load(f)


def estimate_tokens(text: str) -> int:
    # Rough estimate: 1 token ≈ 4 chars
    return len(text) // 4


def split_into_sentences(text: str) -> list[str]:
    # Simple sentence splitter on period/exclamation/question followed by space+capital
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_section(text: str, section_name: str) -> str:
    """Extract a named section (Charmides, Laches, Lysis) from a combined file."""
    # Find the section header
    pattern = rf'\b{re.escape(section_name.upper())}\b'
    match = re.search(pattern, text)
    if not match:
        # Try title case
        pattern = rf'\b{re.escape(section_name)}\b'
        match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return text

    start = match.start()
    # Find next major section (another dialogue title at the start of a line)
    dialogue_titles = ["CHARMIDES", "LACHES", "LYSIS"]
    other_titles = [t for t in dialogue_titles if t != section_name.upper()]
    next_pos = len(text)
    for title in other_titles:
        m = re.search(rf'\n\s*{title}\b', text[start + 100:])
        if m:
            pos = start + 100 + m.start()
            if pos < next_pos:
                next_pos = pos
    return text[start:next_pos]


def chunk_prose_by_paragraph(text: str, book: str, settings: dict) -> list[Chunk]:
    """
    Chunk prose text by paragraphs. Merge small paragraphs, split large ones.
    Used for books without clear speaker labels.
    """
    max_tokens = settings["chunking"]["max_tokens"]
    min_tokens = settings["chunking"]["min_tokens"]
    overlap_sents = settings["chunking"]["overlap_sentences"]

    # Split on double newlines (paragraph breaks)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks = []
    buffer = ""
    chunk_index = 0

    for para in paragraphs:
        if not para or len(para) < 20:
            continue

        candidate = (buffer + " " + para).strip() if buffer else para

        if estimate_tokens(candidate) <= max_tokens:
            buffer = candidate
        else:
            # Flush buffer if it meets min size
            if buffer and estimate_tokens(buffer) >= min_tokens:
                chunks.append(Chunk(
                    text=buffer,
                    book=book,
                    chunk_index=chunk_index,
                    speaker=None,
                    section_hint=book,
                ))
                chunk_index += 1
                # Start new buffer with overlap
                sents = split_into_sentences(buffer)
                overlap = " ".join(sents[-overlap_sents:]) if len(sents) >= overlap_sents else buffer
                buffer = (overlap + " " + para).strip()
            else:
                # Buffer too small — just append even if over limit
                buffer = candidate

    if buffer and estimate_tokens(buffer) >= min_tokens:
        chunks.append(Chunk(
            text=buffer,
            book=book,
            chunk_index=chunk_index,
            speaker=None,
            section_hint=book,
        ))

    return chunks


def chunk_labeled_dialogue(text: str, book: str, speakers: list[str], settings: dict) -> list[Chunk]:
    """
    Parse dialogue with 'Speaker. text' or 'Speaker:' labeled turns.
    Groups turns into chunks of 300-800 tokens.
    """
    max_tokens = settings["chunking"]["max_tokens"]
    min_tokens = settings["chunking"]["min_tokens"]
    overlap_sents = settings["chunking"]["overlap_sentences"]

    # Build pattern for speaker labels
    # Matches: "  Socrates. " or "  Meno. " at start of a line (with optional leading whitespace)
    speaker_pattern = r'(?:^|\n)\s{0,4}(' + '|'.join(re.escape(s) for s in speakers) + r')\.\s+'

    turns = []
    pos = 0
    for m in re.finditer(speaker_pattern, text, re.IGNORECASE):
        if pos < m.start():
            # text before first speaker — skip (preamble)
            pass
        speaker = m.group(1).capitalize()
        content_start = m.end()
        turns.append((speaker, content_start, m.start()))
        pos = m.end()

    if not turns:
        # Fallback to prose chunking
        return chunk_prose_by_paragraph(text, book, settings)

    # Extract text for each turn
    turn_texts = []
    for i, (speaker, content_start, _) in enumerate(turns):
        end = turns[i + 1][2] if i + 1 < len(turns) else len(text)
        turn_text = text[content_start:end].strip()
        if turn_text:
            turn_texts.append((speaker, turn_text))

    # Group turns into chunks
    chunks = []
    chunk_index = 0
    buffer_text = ""
    buffer_speaker = None
    last_sents_for_overlap = []

    for speaker, turn_text in turn_texts:
        entry = f"{speaker}: {turn_text}"
        candidate = (buffer_text + "\n\n" + entry).strip() if buffer_text else entry

        if estimate_tokens(candidate) <= max_tokens:
            buffer_text = candidate
            buffer_speaker = speaker
            last_sents_for_overlap = split_into_sentences(turn_text)
        else:
            if buffer_text and estimate_tokens(buffer_text) >= min_tokens:
                chunks.append(Chunk(
                    text=buffer_text,
                    book=book,
                    chunk_index=chunk_index,
                    speaker=buffer_speaker,
                    section_hint=book,
                ))
                chunk_index += 1
                # Overlap: last N sentences of buffer
                overlap_text = " ".join(last_sents_for_overlap[-overlap_sents:]) if last_sents_for_overlap else ""
                buffer_text = (overlap_text + "\n\n" + entry).strip() if overlap_text else entry
                buffer_speaker = speaker
            else:
                buffer_text = candidate
                buffer_speaker = speaker
            last_sents_for_overlap = split_into_sentences(turn_text)

    if buffer_text and estimate_tokens(buffer_text) >= min_tokens:
        chunks.append(Chunk(
            text=buffer_text,
            book=book,
            chunk_index=chunk_index,
            speaker=buffer_speaker,
            section_hint=book,
        ))

    return chunks


def parse_book(book_name: str, book_config: dict, settings: dict) -> list[Chunk]:
    books_dir = PROJECT_ROOT / settings["paths"]["books_dir"]
    filepath = books_dir / book_config["file"]

    if not filepath.exists():
        print(f"  WARNING: {filepath} not found, skipping.")
        return []

    text = filepath.read_text(encoding="utf-8")

    # Extract section if needed
    if "section" in book_config:
        text = extract_section(text, book_config["section"])

    fmt = book_config.get("format", "dialogue_prose")
    speakers = book_config.get("speakers", [])

    if fmt == "dialogue_labeled":
        chunks = chunk_labeled_dialogue(text, book_name, speakers, settings)
    else:
        # prose or unlabeled dialogue — use paragraph chunking
        chunks = chunk_prose_by_paragraph(text, book_name, settings)

    return chunks


def embed_and_store(chunks: list[Chunk], settings: dict):
    import chromadb
    from sentence_transformers import SentenceTransformer

    vectorstore_path = PROJECT_ROOT / settings["paths"]["vectorstore_dir"]
    vectorstore_path.mkdir(exist_ok=True)

    print("Loading embedding model...")
    model = SentenceTransformer(settings["embedding"]["model"])

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(vectorstore_path))

    # Delete existing collection to rebuild cleanly
    try:
        client.delete_collection(settings["collection_name"])
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=settings["collection_name"],
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 64
    total = len(chunks)
    print(f"Embedding and storing {total} chunks...")

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        ids = [c.chunk_id() for c in batch]
        metadatas = [
            {
                "book": c.book,
                "chunk_index": c.chunk_index,
                "speaker": c.speaker or "",
                "section_hint": c.section_hint,
                "is_overlap": str(c.is_overlap),
            }
            for c in batch
        ]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        print(f"  Stored {min(i + batch_size, total)}/{total}", end="\r")

    print(f"\nDone. {total} chunks in ChromaDB.")
    return collection


def main():
    settings = load_settings()
    speakers_config = load_speakers()

    all_chunks = []
    books = speakers_config["books"]

    for book_name, book_config in books.items():
        print(f"Parsing {book_name}...")
        chunks = parse_book(book_name, book_config, settings)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Save chunk stats for debugging
    logs_dir = PROJECT_ROOT / settings["paths"]["logs_dir"]
    logs_dir.mkdir(exist_ok=True)
    stats = {}
    for c in all_chunks:
        stats[c.book] = stats.get(c.book, 0) + 1
    print("Chunks per book:", json.dumps(stats, indent=2))

    embed_and_store(all_chunks, settings)


if __name__ == "__main__":
    main()
