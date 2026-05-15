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
import datetime
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



def infer_dominant_speaker(chunk_text: str, speakers: list[str]) -> str:
    """
    Heuristic speaker attribution for prose-format chunks.
    Scans for 'X said', 'said X', 'X replied', 'X answered', 'X continued' patterns.
    Also counts 'I said'/'I replied' as potential Socrates cues when Socrates is a speaker.
    Returns the most-mentioned speaker name, 'mixed' if tied, or '' if undetectable.
    """
    counts: dict[str, int] = {s: 0 for s in speakers}
    lower = chunk_text.lower()

    for speaker in speakers:
        s = speaker.lower()
        # "Socrates said", "said Socrates", "Socrates replied", etc.
        patterns = [
            rf'\b{re.escape(s)}\s+said\b',
            rf'\bsaid\s+{re.escape(s)}\b',
            rf'\b{re.escape(s)}\s+replied\b',
            rf'\breplied\s+{re.escape(s)}\b',
            rf'\b{re.escape(s)}\s+answered\b',
            rf'\banswered\s+{re.escape(s)}\b',
            rf'\b{re.escape(s)}\s+continued\b',
            rf'\b{re.escape(s)}\s+asked\b',
            rf'\b{re.escape(s)}\s+observed\b',
            rf'\b{re.escape(s)}\s+rejoined\b',
            rf'\b{re.escape(s)}\s+spoke\b',
            rf'\b{re.escape(s)}\s+began\b',
            rf'\b{re.escape(s)}\s+went\s+on\b',
            rf'\b{re.escape(s)}\s+added\b',
            rf'\b{re.escape(s)}\s+exclaimed\b',
            rf'\b{re.escape(s)}\s+urged\b',
        ]
        for pat in patterns:
            counts[speaker] += len(re.findall(pat, lower))

    # First-person narration cues: "I said", "I replied" etc. → Socrates if present
    socrates_key = next((s for s in speakers if s.lower() == "socrates"), None)
    if socrates_key:
        first_person_patterns = [
            r'\bI said\b', r'\bI replied\b', r'\bI answered\b',
            r'\bI continued\b', r'\bI asked\b', r'\bI rejoined\b',
            r'\bI spoke\b', r'\bI began\b', r'\bI went on\b',
            r'\bI added\b', r'\bI exclaimed\b', r'\bI urged\b',
        ]
        for pat in first_person_patterns:
            counts[socrates_key] += len(re.findall(pat, chunk_text))

    total = sum(counts.values())
    if total == 0:
        return ""

    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    best_name, best_count = top[0]
    second_count = top[1][1] if len(top) > 1 else 0

    # Require at least 60% share to call it a dominant speaker
    if best_count == 0:
        return ""
    if total > 0 and best_count / total >= 0.6:
        return best_name
    return "mixed"


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
                # Start new buffer with overlap; if new buffer already oversized,
                # sentence-split it and flush any full sub-chunks immediately.
                sents = split_into_sentences(buffer)
                overlap = " ".join(sents[-overlap_sents:]) if len(sents) >= overlap_sents else buffer
                new_buffer = (overlap + " " + para).strip()
                if estimate_tokens(new_buffer) <= max_tokens:
                    buffer = new_buffer
                else:
                    # Para itself is huge — sentence-split it
                    sub_sents = split_into_sentences(new_buffer)
                    sub_buf = ""
                    for sent in sub_sents:
                        sub_candidate = (sub_buf + " " + sent).strip() if sub_buf else sent
                        if estimate_tokens(sub_candidate) <= max_tokens:
                            sub_buf = sub_candidate
                        else:
                            if sub_buf and estimate_tokens(sub_buf) >= min_tokens:
                                chunks.append(Chunk(
                                    text=sub_buf,
                                    book=book,
                                    chunk_index=chunk_index,
                                    speaker=None,
                                    section_hint=book,
                                ))
                                chunk_index += 1
                            sub_buf = sent
                    buffer = sub_buf
            else:
                # Buffer too small to flush — sentence-split the oversized candidate
                sents = split_into_sentences(candidate)
                sub_buf = ""
                for sent in sents:
                    sub_candidate = (sub_buf + " " + sent).strip() if sub_buf else sent
                    if estimate_tokens(sub_candidate) <= max_tokens:
                        sub_buf = sub_candidate
                    else:
                        if sub_buf and estimate_tokens(sub_buf) >= min_tokens:
                            chunks.append(Chunk(
                                text=sub_buf,
                                book=book,
                                chunk_index=chunk_index,
                                speaker=None,
                                section_hint=book,
                            ))
                            chunk_index += 1
                        sub_buf = sent
                buffer = sub_buf

    if buffer and estimate_tokens(buffer) >= min_tokens:
        chunks.append(Chunk(
            text=buffer,
            book=book,
            chunk_index=chunk_index,
            speaker=None,
            section_hint=book,
        ))

    return chunks


def build_speaker_pattern(speakers: list[str]) -> str:
    """
    Build regex that matches full names (Socrates.) and common abbreviations (Soc., La.).
    Generates 2, 3, 4-char prefixes to cover Jowett's varied abbreviation styles.
    """
    alts = []
    for s in speakers:
        alts.append(re.escape(s))
        for length in (4, 3, 2):
            if len(s) > length:
                alts.append(re.escape(s[:length]))
    # Deduplicate, longest first to avoid partial matches
    alts = sorted(set(alts), key=len, reverse=True)
    return r'(?:^|\n)\s{0,6}(' + '|'.join(alts) + r')\.\s+'


def normalize_speaker(abbrev: str, speakers: list[str]) -> str:
    """Map an abbreviation back to the full speaker name."""
    abbrev_lower = abbrev.lower()
    for s in speakers:
        if s.lower().startswith(abbrev_lower) or abbrev_lower.startswith(s[:3].lower()):
            return s
    return abbrev.capitalize()


def chunk_labeled_dialogue(
    text: str,
    book: str,
    speakers: list[str],
    settings: dict,
    pattern_builder=None,
) -> list[Chunk]:
    """
    Parse dialogue with 'Speaker. text' or 'Abbrev. text' labeled turns.
    Groups turns into chunks of 300-800 tokens.
    pattern_builder: optional callable(speakers) -> regex str.
                     Defaults to build_speaker_pattern (period-style).
    """
    max_tokens = settings["chunking"]["max_tokens"]
    min_tokens = settings["chunking"]["min_tokens"]
    overlap_sents = settings["chunking"]["overlap_sentences"]

    if pattern_builder is None:
        pattern_builder = build_speaker_pattern
    speaker_pattern = pattern_builder(speakers)

    turns = []
    for m in re.finditer(speaker_pattern, text, re.IGNORECASE):
        speaker = normalize_speaker(m.group(1), speakers)
        content_start = m.end()
        turns.append((speaker, content_start, m.start()))

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

    # Split oversized individual turns at sentence boundaries before grouping
    split_turns = []
    for speaker, turn_text in turn_texts:
        if estimate_tokens(turn_text) <= max_tokens:
            split_turns.append((speaker, turn_text))
        else:
            # Turn is too long — split into sentence-boundary sub-chunks
            sents = split_into_sentences(turn_text)
            sub_buf = ""
            for sent in sents:
                candidate = (sub_buf + " " + sent).strip() if sub_buf else sent
                if estimate_tokens(candidate) <= max_tokens:
                    sub_buf = candidate
                else:
                    if sub_buf:
                        split_turns.append((speaker, sub_buf))
                    sub_buf = sent
            if sub_buf:
                split_turns.append((speaker, sub_buf))

    # Group turns into chunks
    chunks = []
    chunk_index = 0
    buffer_text = ""
    buffer_speakers: set[str] = set()
    last_sents_for_overlap = []

    for speaker, turn_text in split_turns:
        entry = f"{speaker}: {turn_text}"
        candidate = (buffer_text + "\n\n" + entry).strip() if buffer_text else entry

        if estimate_tokens(candidate) <= max_tokens:
            buffer_text = candidate
            buffer_speakers.add(speaker)
            last_sents_for_overlap = split_into_sentences(turn_text)
        else:
            if buffer_text and estimate_tokens(buffer_text) >= min_tokens:
                speaker_val = next(iter(buffer_speakers)) if len(buffer_speakers) == 1 else "mixed"
                chunks.append(Chunk(
                    text=buffer_text,
                    book=book,
                    chunk_index=chunk_index,
                    speaker=speaker_val,
                    section_hint=book,
                ))
                chunk_index += 1
                overlap_text = " ".join(last_sents_for_overlap[-overlap_sents:]) if last_sents_for_overlap else ""
                buffer_text = (overlap_text + "\n\n" + entry).strip() if overlap_text else entry
                # Guard: drop overlap if it pushes the seed buffer over the size limit
                if overlap_text and estimate_tokens(buffer_text) > max_tokens:
                    buffer_text = entry
                buffer_speakers = {speaker}
            else:
                buffer_text = candidate
                buffer_speakers.add(speaker)
            last_sents_for_overlap = split_into_sentences(turn_text)

    if buffer_text and estimate_tokens(buffer_text) >= min_tokens:
        speaker_val = next(iter(buffer_speakers)) if len(buffer_speakers) == 1 else "mixed"
        chunks.append(Chunk(
            text=buffer_text,
            book=book,
            chunk_index=chunk_index,
            speaker=speaker_val,
            section_hint=book,
        ))

    return chunks


def _is_front_matter_line(line: str) -> bool:
    """
    Return True if a line looks like front-matter (blank, cast roster, scene block)
    rather than actual dialogue or prose.
    """
    stripped = line.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if lower.startswith("scene") or lower.startswith("place"):
        return True
    # "And others who are mute auditors." — Jowett cast footnote
    if lower.startswith("and others"):
        return True
    # First line of a scene description paragraph — caller tracks continuation
    if lower.startswith("the scene"):
        return True
    # Cast rosters: 2+ all-caps words of length ≥3
    # (e.g., "SOCRATES, who is the narrator.  CEPHALUS." or "GLAUCON. THRASYMACHUS.")
    # Lines with a colon but no caps words are speaker labels — not front matter.
    caps_words = re.findall(r'\b[A-Z]{3,}\b', stripped)
    if not caps_words and ":" in stripped:
        return False  # speaker label like "SOCRATES: ..."  ← actually has caps, handled above
    if len(caps_words) >= 2:
        return True
    # Single isolated name line (e.g., "POLEMARCHUS." with nothing else)
    if re.match(r'^[A-Z]{3,}[.,]?$', stripped):
        return True
    return False


def strip_front_matter(text: str) -> str:
    """
    Remove edition metadata, introductions, and PERSONS OF THE DIALOGUE headers,
    including cast rosters and multi-line scene description blocks.
    Uses the last occurrence of 'persons of the dialogue' as the anchor point.
    Falls back to stripping past the 'translated by' line if no PERSONS found.
    """
    lines = text.split('\n')

    # Find LAST "persons of the dialogue" line
    persons_idx = -1
    for i, line in enumerate(lines):
        if "persons of the dialogue" in line.lower():
            persons_idx = i

    if persons_idx >= 0:
        start = persons_idx + 1

        # Skip front-matter after the PERSONS anchor.
        # Track multi-line scene description paragraphs so their continuation
        # lines are not mistaken for actual content (e.g., Republic's
        # "The scene is laid... / the whole dialogue is narrated by Socrates...").
        in_scene_para = False
        while start < len(lines):
            stripped = lines[start].strip()
            fm = _is_front_matter_line(lines[start])

            if not stripped:
                # Blank line ends any ongoing scene paragraph
                in_scene_para = False
                start += 1
            elif in_scene_para:
                # Continuation of a scene description — skip regardless
                start += 1
            elif fm:
                # Mark scene-description paragraph start so continuations are skipped
                if stripped.lower().startswith("the scene"):
                    in_scene_para = True
                start += 1
            else:
                break

        return '\n'.join(lines[start:])

    # Fallback: skip past "translated by" line + blanks
    for i, line in enumerate(lines):
        if "translated by" in line.lower():
            start = i + 1
            while start < len(lines) and lines[start].strip() == "":
                start += 1
            return '\n'.join(lines[start:])

    return text  # no front matter detected


def build_speaker_pattern_caps(speakers: list[str]) -> str:
    """
    Build regex that matches ALL-CAPS speaker labels: 'SOCRATES:  text'
    Used for books like Euthydemus, Menexenus, Alcibiades I/II, Hippias Minor.
    """
    alts = [re.escape(s.upper()) for s in speakers]
    alts = sorted(set(alts), key=len, reverse=True)
    return r'(?:^|\n)\s{0,6}(' + '|'.join(alts) + r'):\s+'


def parse_book(book_name: str, book_config: dict, settings: dict) -> list[Chunk]:
    books_dir = PROJECT_ROOT / settings["paths"]["books_dir"]
    filepath = books_dir / book_config["file"]

    if not filepath.exists():
        print(f"  WARNING: {filepath} not found, skipping.")
        return []

    text = filepath.read_text(encoding="utf-8")
    text = strip_front_matter(text)

    fmt = book_config.get("format", "dialogue_prose")
    speakers = book_config.get("speakers", [])

    if fmt == "dialogue_labeled":
        chunks = chunk_labeled_dialogue(text, book_name, speakers, settings)
    elif fmt == "dialogue_labeled_caps":
        chunks = chunk_labeled_dialogue(
            text, book_name, speakers, settings,
            pattern_builder=build_speaker_pattern_caps,
        )
    else:
        # prose or unlabeled dialogue — use paragraph chunking
        chunks = chunk_prose_by_paragraph(text, book_name, settings)
        # Infer speaker for each prose chunk via attribution heuristic
        if speakers:
            for chunk in chunks:
                chunk.speaker = infer_dominant_speaker(chunk.text, speakers)

    return chunks


def embed_and_store(chunks: list[Chunk], settings: dict):
    import chromadb
    from sentence_transformers import SentenceTransformer

    vectorstore_path = PROJECT_ROOT / settings["paths"]["vectorstore_dir"]
    vectorstore_path.mkdir(exist_ok=True)

    print("Loading embedding model...")
    model_name = settings["embedding"]["model"]
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except OSError:
        model = SentenceTransformer(model_name)

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

    # Precompute total chunks per book for char_start_ratio calculation
    book_totals: dict[str, int] = {}
    for c in chunks:
        book_totals[c.book] = book_totals.get(c.book, 0) + 1

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
                # Fractional position within this book (0.0 = start, 1.0 = end)
                # Used to compute approximate Stephanus references at retrieval time.
                "char_start_ratio": round(c.chunk_index / max(book_totals[c.book] - 1, 1), 4),
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
    build_bm25_index(all_chunks)
    write_build_info(all_chunks, settings)


def build_bm25_index(all_chunks: list):
    import pickle
    from rank_bm25 import BM25Okapi

    idx_dir = PROJECT_ROOT / "data" / "indexes"
    idx_dir.mkdir(parents=True, exist_ok=True)

    print("Building BM25 lexical index...")
    tokenized = [c.text.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    payload = {
        "bm25": bm25,
        "chunk_ids": [c.chunk_id() for c in all_chunks],
        "books": [c.book for c in all_chunks],
        "speakers": [c.speaker or "" for c in all_chunks],
        "texts": [c.text for c in all_chunks],
        "chunk_indexes": [c.chunk_index for c in all_chunks],
        "char_starts": [c.char_start for c in all_chunks],
    }
    out_path = idx_dir / "bm25.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"BM25 index written to {out_path} ({len(all_chunks)} chunks)")


def write_build_info(all_chunks: list, settings: dict):
    build_dir = PROJECT_ROOT / "data" / "corpus"
    build_dir.mkdir(parents=True, exist_ok=True)

    settings_hash = hashlib.sha256(
        (PROJECT_ROOT / "config" / "settings.yaml").read_bytes()
    ).hexdigest()

    books_dir = PROJECT_ROOT / "Books"
    book_hashes = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(books_dir.glob("*.txt"))
    }

    info = {
        "settings_hash": settings_hash,
        "book_file_hashes": book_hashes,
        "chunk_count": len(all_chunks),
        "build_timestamp": datetime.datetime.now().isoformat(),
        "embedding_model": settings["embedding"]["model"],
        "collection_name": settings["collection_name"],
    }
    with open(build_dir / "build_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Build manifest written to data/corpus/build_info.json ({len(all_chunks)} chunks)")


if __name__ == "__main__":
    main()
