"""
retrieve.py — Given a user question, retrieve the most relevant dialogue passages.
Usage: python scripts/retrieve.py "What is courage?"
Returns formatted passages ready for injection into the agent prompt.
"""

import re
import sys
import json
import yaml
import datetime
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.theme_classifier import classify, themes_to_books

# Module-level caches — initialized on first use, reused for all subsequent calls
_model_cache: dict = {}
_collection_cache: dict = {}
_settings_cache: dict = {}
_bm25_cache: dict = {}  # key "payload" → loaded bm25 index payload
_entity_cache: dict = {}  # key "registry" → entity name → {books, is_speaker}

# Non-speaker entities mentioned in dialogue text but not labeled as speakers in speakers.yaml.
# Short/ambiguous entries carry a context_required regex — the entity only fires when the
# pattern matches the full question, preventing hesitation phrasing ("Er, ...") from
# triggering a book-level misroute.
_EXTRA_ENTITIES: dict[str, dict] = {
    "Diotima": {"books": ["Symposium"]},
    "Gyges": {"books": ["Republic"], "context_required": r"(?:ring|story|tale|myth)\s+of\s+Gyges\b"},
    "Er": {"books": ["Republic"], "context_required": r"(?:myth|story|tale|vision)\s+of\s+Er\b"},
    "Homer": {"books": ["Ion", "Republic", "Protagoras"]},
}


@dataclass
class Passage:
    text: str
    book: str
    speaker: str
    chunk_index: int
    score: float
    chunk_id: str
    char_start_ratio: float = 0.0


# --- Stephanus reference helpers ---

_SECTION_LETTERS = "abcde"
_SECTION_MAP = {"a": 0.0, "b": 0.2, "c": 0.4, "d": 0.6, "e": 0.8}


def _parse_stephanus(ref: str) -> float:
    """Convert Stephanus ref string to float. '80d' → 80.6"""
    if not ref or len(ref) < 2:
        return 0.0
    letter = ref[-1].lower()
    try:
        page = int(ref[:-1])
    except ValueError:
        return 0.0
    return page + _SECTION_MAP.get(letter, 0.0)


def _format_stephanus(val: float) -> str:
    """Convert float to Stephanus ref string. 80.6 → '80d'"""
    page = int(val)
    frac = val - page
    idx = min(round(frac / 0.2), 4)
    return f"{page}{_SECTION_LETTERS[idx]}"


def approx_stephanus(book: str, ratio: float, stephanus_map: dict) -> str:
    """Return approximate Stephanus ref like '~80d', or '' if book not in map."""
    entry = stephanus_map.get(book)
    if not entry:
        return ""
    start_val = _parse_stephanus(entry.get("start", ""))
    end_val = _parse_stephanus(entry.get("end", ""))
    if start_val == 0.0 and end_val == 0.0:
        return ""
    approx = start_val + ratio * (end_val - start_val)
    return f"~{_format_stephanus(approx)}"


def _load_settings() -> tuple[dict, dict]:
    """Load and cache settings.yaml, theme_book_map.yaml, and stephanus_map.yaml."""
    if "s" not in _settings_cache:
        with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
            _settings_cache["s"] = yaml.safe_load(f)
        with open(PROJECT_ROOT / "config" / "theme_book_map.yaml") as f:
            _settings_cache["t"] = yaml.safe_load(f)
        stephanus_path = PROJECT_ROOT / "config" / "stephanus_map.yaml"
        if stephanus_path.exists():
            with open(stephanus_path) as f:
                _settings_cache["stephanus"] = yaml.safe_load(f) or {}
        else:
            _settings_cache["stephanus"] = {}
    return _settings_cache["s"], _settings_cache["t"]


def _get_stephanus_map() -> dict:
    _load_settings()
    return _settings_cache.get("stephanus", {})


def _load_entity_registry() -> dict:
    """Load and cache entity registry from speakers.yaml + _EXTRA_ENTITIES.

    Returns {entity_name: {"books": [...], "is_speaker": bool}}.
    Excludes "Socrates" — too ubiquitous to be useful as a filter.
    """
    if "registry" in _entity_cache:
        return _entity_cache["registry"]

    speakers_path = PROJECT_ROOT / "config" / "speakers.yaml"
    with open(speakers_path) as f:
        speakers_config = yaml.safe_load(f)

    registry: dict = {}
    for book_name, book_cfg in speakers_config["books"].items():
        for speaker in book_cfg.get("speakers", []):
            if speaker == "Socrates":
                continue
            if speaker not in registry:
                registry[speaker] = {"books": [], "is_speaker": True}
            if book_name not in registry[speaker]["books"]:
                registry[speaker]["books"].append(book_name)

    for entity, cfg in _EXTRA_ENTITIES.items():
        if entity not in registry:
            entry: dict = {"books": cfg["books"], "is_speaker": False}
            if "context_required" in cfg:
                entry["context_required"] = cfg["context_required"]
            registry[entity] = entry

    _entity_cache["registry"] = registry
    return registry


def detect_entities(question: str) -> list[dict]:
    """Detect named entities (speakers/characters) in the question.

    Returns list of {"name": str, "books": list[str], "is_speaker": bool},
    sorted by name length (longer names first to prefer specific matches).
    Uses a precompiled word-boundary regex — no API calls.
    """
    registry = _load_entity_registry()

    if "pattern" not in _entity_cache:
        # Sort by length descending so longer names match before shorter substrings
        names = sorted(registry.keys(), key=len, reverse=True)
        pat = r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b'
        _entity_cache["pattern"] = re.compile(pat, re.IGNORECASE)

    matches = _entity_cache["pattern"].findall(question)

    results = []
    seen: set[str] = set()
    for match in matches:
        for name, info in registry.items():
            if name.lower() == match.lower() and name not in seen:
                # Context guard: short/ambiguous entities require a supporting phrase
                ctx_pat = info.get("context_required")
                if ctx_pat and not re.search(ctx_pat, question, re.IGNORECASE):
                    break  # match found but context absent — skip this entity
                seen.add(name)
                results.append({"name": name, **info})
                break

    return results


def _get_collection(settings: dict):
    """Return cached ChromaDB collection, creating it on first call."""
    key = settings["collection_name"]
    if key not in _collection_cache:
        import chromadb
        vectorstore_path = PROJECT_ROOT / settings["paths"]["vectorstore_dir"]
        client = chromadb.PersistentClient(path=str(vectorstore_path))
        _collection_cache[key] = client.get_collection(key)
    return _collection_cache[key]


def _get_model(model_name: str):
    """Return cached SentenceTransformer model, loading it on first call.

    Tries local-only first so CI/offline environments don't hit the network.
    Falls back to downloading if the model hasn't been cached yet.
    Run scripts/download_model.py once to pre-cache.
    """
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        try:
            _model_cache[model_name] = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            # Model not in local cache — download it (requires network)
            _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_question(question: str, settings: dict):
    return _get_model(settings["embedding"]["model"]).encode([question])[0].tolist()


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
            char_start_ratio=float(meta.get("char_start_ratio", 0.0)),
        ))
    return passages


def query_collection_by_speaker(
    collection, embedding, top_k: int, speaker_name: str, fallback_books: list[str]
) -> list[Passage]:
    """Query ChromaDB filtering by speaker metadata field.

    Falls back to a book-level filter if the speaker filter returns no results
    (e.g. the speaker name doesn't match any stored metadata exactly).
    """
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where={"speaker": speaker_name},
        )
        if not results["documents"][0]:
            raise ValueError("no results")
    except Exception:
        # Fall back to book filter for this entity's books
        return query_collection(collection, embedding, top_k, fallback_books or None)

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
            score=1.0 - dist,
            chunk_id=results["ids"][0][len(passages)],
            char_start_ratio=float(meta.get("char_start_ratio", 0.0)),
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


def _load_bm25() -> dict:
    """Load and cache BM25 index payload. Raises if index not built."""
    if "payload" not in _bm25_cache:
        bm25_path = PROJECT_ROOT / "data" / "indexes" / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {bm25_path}. Run: python3 scripts/ingest.py"
            )
        import pickle
        with open(bm25_path, "rb") as f:
            _bm25_cache["payload"] = pickle.load(f)
    return _bm25_cache["payload"]


def query_bm25(question: str, top_k: int, book_filter: list[str] | None = None) -> list[Passage]:
    """Query BM25 index."""
    payload = _load_bm25()

    try:
        bm25 = payload["bm25"]
        chunk_ids = payload["chunk_ids"]
        books = payload["books"]
        speakers = payload["speakers"]
        texts = payload["texts"]
        chunk_indexes = payload["chunk_indexes"]

        tokens = question.lower().split()
        scores = bm25.get_scores(tokens)

        # Gather candidates with score > 0, apply book filter
        candidates = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            if book_filter and books[i] not in book_filter:
                continue
            candidates.append((i, score))

        # Sort by score descending, take top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]

        passages = []
        for i, score in candidates:
            passages.append(Passage(
                text=texts[i],
                book=books[i],
                speaker=speakers[i],
                chunk_index=chunk_indexes[i],
                score=score,
                chunk_id=chunk_ids[i],
                char_start_ratio=0.0,
            ))
        return passages
    except Exception:
        return []


def reciprocal_rank_fusion(lists: list[list[Passage]], k: int = 60) -> list[Passage]:
    """Merge multiple ranked passage lists using Reciprocal Rank Fusion.

    Score for each passage = sum(1 / (k + rank_i)) across all lists where it appears.
    k=60 is the standard constant (Cormack et al. 2009).
    """
    scores: dict[str, float] = {}
    passage_map: dict[str, Passage] = {}

    for ranked_list in lists:
        for rank, passage in enumerate(ranked_list, 1):
            cid = passage.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in passage_map:
                passage_map[cid] = passage

    merged = sorted(passage_map.values(), key=lambda p: scores[p.chunk_id], reverse=True)
    # Attach RRF score for downstream use
    for p in merged:
        p.score = scores[p.chunk_id]
    return merged


def get_primary_books(themes: list[str], theme_map: dict) -> list[str]:
    """Return only the PRIMARY books for the given themes."""
    books = []
    for theme in themes:
        if theme in theme_map.get("themes", {}):
            books.extend(theme_map["themes"][theme].get("primary", []))
    return list(dict.fromkeys(books))


def get_passages(question: str) -> list[Passage]:
    settings, theme_map = _load_settings()

    # Step 1: Classify themes → determine book filters
    themes = classify(question)
    primary_books = get_primary_books(themes, theme_map)
    all_books = themes_to_books(themes, theme_map)  # primary + secondary

    # Step 1b: Detect named entities (speakers/characters) in the question
    entities = detect_entities(question)

    # Step 2: Embed question
    embedding = embed_question(question, settings)

    # Step 3: Load collection
    collection = _get_collection(settings)

    # Step 4a: Primary-books-only query (guaranteed top-source coverage)
    primary_passages = []
    if primary_books:
        primary_passages = query_collection(
            collection, embedding,
            top_k=settings["retrieval"]["top_k_filtered"],
            book_filter=primary_books,
        )

    # Step 4b: Broader filtered query (primary + secondary books)
    filtered_passages = []
    if all_books and all_books != primary_books:
        filtered_passages = query_collection(
            collection, embedding,
            top_k=settings["retrieval"]["top_k_filtered"],
            book_filter=all_books,
        )

    # Step 5: Unfiltered fallback
    unfiltered_passages = query_collection(
        collection, embedding,
        top_k=settings["retrieval"]["top_k_unfiltered"],
        book_filter=None,
    )

    # Step 5b: Entity-filtered query (supplementary, only when entities detected)
    # For labeled speakers: filter by speaker metadata field (precise)
    # For non-speaker entities (Diotima, Gyges): filter by their known books
    # Cap at 2 entities so comparison queries ("Meno and Protagoras") get both covered.
    entity_passage_groups: list[list] = []
    for entity in entities[:2]:
        if entity["is_speaker"]:
            ep = query_collection_by_speaker(
                collection, embedding,
                top_k=settings["retrieval"]["top_k_filtered"],
                speaker_name=entity["name"],
                fallback_books=entity["books"],
            )
        else:
            ep = query_collection(
                collection, embedding,
                top_k=settings["retrieval"]["top_k_filtered"],
                book_filter=entity["books"],
            )
        entity_passage_groups.append(ep)

    # Step 5c: BM25 lexical retrieval
    bm25_passages = query_bm25(question, top_k=settings["retrieval"]["top_k_unfiltered"], book_filter=None)

    # Step 5d: Civil-disobedience boost — when no entities are active and the question
    # contains law/obedience keywords, add a Crito-only retrieval pass to the RRF fusion.
    # This pushes Crito to rank-1 for queries like "Is it right to break an unjust law?"
    # without a special-case reranker. Guard: only fires when Crito is already in all_books
    # (theme classifier matched justice/obedience) to avoid false positives elsewhere.
    _OBEDIENCE_RE = re.compile(r'\b(?:break|obey|disobey|disobedience|escape|flee|prison)\b', re.IGNORECASE)
    crito_boost_passages: list = []
    if not entities and _OBEDIENCE_RE.search(question) and "Crito" in all_books:
        crito_boost_passages = query_collection(
            collection, embedding,
            top_k=settings["retrieval"]["top_k_filtered"],
            book_filter=["Crito"],
        )

    # Step 6: Assemble final selection.
    # Use RRF to merge the three dense retrieval lists (primary, filtered, unfiltered) —
    # each list targets a different book scope, and RRF resolves ranking conflicts between
    # them more accurately than a naive sort-by-score merge.
    # BM25 is additive overflow only: it fills remaining slots after dense selection.
    # This prevents BM25 proper-noun bias (e.g. "Alcibiades" saturating Alcibiades I)
    # from displacing semantically correct dense results.
    final_k = settings["retrieval"]["final_top_k"]

    # RRF merge of the theme-based dense lists (+ optional Crito boost)
    theme_lists = [l for l in [primary_passages, filtered_passages, unfiltered_passages, crito_boost_passages] if l]
    fused_theme = reciprocal_rank_fusion(theme_lists) if len(theme_lists) > 1 else deduplicate(primary_passages + filtered_passages + unfiltered_passages)

    # Step 6a: Define speaker reranker key (used for both slot reservation and reranking).
    # Five-tier key: (exact metadata, early position, attribution, entity ratio, count).
    # Tier 1: exact p.speaker == entity name (best signal, rare due to speaker=mixed ingest).
    # Tier 2: entity name in opening 150 chars — chunk likely starts with their speech.
    # Tier 3: dialogue attribution patterns ("X said", "said X") — text-based speaker signal.
    # Tier 4: entity-to-Socrates ratio — penalizes Socrates-dominated chunks for entity queries.
    # Tier 5: raw entity name mention count (weakest proxy).
    _speaker_rerank_key = None
    if entities:
        entity_names_lower = [e["name"].lower() for e in entities]
        entity_names_set = {e["name"] for e in entities}

        # Detect "X about Y" pattern — Y is typically Socrates (excluded from entity registry).
        # Captures queries like "What does Alcibiades say about Socrates?" where we want chunks
        # containing both the primary entity AND the secondary target together.
        _ABOUT_RE = re.compile(r'\b(?:about|regarding|concerning)\s+([A-Za-z]+)', re.IGNORECASE)
        _about_match = _ABOUT_RE.search(question)
        _secondary_target = _about_match.group(1).lower() if _about_match else None

        _ATTRIB_VERBS = [
            "said", "replied", "answered", "continued", "asked",
            "observed", "rejoined", "spoke", "began", "added",
            "exclaimed", "urged",
        ]

        def _attribution_score(text_lower: str, name_lower: str) -> int:
            count = 0
            for verb in _ATTRIB_VERBS:
                if re.search(rf'\b{re.escape(name_lower)}\s+{verb}\b', text_lower):
                    count += 1
                if re.search(rf'\b{verb}\s+{re.escape(name_lower)}\b', text_lower):
                    count += 1
            return count

        def _speaker_rerank_key(p: Passage) -> tuple:
            # Tier 1: exact speaker metadata match
            exact = 1 if p.speaker in entity_names_set else 0
            text_lower = p.text.lower()
            # Tier 2: entity name in opening 150 chars
            opening = text_lower[:150]
            early = 1 if any(n in opening for n in entity_names_lower) else 0
            # Tier 3: co-occurrence of primary entity + secondary target ("X about Y")
            if _secondary_target:
                has_both = 1 if (
                    any(n in text_lower for n in entity_names_lower)
                    and _secondary_target in text_lower
                ) else 0
            else:
                has_both = 0
            # Tier 4: dialogue attribution patterns ("X said", "said X")
            attrib = sum(_attribution_score(text_lower, n) for n in entity_names_lower)
            # Tier 5: entity-to-Socrates ratio — penalizes Socrates-dominated chunks
            entity_count = sum(text_lower.count(n) for n in entity_names_lower)
            soc_count = text_lower.count("socrates")
            total = entity_count + soc_count
            entity_ratio = entity_count / total if total > 0 else 0.0
            # Tier 6: raw entity mention count (tiebreaker)
            return (exact, early, has_both, attrib, entity_ratio, entity_count)

    selected = []
    seen_ids = set()

    if entity_passage_groups:
        # Entity query detected: reserve slots from entity-filtered passages (by dense score).
        # 1 entity → 2 reserved slots; 2 entities → 1 slot each.
        slots_per = 2 if len(entity_passage_groups) == 1 else 1
        for ep_group in entity_passage_groups:
            ep_dedup = deduplicate(ep_group)
            ep_dedup.sort(key=lambda p: p.score, reverse=True)
            added = 0
            for p in ep_dedup:
                if added >= slots_per:
                    break
                if p.chunk_id not in seen_ids:
                    selected.append(p)
                    seen_ids.add(p.chunk_id)
                    added += 1
    else:
        # No entity detected: reserve 1 slot for the top primary-book passage (by RRF score)
        if primary_books:
            for p in fused_theme:
                if p.book in primary_books and p.chunk_id not in seen_ids:
                    selected.append(p)
                    seen_ids.add(p.chunk_id)
                    break

    # Record how many slots are entity/primary reserved — reranker must not touch these
    n_reserved = len(selected)

    # Fill remaining slots by theme-fused RRF score
    for p in fused_theme:
        if len(selected) >= final_k:
            break
        if p.chunk_id not in seen_ids:
            selected.append(p)
            seen_ids.add(p.chunk_id)

    # BM25 overflow: fill any remaining slots with lexically-matched passages
    for p in bm25_passages:
        if len(selected) >= final_k:
            break
        if p.chunk_id not in seen_ids:
            selected.append(p)
            seen_ids.add(p.chunk_id)

    # Step 6b: Speaker-intent rerank — sort unreserved slots by speaker signals.
    # Reserved positions stay fixed to preserve entity/primary slot guarantees.
    if _speaker_rerank_key:
        unreserved = selected[n_reserved:]
        unreserved.sort(key=_speaker_rerank_key, reverse=True)
        selected = selected[:n_reserved] + unreserved

    # Step 7: Log retrieval
    log_retrieval(question, themes, all_books, selected, settings, entities=entities)

    return selected


def format_passages(passages: list[Passage]) -> str:
    stephanus_map = _get_stephanus_map()
    lines = ["=== RETRIEVED PASSAGES ===\n"]
    for i, p in enumerate(passages, 1):
        speaker_tag = f" | speaker: {p.speaker}" if p.speaker and p.speaker not in ("", "mixed") else ""
        steph_ref = approx_stephanus(p.book, p.char_start_ratio, stephanus_map)
        steph_tag = f" | {steph_ref}" if steph_ref else ""
        lines.append(f"[Passage {i} | {p.book}{speaker_tag}{steph_tag} | score: {p.score:.3f}]")
        lines.append(p.text)
        lines.append("")
    lines.append("=== END RETRIEVED PASSAGES ===")
    return "\n".join(lines)


def log_retrieval(question: str, themes: list, books: list, passages: list[Passage], settings: dict, entities: list | None = None):
    logs_dir = PROJECT_ROOT / settings["paths"]["logs_dir"]
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / "retrieval_log.jsonl"
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "themes": themes,
        "entities_detected": [e["name"] for e in (entities or [])],
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
