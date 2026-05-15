# Socrates Agent — Handoff Document

## Goal

A RAG-based Socratic dialogue agent that speaks as Socrates of Athens using Platonic dialogue passages retrieved from ChromaDB. The agent runs via `python3 scripts/socrates_loop.py`.

---

## Current State

### Vectorstore
- **1085 chunks** across 27 Platonic dialogues
- ChromaDB at `vectorstore/` — rebuild with `python3 scripts/ingest.py`
- Embedding model: `sentence-transformers/all-mpnet-base-v2` (locally cached)

### Architecture
- `scripts/socrates_loop.py` — main conversation loop with dialectical mode cycling
- `scripts/retrieve.py` — RAG retrieval with theme classification + entity detection + Stephanus references
- `scripts/ingest.py` — ingestion pipeline (chunking + embedding + ChromaDB storage)
- `scripts/response_validator.py` — character guardrail checks
- `scripts/memory_writer.py` — session digest persistence
- `scripts/eval_retrieval.py` — evaluates retrieval against gold set (38 questions)
- `tests/test_retrieve.py` — spot-checks 10 key queries with PASS/WEAK/SOFT/FAIL/PASS* status
- `tests/retrieval_gold.yaml` — 38-question gold evaluation set
- `config/settings.yaml` — all tunable parameters
- `config/theme_book_map.yaml` — theme-to-book routing (primary/secondary) + keyword index
- `config/speakers.yaml` — book list, file names, speaker lists, format types

---

## What Was Done (Prior Sessions)

### Sessions 1-2: Core fixes (socrates_loop.py, ingest.py, CLAUDE.md)
See prior content — synthesis loop, vocative monotony, ingest speaker attribution, caps-format books, front matter stripping, oversized chunk handling. All fixed. Vectorstore rebuilt to 1085 chunks.

### Session 3: Retrieval quality improvements
1. **SOCRATES_SKIP_API offline path** — `SOCRATES_SKIP_API=1` env var forces keyword-only theme classification in tests/eval, preventing API noise
2. **Honest subset-based metrics** — eval script computes `entity_text_hit@1` and `speaker_exact@1` only over their applicable subsets (not inflated by vacuous truths)
3. **3-tier speaker reranker** — `_speaker_rerank_key` with exact metadata → attribution verbs → mention count

### Session 4: Reranker expansion + test reporting
1. **5-tier speaker reranker** (`scripts/retrieve.py` ~line 490):
   - Tier 1: exact `p.speaker` metadata match
   - Tier 2: entity name in first 150 chars (chunk likely starts with their speech)
   - Tier 3: dialogue attribution patterns ("X said", "said X" — 12 verbs)
   - Tier 4: entity-to-Socrates ratio (penalizes Socrates-dominated chunks)
   - Tier 5: raw entity mention count (tiebreaker)
2. **SOFT/PASS\* status** (`tests/test_retrieve.py`) — SOFT = expected book in top-5 but not rank-1 (speaker queries); PASS\* = same for non-speaker queries

### Session 5 (most recent): Crito rank-1 fix + Alcibiades best-effort
**Result**: hit@1 94.7% → **97.4%**, MRR 0.967 → **0.980**, test suite PASS=9 PASS\*=0 SOFT=1 FAIL=0

1. **`obedience` sub-theme** (`config/theme_book_map.yaml`):
   - Primary: Crito only; secondary: Republic, Apology
   - Keywords: obey, obedience, disobey, disobedience, escape, flee, prison, break
   - Added `"obedience"` to THEMES list and CLASSIFY_PROMPT in `scripts/theme_classifier.py`

2. **Crito-only retrieval pass** (`scripts/retrieve.py`, Step 5d ~line 447):
   - Fires when: no entities AND obedience keywords match AND Crito in all_books
   - Adds a Crito-filtered dense query to the RRF fusion list
   - Effect: "Is it right to break an unjust law?" now returns Crito at rank-1

3. **6-tier reranker** — added Tier 3: "X about Y" co-occurrence signal:
   - Detects `about|regarding|concerning Y` pattern in question
   - Boosts chunks containing both primary entity AND secondary target (Y)
   - Intended for "What does Alcibiades say about Socrates?"

4. **Strict rank-1 test** (`tests/test_retrieve.py`):
   - Added `expected_rank1: "Crito"` to the civil disobedience case
   - Status logic: if `expected_rank1` present and rank-1 book doesn't match → FAIL

---

## What Worked

- Crito-only RRF boost pass cleanly fixes rank-1 without disturbing other results
- 5-tier (now 6-tier) reranker improved Euthyphro, Meno, Thrasymachus, Diotima from FAIL/WEAK to PASS
- `obedience` sub-theme is clean architecture — no special-case code in retrieve.py beyond the keyword check
- RRF (reciprocal rank fusion) with multiple targeted retrieval passes is the right pattern for boosting specific books

## What Didn't Work / Gotchas

- **Entity slot reservation with reranker key** (reverted): Attempted to use the 5-tier reranker key to SELECT which entity passages to reserve (instead of dense score). This caused Alcibiades to FAIL — all 5 slots went to Alcibiades I because the entity_ratio signal can't distinguish Alcibiades I vs Symposium (both have dense Alcibiades+Socrates co-occurrence). Reverted to dense score for slot selection.

- **Alcibiades "X about Y" fix didn't work**: The co-occurrence signal (Tier 3 in 6-tier key) is theoretically correct but practically useless here. Both Alcibiades I (Socrates questions Alcibiades) and Symposium (Alcibiades praises Socrates) have dense co-occurrence of both names. The reranker cannot distinguish who is speaking about whom without ingest-level metadata.

- **speaker_exact@1 ceiling**: Stuck at 50%. Root cause: ingest correctly labels chunks by speaker metadata (Socrates for narrated sections), but Thrasymachus/Diotima speeches are NARRATED by Socrates in the text, so they get `speaker=Socrates`. No reranker signal can fix this — only ingest metadata redesign can.

---

## Remaining Known Issues

### Alcibiades SOFT (rank-4 in tests, rank-3 in eval)
- Query: "What does Alcibiades say about Socrates?"
- Symptom: Alcibiades I ranks 1-3, Symposium ranks 4
- Root cause: entity "Alcibiades" maps to [Alcibiades I, Symposium, Protagoras]; dense score strongly prefers Alcibiades I (entire dialogue named after him). The query is semantically closer to Alcibiades I passages (Socrates addressing Alcibiades about his ambitions) than Symposium passages.
- Fix path: targeted ingest metadata — add `narrated_subject` or `about_entities` field so Symposium's Alcibiades speech can be tagged as "Alcibiades speaking about Socrates"

### speaker_exact@1 = 50% ceiling
- 5 of 10 speaker queries return `speaker=Socrates` or `speaker=mixed` at rank-1 despite the entity being the intended subject
- Root cause: Thrasymachus/Diotima speak IN passages narrated by Socrates — ingest labels these correctly as Socrates, not a labeling error
- Fix path: ingest metadata redesign — add `speakers_present` list field (separate from narrator) so we can filter by "who appears" vs "who narrates"

### Potential regressions to watch
- The `obedience` keyword list includes `break` which is a common word. Watch for false-positive Crito boosts on unrelated queries. Current guard: `not entities AND "Crito" in all_books` (obedience theme must already be active).

---

## Current Eval Metrics (Session 5 end state)

| Metric | Value |
|--------|-------|
| hit@1 | 97.4% |
| hit@5 | 100.0% |
| MRR | 0.980 |
| primary_recall | 93.4% |
| substring_hit_rate | 100.0% |
| entity_text_hit@1 | 92.3% (n=13) |
| speaker_exact@1 | 50.0% (n=10) |
| duplicate_rate | 0.0% |

Test suite: `PASS=9 PASS*=0 WEAK=0 SOFT=1 FAIL=0`

---

## Next Steps (Recommended)

### Option A: Targeted ingest metadata for narrated speeches (highest impact)
Codex (independent reviewer) recommends this as the next step after reranker improvements plateau.

In `scripts/ingest.py`, add lightweight metadata to chunks:
- `speakers_present: list[str]` — all characters who appear/speak in the chunk (vs just the narrator)
- `about_entities: list[str]` — entities that are the SUBJECT of discussion (for eulogy-style passages)

Re-ingest only Symposium first (Alcibiades speech), then Republic (Thrasymachus sections), Meno, Euthyphro if needed.

In `scripts/retrieve.py`, update speaker-filtered queries to use `speakers_present` instead of (or in addition to) `speaker`.

**Success criteria**: Alcibiades query returns Symposium at rank-1; speaker_exact@1 improves from 50%.

### Option B: Passage grounding audit tool
Passage metadata is stored per turn in transcript (since Session 1). No tooling exists to check which philosophical claims trace to actual retrieved passages vs LLM training data. A post-hoc audit script would close this loop.

### To run a live session
```bash
python3 scripts/socrates_loop.py
```

### To verify retrieval health
```bash
python3 tests/test_retrieve.py        # should be PASS=9 SOFT=1 FAIL=0
python3 scripts/eval_retrieval.py     # should be hit@1=97.4%, MRR=0.980
python3 tests/test_validator.py       # should be 18/18 pass
```
