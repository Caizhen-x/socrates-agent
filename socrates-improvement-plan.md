# Socrates Persona Agent — Improvement Plan

**Baseline:** Architecture spec v5 (Session 5 end state). 27 dialogues, 1085 chunks, 97.4% hit@1, 6-mode dialectical system, Jowett voice rules + 5 few-shot exemplars, response validator with retry, mood-aware stage directions, YAML session memory with dialectical trace.

---

## Part A: Voice Authenticity (Making Socrates Sound Like Socrates)

### A1. Expand the Exemplar Bank (high impact, moderate effort)

**Problem:** 5 few-shot exemplars in the system prompt is a good start, but the LLM sees the same 5 examples every turn. Over a long session, its outputs converge toward those 5 patterns rather than the full range of Socratic expression.

**Solution:** Build a **rotating exemplar bank** of 30–40 curated exchanges, organized by dialectical mode × theme. Each turn, select 2–3 exemplars that match the current mode and retrieved book — not the same 5 every time.

**Implementation:**
- File: `config/exemplars.yaml` — structured as:
  ```yaml
  - mode: elenctic
    theme: courage
    book: Laches
    exchange: |
      INTERLOCUTOR: Courage is endurance of the soul.
      SOCRATES: And is all endurance courage? ...
  ```
- `socrates_loop.py` loads all exemplars at startup, filters by current `dialectical_mode` and primary retrieved book, selects 2–3 randomly (with no-repeat-in-last-3-turns constraint).
- The `## Exemplary Exchanges` section in `SYSTEM_PROMPT_TEMPLATE` becomes dynamic rather than static.
- **Sourcing:** Have Opus read each of the 27 dialogues and extract the 1–2 most characteristic Socratic exchanges per dialogue. Hand-edit for length (target: 4–8 turns, 150–300 words each). This gives ~40 exemplars grounded in actual Plato, covering all modes.

**Why it works:** LLMs calibrate register from examples more reliably than from rules. More diverse examples = more varied, authentic output. Mode-matched examples mean the model sees "how Socrates sounds when being constructive" right when it needs to be constructive.

---

### A2. Quantitative Voice Profile (medium impact, moderate effort)

**Problem:** The current voice rules are qualitative ("alternate short questions with longer analogical passages"). The LLM interprets these loosely. There's no way to measure whether a response actually matches Socratic speech patterns.

**Solution:** Analyze all 27 dialogues quantitatively and extract measurable voice parameters. Use these both as generation guidance and as post-hoc evaluation metrics.

**Analysis to run (one-time, via Opus):**
- **Question-to-assertion ratio** per dialogue and overall (expect ~3:1 in elenctic, ~1:2 in constructive)
- **Average sentence length** in words (Jowett Socrates tends toward 15–25 words per sentence)
- **Analogy frequency** (how often does Socrates use a craft/trade comparison per page?)
- **Vocative frequency** (how often does he address the interlocutor by name/epithet?)
- **Turn length distribution** (short turns vs. long speeches — expect bimodal: many short, few long)
- **Question types** taxonomy: yes/no leading questions, "is it not...?" constructions, open definitional questions, reductio questions

**Output:** `config/voice_profile.yaml` — quantitative targets:
```yaml
elenctic:
  question_ratio: 0.75          # 75% of sentences are questions
  avg_sentence_length: 18       # words
  max_consecutive_assertions: 2
  analogy_per_response: 0.3     # ~1 every 3 responses
constructive:
  question_ratio: 0.40
  avg_sentence_length: 22
  max_consecutive_assertions: 4
  ...
```

**Usage:**
1. Inject key parameters into mode instructions: "In elenctic mode, roughly 3 of every 4 sentences should be questions."
2. Add a lightweight `voice_metrics()` function in `response_validator.py` that computes question ratio, sentence length, etc. on each response. Log to `validation_log.jsonl` as `type: voice_drift`. No retry — just monitoring.
3. Drift audit can flag systematic voice drift: "Last 10 responses averaged 35% questions in elenctic mode; target is 75%."

---

### A3. Contrastive Voice Examples (medium impact, low effort)

**Problem:** The system prompt tells Socrates what NOT to say (no contractions, no "basically", etc.), but negative rules are weaker than positive demonstrations. The LLM doesn't see *what the wrong version looks like* side-by-side with the right one.

**Solution:** Add 3–4 **contrastive pairs** to the system prompt — each showing a BAD response and a GOOD response to the same question.

**Example:**
```
## What Socrates Does NOT Sound Like

BAD: "So basically, courage is about facing your fears, right? I think
we can all agree that's a pretty solid definition. Let me push back on
that a bit though — what about situations where..."

GOOD: "Tell me then — you say that courage is the facing of fears. But
does the physician who knows the danger of a wound, and faces it
nonetheless, show the same courage as the fool who rushes in knowing
nothing? Would you not say that knowledge plays some part?"
```

**Why it works:** Contrastive examples are one of the most effective prompting techniques. The model sees the exact failure mode (modern register, casual filler, lecture structure) next to the correct alternative (question-led, Jowett vocabulary, analogy-first). Three pairs covering the most common failure modes (modern casual, lecture mode, over-assertion) would significantly tighten the register.

**Placement:** After the current `## Exemplary Exchanges` section. Keep to 3 pairs max — contrastive examples are high-signal but token-expensive.

---

### A4. Dialogue-Phase Voice Variation (lower impact, medium effort)

**Problem:** The current system treats "Socrates' voice" as uniform. In reality, Jowett-register Socrates sounds noticeably different in early dialogues (Euthyphro, Laches — shorter, more combative, faster to refute) versus middle dialogues (Symposium, Phaedo — longer speeches, more poetic, willing to build positive accounts) versus late dialogues (Republic Books V–VII — sustained constructive argument, myth, extended analogy).

**Solution:** Tag each exemplar and each book in `config/theme_book_map.yaml` with a `period` field: `early`, `middle`, `late`. When retrieval pulls from a specific period, the system prompt gets a one-line register hint:

```
Early dialogues register: Keep turns short. Refute quickly. Use
concrete craft analogies (cobbler, physician). Rarely speak more
than 4 sentences before returning to questions.

Middle dialogues register: Allow longer passages. Use myth and
poetic imagery when apt. Build positive accounts alongside
refutation. The interlocutor is a partner, not a target.
```

This is a refinement, not a priority — implement after A1–A3 are working.

---

### A5. Voice Evaluation Suite (medium impact, medium effort)

**Problem:** You have a 38-question retrieval eval and an 18-case validator test. You have no systematic evaluation of *voice quality*. You can't measure whether changes to the prompt actually make Socrates sound more authentic.

**Solution:** Build `tests/test_voice.py` — a voice evaluation suite.

**Design:**
1. **Gold responses:** For 10 questions, write (or have Opus write) a gold-standard Socratic response in perfect Jowett register. These are the reference.
2. **Automated metrics** (no API needed):
   - Question ratio (% of sentences ending in ?)
   - Vocative presence (does the response address the interlocutor?)
   - Prohibited term scan (contractions, modern filler, asterisk actions)
   - Average sentence length
   - Craft analogy detection (keyword scan for trade/craft terms)
3. **LLM-as-judge** (optional, uses Haiku):
   - Feed each test response + the gold response to Haiku: "Rate on a 1–5 scale how closely the test response matches the register, vocabulary, and rhetorical structure of the reference."
   - Track aggregate score over time.

**Run:** `python3 tests/test_voice.py` — outputs a voice scorecard. Run after any prompt change to detect regressions.

---

## Part B: System Improvements (General)

### B1. Fix the Alcibiades SOFT (documented, straightforward)

**Problem:** "What does Alcibiades say about Socrates?" — Symposium ranks 4th because Alcibiades I ranks 1–3. The reranker can't distinguish speaker-vs-subject without ingest-level metadata.

**Fix:** Add `about_entities: list[str]` metadata to Symposium's Alcibiades speech chunks at ingest time (roughly chunks covering Symposium 212c–222b). Then add an `about_entities` filter in the entity retrieval pass (Step 5b) that boosts chunks where the queried entity is in `about_entities`.

**Effort:** Small — modify `ingest.py` to tag ~20 Symposium chunks, add one filter condition in `retrieve.py`.

---

### B2. Speaker-Present Metadata (documented ceiling fix)

**Problem:** `speaker_exact@1 = 50%` because Thrasymachus/Diotima speeches are narrated by Socrates, so ingest correctly labels them `speaker=Socrates`. The reranker has no signal to distinguish "Socrates narrating Diotima's speech" from "Socrates speaking as himself."

**Fix:** Add `speakers_present: list[str]` metadata field at ingest time — who appears or speaks *within* the chunk, separate from the narrator. For Republic, a chunk where Thrasymachus argues would have `speaker=Socrates` (narrator) but `speakers_present=[Thrasymachus, Socrates]`.

**Implementation:**
- `ingest.py`: after `infer_dominant_speaker()`, run a second pass `detect_speakers_present()` that scans each chunk for all known speaker names from `speakers.yaml`.
- `retrieve.py`: in the 6-tier speaker reranker, add a new tier (T1.5) that checks `speakers_present` metadata — ranked above current T2 (text substring) but below T1 (exact speaker match).

**Effort:** Medium — requires re-ingestion after the metadata change.

---

### B3. End-to-End Dialogue Quality Evaluation (high impact, high effort)

**Problem:** You evaluate retrieval (38-question gold set) and validate individual responses (validator + test suite). You don't evaluate *multi-turn dialogue quality* — does a 10-turn conversation feel like talking to Socrates?

**Solution:** Build `tests/test_dialogue.py` — an automated multi-turn eval.

**Design:**
1. Define 5 scripted conversations (10 turns each) with pre-written user messages that exercise different modes:
   - A courage definition dialogue (elenctic → constructive → synthesis)
   - A hostile interlocutor who pushes back (elenctic → ironic → elenctic)
   - A confused beginner (opening → concessive_teaching → constructive)
   - A question about post-Socratic content (aporia fallback)
   - A returning interlocutor (memory continuity)
2. Run each conversation through `socrates_loop.py` programmatically (mock the terminal input).
3. Feed the full transcript to Opus with a rubric:
   - Character consistency (1–5): Does Socrates break character?
   - Dialectical progression (1–5): Does the argument advance logically?
   - Voice authenticity (1–5): Does it sound like Jowett-register Socrates?
   - Passage grounding (1–5): Are claims traceable to retrieved passages?
   - Mode appropriateness (1–5): Does the mode match what the conversation needs?
4. Output a scorecard. Track over time.

**Effort:** High — requires programmatic conversation runner + Opus eval. But this is the only way to measure the thing that actually matters: is the agent good to talk to?

---

### B4. BM25 Index Robustness (low impact, low effort)

**Problem:** Step 5c (BM25 lexical retrieval) has "graceful degradation if index missing." This means if the BM25 index isn't built, the system silently skips lexical retrieval. BM25 catches cases where semantic embedding misses exact keyword matches (e.g., proper names, specific Greek terms).

**Fix:** Make `ingest.py` build the BM25 index alongside the ChromaDB embeddings. Store as a pickle or JSON file in `vectorstore/`. Remove the graceful degradation — if the index is missing, raise an error.

---

### B5. Prompt Token Budget Monitoring (medium impact, low effort)

**Problem:** The system prompt is already large: identity + rules + voice table + 5 exemplars + characteristic devices + memory context + retrieved passages. There's no monitoring of total prompt size. As you add more exemplars (A1), contrastive pairs (A3), and register hints (A4), you risk hitting context limits or degrading response quality from prompt overload.

**Fix:** Add a `_log_prompt_stats()` function in `socrates_loop.py` that logs:
- System prompt token count (approximate via char/4)
- Retrieved passages token count
- Memory context token count
- Total input tokens
- Ratio of instruction tokens to passage tokens

Log to `logs/prompt_stats.jsonl`. Set a warning threshold (e.g., if system prompt exceeds 4000 tokens, log a warning). This gives you data to make tradeoffs when adding new prompt content.

---

### B6. Cross-Session Argument Threading (medium impact, medium effort)

**Problem:** Memory currently loads the last 2 session digests. If a user returns after 10 sessions to revisit a topic from session 3, Socrates has no memory of that earlier conversation. The `summary.yaml` tracks themes explored but not the specific arguments.

**Fix:** Add a **topic index** to `memory/`:
```yaml
# memory/topic_index.yaml
courage:
  - session: "2026-04-08_001"
    conclusion: "aporia — endurance requires wisdom"
    definitions_tried: ["endurance of the soul"]
  - session: "2026-04-22_003"
    conclusion: "provisional — courage is knowledge of what is truly fearful"
    definitions_tried: ["knowledge of danger", "knowledge of good and evil"]
```

`memory_writer.py` updates this index after each session. `memory_reader.py` loads the current session's themes and pulls relevant entries from the topic index — not just the last 2 sessions, but any past session that discussed the same theme. Capped at 200 tokens of topic context.

This lets Socrates say: "We have examined courage before, and found that endurance alone was insufficient. Shall we begin from where we left off — with the question of whether courage requires knowledge of good and evil entire?"

---

### B7. Response Length Calibration (low-medium impact, low effort)

**Problem:** The system prompt says "100–300 words. Longer only if the argument requires it." In practice, LLMs tend toward the upper end of any range. Socrates in the early dialogues often responds in 2–3 sentences (30–60 words) — especially in rapid elenctic exchange.

**Fix:** Make response length guidance mode-dependent:
```
opening:              30–80 words (casual, brief)
elenctic:             50–150 words (sharp, question-heavy)
constructive:         100–250 words (building, more exposition needed)
ironic:               40–100 words (the punch lands faster when short)
concessive_teaching:  100–200 words (patient, step-by-step)
synthesis:            150–300 words (drawing threads together)
```

Inject the appropriate range into the mode instruction. Add a `response_length` check in the voice metrics (B5/A2) to flag responses that consistently exceed the upper bound.

---

## Priority Order

| Priority | Item | Impact | Effort | Dependencies |
|----------|------|--------|--------|--------------|
| 1 | **A1** Rotating exemplar bank (30–40 exemplars) | High | Medium | None |
| 2 | **A3** Contrastive voice examples (3 pairs) | Medium | Low | None |
| 3 | **B1** Fix Alcibiades SOFT | Medium | Low | Re-ingest |
| 4 | **A2** Quantitative voice profile | Medium | Medium | None |
| 5 | **B7** Response length calibration per mode | Low-Med | Low | None |
| 6 | **B5** Prompt token budget monitoring | Medium | Low | None |
| 7 | **A5** Voice evaluation suite | Medium | Medium | A2 (metrics) |
| 8 | **B2** Speaker-present metadata | Medium | Medium | Re-ingest |
| 9 | **B6** Cross-session argument threading | Medium | Medium | None |
| 10 | **B4** BM25 index robustness | Low | Low | Re-ingest |
| 11 | **A4** Dialogue-phase voice variation | Lower | Medium | A1 (exemplars) |
| 12 | **B3** End-to-end dialogue eval | High | High | A5, B7 |

**Suggested batches:**
- **Batch 1 (voice tightening):** A1 + A3 + B7 — biggest voice improvement for least effort
- **Batch 2 (measurement):** A2 + A5 + B5 — build the instruments to measure whether changes help
- **Batch 3 (retrieval polish):** B1 + B2 + B4 — close the documented gaps, requires one re-ingest
- **Batch 4 (depth):** B6 + A4 + B3 — cross-session memory, period-aware voice, end-to-end eval
