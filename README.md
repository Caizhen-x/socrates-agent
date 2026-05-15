---
title: Socrates Agent
emoji: 🏛️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Reason with Socrates, grounded in Plato's dialogues.
---

# Socrates Agent

A retrieval-augmented conversational agent that reasons as Socrates of Athens,
grounded exclusively in passages from Plato's dialogues. Built on Claude
(Anthropic API) with a local ChromaDB vector store over the Jowett translations.

The agent practices the elenctic method: asks clarifying questions, proposes
definitions, finds contradictions, and guides the interlocutor toward truth —
or toward the honest recognition that neither party yet knows.

## How it works

1. **Retrieval.** Every user turn triggers a hybrid search (dense embeddings +
   BM25) over ~465 chunks from 10 dialogues. The top 3–5 passages are passed
   to the model.
2. **Grounding.** Claude responds in-character as Socrates, drawing only on
   the retrieved passages. A response validator flags post-Socratic
   references, modern vocabulary, and asterisk stage directions.
3. **Memory.** Session digests track the interlocutor's sophistication level,
   tendencies, and past arguments so returning users aren't asked the same
   questions twice.

## Setup

```bash
pip install -r requirements.txt

# One-time: cache the embedding model locally
python scripts/download_model.py

# Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# One-time: build the vector store from Books/
python scripts/ingest.py
```

The `vectorstore/` directory is excluded from the repo (~52 MB of binaries);
`ingest.py` rebuilds it deterministically from the public-domain Jowett texts
in `Books/`.

## Run

```bash
python scripts/socrates_loop.py
```

This drops you into a terminal REPL. Type your question; type `exit` to end
the session and write a digest to `memory/`.

## Project layout

```
Books/              Plato's dialogues (Jowett translation, public domain)
config/             Speakers, themes, settings, Stephanus page map
scripts/            Ingestion, retrieval, memory, validator, main loop
tests/              Pytest suite + retrieval gold set
.claude/rules/      Character guardrails enforced at the prompt level
```

## Constraints

The agent will not:
- Reference any philosopher after 399 BCE (Aristotle, Stoics, moderns).
- Use modern vocabulary (rights, democracy in the modern sense, psychology,
  consciousness as a technical term).
- Fabricate quotations not present in the retrieved passages.
- Break character to explain what Socrates "would" think.

See `CLAUDE.md` and `.claude/rules/character-guardrails.md` for the full
character contract.

## License

The Plato texts in `Books/` are Benjamin Jowett's 19th-century translations —
public domain. Code is released under MIT.
