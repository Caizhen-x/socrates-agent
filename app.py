"""
app.py — Gradio web UI for the Socrates agent.
Entry point for Hugging Face Spaces.
"""

import io
import os
import random
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings("ignore")

import logging
for _noisy in ("sentence_transformers", "transformers", "huggingface_hub", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


def _ensure_vectorstore():
    vs = PROJECT_ROOT / "vectorstore"
    if vs.exists() and any(vs.iterdir()):
        return
    print("[Vectorstore not found — running first-boot ingestion. Takes a few minutes.]")
    from scripts import ingest
    ingest.main()
    print("[Ingestion complete.]")


_ensure_vectorstore()

import yaml
import anthropic
import gradio as gr

from scripts.env_loader import load_env
load_env(PROJECT_ROOT)

# Silence sentence-transformers during retrieval module import
_stderr = sys.stderr
sys.stderr = io.StringIO()
from scripts.socrates_loop import (
    SYSTEM_PROMPT_TEMPLATE,
    SOCRATES_ACTIONS,
    classify_mood,
    get_dialectical_mode,
    build_user_turn,
    _MODE_INSTRUCTIONS,
    _strip_old_passages,
)
from scripts.response_validator import (
    validate_response,
    _VOCATIVE_PATTERN,
    _TRANSITION_PATTERN,
    _DIALOGUE_NAMES,
    _SPEAKER_NAMES,
)
sys.stderr = _stderr

with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
    SETTINGS = yaml.safe_load(f)

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(
    memory_context="This appears to be our first conversation. Begin fresh."
)
CLIENT = anthropic.Anthropic()


def _repetition_hints(transcript: list[dict]) -> list[str]:
    recent_assistant = [t["content"] for t in transcript if t["role"] == "assistant"]
    if not recent_assistant:
        return []
    hints: list[str] = []
    last_resp = recent_assistant[-1]
    voc = _VOCATIVE_PATTERN.search(last_resp)
    if voc:
        hints.append(
            f'VOCATIVE: Do NOT use "{voc.group()}" — you used it last turn. '
            f"Choose a different form of address."
        )
    recent_transitions: set[str] = set()
    for prev in recent_assistant[-2:]:
        for m in _TRANSITION_PATTERN.finditer(prev):
            recent_transitions.add(m.group())
    if recent_transitions:
        avoid_list = ", ".join(f'"{t}"' for t in sorted(recent_transitions))
        hints.append(
            f"TRANSITION: Do NOT open with {avoid_list} — vary your pivot phrase."
        )
    for name in set(_DIALOGUE_NAMES) | set(_SPEAKER_NAMES):
        count = sum(1 for r in recent_assistant[-3:] if name in r.lower())
        if count >= 2:
            hints.append(
                f"PASSAGE: You have cited '{name}' {count} times recently — "
                f"use different material or reasoning."
            )
    return hints


def respond(user_message: str, history: list[dict]):
    transcript: list[dict] = []
    for msg in history:
        if msg["role"] == "user":
            transcript.append({"role": "user", "content": msg["content"], "mode": "", "passages": []})
        else:
            transcript.append({"role": "assistant", "content": msg["content"]})

    turn_number = sum(1 for t in transcript if t["role"] == "user") + 1
    mood = classify_mood(user_message)
    action = random.choice(SOCRATES_ACTIONS[mood])
    mode = get_dialectical_mode(turn_number, transcript, user_message)

    mood_block = f"[Your disposition: {mood} — {action}]"
    mode_block = f"[Dialectical mode: {mode} — {_MODE_INSTRUCTIONS[mode]}]"

    _stderr_local = sys.stderr
    sys.stderr = io.StringIO()
    try:
        augmented_input, _ = build_user_turn(user_message)
    finally:
        sys.stderr = _stderr_local

    hints = _repetition_hints(transcript)
    if hints:
        hint_block = "[VARIETY: " + " | ".join(hints) + "]"
        augmented_input = f"{mood_block}\n{mode_block}\n{hint_block}\n\n{augmented_input}"
    else:
        augmented_input = f"{mood_block}\n{mode_block}\n\n{augmented_input}"

    api_messages = [{"role": m["role"], "content": m["content"]} for m in history]
    api_messages = _strip_old_passages(api_messages)
    api_messages.append({"role": "user", "content": augmented_input})

    accumulated = f"*{action.rstrip('.').rstrip()}*\n\n"
    yield accumulated

    try:
        with CLIENT.messages.stream(
            model=SETTINGS["llm"]["agent_model"],
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=api_messages,
        ) as stream:
            for chunk in stream.text_stream:
                accumulated += chunk
                yield accumulated
    except Exception as e:
        print(f"[Anthropic stream failed: {type(e).__name__}: {e}]", file=sys.stderr)
        yield accumulated + "\n\n*[The connection to the agora has faltered. Please try again shortly.]*"
        return

    response_text = accumulated.split("\n\n", 1)[1] if "\n\n" in accumulated else accumulated
    issues = validate_response(response_text)
    if any(i["severity"] == "critical" for i in issues):
        retry_messages = api_messages[:-1] + [{
            "role": "user",
            "content": (
                "[IMPORTANT: Stay entirely in character as Socrates of Athens. "
                "Do not reference AI, retrieval systems, Plato as an author, or "
                "modern concepts. Respond as Socrates speaking from memory.]\n\n"
                + api_messages[-1]["content"]
            ),
        }]
        retry_accum = f"*{action.rstrip('.').rstrip()}*\n\n"
        yield retry_accum
        try:
            with CLIENT.messages.stream(
                model=SETTINGS["llm"]["agent_model"],
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=retry_messages,
            ) as stream:
                for chunk in stream.text_stream:
                    retry_accum += chunk
                    yield retry_accum
        except Exception as e:
            print(f"[Anthropic retry failed: {type(e).__name__}: {e}]", file=sys.stderr)
            pass


with gr.Blocks(title="Socrates of Athens", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Socrates of Athens\n"
        "*A retrieval-augmented dialogue agent, grounded in Plato's dialogues.*\n\n"
        "Ask a philosophical question — about virtue, justice, courage, knowledge, "
        "love, the soul — and Socrates will reply using his own words, as recorded "
        "by Plato. Expect questions back, not answers."
    )
    gr.ChatInterface(
        fn=respond,
        type="messages",
        examples=[
            "What is courage?",
            "Is the unexamined life truly not worth living?",
            "Can virtue be taught?",
            "What does it mean to know oneself?",
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
