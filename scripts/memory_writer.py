"""
memory_writer.py — Generate a session digest YAML from the conversation transcript,
then update the rolling summary.yaml.
Usage: python scripts/memory_writer.py --transcript /tmp/socrates_transcript.json
"""

import sys
import json
import os
import yaml
import argparse
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.env_loader import load_env
load_env(PROJECT_ROOT)

MEMORY_DIR = PROJECT_ROOT / "memory"
SESSIONS_DIR = MEMORY_DIR / "sessions"
SUMMARY_PATH = MEMORY_DIR / "summary.yaml"

DIGEST_PROMPT = """You are writing a structured memory digest from a Socratic dialogue session.
Given the conversation transcript below, produce a YAML document with this exact schema:

session_id: "{session_id}"
date: "{date}"
timestamp: "{timestamp}"

interlocutor_inquiry:
  topic: "<one-sentence description of what the interlocutor was exploring>"
  themes: [<list of themes from: justice, courage, piety, love, knowledge, death, virtue, temperance, friendship, truth, beauty, soul, good, rhetoric, education>]
  books_cited: [<list of book names mentioned or drawn from>]

positions_taken:
  - claim: "<philosophical claim Socrates made>"
    grounding: "<textual reference if any, e.g. 'Laches 194e' or '' if none>"
    support_chunk_ids: [<list of chunk_ids from AVAILABLE_CHUNKS lines that directly support this claim, or []>]
  # ... up to 4 positions
  # IMPORTANT: only include chunk_ids that actually appear in the AVAILABLE_CHUNKS lines above.
  # A position without any support_chunk_ids will be treated as ungrounded and not stored in memory.

dialectical_state: "<one of: aporia, provisional_conclusion, refutation_complete, ongoing>"
conclusion_if_any: "<brief description or empty string>"

interlocutor_profile:
  sophistication: "<one of: beginner, intermediate, advanced>"
  tendencies: [<list of up to 3 intellectual tendencies observed>]
  growth_areas: [<list of any improvements noted, or []>]

ungrounded_claims: [<list of any claims Socrates made without textual grounding, or []>]

definitions_examined:
  - term: "<the concept being defined, e.g. 'courage'>"
    proposed_definition: "<what the interlocutor proposed>"
    status: "<one of: accepted, refuted, modified, abandoned>"
    refutation_reason: "<brief reason if refuted, else empty string>"
  # ... list all definitions proposed in this session, or []

concessions_extracted:
  - "<a statement the interlocutor explicitly agreed to>"
  # ... up to 5 concessions, or []

contradictions_found:
  - premises: ["<premise 1>", "<premise 2>"]
    conclusion: "<the contradiction or tension revealed>"
  # ... up to 3, or []

Return ONLY the YAML, no explanation, no markdown fences.

TRANSCRIPT:
{transcript}"""


def get_session_id(date: str) -> str:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(SESSIONS_DIR.glob(f"{date}_*.yaml"))
    seq = len(existing) + 1
    return f"{date}_{seq:03d}"


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Prefer sentence-ending punctuation as the cut point
    for sep in (".\n", "?\n", "!\n", ". ", "? ", "! ", "\n", ".", "?", "!"):
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:  # Don't cut away more than half
            return truncated[: idx + len(sep)].rstrip()
    return truncated


def _normalize_digest(data: dict) -> dict:
    """
    Enforce the grounding contract: positions_taken entries with no textual
    grounding reference are moved to ungrounded_claims so they can't be
    baked into memory as grounded positions.
    """
    if not isinstance(data, dict):
        return data

    positions = data.get("positions_taken") or []
    ungrounded = list(data.get("ungrounded_claims") or [])

    grounded = []
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        claim = (pos.get("claim") or "").strip()
        if not claim:
            continue
        chunk_ids = pos.get("support_chunk_ids") or []
        if isinstance(chunk_ids, list) and len(chunk_ids) > 0:
            grounded.append(pos)
        else:
            # No chunk-level grounding — move to ungrounded_claims
            if claim not in ungrounded:
                ungrounded.append(claim)

    data["positions_taken"] = grounded
    data["ungrounded_claims"] = ungrounded
    return data


def generate_digest(transcript: list[dict], session_id: str, date: str, timestamp: str) -> dict:
    import anthropic

    # Format transcript for the prompt, including retrieved chunk_ids per user turn
    lines = []
    for turn in transcript:
        lines.append(f"{turn['role'].upper()}: {turn['content']}")
        if turn.get("passages"):
            ids = [p["chunk_id"] for p in turn["passages"] if "chunk_id" in p]
            if ids:
                lines.append(f"  [AVAILABLE_CHUNKS: {', '.join(ids)}]")
    transcript_text = "\n".join(lines)

    prompt = DIGEST_PROMPT.format(
        session_id=session_id,
        date=date,
        timestamp=timestamp,
        transcript=_truncate_at_sentence(transcript_text, 8000),
    )

    client = anthropic.Anthropic()
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    response = client.messages.create(
        model=settings["llm"].get("digest_model", settings["llm"]["theme_model"]),
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_yaml = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw_yaml.startswith("```"):
        raw_yaml = "\n".join(raw_yaml.split("\n")[1:])
    if raw_yaml.endswith("```"):
        raw_yaml = "\n".join(raw_yaml.split("\n")[:-1])

    return _normalize_digest(yaml.safe_load(raw_yaml))


def update_summary(digest: dict):
    summary = {}
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            summary = yaml.safe_load(f) or {}

    summary["total_sessions"] = summary.get("total_sessions", 0) + 1
    summary["last_session"] = digest.get("session_id", "")

    # Merge themes
    themes_explored = summary.get("themes_explored", {})
    for theme in digest.get("interlocutor_inquiry", {}).get("themes", []):
        themes_explored[theme] = themes_explored.get(theme, 0) + 1
    summary["themes_explored"] = themes_explored

    # Update interlocutor profile
    new_profile = digest.get("interlocutor_profile", {})
    existing_profile = summary.get("interlocutor_profile", {})

    # Weighted sophistication: majority vote over last 10 sessions
    from collections import Counter
    _VALID_SOPHISTICATION = {"beginner", "intermediate", "advanced"}
    sophistication_history = existing_profile.get("sophistication_history", [])
    new_soph = new_profile.get("sophistication", "")
    if new_soph in _VALID_SOPHISTICATION:
        sophistication_history.append(new_soph)
        sophistication_history = sophistication_history[-10:]
    existing_profile["sophistication_history"] = sophistication_history
    if sophistication_history:
        existing_profile["sophistication"] = Counter(sophistication_history).most_common(1)[0][0]

    # Merge tendencies (keep unique, cap at 10) — stored as "tendencies", not "recurring_errors"
    existing_tendencies = existing_profile.get("tendencies", [])
    new_tendencies = new_profile.get("tendencies", [])
    merged = list(dict.fromkeys(existing_tendencies + new_tendencies))[:10]
    existing_profile["tendencies"] = merged

    # Merge growth areas
    existing_growth = existing_profile.get("growth_areas", [])
    new_growth = new_profile.get("growth_areas", [])
    merged_growth = list(dict.fromkeys(existing_growth + new_growth))[:10]
    existing_profile["growth_areas"] = merged_growth

    summary["interlocutor_profile"] = existing_profile

    with open(SUMMARY_PATH, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

    print(f"Updated summary.yaml (total sessions: {summary['total_sessions']})")


def write_digest(transcript_path: str | None = None, transcript_data: list | None = None):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    if transcript_data is None:
        if transcript_path is None:
            print("No transcript provided.", file=sys.stderr)
            sys.exit(1)
        with open(transcript_path) as f:
            transcript_data = json.load(f)

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    timestamp = now.isoformat()
    session_id = get_session_id(date)

    print(f"Generating session digest for {session_id}...")
    try:
        digest = generate_digest(transcript_data, session_id, date, timestamp)
    except Exception as e:
        print(f"Error generating digest: {e}", file=sys.stderr)
        # Write a minimal fallback digest
        digest = {
            "session_id": session_id,
            "date": date,
            "timestamp": timestamp,
            "interlocutor_inquiry": {"topic": "unknown", "themes": [], "books_cited": []},
            "positions_taken": [],
            "dialectical_state": "ongoing",
            "conclusion_if_any": "",
            "interlocutor_profile": {"sophistication": "unknown", "tendencies": [], "growth_areas": []},
            "ungrounded_claims": [],
            "error": str(e),
        }

    session_file = SESSIONS_DIR / f"{session_id}.yaml"
    with open(session_file, "w") as f:
        yaml.dump(digest, f, default_flow_style=False, allow_unicode=True)
    print(f"Session digest saved to {session_file}")

    update_summary(digest)
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", help="Path to transcript JSON file")
    args = parser.parse_args()
    write_digest(transcript_path=args.transcript)
