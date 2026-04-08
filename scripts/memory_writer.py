"""
memory_writer.py — Generate a session digest YAML from the conversation transcript,
then update the rolling summary.yaml.
Usage: python scripts/memory_writer.py --transcript /tmp/socrates_transcript.json
"""

import sys
import json
import yaml
import argparse
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
  # ... up to 4 positions

dialectical_state: "<one of: aporia, provisional_conclusion, refutation_complete, ongoing>"
conclusion_if_any: "<brief description or empty string>"

interlocutor_profile:
  sophistication: "<one of: beginner, intermediate, advanced>"
  tendencies: [<list of up to 3 intellectual tendencies observed>]
  growth_areas: [<list of any improvements noted, or []>]

ungrounded_claims: [<list of any claims Socrates made without textual grounding, or []>]

Return ONLY the YAML, no explanation, no markdown fences.

TRANSCRIPT:
{transcript}"""


def get_session_id(date: str) -> str:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(SESSIONS_DIR.glob(f"{date}_*.yaml"))
    seq = len(existing) + 1
    return f"{date}_{seq:03d}"


def generate_digest(transcript: list[dict], session_id: str, date: str, timestamp: str) -> dict:
    import anthropic

    # Format transcript for the prompt
    transcript_text = "\n".join(
        f"{turn['role'].upper()}: {turn['content']}" for turn in transcript
    )

    prompt = DIGEST_PROMPT.format(
        session_id=session_id,
        date=date,
        timestamp=timestamp,
        transcript=transcript_text[:8000],  # Truncate very long sessions
    )

    client = anthropic.Anthropic()
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    response = client.messages.create(
        model=settings["llm"]["agent_model"],
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_yaml = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw_yaml.startswith("```"):
        raw_yaml = "\n".join(raw_yaml.split("\n")[1:])
    if raw_yaml.endswith("```"):
        raw_yaml = "\n".join(raw_yaml.split("\n")[:-1])

    return yaml.safe_load(raw_yaml)


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

    if new_profile.get("sophistication"):
        existing_profile["sophistication"] = new_profile["sophistication"]

    # Merge tendencies (keep unique, cap at 10)
    existing_tendencies = existing_profile.get("recurring_errors", [])
    new_tendencies = new_profile.get("tendencies", [])
    merged = list(dict.fromkeys(existing_tendencies + new_tendencies))[:10]
    existing_profile["recurring_errors"] = merged

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
