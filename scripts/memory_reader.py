"""
memory_reader.py — Load memory context (summary + last 2 sessions) for session start.
Outputs a formatted text block for injection into the system prompt.
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MEMORY_DIR = PROJECT_ROOT / "memory"
SESSIONS_DIR = MEMORY_DIR / "sessions"
SUMMARY_PATH = MEMORY_DIR / "summary.yaml"
MAX_CONTEXT_TOKENS = 800


def load_summary() -> dict | None:
    if not SUMMARY_PATH.exists():
        return None
    with open(SUMMARY_PATH) as f:
        return yaml.safe_load(f)


def load_recent_sessions(n: int = 2) -> list[dict]:
    if not SESSIONS_DIR.exists():
        return []
    files = sorted(SESSIONS_DIR.glob("*.yaml"), reverse=True)[:n]
    sessions = []
    for f in files:
        with open(f) as fh:
            data = yaml.safe_load(fh)
            if data:
                sessions.append(data)
    return sessions


def format_summary(summary: dict) -> str:
    lines = []
    total = summary.get("total_sessions", 0)
    if total:
        lines.append(f"You have spoken with this interlocutor across {total} previous session(s).")

    themes = summary.get("themes_explored", {})
    if themes:
        top = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append("Themes most explored: " + ", ".join(f"{t} ({c}x)" for t, c in top) + ".")

    profile = summary.get("interlocutor_profile", {})
    if profile:
        soph = profile.get("sophistication")
        if soph:
            lines.append(f"The interlocutor's level: {soph}.")
        tendencies = profile.get("tendencies", [])
        if tendencies:
            lines.append("Tendencies observed: " + "; ".join(tendencies) + ".")
        growth = profile.get("growth_areas", [])
        if growth:
            lines.append("Areas of growth noted: " + "; ".join(growth) + ".")

    return "\n".join(lines)


def format_session(session: dict) -> str:
    lines = []
    date = session.get("date", "unknown date")
    inquiry = session.get("interlocutor_inquiry", {})
    if inquiry:
        topic = inquiry.get("topic", "")
        if topic:
            lines.append(f"Last explored ({date}): {topic}.")

    positions = session.get("positions_taken", [])
    grounded_positions = [
        p for p in positions
        if isinstance(p, dict) and (
            # New sessions: grounded by chunk_ids
            (p.get("support_chunk_ids") or [])
            # Old sessions: grounded by free-text Stephanus ref (backward compat)
            or (p.get("grounding") or "").strip()
        )
    ]
    if grounded_positions:
        lines.append("Key points established:")
        for pos in grounded_positions[:3]:
            claim = pos.get("claim", "")
            grounding = pos.get("grounding", "")
            if claim:
                entry = f"  - {claim} (cf. {grounding})"
                lines.append(entry)

    state = session.get("dialectical_state", "")
    conclusion = session.get("conclusion_if_any", "")
    if state:
        lines.append(f"Where we ended: {state}." + (f" {conclusion}" if conclusion else ""))

    definitions = session.get("definitions_examined", []) or []
    if definitions:
        lines.append("Definitions examined:")
        for d in definitions[:3]:
            term = d.get("term", "")
            defn = d.get("proposed_definition", "")
            status = d.get("status", "")
            reason = d.get("refutation_reason", "")
            if term and defn:
                entry = f"  - {term}: '{defn}' ({status})"
                if reason:
                    entry += f" — {reason}"
                lines.append(entry)

    concessions = session.get("concessions_extracted", []) or []
    if concessions:
        lines.append("Points already agreed upon:")
        for c in concessions[:3]:
            if c:
                lines.append(f"  - {c}")

    contradictions = session.get("contradictions_found", []) or []
    if contradictions:
        lines.append("Contradictions found:")
        for ct in contradictions[:2]:
            premises = ct.get("premises", [])
            conclusion_ct = ct.get("conclusion", "")
            if conclusion_ct:
                lines.append(f"  - {conclusion_ct}")

    return "\n".join(lines)


def load_context() -> str:
    parts = []
    summary = load_summary()
    recent = load_recent_sessions(2)

    if not summary and not recent:
        return ""  # First session — no memory block needed

    parts.append("=== YOUR MEMORY OF PAST CONVERSATIONS ===")

    if summary:
        s = format_summary(summary)
        if s:
            parts.append(s)

    for session in recent:
        s = format_session(session)
        if s:
            parts.append(s)

    parts.append("=== END MEMORY ===")
    context = "\n".join(parts)
    # Enforce token limit (rough estimate: 4 chars ≈ 1 token)
    max_chars = MAX_CONTEXT_TOKENS * 4
    if len(context) > max_chars:
        truncated = context[:max_chars]
        # Cut at the last complete line to avoid mid-sentence breaks
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars // 2:
            context = truncated[:last_newline] + "\n[Memory truncated]"
        else:
            context = truncated + "\n[Memory truncated]"
    return context


if __name__ == "__main__":
    context = load_context()
    if context:
        print(context)
    else:
        print("[No prior memory — this is the first session.]")
