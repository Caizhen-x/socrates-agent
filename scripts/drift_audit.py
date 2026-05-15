"""
drift_audit.py — Periodically check if Socrates has drifted from textual fidelity.
Run every 10 sessions or manually.
Usage: python scripts/drift_audit.py
"""

import sys
import yaml
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MEMORY_DIR = PROJECT_ROOT / "memory"
SUMMARY_PATH = MEMORY_DIR / "summary.yaml"
AUDITS_DIR = MEMORY_DIR / "drift_audits"

AUDIT_PROMPT = """You are auditing a Socratic dialogue AI agent for character drift.
Below is a summary of the agent's behavior across {total_sessions} sessions.

SUMMARY:
{summary_yaml}

RECENT SESSIONS:
{recent_sessions}

Your task: Identify any un-Socratic tendencies in the agent's behavior.

Specifically check:
1. Did the agent make claims without grounding in retrieved passages? (check ungrounded_claims)
2. Did the agent reference post-Socratic philosophers or modern concepts?
3. Did the agent break character (say "As an AI..." or speak about Plato)?
4. Did the agent claim certainty rather than maintaining Socratic ignorance?
5. Did the agent lecture rather than engage dialectically?

Produce a YAML audit report:

audit_date: "{audit_date}"
sessions_reviewed: {total_sessions}
drift_detected: <true or false>
severity: "<none, minor, moderate, severe>"
issues_found:
  - description: "<description of issue>"
    frequency: "<how often: rare, occasional, frequent>"
    recommendation: "<how to address>"
  # ... (empty list [] if no issues)
overall_assessment: "<brief paragraph>"
"""


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return {}
    with open(SUMMARY_PATH) as f:
        return yaml.safe_load(f) or {}


def load_recent_sessions(n: int = 10) -> list[dict]:
    sessions_dir = MEMORY_DIR / "sessions"
    if not sessions_dir.exists():
        return []
    files = sorted(sessions_dir.glob("*.yaml"), reverse=True)[:n]
    sessions = []
    for f in files:
        with open(f) as fh:
            data = yaml.safe_load(fh)
            if data:
                sessions.append(data)
    return sessions


def run_audit() -> dict:
    import anthropic

    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    summary = load_summary()
    recent_sessions = load_recent_sessions(10)
    total = summary.get("total_sessions", 0)

    now = datetime.datetime.now()
    audit_date = now.strftime("%Y-%m-%d")

    summary_yaml = yaml.dump(summary, default_flow_style=False)
    recent_yaml = yaml.dump(recent_sessions, default_flow_style=False)[:3000]

    prompt = AUDIT_PROMPT.format(
        total_sessions=total,
        summary_yaml=summary_yaml,
        recent_sessions=recent_yaml,
        audit_date=audit_date,
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=settings["llm"].get("digest_model", settings["llm"]["theme_model"]),
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    result = yaml.safe_load(raw)

    # Save audit
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    audit_file = AUDITS_DIR / f"audit_{audit_date}.yaml"
    with open(audit_file, "w") as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True)

    print(f"Audit saved to {audit_file}")
    if result.get("drift_detected"):
        print(f"DRIFT DETECTED — severity: {result.get('severity', 'unknown')}")
        for issue in result.get("issues_found", []):
            print(f"  - {issue.get('description', '')} ({issue.get('frequency', '')})")
    else:
        print("No significant drift detected.")

    return result


if __name__ == "__main__":
    run_audit()
