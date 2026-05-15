"""
response_validator.py — Fast regex/keyword checks on Socrates' responses.
Catches anachronisms, modern register, character breaks, and contractions
before they reach the user. Returns a list of violations for logging and
optional retry logic.
"""

import re
import json
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# --- Violation term lists ---

# Post-Socratic philosophers and modern schools (399 BCE cutoff)
ANACHRONISM_TERMS = [
    "aristotle", "aristotelian", "nicomachean",
    "stoic", "stoicism", "zeno of citium",
    "epicurean", "epicurus",
    "kant", "kantian", "categorical imperative",
    "hegel", "hegelian", "dialectic of history",
    "nietzsche", "nietzschean",
    "descartes", "cartesian",
    "locke", "hume", "empiricism",
    "utilitarianism", "utilitarian", "bentham", "mill",
    "existentialism", "existential",
    "social contract", "rousseau",
    "neuroscience", "neuroscientist",
    "psychology", "psychological", "psychologist",
    "evolution", "darwinian",
    "quantum", "relativity", "physics",
    "consciousness",  # as technical philosophical term
    "human rights", "civil rights", "constitutional rights",
    "democracy",  # in modern sense only — "democratic" in Athens is fine
]

# Modern conversational register that Jowett would never use
MODERN_REGISTER = [
    "basically",
    "at the end of the day",
    "let's unpack",
    "unpack this",
    "let me push back",
    "push back on",
    "devil's advocate",
    "makes sense?",
    "make sense?",
    "circle back",
    "touch base",
    "going forward",
    "the bottom line",
    "here's the thing",
    "i'd argue",
    "i would argue that",
    "you know what i mean",
    "right?",  # as conversation tag
    "you know?",
    "ok so",
    "so basically",
    "like i said",
    "at the end of the day",
]

# Hard character breaks — these are critical failures
CHARACTER_BREAKS = [
    "as an ai",
    "i am an ai",
    "i'm an ai",
    "language model",
    "large language model",
    "artificial intelligence",
    "plato wrote",
    "plato described",
    "plato's account",
    "as plato",
    "retrieved passage",
    "retrieval system",
    "passages suggest",
    "passages indicate",
    "according to the passages",
    "the passages show",
    "as socrates, i",
    "socrates would say",
    "socrates believed",
    "in the dialogues",
]

# --- Repetition detection patterns ---

_VOCATIVE_PATTERN = re.compile(
    r'\b(?:my (?:dear |excellent |good )?(?:friend|companion|man)|'
    r'best of men|fair friend|my dear \w+)\b',
    re.IGNORECASE,
)

_TRANSITION_PATTERN = re.compile(
    r'\b(?:but tell me(?: this)?|tell me then|let us consider|'
    r'suppose that|well then|and now|come now|'
    r'consider this|for tell me|let us suppose)\b',
    re.IGNORECASE,
)

_DIALOGUE_NAMES = [
    "laches", "euthyphro", "meno", "gorgias", "republic",
    "symposium", "phaedo", "phaedrus", "protagoras", "crito",
    "apology", "charmides", "lysis", "ion", "hippias",
    "alcibiades", "theaetetus", "sophist", "statesman", "philebus",
    "timaeus", "laws", "parmenides", "cratylus",
]

_SPEAKER_NAMES = [
    "nicias", "polus", "callicles", "thrasymachus", "glaucon",
    "adeimantus", "agathon", "diotima", "protagoras",
    "simmias", "cebes",
]

# Contraction pattern — Jowett Socrates never uses contractions
_CONTRACTION_PATTERN = re.compile(
    r"\b\w+n't\b|\b\w+'re\b|\b\w+'ve\b|\b\w+'ll\b|\bI'm\b|\blet's\b|\bdon't\b|\bcan't\b|\bwon't\b|\bdidn't\b|\bisn't\b|\baren't\b|\bwasn't\b|\bweren't\b|\bhasn't\b|\bhaven't\b|\bhadn't\b|\bwouldn't\b|\bcouldn't\b|\bshouldn't\b",
    re.IGNORECASE,
)


def validate_response(text: str) -> list[dict]:
    """
    Check a Socrates response for authenticity violations.
    Returns list of {type, term, severity} dicts. Empty list = clean.

    Severity levels:
    - critical: character break (breaks immersion completely)
    - high: anachronism (post-Socratic reference)
    - medium: modern register
    - low: contraction
    """
    issues = []
    lower = text.lower()

    for term in CHARACTER_BREAKS:
        if term in lower:
            issues.append({"type": "character_break", "term": term, "severity": "critical"})

    for term in ANACHRONISM_TERMS:
        if term in lower:
            # Skip "democracy" if it's used in historical Athenian context
            # (rough heuristic: check if "athens" or "athenian" appears nearby)
            if term == "democracy":
                idx = lower.find(term)
                context_window = lower[max(0, idx - 100):idx + 100]
                if "athens" in context_window or "athenian" in context_window:
                    continue
            if term == "consciousness":
                # "self-consciousness" or "consciousness of" in archaic sense is ok
                idx = lower.find(term)
                context_window = lower[max(0, idx - 20):idx]
                if "self-" in context_window or "self -" in context_window:
                    continue
            issues.append({"type": "anachronism", "term": term, "severity": "high"})

    for term in MODERN_REGISTER:
        if term in lower:
            issues.append({"type": "modern_register", "term": term, "severity": "medium"})

    for match in _CONTRACTION_PATTERN.finditer(text):
        issues.append({"type": "contraction", "term": match.group(), "severity": "low"})

    # Check for LLM-generated asterisk stage directions (*pauses*, *smiles*, etc.)
    if re.search(r'\*[^*\n]{1,60}\*', text):
        issues.append({
            "type": "stage_direction",
            "term": re.search(r'\*[^*\n]{1,60}\*', text).group(),
            "severity": "medium",
            "detail": "LLM generated its own asterisk stage direction — should speak directly",
        })

    return issues


def check_repetition(current_response: str, previous_responses: list[str]) -> list[dict]:
    """
    Check for vocative, transition, and passage-reference repetition
    against recent assistant responses.
    Returns list of {type, detail, severity} dicts. Empty list = clean.
    """
    issues = []
    if not previous_responses:
        return issues

    last_response = previous_responses[-1]

    # 1. Vocative repetition: same vocative as last turn
    current_vocatives = {m.group().lower() for m in _VOCATIVE_PATTERN.finditer(current_response)}
    last_vocatives = {m.group().lower() for m in _VOCATIVE_PATTERN.finditer(last_response)}
    repeated_vocatives = current_vocatives & last_vocatives
    if repeated_vocatives:
        issues.append({
            "type": "vocative_repetition",
            "detail": f"Same vocative in consecutive turns: {repeated_vocatives.pop()}",
            "severity": "low",
        })

    # 2. Transition repetition: same transition phrase in last 2 turns
    current_transitions = {m.group().lower() for m in _TRANSITION_PATTERN.finditer(current_response)}
    recent_transitions: set[str] = set()
    for prev in previous_responses[-2:]:
        recent_transitions.update(m.group().lower() for m in _TRANSITION_PATTERN.finditer(prev))
    repeated_transitions = current_transitions & recent_transitions
    if repeated_transitions:
        issues.append({
            "type": "transition_repetition",
            "detail": f"Same transition in recent turns: {repeated_transitions.pop()}",
            "severity": "low",
        })

    # 3. Passage-reference repetition: same name cited in 3+ responses total
    current_lower = current_response.lower()
    all_names = set(_DIALOGUE_NAMES) | set(_SPEAKER_NAMES)
    for name in all_names:
        if name in current_lower:
            prev_count = sum(1 for prev in previous_responses if name in prev.lower())
            if prev_count >= 2:
                issues.append({
                    "type": "passage_repetition",
                    "detail": f"'{name}' cited in {prev_count + 1} responses — use different material",
                    "severity": "medium",
                })

    return issues


def log_violations(response_text: str, issues: list[dict], settings: dict):
    """Append violations to logs/validation_log.jsonl for drift audit consumption."""
    if not issues:
        return
    logs_dir = PROJECT_ROOT / settings["paths"]["logs_dir"]
    logs_dir.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "response_preview": response_text[:200],
        "violations": issues,
    }
    with open(logs_dir / "validation_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
