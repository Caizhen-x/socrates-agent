"""
theme_classifier.py — Extract philosophical themes from a user question.
Tries Claude Haiku API first; falls back to keyword index.
"""

import os
import re
import sys
import json
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.env_loader import load_env
load_env(PROJECT_ROOT)


def load_config():
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "theme_book_map.yaml") as f:
        theme_map = yaml.safe_load(f)
    return settings, theme_map


THEMES = [
    "justice", "courage", "piety", "love", "knowledge", "death",
    "virtue", "temperance", "friendship", "truth", "beauty",
    "soul", "good", "rhetoric", "education", "obedience",
]

CLASSIFY_PROMPT = """Given this question: "{question}"
Which 1-3 of these philosophical themes are most relevant?
[justice, courage, piety, love, knowledge, death, virtue, temperance,
 friendship, truth, beauty, soul, good, rhetoric, education, obedience]
Return ONLY a JSON list of theme strings. Example: ["courage", "knowledge"]"""


def classify_via_api(question: str, model: str) -> list[str] | None:
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(question=question)}],
        )
        raw = message.content[0].text.strip()
        themes = json.loads(raw)
        # Validate
        valid = [t for t in themes if t in THEMES]
        return valid if valid else None
    except Exception as e:
        print(f"  [theme_classifier] API call failed: {e}", file=sys.stderr)
        return None


# Precompiled keyword patterns keyed by (theme, keyword) for word-boundary matching.
# Built lazily on first classify_via_keywords call and reused thereafter.
_keyword_patterns: dict[str, re.Pattern] = {}


def _get_theme_pattern(theme: str, keywords: list[str]) -> re.Pattern:
    if theme not in _keyword_patterns:
        pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
        _keyword_patterns[theme] = re.compile(pattern, re.IGNORECASE)
    return _keyword_patterns[theme]


def classify_via_keywords(question: str, keyword_map: dict) -> list[str]:
    scores = {}
    for theme, keywords in keyword_map.items():
        pat = _get_theme_pattern(theme, keywords)
        count = len(pat.findall(question))
        if count > 0:
            scores[theme] = count
    sorted_themes = sorted(scores, key=scores.get, reverse=True)
    return sorted_themes[:3]


def classify(question: str) -> list[str]:
    settings, theme_map = load_config()

    # Skip API attempt when no key is configured OR when SOCRATES_SKIP_API is set.
    # SOCRATES_SKIP_API lets eval/test scripts force keyword fallback even when
    # .env populates ANTHROPIC_API_KEY at import time via env_loader.
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    skip_api = os.environ.get("SOCRATES_SKIP_API", "")
    if api_key and not skip_api:
        model = settings["llm"]["theme_model"]
        themes = classify_via_api(question, model)
        if themes:
            return themes

    print("  [theme_classifier] Using keyword fallback.", file=sys.stderr)
    return classify_via_keywords(question, theme_map.get("keywords", {}))


def themes_to_books(themes: list[str], theme_map: dict) -> list[str]:
    """Return primary + secondary books for the given themes."""
    books = []
    for theme in themes:
        entry = theme_map.get("themes", {}).get(theme, {})
        books.extend(entry.get("primary", []))
        books.extend(entry.get("secondary", []))
    return list(dict.fromkeys(books))  # deduplicate, preserve order


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is courage?"
    print(f"Question: {question}")
    themes = classify(question)
    print(f"Themes: {themes}")
    _, theme_map = load_config()
    books = themes_to_books(themes, theme_map)
    print(f"Primary books: {books}")
