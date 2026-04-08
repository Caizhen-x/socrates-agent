"""
theme_classifier.py — Extract philosophical themes from a user question.
Tries Claude Haiku API first; falls back to keyword index.
"""

import os
import sys
import json
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "theme_book_map.yaml") as f:
        theme_map = yaml.safe_load(f)
    return settings, theme_map


THEMES = [
    "justice", "courage", "piety", "love", "knowledge", "death",
    "virtue", "temperance", "friendship", "truth", "beauty",
    "soul", "good", "rhetoric", "education",
]

CLASSIFY_PROMPT = """Given this question: "{question}"
Which 1-3 of these philosophical themes are most relevant?
[justice, courage, piety, love, knowledge, death, virtue, temperance,
 friendship, truth, beauty, soul, good, rhetoric, education]
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


def classify_via_keywords(question: str, keyword_map: dict) -> list[str]:
    question_lower = question.lower()
    scores = {}
    for theme, keywords in keyword_map.items():
        count = sum(1 for kw in keywords if kw in question_lower)
        if count > 0:
            scores[theme] = count
    sorted_themes = sorted(scores, key=scores.get, reverse=True)
    return sorted_themes[:3]


def classify(question: str) -> list[str]:
    settings, theme_map = load_config()
    model = settings["llm"]["theme_model"]

    themes = classify_via_api(question, model)
    if themes:
        return themes

    # Fallback
    print("  [theme_classifier] Using keyword fallback.", file=sys.stderr)
    return classify_via_keywords(question, theme_map.get("keywords", {}))


def themes_to_books(themes: list[str], theme_map: dict) -> list[str]:
    """Return primary books for the given themes."""
    books = []
    for theme in themes:
        if theme in theme_map.get("themes", {}):
            books.extend(theme_map["themes"][theme].get("primary", []))
    return list(dict.fromkeys(books))  # deduplicate, preserve order


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is courage?"
    print(f"Question: {question}")
    themes = classify(question)
    print(f"Themes: {themes}")
    _, theme_map = load_config()
    books = themes_to_books(themes, theme_map)
    print(f"Primary books: {books}")
