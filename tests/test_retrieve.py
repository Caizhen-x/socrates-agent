"""
test_retrieve.py — Verify passage relevance for known questions.
Requires vectorstore to be built first (run ingest.py).
Usage: python tests/test_retrieve.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrieve import get_passages

TEST_CASES = [
    {
        "question": "What is courage?",
        "expected_books": ["Laches"],
        "description": "Courage question should retrieve Laches",
    },
    {
        "question": "Is it right to break an unjust law?",
        "expected_books": ["Crito"],
        "description": "Civil disobedience → Crito",
    },
    {
        "question": "What happens to the soul after death?",
        "expected_books": ["Phaedo"],
        "description": "Soul after death → Phaedo",
    },
    {
        "question": "Can virtue be taught?",
        "expected_books": ["Meno"],
        "description": "Teachability of virtue → Meno",
    },
    {
        "question": "What is love?",
        "expected_books": ["Symposium"],
        "description": "Love → Symposium",
    },
]


def test_retrieval():
    all_pass = True
    for case in TEST_CASES:
        q = case["question"]
        expected = case["expected_books"]
        desc = case["description"]

        passages = get_passages(q)
        retrieved_books = [p.book for p in passages]

        hit = any(b in retrieved_books for b in expected)
        status = "PASS" if hit else "FAIL"
        if not hit:
            all_pass = False

        print(f"{status} {desc}")
        print(f"  Q: {q}")
        print(f"  Expected books: {expected}")
        print(f"  Got: {retrieved_books}")
        if passages:
            print(f"  Top passage ({passages[0].book}, score={passages[0].score:.3f}): {passages[0].text[:100]!r}")
        print()

    return all_pass


if __name__ == "__main__":
    print("=== test_retrieve.py ===\n")
    ok = test_retrieval()
    sys.exit(0 if ok else 1)
