"""
test_retrieve.py — Verify passage relevance for known questions.
Requires vectorstore to be built first (run ingest.py).
Usage: python tests/test_retrieve.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["SOCRATES_SKIP_API"] = "1"

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
        "expected_rank1": "Crito",
        "description": "Civil disobedience → Crito (rank-1 required)",
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
    # Speaker-specific regression tests — book hit is the pass/fail criterion.
    # expected_speaker is a soft check: warns if entity name is absent from rank-1 text.
    {
        "question": "What does Thrasymachus say about justice?",
        "expected_books": ["Republic"],
        "expected_speaker": "Thrasymachus",
        "description": "Thrasymachus speaker query → Republic",
    },
    {
        "question": "What does Diotima teach about love?",
        "expected_books": ["Symposium"],
        "expected_speaker": "Diotima",
        "description": "Diotima speaker query → Symposium",
    },
    {
        "question": "What does Alcibiades say about Socrates?",
        "expected_books": ["Symposium"],
        "expected_speaker": "Alcibiades",
        "description": "Alcibiades speaker query → Symposium",
    },
    {
        "question": "What does Euthyphro say about piety?",
        "expected_books": ["Euthyphro"],
        "expected_speaker": "Euthyphro",
        "description": "Euthyphro speaker query → Euthyphro",
    },
    {
        "question": "What does Meno ask about virtue?",
        "expected_books": ["Meno"],
        "expected_speaker": "Meno",
        "description": "Meno speaker query → Meno",
    },
]


def test_retrieval():
    all_pass = True
    counts = {"PASS": 0, "WEAK": 0, "SOFT": 0, "FAIL": 0}
    for case in TEST_CASES:
        q = case["question"]
        expected = case["expected_books"]
        desc = case["description"]
        expected_speaker = case.get("expected_speaker")

        passages = get_passages(q)
        retrieved_books = [p.book for p in passages]

        hit = any(b in retrieved_books for b in expected)
        expected_rank1 = case.get("expected_rank1")

        if not hit:
            status = "FAIL"
            all_pass = False
        elif expected_rank1:
            # Strict rank-1 check: the specific book must be at position 0
            if passages and passages[0].book == expected_rank1:
                status = "PASS"
            else:
                status = "FAIL"
                all_pass = False
        elif expected_speaker:
            rank1_book_hit = bool(passages) and passages[0].book in expected
            rank1_text_hit = bool(passages) and expected_speaker.lower() in passages[0].text.lower()
            if rank1_book_hit and rank1_text_hit:
                status = "PASS"
            elif rank1_book_hit:
                status = "WEAK"   # right book at rank-1, entity name absent from rank-1 text
            else:
                status = "SOFT"   # expected book in top-5 but not at rank-1
        else:
            rank1_book_hit = bool(passages) and passages[0].book in expected
            status = "PASS" if rank1_book_hit else "PASS*"  # PASS* = book in top-5 but not rank-1

        counts[status] = counts.get(status, 0) + 1

        print(f"{status} {desc}")
        print(f"  Q: {q}")
        print(f"  Expected books: {expected}")
        print(f"  Got: {retrieved_books}")
        if passages:
            print(f"  Top passage ({passages[0].book}, speaker={passages[0].speaker}, score={passages[0].score:.3f}): {passages[0].text[:100]!r}")
        print()

    star = counts.get("PASS*", 0)
    print(f"Summary: PASS={counts['PASS']} PASS*={star} WEAK={counts['WEAK']} SOFT={counts['SOFT']} FAIL={counts['FAIL']}")
    print("  (PASS*=book in top-5 but not rank-1  WEAK=rank-1 book ok but entity absent  SOFT=speaker book not at rank-1)")
    return all_pass


if __name__ == "__main__":
    print("=== test_retrieve.py ===\n")
    ok = test_retrieval()
    sys.exit(0 if ok else 1)
