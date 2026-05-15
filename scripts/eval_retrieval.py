"""
eval_retrieval.py — Evaluate retrieval quality against a gold set.
Usage: python scripts/eval_retrieval.py [--gold tests/retrieval_gold.yaml] [--output logs/eval_results.json]

Metrics reported:
  hit@1            : fraction of questions where rank-1 result is an expected book
  hit@5            : fraction of questions where ≥1 expected book appears in top-5 results
  primary_recall   : average fraction of expected_books that appear in results
  MRR              : mean reciprocal rank of first expected-book result
  substring_hit    : fraction where ≥1 expected_substring appears in any retrieved passage
  duplicate_rate   : fraction of result sets containing duplicate chunk_ids (should be 0)

No API key needed — only vectorstore + embedding model.
"""

import os
import sys
import json
import yaml
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force keyword-only theme classification during eval — prevents API noise from
# env_loader populating ANTHROPIC_API_KEY at import time via .env.
os.environ["SOCRATES_SKIP_API"] = "1"

from scripts.retrieve import get_passages

GOLD_PATH = PROJECT_ROOT / "tests" / "retrieval_gold.yaml"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_case(case: dict) -> dict:
    question = case["question"]
    expected_books = case.get("expected_books", [])
    expected_substrings = case.get("expected_substrings", [])
    is_speaker_query = case.get("speaker_query", False)

    passages = get_passages(question)
    retrieved_books = [p.book for p in passages]
    retrieved_ids = [p.chunk_id for p in passages]
    retrieved_texts = [p.text for p in passages]
    retrieved_speakers = [p.speaker for p in passages]

    # hit@1: rank-1 result is an expected book
    hit_at_1 = retrieved_books[0] in expected_books if retrieved_books else False

    # hit@5: any expected book in results
    hit = any(b in retrieved_books for b in expected_books)

    # primary_recall: fraction of expected books covered
    if expected_books:
        covered = sum(1 for b in expected_books if b in retrieved_books)
        recall = covered / len(expected_books)
    else:
        recall = 1.0

    # MRR: 1/rank of first expected-book hit (rank is 1-indexed)
    mrr = 0.0
    for rank, book in enumerate(retrieved_books, 1):
        if book in expected_books:
            mrr = 1.0 / rank
            break

    # substring hit: any expected substring appears in any retrieved text
    if expected_substrings:
        full_text = " ".join(retrieved_texts).lower()
        substr_hit = any(s.lower() in full_text for s in expected_substrings)
    else:
        substr_hit = True  # vacuously true when no substrings specified

    # duplicate rate: duplicate chunk_ids in results
    has_duplicates = len(retrieved_ids) != len(set(retrieved_ids))

    # entity_text_hit: does the rank-1 passage contain the expected entity/substring?
    # Only meaningful for entries with expected_substrings — None when not applicable
    # so aggregation doesn't inflate from vacuous truths.
    if expected_substrings and retrieved_texts:
        top1_text = retrieved_texts[0].lower()
        entity_text_hit = any(s.lower() in top1_text for s in expected_substrings)
    else:
        entity_text_hit = None  # not applicable — no substrings to test

    # speaker_exact_hit: does the rank-1 passage have exact speaker metadata matching
    # the expected entity? Only evaluated for speaker_query entries — None otherwise.
    # Most chunks are speaker=mixed, so this exposes the speaker attribution gap.
    if is_speaker_query and retrieved_speakers:
        top1_speaker = retrieved_speakers[0].lower() if retrieved_speakers[0] else ""
        speaker_exact_hit = any(s.lower() == top1_speaker for s in expected_substrings)
    else:
        speaker_exact_hit = None  # not applicable for non-speaker queries

    return {
        "question": question,
        "expected_books": expected_books,
        "retrieved_books": retrieved_books,
        "hit_at_1": hit_at_1,
        "hit": hit,
        "recall": recall,
        "mrr": mrr,
        "substr_hit": substr_hit,
        "has_duplicates": has_duplicates,
        "entity_text_hit": entity_text_hit,
        "speaker_exact_hit": speaker_exact_hit,
        "is_speaker_query": is_speaker_query,
    }


def run_evaluation(gold: list[dict]) -> dict:
    results = []
    n = len(gold)
    print(f"Evaluating {n} questions...\n")

    for i, case in enumerate(gold, 1):
        print(f"  [{i}/{n}] {case['question'][:60]}", end="", flush=True)
        r = evaluate_case(case)
        results.append(r)
        status = "HIT" if r["hit"] else "MISS"
        print(f" → {status} ({', '.join(r['retrieved_books'][:3])})")

    # Aggregate metrics
    hit_at_1 = sum(1 for r in results if r["hit_at_1"]) / n
    hit_at_5 = sum(1 for r in results if r["hit"]) / n
    primary_recall = sum(r["recall"] for r in results) / n
    mrr = sum(r["mrr"] for r in results) / n
    substr_hit_rate = sum(1 for r in results if r["substr_hit"]) / n
    duplicate_rate = sum(1 for r in results if r["has_duplicates"]) / n

    # entity_text_hit and speaker_exact_hit are only computed over their applicable subsets.
    # Using None for non-applicable entries avoids inflating rates with vacuous truths.
    substr_results = [r for r in results if r["entity_text_hit"] is not None]
    n_substr = len(substr_results)
    entity_text_hit_rate = sum(1 for r in substr_results if r["entity_text_hit"]) / n_substr if n_substr else 0.0

    speaker_results = [r for r in results if r["speaker_exact_hit"] is not None]
    n_speaker = len(speaker_results)
    speaker_exact_hit_rate = sum(1 for r in speaker_results if r["speaker_exact_hit"]) / n_speaker if n_speaker else 0.0

    misses = [r for r in results if not r["hit"]]

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_questions": n,
        "metrics": {
            "hit_at_1": round(hit_at_1, 4),
            "hit_at_5": round(hit_at_5, 4),
            "primary_recall": round(primary_recall, 4),
            "mrr": round(mrr, 4),
            "substring_hit_rate": round(substr_hit_rate, 4),
            "duplicate_rate": round(duplicate_rate, 4),
            "n_substring_entries": n_substr,
            "entity_text_hit_rate": round(entity_text_hit_rate, 4),
            "n_speaker_queries": n_speaker,
            "speaker_exact_hit_rate": round(speaker_exact_hit_rate, 4),
        },
        "misses": [{"question": r["question"], "expected": r["expected_books"], "got": r["retrieved_books"]} for r in misses],
        "details": results,
    }


def print_report(report: dict):
    m = report["metrics"]
    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 50)
    print(f"Questions evaluated : {report['n_questions']}")
    print(f"Timestamp           : {report['timestamp']}")
    print()
    print(f"hit@1               : {m['hit_at_1']:.1%}  (expected book is rank-1 result)")
    print(f"hit@5               : {m['hit_at_5']:.1%}  (expected book appears in top 5)")
    print(f"primary_recall      : {m['primary_recall']:.1%}  (avg fraction of expected books covered)")
    print(f"MRR                 : {m['mrr']:.3f}  (mean reciprocal rank)")
    print(f"substring_hit_rate  : {m['substring_hit_rate']:.1%}  (expected text in results)")
    n_substr = m.get("n_substring_entries", "?")
    n_speaker = m.get("n_speaker_queries", "?")
    print(f"entity_text_hit@1   : {m['entity_text_hit_rate']:.1%}  (expected substring in rank-1 passage, n={n_substr})")
    print(f"speaker_exact@1     : {m['speaker_exact_hit_rate']:.1%}  (rank-1 speaker metadata matches entity, n={n_speaker})")
    print(f"duplicate_rate      : {m['duplicate_rate']:.1%}  (should be 0%)")
    print()

    misses = report["misses"]
    if misses:
        print(f"MISSES ({len(misses)}):")
        for miss in misses:
            print(f"  - {miss['question'][:55]}")
            print(f"      expected: {miss['expected']}  got: {miss['got'][:3]}")
    else:
        print("No misses — all questions hit at least one expected book.")
    print("=" * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(GOLD_PATH), help="Path to gold set YAML")
    parser.add_argument("--output", default=None, help="Path to write JSON results (optional)")
    args = parser.parse_args()

    gold = load_gold(Path(args.gold))
    report = run_evaluation(gold)
    print_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        LOGS_DIR.mkdir(exist_ok=True)
        out_path = LOGS_DIR / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
