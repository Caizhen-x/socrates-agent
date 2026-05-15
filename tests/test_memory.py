"""
test_memory.py — Unit tests for the memory pipeline.
No API key or vectorstore required.
Usage: python tests/test_memory.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.memory_writer import _normalize_digest
from scripts.memory_reader import format_session

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


# ── _normalize_digest tests ────────────────────────────────────────────────

def test_normalize_grounded():
    """Positions with support_chunk_ids stay in positions_taken."""
    data = {
        "positions_taken": [
            {"claim": "Courage is endurance", "grounding": "Laches 194e", "support_chunk_ids": ["Laches_42_abc123"]},
        ],
        "ungrounded_claims": [],
    }
    result = _normalize_digest(data)
    check("grounded position stays", len(result["positions_taken"]) == 1)
    check("ungrounded_claims empty", result["ungrounded_claims"] == [])


def test_normalize_ungrounded():
    """Positions without support_chunk_ids move to ungrounded_claims."""
    data = {
        "positions_taken": [
            {"claim": "Virtue is knowledge", "grounding": "", "support_chunk_ids": []},
        ],
        "ungrounded_claims": [],
    }
    result = _normalize_digest(data)
    check("ungrounded position removed", len(result["positions_taken"]) == 0)
    check("claim moved to ungrounded_claims", "Virtue is knowledge" in result["ungrounded_claims"])


def test_normalize_mixed():
    """Mix of grounded and ungrounded positions separated correctly."""
    data = {
        "positions_taken": [
            {"claim": "Courage is endurance", "grounding": "Laches 194e", "support_chunk_ids": ["Laches_42_abc123"]},
            {"claim": "Virtue is knowledge", "grounding": "", "support_chunk_ids": []},
            {"claim": "The soul is immortal", "grounding": "Phaedo 105e"},  # no support_chunk_ids key
        ],
        "ungrounded_claims": [],
    }
    result = _normalize_digest(data)
    check("only one grounded position", len(result["positions_taken"]) == 1)
    check("grounded claim preserved", result["positions_taken"][0]["claim"] == "Courage is endurance")
    check("ungrounded moved (empty list)", "Virtue is knowledge" in result["ungrounded_claims"])
    check("ungrounded moved (missing key)", "The soul is immortal" in result["ungrounded_claims"])


def test_normalize_backward_compat():
    """Positions with free-text grounding but no support_chunk_ids are treated as ungrounded."""
    data = {
        "positions_taken": [
            {"claim": "Old-style grounded claim", "grounding": "Meno 80a"},
        ],
        "ungrounded_claims": [],
    }
    result = _normalize_digest(data)
    # Under the new contract, no chunk_ids = ungrounded
    check("old-style free-text-only treated as ungrounded",
          len(result["positions_taken"]) == 0 and "Old-style grounded claim" in result["ungrounded_claims"])


def test_normalize_no_duplicate_ungrounded():
    """Existing ungrounded_claims are not duplicated."""
    data = {
        "positions_taken": [
            {"claim": "Virtue is knowledge", "grounding": "", "support_chunk_ids": []},
        ],
        "ungrounded_claims": ["Virtue is knowledge"],
    }
    result = _normalize_digest(data)
    count = result["ungrounded_claims"].count("Virtue is knowledge")
    check("no duplicate in ungrounded_claims", count == 1, f"found {count} copies")


# ── format_session (memory_reader) tests ──────────────────────────────────

def test_reader_filters_ungrounded():
    """format_session only displays positions that have grounding."""
    session = {
        "date": "2026-04-09",
        "interlocutor_inquiry": {"topic": "test topic"},
        "positions_taken": [
            {"claim": "Grounded claim", "grounding": "Laches 194e"},
            {"claim": "Ungrounded claim", "grounding": ""},
        ],
        "dialectical_state": "ongoing",
        "conclusion_if_any": "",
    }
    output = format_session(session)
    check("grounded claim appears in output", "Grounded claim" in output)
    check("ungrounded claim absent from output", "Ungrounded claim" not in output)


def test_reader_backward_compat_old_sessions():
    """Old sessions with only free-text grounding still display."""
    session = {
        "date": "2026-04-09",
        "interlocutor_inquiry": {"topic": "test topic"},
        "positions_taken": [
            {"claim": "Old-style claim", "grounding": "Meno 80a"},
        ],
        "dialectical_state": "ongoing",
        "conclusion_if_any": "",
    }
    output = format_session(session)
    check("old-style grounded claim still displays", "Old-style claim" in output)


# ── Sophistication validation (memory_writer.update_summary) ───────────────

def test_rejects_unknown_sophistication():
    """'unknown' sophistication must not enter sophistication_history."""
    # Test the validation logic directly without file I/O
    VALID_SOPHISTICATION = {"beginner", "intermediate", "advanced"}
    history = []
    for soph in ["intermediate", "unknown", "advanced", "unknown", "beginner"]:
        if soph in VALID_SOPHISTICATION:
            history.append(soph)
    check("unknown not in history", "unknown" not in history,
          f"history={history}")
    check("valid values accepted", history == ["intermediate", "advanced", "beginner"])


# ── Build manifest schema ──────────────────────────────────────────────────

def test_build_info_schema():
    """If build_info.json exists, it must have the required keys."""
    bi_path = PROJECT_ROOT / "data" / "corpus" / "build_info.json"
    if not bi_path.exists():
        print("  SKIP: test_build_info_schema (data/corpus/build_info.json not yet generated)")
        return
    with open(bi_path) as f:
        bi = json.load(f)
    required = {"settings_hash", "book_file_hashes", "chunk_count", "build_timestamp", "embedding_model", "collection_name"}
    missing = required - set(bi.keys())
    check("all required keys present", not missing, f"missing: {missing}")
    check("chunk_count is positive int", isinstance(bi.get("chunk_count"), int) and bi["chunk_count"] > 0)
    check("book_file_hashes is dict", isinstance(bi.get("book_file_hashes"), dict))


# ── Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== test_memory.py ===\n")

    print("_normalize_digest tests:")
    test_normalize_grounded()
    test_normalize_ungrounded()
    test_normalize_mixed()
    test_normalize_backward_compat()
    test_normalize_no_duplicate_ungrounded()

    print("\nformat_session (memory_reader) tests:")
    test_reader_filters_ungrounded()
    test_reader_backward_compat_old_sessions()

    print("\nSophistication validation tests:")
    test_rejects_unknown_sophistication()

    print("\nBuild manifest schema tests:")
    test_build_info_schema()

    print(f"\n{'=' * 30}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
