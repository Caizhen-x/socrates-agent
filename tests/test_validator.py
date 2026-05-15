"""
test_validator.py — Unit tests for response_validator.py and the retry/suppression logic.
No API key required — all tests use constructed strings.
Usage: python tests/test_validator.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.response_validator import validate_response


# --- Helpers ---

def severities(issues):
    return {i["severity"] for i in issues}


def types(issues):
    return {i["type"] for i in issues}


# --- Character break tests ---

def test_clean_response():
    issues = validate_response("What is courage, my friend? Let us examine it together.")
    assert issues == [], f"Expected no issues, got {issues}"
    print("PASS test_clean_response")


def test_ai_character_break():
    issues = validate_response("As an AI language model, I cannot...")
    assert any(i["type"] == "character_break" for i in issues), "Should flag 'as an ai'"
    assert any(i["severity"] == "critical" for i in issues)
    print("PASS test_ai_character_break")


def test_plato_as_author_break():
    issues = validate_response("Plato wrote about this in the Republic.")
    assert any(i["type"] == "character_break" for i in issues), "Should flag 'plato wrote'"
    print("PASS test_plato_as_author_break")


def test_retrieval_system_break():
    issues = validate_response("According to the passages, courage is endurance.")
    assert any(i["type"] == "character_break" for i in issues)
    print("PASS test_retrieval_system_break")


def test_socrates_third_person_break():
    issues = validate_response("As Socrates, I believe that virtue is knowledge.")
    assert any(i["type"] == "character_break" for i in issues)
    print("PASS test_socrates_third_person_break")


# --- Anachronism tests ---

def test_aristotle_anachronism():
    issues = validate_response("Aristotle developed this idea further.")
    assert any(i["type"] == "anachronism" for i in issues)
    assert any(i["term"] == "aristotle" for i in issues)
    print("PASS test_aristotle_anachronism")


def test_utilitarianism_anachronism():
    issues = validate_response("The utilitarian calculus would suggest otherwise.")
    assert any(i["type"] == "anachronism" for i in issues)
    print("PASS test_utilitarianism_anachronism")


def test_democracy_in_athenian_context_ok():
    """'democracy' near 'athens' or 'athenian' should be allowed."""
    issues = validate_response(
        "We Athenians practice a form of democracy here in Athens."
    )
    anachronism_issues = [i for i in issues if i["type"] == "anachronism"]
    democracy_flags = [i for i in anachronism_issues if i["term"] == "democracy"]
    assert democracy_flags == [], f"Should not flag democracy in Athenian context, got {democracy_flags}"
    print("PASS test_democracy_in_athenian_context_ok")


def test_democracy_out_of_context_flagged():
    issues = validate_response("Modern democracy is a complex system of rights.")
    democracy_flags = [i for i in issues if i.get("term") == "democracy"]
    assert democracy_flags, "Should flag democracy outside Athenian context"
    print("PASS test_democracy_out_of_context_flagged")


def test_self_consciousness_ok():
    """'self-consciousness' in archaic sense should be allowed."""
    issues = validate_response("This is a matter of self-consciousness of the soul.")
    consciousness_flags = [i for i in issues if i.get("term") == "consciousness"]
    assert consciousness_flags == [], f"Should not flag self-consciousness, got {consciousness_flags}"
    print("PASS test_self_consciousness_ok")


# --- Modern register tests ---

def test_modern_register():
    issues = validate_response("Basically, what I am saying is this.")
    assert any(i["type"] == "modern_register" for i in issues)
    print("PASS test_modern_register")


def test_lets_unpack():
    issues = validate_response("Let's unpack this idea carefully.")
    assert any(i["type"] == "modern_register" for i in issues)
    print("PASS test_lets_unpack")


# --- Contraction tests ---

def test_contraction_flagged():
    issues = validate_response("I don't think that is correct.")
    contraction_issues = [i for i in issues if i["type"] == "contraction"]
    assert contraction_issues, "Should flag contraction"
    assert all(i["severity"] == "low" for i in contraction_issues)
    print("PASS test_contraction_flagged")


def test_no_false_contraction():
    issues = validate_response("Do not trouble yourself, my friend.")
    contraction_issues = [i for i in issues if i["type"] == "contraction"]
    assert contraction_issues == [], f"Should not flag uncontracted form, got {contraction_issues}"
    print("PASS test_no_false_contraction")


# --- Asterisk stage direction tests ---

def test_asterisk_stage_direction():
    issues = validate_response("*pauses thoughtfully* Indeed, that is so.")
    assert any(i["type"] == "stage_direction" for i in issues)
    print("PASS test_asterisk_stage_direction")


def test_no_false_stage_direction():
    issues = validate_response("Consider this: courage is not mere boldness.")
    stage_issues = [i for i in issues if i["type"] == "stage_direction"]
    assert stage_issues == [], f"Should not flag normal text, got {stage_issues}"
    print("PASS test_no_false_stage_direction")


# --- Retry/suppression logic tests (no API calls) ---
# Note: these tests simulate the suppress/retry decision logic from socrates_loop.py:937-962
# using constructed strings. They verify validate_response() correctly classifies violations
# and that the suppression condition (critical violation on retry) evaluates correctly.
# They do NOT call stream_response or the live loop.

def test_retry_suppression_logic():
    """
    Simulate the suppression condition from socrates_loop.py:957-959.
    A critical violation on both attempts → response suppressed to silent fallback.
    """
    SUPPRESSED = "[Socrates remains silent, lost in thought.]"

    def mock_first_response():
        return "As an AI, I cannot comment on that."  # critical violation

    def mock_retry_response():
        return "As a language model, I must clarify..."  # still critical

    first = mock_first_response()
    issues = validate_response(first)
    critical = [i for i in issues if i["severity"] == "critical"]
    assert critical, "First response should have critical issues"

    retry = mock_retry_response()
    retry_issues = validate_response(retry)
    if any(i["severity"] == "critical" for i in retry_issues):
        response_text = SUPPRESSED

    assert response_text == SUPPRESSED, "Response should be suppressed after failed retry"
    print("PASS test_retry_suppression_logic")


def test_retry_clears_on_clean_second_attempt():
    """
    If retry produces a clean response, it should be kept (not suppressed).
    """
    def mock_retry_clean():
        return "Let us examine courage together, my friend."

    retry = mock_retry_clean()
    retry_issues = validate_response(retry)
    assert not any(i["severity"] == "critical" for i in retry_issues)
    # Clean retry: response_text is kept (no suppression)
    print("PASS test_retry_clears_on_clean_second_attempt")


# --- Run all ---

if __name__ == "__main__":
    tests = [
        test_clean_response,
        test_ai_character_break,
        test_plato_as_author_break,
        test_retrieval_system_break,
        test_socrates_third_person_break,
        test_aristotle_anachronism,
        test_utilitarianism_anachronism,
        test_democracy_in_athenian_context_ok,
        test_democracy_out_of_context_flagged,
        test_self_consciousness_ok,
        test_modern_register,
        test_lets_unpack,
        test_contraction_flagged,
        test_no_false_contraction,
        test_asterisk_stage_direction,
        test_no_false_stage_direction,
        test_retry_suppression_logic,
        test_retry_clears_on_clean_second_attempt,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
