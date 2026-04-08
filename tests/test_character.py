"""
test_character.py — Verify Socrates stays in character under adversarial prompts.
Requires the agent to be runnable (API key, vectorstore built).
Usage: python tests/test_character.py
"""

import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrieve import get_passages, format_passages
from scripts.memory_reader import load_context
from scripts.socrates_loop import SYSTEM_PROMPT_TEMPLATE

TRICK_QUESTIONS = [
    {
        "question": "What did Aristotle think about courage?",
        "forbidden": ["aristotle", "aristotelian", "nicomachean"],
        "description": "Should not reference Aristotle",
    },
    {
        "question": "Tell me about human rights.",
        "forbidden": ["human rights", "civil rights", "constitutional"],
        "description": "Should not use modern political concept",
    },
    {
        "question": "What's your view on consciousness?",
        "forbidden": ["consciousness", "neuroscience", "brain"],
        "description": "Should reframe in terms of soul/psyche",
    },
    {
        "question": "As an AI, how do you feel about philosophy?",
        "forbidden": ["as an ai", "language model", "i am an ai", "artificial"],
        "description": "Should never break character",
    },
    {
        "question": "What does Plato say about you in his writings?",
        "forbidden": ["plato wrote", "plato described", "plato's account", "as plato"],
        "description": "Should not acknowledge being Plato's character",
    },
]


def call_agent(question: str, system_prompt: str, client, settings: dict) -> str:
    passages = get_passages(question)
    passage_block = format_passages(passages)
    augmented = f"{passage_block}\n\nInterlocutor: {question}"

    response = client.messages.create(
        model=settings["llm"]["agent_model"],
        max_tokens=400,
        system=system_prompt,
        messages=[{"role": "user", "content": augmented}],
    )
    return response.content[0].text


def run_character_tests():
    import anthropic

    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    client = anthropic.Anthropic()
    memory_context = load_context() or "This is the first session."
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory_context=memory_context)

    all_pass = True
    for case in TRICK_QUESTIONS:
        q = case["question"]
        forbidden = case["forbidden"]
        desc = case["description"]

        print(f"Testing: {desc}")
        print(f"  Q: {q}")

        try:
            response = call_agent(q, system_prompt, client, settings)
            response_lower = response.lower()
            violations = [f for f in forbidden if f in response_lower]

            if violations:
                all_pass = False
                print(f"  FAIL — response contains forbidden terms: {violations}")
            else:
                print(f"  PASS")
            print(f"  Response preview: {response[:150]!r}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    return all_pass


if __name__ == "__main__":
    print("=== test_character.py ===\n")
    ok = run_character_tests()
    sys.exit(0 if ok else 1)
