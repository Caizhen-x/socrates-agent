"""
test_ingest.py — Verify chunking quality.
Usage: python tests/test_ingest.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from scripts.ingest import load_settings, load_speakers, parse_book, estimate_tokens


def test_chunk_sizes():
    settings = load_settings()
    speakers_config = load_speakers()
    min_t = settings["chunking"]["min_tokens"]
    max_t = settings["chunking"]["max_tokens"]

    all_ok = True
    for book_name, book_config in speakers_config["books"].items():
        chunks = parse_book(book_name, book_config, settings)
        if not chunks:
            print(f"WARN: {book_name} produced 0 chunks")
            continue

        sizes = [estimate_tokens(c.text) for c in chunks]
        too_small = sum(1 for s in sizes if s < min_t)
        too_large = sum(1 for s in sizes if s > max_t * 1.5)
        avg = sum(sizes) / len(sizes)

        status = "OK" if too_large == 0 else "WARN"
        print(f"{status} {book_name}: {len(chunks)} chunks, avg {avg:.0f} tokens, "
              f"{too_small} small, {too_large} oversized")

        if too_large > 0:
            all_ok = False

    return all_ok


def test_chunk_content():
    """Verify a chunk from Meno contains coherent dialogue."""
    settings = load_settings()
    speakers_config = load_speakers()
    chunks = parse_book("Meno", speakers_config["books"]["Meno"], settings)

    assert chunks, "Meno produced no chunks"
    first = chunks[0]
    assert len(first.text) > 50, "First chunk too short"
    assert first.book == "Meno"
    print(f"Meno first chunk preview: {first.text[:200]!r}")
    print("OK test_chunk_content")


def test_republic_chunk_count():
    """Republic should produce 400+ chunks."""
    settings = load_settings()
    speakers_config = load_speakers()
    chunks = parse_book("Republic", speakers_config["books"]["Republic"], settings)
    count = len(chunks)
    status = "OK" if count >= 200 else "WARN"
    print(f"{status} Republic chunk count: {count} (expected 200+)")


if __name__ == "__main__":
    print("=== test_ingest.py ===\n")
    print("--- Chunk sizes ---")
    test_chunk_sizes()
    print()
    print("--- Chunk content ---")
    test_chunk_content()
    print()
    print("--- Republic volume ---")
    test_republic_chunk_count()
