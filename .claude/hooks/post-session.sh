#!/bin/bash
# Post-session hook: write session digest
cd ~/Desktop/Socrates
echo "Writing session digest..."
python scripts/memory_writer.py --transcript logs/last_transcript.json
echo "Session digest saved."
