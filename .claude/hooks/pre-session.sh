#!/bin/bash
# Pre-session hook: load memory context
cd ~/Desktop/Socrates
echo "Loading Socratic memory..."
python scripts/memory_reader.py > /tmp/socrates_memory_context.txt
echo "Memory loaded. $(wc -l < /tmp/socrates_memory_context.txt) lines of context ready."
