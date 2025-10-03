#!/usr/bin/env bash
INPUT="$1"
if [ -z "$INPUT" ]; then
  echo ""
  exit 0
fi
echo "[LOCAL TRANSCRIBE] Replace transcribe_local.sh with a script that calls your local model and prints a plain-text transcript to stdout."
echo "[LOCAL TRANSCRIBE] Input file: $INPUT"   
