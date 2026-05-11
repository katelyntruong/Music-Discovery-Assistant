#!/bin/bash
# Weekly scraper runner — invoked by cron
# Logs to backend/logs/scraper.log (rotated, keeps last 10 runs)

BACKEND="/Users/nhantruong/Desktop/App Partify/backend"
LOG_DIR="$BACKEND/logs"
LOG_FILE="$LOG_DIR/scraper.log"

mkdir -p "$LOG_DIR"

echo "======================================" >> "$LOG_FILE"
echo "Run started: $(date)" >> "$LOG_FILE"

"$BACKEND/.venv/bin/python" "$BACKEND/scraper.py" --pages 5 >> "$LOG_FILE" 2>&1
EXIT=$?

echo "Exit code: $EXIT" >> "$LOG_FILE"
echo "Run finished: $(date)" >> "$LOG_FILE"

# Keep only the last 10 run blocks (each block = 5 lines)
tail -50 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
