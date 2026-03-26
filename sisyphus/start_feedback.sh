#!/bin/bash
# Start the redpill feedback service in the background.
# Safe to run multiple times — won't start a second instance if already running.

PROJ="/home/ching/projects/redpill"
LOG="$PROJ/data/feedback.log"
BINARY="/home/ching/miniconda3/envs/tenv/bin/redpill-feedback"

if pgrep -f "redpill-feedback" > /dev/null; then
    echo "redpill-feedback is already running"
    exit 0
fi

mkdir -p "$PROJ/data"
cd "$PROJ" || exit 1

nohup "$BINARY" >> "$LOG" 2>&1 &
echo "redpill-feedback started (pid $!), logging to $LOG"
