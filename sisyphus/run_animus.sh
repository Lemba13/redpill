#!/bin/bash
# Run redpill animus to synthesise DB state into data/memory/KNOWLEDGE.md.
# Loads .env so API keys are available regardless of how the script is invoked
# (cron, anacron, or manually).

PROJ="/home/ching/projects/redpill"
LOG="$PROJ/data/animus.log"
BINARY="/home/ching/miniconda3/envs/tenv/bin/redpill"
ENV_FILE="$PROJ/.env"

mkdir -p "$PROJ/data"

# Load .env — cron/anacron don't inherit your shell environment
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

cd "$PROJ" || exit 1

echo "--- redpill animus started at $(date) ---" >> "$LOG"
"$BINARY" animus --config "$PROJ/config.yaml" >> "$LOG" 2>&1
echo "--- redpill animus finished at $(date) ---" >> "$LOG"
