#!/bin/bash
# Run the redpill daily pipeline.
# Loads .env so API keys are available regardless of how the script is invoked
# (cron, anacron, or manually).

PROJ="/home/ching/projects/redpill"
LOG="$PROJ/data/redpill.log"
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

echo "--- redpill run started at $(date) ---" >> "$LOG"
"$BINARY" run --config "$PROJ/config.yaml" >> "$LOG" 2>&1
echo "--- redpill run finished at $(date) ---" >> "$LOG"
