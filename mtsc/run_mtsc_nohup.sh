#!/bin/bash
# Run mtsc_pipeline.py in the background with nohup, saving output to a log file
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

LOG_FILE="mtsc_pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Starting mtsc_pipeline.py in background..."
echo "Logs will be written to $LOG_FILE"

nohup python3 mtsc_pipeline.py > "$LOG_FILE" 2>&1 &

echo "Process started with PID $!"
