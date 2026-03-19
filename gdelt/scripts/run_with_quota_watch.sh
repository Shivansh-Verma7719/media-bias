#!/bin/bash
# =============================================================================
# run_with_quota_watch.sh
#
# Runs gdelt_pipeline.py and monitors pipeline.log for BigQuery quota exceeded
# errors. On detection, the pipeline is killed and the watcher sleeps until
# midnight Pacific Time (when the BigQuery daily quota resets), then restarts.
# The pipeline itself uses a checkpoint file so it resumes from where it
# stopped rather than re-scanning from 2015.
#
# Safety features:
#   - Ctrl+C (SIGINT) / terminal close (SIGHUP/SIGTERM) stop everything.
#   - Quota-reset wait is interruptible — Ctrl+C works during the sleep.
#   - The restart count is always visible.
#
# SSH-disconnect-proof mode (--detach):
#   - Runs the watcher inside a named tmux session (falls back to screen).
#   - Survives SSH disconnections completely.
#   - Use --attach to reconnect to the running session.
#   - Use --stop to gracefully shut everything down from any terminal.
#   - Use --status to check if the watcher is currently running.
#
# Usage:
#   ./run_with_quota_watch.sh              # foreground (normal)
#   ./run_with_quota_watch.sh --detach     # run in background tmux/screen session
#   ./run_with_quota_watch.sh --attach     # re-attach to running session
#   ./run_with_quota_watch.sh --stop       # gracefully stop the detached watcher
#   ./run_with_quota_watch.sh --status     # show whether the watcher is running
# =============================================================================

# ---- Configuration ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIPELINE_SCRIPT="$PROJECT_DIR/gdelt/gdelt_pipeline.py"
LOG_FILE="$PROJECT_DIR/pipeline.log"
CREDENTIALS_FILE="$PROJECT_DIR/gdelt/ism-gdelt-key.json"

QUOTA_ERROR_PATTERN="quotaExceeded"   # string to scan for in pipeline.log
QUOTA_RESET_BUFFER_SECS=120           # extra seconds to wait after midnight PT

# Detach / session management
SESSION_NAME="gdelt_watcher"          # tmux/screen session name
PID_FILE="/tmp/gdelt_watcher.pid"     # watcher PID stored here when detached
# -----------------------------------------------------------------------------

export GOOGLE_APPLICATION_CREDENTIALS="$CREDENTIALS_FILE"

# ---- Detect session backend (tmux preferred, screen as fallback) ------------
detect_backend() {
    if command -v tmux &>/dev/null; then
        echo "tmux"
    elif command -v screen &>/dev/null; then
        echo "screen"
    else
        echo "none"
    fi
}

# ---- Session helpers ---------------------------------------------------------

session_exists() {
    local backend="$1"
    case "$backend" in
        tmux)   tmux has-session -t "$SESSION_NAME" 2>/dev/null ;;
        screen) screen -list | grep -q "$SESSION_NAME" ;;
        *)      return 1 ;;
    esac
}

start_detached_session() {
    local backend="$1"
    case "$backend" in
        tmux)
            tmux new-session -d -s "$SESSION_NAME" \
                "bash '$BASH_SOURCE' --_run_watcher; echo '[watcher] Session ended. Press Enter to close.'; read"
            ;;
        screen)
            screen -dmS "$SESSION_NAME" \
                bash -c "bash '$BASH_SOURCE' --_run_watcher; echo '[watcher] Session ended. Press Enter to close.'; read"
            ;;
    esac
}

attach_to_session() {
    local backend="$1"
    case "$backend" in
        tmux)   tmux attach-session -t "$SESSION_NAME" ;;
        screen) screen -r "$SESSION_NAME" ;;
    esac
}

kill_session() {
    local backend="$1"
    case "$backend" in
        tmux)   tmux kill-session -t "$SESSION_NAME" 2>/dev/null ;;
        screen) screen -S "$SESSION_NAME" -X quit 2>/dev/null ;;
    esac
}

# ---- --status ----------------------------------------------------------------
cmd_status() {
    local backend
    backend="$(detect_backend)"

    echo "[watcher] Session backend : ${backend}"

    if [[ "$backend" == "none" ]]; then
        echo "[watcher] Neither tmux nor screen is installed — detach mode unavailable."
        exit 0
    fi

    if session_exists "$backend"; then
        echo "[watcher] Session '${SESSION_NAME}' : RUNNING"
    else
        echo "[watcher] Session '${SESSION_NAME}' : not found"
    fi

    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[watcher] Watcher PID ${pid}           : alive"
        else
            echo "[watcher] Watcher PID ${pid}           : stale (process gone)"
        fi
    else
        echo "[watcher] PID file                   : not found"
    fi
}

# ---- --stop ------------------------------------------------------------------
cmd_stop() {
    local backend
    backend="$(detect_backend)"
    local stopped=0

    echo "[watcher] Stopping detached watcher..."

    # 1) Try to SIGTERM the watcher PID first so cleanup() runs properly
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[watcher] Sending SIGTERM to watcher PID $pid ..."
            kill -TERM "$pid" 2>/dev/null
            # Wait up to 10 seconds for it to exit
            local i=0
            while kill -0 "$pid" 2>/dev/null && [[ $i -lt 10 ]]; do
                sleep 1
                i=$(( i + 1 ))
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "[watcher] Watcher still alive, sending SIGKILL..."
                kill -KILL "$pid" 2>/dev/null
            else
                echo "[watcher] Watcher exited cleanly."
            fi
            stopped=1
        else
            echo "[watcher] PID $pid in PID file is no longer alive."
        fi
        rm -f "$PID_FILE"
    else
        echo "[watcher] No PID file found at $PID_FILE."
    fi

    # 2) Kill the tmux/screen session if it still exists
    if [[ "$backend" != "none" ]] && session_exists "$backend"; then
        echo "[watcher] Killing ${backend} session '${SESSION_NAME}'..."
        kill_session "$backend"
        stopped=1
    fi

    if [[ "$stopped" -eq 1 ]]; then
        echo "[watcher] Done. Pipeline watcher stopped."
    else
        echo "[watcher] Nothing to stop (no running session or PID file found)."
    fi
}

# ---- --detach ----------------------------------------------------------------
cmd_detach() {
    local backend
    backend="$(detect_backend)"

    if [[ "$backend" == "none" ]]; then
        echo "[watcher] ERROR: Neither tmux nor screen is installed."
        echo "[watcher] Install one of them (e.g. 'sudo apt install tmux') and try again."
        exit 1
    fi

    if session_exists "$backend"; then
        echo "[watcher] A session named '${SESSION_NAME}' is already running."
        echo "[watcher] Use --attach to connect to it, or --stop to stop it first."
        exit 1
    fi

    echo "[watcher] Starting detached session '${SESSION_NAME}' via ${backend}..."
    start_detached_session "$backend"

    sleep 1   # give the session a moment to spin up

    if session_exists "$backend"; then
        echo "[watcher] Session started successfully."
        echo ""
        echo "  To watch live output : $0 --attach"
        echo "  To stop the pipeline : $0 --stop"
        echo "  To check status      : $0 --status"
    else
        echo "[watcher] ERROR: Session did not start. Check your tmux/screen installation."
        exit 1
    fi
}

# ---- --attach ----------------------------------------------------------------
cmd_attach() {
    local backend
    backend="$(detect_backend)"

    if [[ "$backend" == "none" ]]; then
        echo "[watcher] ERROR: tmux/screen not installed — no session to attach to."
        exit 1
    fi

    if ! session_exists "$backend"; then
        echo "[watcher] No running session named '${SESSION_NAME}' found."
        echo "[watcher] Start one with: $0 --detach"
        exit 1
    fi

    echo "[watcher] Attaching to ${backend} session '${SESSION_NAME}'..."
    echo "[watcher] (Detach without stopping: $([ "$backend" = "tmux" ] && echo "Ctrl+B then D" || echo "Ctrl+A then D"))"
    attach_to_session "$backend"
}

# ---- Argument parsing --------------------------------------------------------
MODE="foreground"  # default

case "${1:-}" in
    --detach|-d)    MODE="detach" ;;
    --attach|-a)    MODE="attach" ;;
    --stop|-s)      MODE="stop" ;;
    --status)       MODE="status" ;;
    --_run_watcher) MODE="watcher" ;;   # internal: called inside the tmux/screen session
    "")             MODE="foreground" ;;
    *)
        echo "Usage: $0 [--detach|-d | --attach|-a | --stop|-s | --status]"
        echo ""
        echo "  (no flag)   Run in foreground (normal mode)"
        echo "  --detach    Run in a background tmux/screen session (SSH-disconnect-proof)"
        echo "  --attach    Re-attach to the running background session"
        echo "  --stop      Gracefully stop the running background watcher"
        echo "  --status    Show whether the watcher is currently running"
        exit 1
        ;;
esac

# Dispatch non-watcher modes immediately
case "$MODE" in
    detach)     cmd_detach; exit 0 ;;
    attach)     cmd_attach; exit 0 ;;
    stop)       cmd_stop;   exit 0 ;;
    status)     cmd_status; exit 0 ;;
esac

# ============================================================================
# From here down: the actual watcher logic.
# Runs when MODE=foreground OR MODE=watcher (inside tmux/screen).
# ============================================================================

# Track pipeline PID and tail PID globally so signal handlers can clean up
PIPELINE_PID=""
TAIL_PID=""
RESTART_COUNT=0

# Flag: set to 1 when a user/terminal signal is received so we do NOT restart
USER_KILLED=0

# Write our own PID to the PID file so --stop can find us
echo "$$" > "$PID_FILE"

# ---- Cleanup -----------------------------------------------------------------
cleanup() {
    USER_KILLED=1  # Prevent any restart loop

    echo ""
    echo "[watcher] Caught signal — shutting down cleanly..."

    # Kill the log tail process
    if [[ -n "$TAIL_PID" ]] && kill -0 "$TAIL_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null
        wait "$TAIL_PID" 2>/dev/null
    fi

    # Kill the pipeline process
    if [[ -n "$PIPELINE_PID" ]] && kill -0 "$PIPELINE_PID" 2>/dev/null; then
        echo "[watcher] Sending SIGTERM to pipeline (PID $PIPELINE_PID)..."
        kill -TERM "$PIPELINE_PID" 2>/dev/null
        # Give it a moment to shut down gracefully
        sleep 2
        if kill -0 "$PIPELINE_PID" 2>/dev/null; then
            echo "[watcher] Pipeline still running, sending SIGKILL..."
            kill -KILL "$PIPELINE_PID" 2>/dev/null
        fi
        wait "$PIPELINE_PID" 2>/dev/null
    fi

    rm -f "$PID_FILE"
    echo "[watcher] Exited after $RESTART_COUNT restart(s)."
    exit 0
}

# Trap Ctrl+C (SIGINT) and terminal close (SIGHUP) for clean exit
trap cleanup SIGINT SIGHUP SIGTERM

# ---- Main loop ---------------------------------------------------------------
cd "$PROJECT_DIR" || { echo "[watcher] ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }

echo "[watcher] Starting GDELT pipeline watcher."
echo "[watcher] Project dir : $PROJECT_DIR"
echo "[watcher] Pipeline    : $PIPELINE_SCRIPT"
echo "[watcher] Log file    : $LOG_FILE"
if [[ "$MODE" == "watcher" ]]; then
    echo "[watcher] Running inside detached session '${SESSION_NAME}'."
    echo "[watcher] To stop: run '$0 --stop' from any terminal."
else
    echo "[watcher] Press Ctrl+C at any time to fully stop."
fi
echo ""

while true; do
    # Don't restart if the user killed us
    if [[ "$USER_KILLED" -eq 1 ]]; then
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "[watcher] --- Attempt #$RESTART_COUNT --- $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[watcher] Launching pipeline..."

    # Truncate or touch the log so we only tail new output from this run
    # (keeps the log file but removes the old content watched by tail)
    # We use a marker approach: note the current end-of-file position so
    # tail --follow only picks up lines written AFTER we start.
    LOG_START_LINE=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)

    # Launch the pipeline in the background
    python3 "$PIPELINE_SCRIPT" &
    PIPELINE_PID=$!
    echo "[watcher] Pipeline PID: $PIPELINE_PID"

    # Give the pipeline a moment to create/open the log file
    sleep 2

    # ---- Monitor the log for quota errors using a background tail ------------
    # We start tail from the current end of the log so we only see new lines.
    QUOTA_HIT=0

    # Use a temp FIFO so we can read lines and react, while tail runs in bg
    FIFO=$(mktemp -u /tmp/gdelt_watcher_XXXXXX)
    mkfifo "$FIFO"

    tail -n 0 -F "$LOG_FILE" > "$FIFO" 2>/dev/null &
    TAIL_PID=$!

    # Read lines from the FIFO until: pipeline exits, user kills, or quota hit
    while IFS= read -r line; do
        # Check if user triggered a shutdown while we were reading
        if [[ "$USER_KILLED" -eq 1 ]]; then
            break
        fi

        if echo "$line" | grep -q "$QUOTA_ERROR_PATTERN"; then
            echo ""
            echo "[watcher] *** Quota exceeded error detected! ***"
            echo "[watcher] Line: $line"
            QUOTA_HIT=1

            # Kill the tail watcher first
            kill "$TAIL_PID" 2>/dev/null
            wait "$TAIL_PID" 2>/dev/null
            TAIL_PID=""

            # Now kill the pipeline process tree
            echo "[watcher] Sending SIGTERM to pipeline (PID $PIPELINE_PID)..."
            kill -TERM "$PIPELINE_PID" 2>/dev/null
            sleep 2
            if kill -0 "$PIPELINE_PID" 2>/dev/null; then
                echo "[watcher] Sending SIGKILL to pipeline..."
                kill -KILL "$PIPELINE_PID" 2>/dev/null
            fi
            wait "$PIPELINE_PID" 2>/dev/null
            PIPELINE_PID=""
            break
        fi

        # Check if the pipeline process has already exited on its own
        if ! kill -0 "$PIPELINE_PID" 2>/dev/null; then
            break
        fi
    done < "$FIFO"

    # Clean up FIFO
    rm -f "$FIFO"

    # Clean up tail if still running
    if [[ -n "$TAIL_PID" ]] && kill -0 "$TAIL_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null
        wait "$TAIL_PID" 2>/dev/null
        TAIL_PID=""
    fi

    # If user killed us, stop the loop entirely
    if [[ "$USER_KILLED" -eq 1 ]]; then
        break
    fi

    # If the pipeline exited naturally (no quota hit), we're done
    if [[ "$QUOTA_HIT" -eq 0 ]]; then
        echo "[watcher] Pipeline exited normally. All done."
        break
    fi

    # ---- Quota was hit — sleep until midnight Pacific Time then restart ------
    #
    # BigQuery free-tier quota resets at midnight PT (PDT = UTC-7, PST = UTC-8).
    # We calculate the offset at runtime: during DST (Mar–Nov) = UTC-7 (-25200s),
    # outside DST (Nov–Mar) = UTC-8 (-28800s).
    # "Midnight PT" expressed in UTC = 07:00 UTC (PDT) or 08:00 UTC (PST).
    # ---
    wait_until_quota_reset() {
        # Determine current UTC offset for Pacific time using Python (available
        # in the venv). Falls back to a fixed 7-hour offset if Python fails.
        local pt_offset_secs
        pt_offset_secs=$(python3 - <<'PYEOF' 2>/dev/null
import datetime, zoneinfo
tz = zoneinfo.ZoneInfo("America/Los_Angeles")
now_pt = datetime.datetime.now(tz)
offset = int(now_pt.utcoffset().total_seconds())
print(offset)
PYEOF
        )
        # Default to PDT = -7h if python failed
        pt_offset_secs=${pt_offset_secs:--25200}

        local now_epoch
        now_epoch=$(date -u +%s)

        # Compute seconds-since-epoch for the NEXT midnight PT.
        # midnight PT in UTC = midnight PT + |offset| seconds.
        # e.g. midnight PDT = 00:00 PDT = 07:00 UTC
        local midnight_utc_secs=$(( -pt_offset_secs ))  # offset is negative, so negate
        local today_midnight_epoch
        today_midnight_epoch=$(date -u -d "$(date -u +%Y-%m-%d) ${midnight_utc_secs}seconds" +%s 2>/dev/null || \
            python3 -c "
import datetime, calendar
d = datetime.date.today()
midnight_utc = datetime.datetime(d.year, d.month, d.day, ${midnight_utc_secs}//3600, 0, 0)
print(int(calendar.timegm(midnight_utc.timetuple())))
")

        local target_epoch=$(( today_midnight_epoch + QUOTA_RESET_BUFFER_SECS ))

        # If target is in the past (e.g. we are running after midnight UTC already),
        # advance by one day
        if [[ "$target_epoch" -le "$now_epoch" ]]; then
            target_epoch=$(( target_epoch + 86400 ))
        fi

        local sleep_secs=$(( target_epoch - now_epoch ))
        local wake_time
        wake_time=$(date -u -d "@$target_epoch" '+%Y-%m-%d %H:%M UTC' 2>/dev/null || \
            python3 -c "import datetime; print(datetime.datetime.utcfromtimestamp($target_epoch).strftime('%Y-%m-%d %H:%M UTC'))")

        local hours=$(( sleep_secs / 3600 ))
        local mins=$(( (sleep_secs % 3600) / 60 ))

        echo "[watcher] BigQuery daily quota resets at midnight Pacific Time."
        echo "[watcher] Sleeping ${hours}h ${mins}m until ${wake_time} (+${QUOTA_RESET_BUFFER_SECS}s buffer)."
        if [[ "$MODE" == "watcher" ]]; then
            echo "[watcher] To stop during this wait: run '$0 --stop' from any terminal."
        else
            echo "[watcher] Press Ctrl+C to abort the wait and stop the watcher."
        fi

        # Sleep in 60-second chunks so USER_KILLED is checked regularly
        local elapsed=0
        while [[ "$elapsed" -lt "$sleep_secs" ]]; do
            if [[ "$USER_KILLED" -eq 1 ]]; then
                echo "[watcher] Sleep interrupted by user signal."
                return
            fi
            local remaining=$(( sleep_secs - elapsed ))
            local chunk=$(( remaining < 60 ? remaining : 60 ))
            sleep "$chunk"
            elapsed=$(( elapsed + chunk ))
            # Print a progress tick every ~5 minutes
            if (( elapsed % 300 == 0 )); then
                local left=$(( sleep_secs - elapsed ))
                echo "[watcher] ... $(( left / 3600 ))h $(( (left % 3600) / 60 ))m remaining until restart ..."
            fi
        done
    }

    wait_until_quota_reset

    # Check once more if the user killed during the wait
    if [[ "$USER_KILLED" -eq 1 ]]; then
        break
    fi
done

rm -f "$PID_FILE"
echo "[watcher] Watcher finished. Total restarts: $((RESTART_COUNT - 1))."
