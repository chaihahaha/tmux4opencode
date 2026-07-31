# Tmux GPU Training Management

## Sequential Training
Use `tmux` to run long GPU training tasks that survive SSH disconnection.

### Start Training in Tmux
```bash
conda activate <env>
tmux kill-session -t <name> 2>/dev/null   # clean old session
rm -f /tmp/<name>.log                      # clean old log
tmux new-session -d -s <name> \
  "python -u train.py --epochs 30 2>&1 | tee /tmp/<name>.log"
```

Key patterns:
- Always use `python -u` for unbuffered output (otherwise logs lag)
- Always `tee` to a log file for later inspection
- Use `tmux new-session -d` to start detached
- Session name should be short and descriptive (e.g., `ps`, `train`, `watchdog`)

### Start Auto-Sequential Watchdog
```bash
tmux new-session -d -s watchdog \
  "while tmux has-session -t train 2>/dev/null; do sleep 120; done; \
   sleep 10; \
   python -u train_next.py 2>&1 | tee /tmp/next.log"
```
The watchdog polls every 120s checking if the main session exited, then auto-starts the next job.

### Check Status
```bash
# List all sessions
tmux ls

# Read latest log output (non-intrusive)
tail -5 /tmp/<name>.log

# Capture current tmux pane content
tmux capture-pane -t <name> -p | tail -5

# Quick GPU check
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

### Parse Logs
```bash
# Count NaN occurrences
grep -c "NaN" /tmp/<name>.log

# Extract epoch losses
grep "Epoch.*Losses" /tmp/<name>.log

# Find current batch progress
grep -oP 'Batch \d+/\d+' /tmp/<name>.log | tail -1

# Check for errors/OOM
grep -i "error\|OOM\|traceback\|killed" /tmp/<name>.log
```

### Kill Training
```bash
# Preferred: kill the tmux session (clean)
tmux kill-session -t <name>

# Fallback: force-kill Python processes on GPU
fuser -k /dev/nvidia0

# Nuclear: kill all training scripts
pkill -9 -f "train_unified\|train\\.py"
```

### Important Notes
- `pkill` from within opencode often hangs due to child process tree; use `tmux kill-session` first
- `fuser -k /dev/nvidia0` is reliable for freeing GPU but kills ALL GPU processes
- When restarting training, always `rm -rf` old checkpoint dirs to avoid confusion
- Check `nvidia-smi` after killing to confirm GPU memory is freed (should be ~50-150 MiB for system processes)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can cause segfaults with CUDA version mismatches; avoid it unless specifically needed

## Self-Supervised Loop: opencode watches opencode

Scenario: an opencode TUI runs inside a tmux session (e.g. `opencode_tmux_session`)
and is designing a neural network and training it. After training starts, the
model usually stops generating and silently waits for the next user instruction,
so training stalls. Fix: start a SECOND opencode session in its own tmux window
(the *supervisor*). The supervisor periodically `sleep`s, captures the target
pane, detects when the target model has finished generating (gone idle), and —
if idle — `send-keys` a follow-up prompt telling it to inspect the results and
improve the model.

### 1. Start the supervisor session

```bash
tmux kill-session -t supervisor 2>/dev/null
tmux new-session -d -s supervisor "opencode"
sleep 4                                        # let the TUI boot
tmux send-keys -t supervisor -l \
  'You are a supervisor. A training opencode session runs in the tmux pane "opencode_tmux_session" and is building/training a neural network. Keep it working autonomously. Loop: (1) sleep 90s; (2) tmux capture-pane -t opencode_tmux_session -p -S - > /tmp/snap_a; sleep 6; capture again > /tmp/snap_b; (3) if cmp shows the snapshots differ, the model is still generating - continue the loop; (4) if identical, sleep 10s and capture a third time - only if still identical is the model idle at the input prompt; (5) then run tmux send-keys -t opencode_tmux_session -l "Look at the log file of training. If loss looks wrong or training crashed, fix it and continue improving the model architecture design." followed by tmux send-keys -t opencode_tmux_session Enter; (6) never send keys while the pane is still generating, and never disturb text the user is typing; (7) after each nudge, log what you sent and wait a full generation before the next check.'
tmux send-keys -t supervisor Enter
```

### 2. Detect idle (finished generating, waiting for input)

The reliable signal is a **stable snapshot**: while the model streams, the pane
changes every few seconds; once it stops, the pane freezes. So:

```bash
tmux capture-pane -t opencode_tmux_session -p -S - > /tmp/snap_a
sleep 5
tmux capture-pane -t opencode_tmux_session -p -S - > /tmp/snap_b

if cmp -s /tmp/snap_a /tmp/snap_b; then
  echo "IDLE (generation stopped)"
else
  echo "BUSY (still generating)"
fi
```

One freeze can be a false positive (e.g. a static "Thinking..." lull). Confirm
by capturing a third snapshot after 10s and requiring it to match too:

```bash
sleep 10
tmux capture-pane -t opencode_tmux_session -p -S - > /tmp/snap_c
if ! cmp -s /tmp/snap_b /tmp/snap_c; then
  echo "not idle after all"
fi
```

Optionally verify the bottom line is the empty input prompt:

```bash
tmux capture-pane -t opencode_tmux_session -p | tail -1   # expect ">"
```

### 3. Send the follow-up prompt

Use `-l` (literal) so tmux does not misread characters as key names, then send
`Enter` as a separate command:

```bash
tmux send-keys -t opencode_tmux_session -l \
  'Look at the log file of training (/tmp/<name>.log). If loss looks wrong or training crashed, fix it, then continue improving the model architecture design.'
tmux send-keys -t opencode_tmux_session Enter
```

### 4. Ready-made watchdog loop

Run inside the supervisor session (or as a bash script); adapt the sleep
interval and follow-up prompt as needed:

```bash
while true; do
  sleep 90                                          # poll interval
  tmux capture-pane -t opencode_tmux_session -p -S - > /tmp/snap_a
  sleep 6
  tmux capture-pane -t opencode_tmux_session -p -S - > /tmp/snap_b
  cmp -s /tmp/snap_a /tmp/snap_b || { echo "[$(date)] still generating"; continue; }

  sleep 10                                          # confirm it is really idle
  tmux capture-pane -t opencode_tmux_session -p -S - > /tmp/snap_c
  cmp -s /tmp/snap_b /tmp/snap_c || { echo "[$(date)] lull, not idle"; continue; }

  echo "[$(date)] idle - nudging"
  tmux send-keys -t opencode_tmux_session -l \
    'Check the log file of training (/tmp/train.log). If loss looks wrong or training crashed, fix it; then continue improving the model architecture design.'
  tmux send-keys -t opencode_tmux_session Enter
done
```

### Supervisor Notes
- `capture-pane -t <name> -p -S -` dumps the full scrollback; `-S -` starts from
  the top of history so long replies don't scroll the interesting lines off.
- Never `send-keys` into the target pane while it is still generating: the text
  queues up and the `Enter` can interrupt the in-flight run. The stable-snapshot
  check above is what prevents this.
- If the user is actively typing in `opencode_tmux_session`, skip the nudge —
  wait for the next idle check instead of hijacking the input box.
- `send-keys -l` sends the prompt literally; without it tmux may interpret parts
  of the text (spaces, quotes, `'`, etc.) as key names. Always follow with a
  separate `Enter`.
- Give the supervisor a different session name than the training session and run
  `tmux ls` first to avoid `send-keys`-ing into the wrong pane.
