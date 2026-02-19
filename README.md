# Redox Flow Battery Leak/Spill Monitor (Convolutional Autoencoder)

A **Windows-friendly**, local monitoring system for a **static USB camera** pointed at a vanadium redox flow battery setup.

It supports the workflow:

1) **Collect “normal” images** (no leaks/spills) from the camera.
2) **Train** a convolutional autoencoder (TensorFlow/Keras) to reconstruct “normal.”
3) **Monitor unattended**: if reconstruction error is high, the system flags a potential anomaly, saves evidence images, writes structured logs, and sends alerts (Slack preferred).

> **Chemists / non-coders:** start with **`USER_GUIDE.md`** and the double‑click **`.bat`** files in the project root.

---

## What’s included

### Core pipeline
- **Capture normals**: `scripts/capture_images.py` (only saves frames when mode is `CAPTURE`)
- **Train model**: `scripts/train_autoencoder.py`
- **Monitor**: `scripts/monitor.py` (only runs detection when mode is `MONITOR`)

### Human-facing tools (new)
- **Reconstruction visualization**: `scripts/visualize_reconstruction.py`
- **Threshold calibration histogram + stats**: `scripts/threshold_calibration.py`
- **Health check**: `scripts/health_check.py`
- **Daily summary report**: `scripts/daily_summary.py`

### No-code launchers (Windows)
Double-clickable `.bat` files are provided for the common tasks:
- `run_capture.bat`, `run_train.bat`, `run_monitor.bat`
- `run_visualize_reconstruction.bat`, `run_threshold_calibration.bat`
- `run_health_check.bat`, `run_daily_summary.bat`
- `set_mode_CAPTURE.bat`, `set_mode_MONITOR.bat`, `set_mode_OFF.bat`

---

## How detection works (short, practical)

- The autoencoder is trained on **normal-only** images.
- For each frame, the model produces a reconstruction and a **reconstruction error score** (mean squared error).
- A threshold is computed as the **99.5th percentile** of reconstruction errors on validation normal images.
- During monitoring:
  - Trigger A (global): a frame is anomalous if `score > threshold`.
  - Trigger B (spatial heatmap): a frame is anomalous if the largest connected high-error blob in the pixel-wise error map exceeds the calibrated blob-area threshold.
  - Alerts trigger only after **N consecutive** anomalous frames (debounce).
  - Alerts are rate-limited by a **cooldown** period to prevent spam (configurable).
  - When an alert triggers, the system saves:
    - the full-size frame (`anomaly_*.jpg`)
    - optional diagnostics (reconstruction, error heatmap, side-by-side panel)
  - Events are logged to `logs/events.jsonl`.
- You can crop the camera feed via `roi` in `config/config.yaml`, and switch to `scoring.method: "topk"` to focus on localized anomalies.

---

## Project layout

```
redox_leak_detection/
  config/
    config.yaml
  data/
    normal/            # collected training images
    anomalies/         # saved anomaly evidence images
  logs/
    events.jsonl       # JSON Lines structured event log
  model/
    autoencoder.keras  # trained model
    threshold.json     # threshold + percentile
    stats.json         # training/validation stats
  reports/
    reconstruction_viz/
    threshold_calibration/
    health_check/
    daily_summary/
  scripts/
    capture_images.py
    train_autoencoder.py
    monitor.py
    visualize_reconstruction.py
    threshold_calibration.py
    health_check.py
    daily_summary.py
    set_mode.py
  src/
    ...implementation modules...
  state/
    mode.txt           # CAPTURE | MONITOR | OFF
```

---

## Quickstart (recommended lab workflow)

### 0) One-time setup (install Python + dependencies)
You only do this once per computer.

1. Install **Python 3.10 or 3.11** (if TensorFlow installation fails on 3.11, try 3.10).
2. Open **PowerShell** in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The `.bat` launchers will automatically use `.venv\Scripts\python.exe` if it exists.

Alternatively, if your system does not permit the use of scripts such as `Activate.ps1`;

```cmd
cd /d "C:\Users\labo\AppData\Local\Programs\VFB Leak Detector"
.\.venv\Scripts\activate.bat
```

This will also activate the virtual environment.

### 1) Configure notifications (Slack recommended)

This project expects secrets via environment variables:

- `SLACK_WEBHOOK_URL` (recommended)
- `SMTP_PASSWORD` (only if email alerts are enabled)

Option A (temporary for this window)
**PowerShell (temporary for this window):**
```powershell
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/XXX/YYY/ZZZ"
```

**CMD (temporary for this window):**
```bat
set SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
```

Option B (permanent, recommended):
- Windows search: “Environment Variables”
- Add a **User** environment variable:
  - Name: `SLACK_WEBHOOK_URL`
  - Value: the webhook URL
Then restart any terminal/monitor windows so they can see the new env var.

> Do **not** put webhook URLs or passwords into `config/config.yaml`.

### 2) Run a health check (recommended before first use)

Double-click:
- `run_health_check.bat`

Optional (CLI):
```powershell
python .\scripts\health_check.py --save_snapshot
python .\scripts\health_check.py --notify
```

### 3) Collect normal images

1. Put the setup into stable, normal operation (no people/tools in frame, lighting like overnight).
2. Set capture mode:
   - Double-click `set_mode_CAPTURE.bat`
3. Start capturing:
   - Double-click `run_capture.bat`

Images will be saved to:
- `data\normal\`

Stop capturing by closing the window (or press `Ctrl+C`). You can also switch the mode to `OFF`:
- Double-click `set_mode_OFF.bat`

### 4) Train the model

Double-click:
- `run_train.bat`

This produces:
- `model\autoencoder.keras`
- `model\threshold.json`
- `model\stats.json`

### 5) Sanity-check reconstructions (recommended)

Double-click:
- `run_visualize_reconstruction.bat`

Outputs:
- `reports\reconstruction_viz\`

### 6) (Optional) Threshold calibration histogram

Double-click:
- `run_threshold_calibration.bat`

Outputs:
- `reports\threshold_calibration\`

Optional (CLI), including overwriting the active threshold (backs up the old one first):
```powershell
python .\scripts\threshold_calibration.py --input_dir data\normal --write_threshold
```

### 7) Monitor unattended

1. Arm monitoring:
   - Double-click `set_mode_MONITOR.bat`
2. Start monitor:
   - Double-click `run_monitor.bat`

When anomalies are detected, evidence frames are saved under:
- `data\anomalies\`

Logs are appended to:
- `logs\events.jsonl`

### 8) Morning review (daily summary)

Double-click:
- `run_daily_summary.bat`

Reports are written to:
- `reports\daily_summary\`

---

## Configuration

Edit `config/config.yaml` to adjust:

- `camera`: index, backend (`DSHOW` recommended on Windows), resolution, FPS
- `capture.interval_seconds`: how often to save images in CAPTURE mode
- `training`: epochs, batch size, augmentation
- `threshold.percentile`: default is **99.5**
- `monitor`: frame interval, debounce (`consecutive_anomalies`), cooldown, diagnostic image saving
- `notify`: enable/disable Slack and email

> Advanced: the code also supports a `paths:` section in config (see `src/config.py` defaults) if you need to move data/model/log directories.

---

## CLI reference (all scripts accept `--config`)

```powershell
# Capture
python .\scripts\capture_images.py               # Starts the image-capture loop for building your NORMAL dataset.
                                                 # Only saves frames when state/mode.txt is set to CAPTURE.
                                                 # Output: data/normal/normal_<timestamp>.jpg (ROI-cropped if ROI is enabled).

# Train
python .\scripts\train_autoencoder.py            # Trains the convolutional autoencoder on images in data/normal/.
                                                 # Splits into train/validation, trains the model, then computes the anomaly
                                                 # threshold from validation NORMAL scores (default p99.5).
                                                 # Output: model/autoencoder.keras, model/threshold.json, model/stats.json.

# Monitor
python .\scripts\monitor.py                      # Starts unattended monitoring (live USB camera -> model -> anomaly score).
                                                 # Only runs detection when state/mode.txt is set to MONITOR.
                                                 # If score exceeds threshold for N consecutive frames, it:
                                                 #   - saves evidence images to data/anomalies/
                                                 #   - writes a structured JSONL event to logs/events.jsonl
                                                 #   - sends a Slack (preferred) or email alert.

# Set mode (writes state/mode.txt)
python .\scripts\set_mode.py CAPTURE             # Arms CAPTURE mode so capture_images.py actually saves frames.
python .\scripts\set_mode.py MONITOR             # Arms MONITOR mode so monitor.py actually runs detection + alerts.
python .\scripts\set_mode.py OFF                 # Disarms the system; capture/monitor scripts will idle (no saving/alerts).

# Reconstruction visualization
python .\scripts\visualize_reconstruction.py --num 16              # Loads sample images from disk (typically data/normal/)
                                                                   # and creates visual panels (input | reconstruction | error/heatmap).
                                                                   # Useful for sanity-checking what the model “learned” as normal.
python .\scripts\visualize_reconstruction.py --use_camera --num 8  # Same visualization, but grabs frames live from the USB camera.
                                                                   # Useful for checking ROI, lighting changes, and reconstruction quality in situ.

# Threshold histogram + stats
python .\scripts\threshold_calibration.py        # Computes reconstruction scores over a folder of NORMAL images (default: data/normal/) and plots a histogram.
                                                 # Shows summary stats and the selected threshold percentile (default 99.5th).
                                                 # (If you add the write flag, it can overwrite model/threshold.json after changing scoring settings like mean vs topk.)
                                                 # Does NOT modify model/threshold.json unless --write_threshold is used.

# Recompute threshold and overwrite model/threshold.json (creates a backup first)
python .\scripts\threshold_calibration.py --write_threshold
                                                 # Overwrites model/threshold.json using the current scoring settings in config.yaml.
                                                 # If model/threshold.json already exists, it creates:
                                                 #   model/threshold_backup_<timestamp>.json

# Health check
python .\scripts\health_check.py --save_snapshot # Runs quick diagnostics before leaving it unattended:
                                                 # checks camera read, disk space, model + threshold files, and notification config.
                                                 # Optionally saves a current camera snapshot for debugging.

# Daily summary
python .\scripts\daily_summary.py                # Summarizes recent events from logs/events.jsonl:
                                                 # counts anomalies, camera errors, and basic run stats (optionally plots).
python .\scripts\daily_summary.py --date 2026-01-27 # Same summary, but for a specific date (YYYY-MM-DD).
```

---

## Troubleshooting (common lab issues)

### Camera won’t open / black frames
- Close other apps that might be using the webcam.
- In `config/config.yaml`, try changing:
  - `camera.backend`: `DSHOW` ↔ `MSMF` ↔ `DEFAULT`
  - `camera.index`: `0` ↔ `1`
- Double-click `run_health_check.bat` (it saves a camera snapshot by default).
- Or run: `python .\scripts\health_check.py --save_snapshot`.

### Too many alerts (false positives)
Common causes: lighting changes, reflections, someone left an object in frame, camera moved.

Things to try:
- Re-capture **more normal images** covering typical overnight lighting conditions.
- Increase debounce: `monitor.consecutive_anomalies`.
- Increase cooldown: `monitor.alert_cooldown_seconds`.
- Increase threshold percentile slightly (e.g., 99.7) **only if** you accept reduced sensitivity.
- Use `run_visualize_reconstruction.bat` and `run_threshold_calibration.bat` to understand what’s happening.
- If you switch between MSE and Top-K% mode, you will need to overwrite the active threshold (Top-K% typically has much higher error).

**NOTE:** Run `threshold_calibration.py --write_threshold` anytime you change config.yaml scoring settings (e.g., scoring.method or scoring.topk_percent), otherwise you’ll be comparing live scores against a threshold computed in a different scoring “space.”

### No Slack alerts
- Ensure `SLACK_WEBHOOK_URL` is set in the environment for the session running the monitor.
- Run: `python .\scripts\health_check.py --notify` to send a test message.

### TensorFlow install fails
- Try Python 3.10.
- Make sure your pip is up to date: `python -m pip install --upgrade pip`.

---

## Safety note

This software is intended as an **assistance tool** for unattended monitoring. It does not replace lab safety practices, physical containment, or required SOPs.
