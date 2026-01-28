# User Guide — Redox Flow Battery Leak/Spill Monitor

This guide is written for chemists and lab staff who **do not write code**.

The system uses a fixed USB camera and an AI model (an autoencoder) trained on **normal operation** images. During unattended periods, it flags frames that look “unusual” (high reconstruction error) as potential leaks/spills and sends alerts.

> **Safety note**
> - This system is an *aid*, not a replacement for lab safety procedures.
> - Treat alerts as “needs human inspection”.
> - Always follow your lab’s SOPs for leaks/spills and chemical exposure.

---

## What you need (one-time setup)

### A) Computer + camera
- Windows desktop with a USB camera mounted **so it cannot move**.
- The camera view should include the battery/tubing area where leaks/spills would appear.

### B) Software
- Python **3.11**
- This project folder (provided by your team)
- IT/admin rights may be needed for installation.

### C) Notifications (Slack recommended)
- Slack incoming webhook URL (provided by your lab/IT)
- Optionally SMTP email settings (less recommended)

---

## Folder map (where files go)

Inside the project folder:

- `data/normal/`  
  Normal images captured for training.

- `model/`  
  Trained model artifacts:
  - `autoencoder.keras`
  - `threshold.json`
  - `stats.json`

- `data/anomalies/`  
  Saved frames when an anomaly alert triggers (including recon/heatmap images).

- `logs/events.jsonl`  
  Structured log of everything (startup, mode changes, alerts, errors).

- `reports/`  
  Human-friendly outputs (reconstruction visualizations, threshold histogram, daily summaries).

- `state/mode.txt`  
  A simple “mode switch” used to prevent capturing/monitoring while people are working:
  - `CAPTURE` = capture normal images
  - `MONITOR` = run unattended monitoring
  - `OFF` = do nothing

---

## Quick daily routine (typical workflow)

### End of day (before leaving the lab)
1) **Ensure the setup is in a stable “normal operation” configuration.**
   - No people in frame
   - No active changes to tubing, containers, etc.
   - Lighting is consistent with overnight lighting

2) **(Optional) Capture more normal images**
   - Run: `set_mode_CAPTURE.bat`
   - Run: `run_capture.bat`
   - Let it run for **10–30 minutes**
   - Run: `set_mode_OFF.bat`

3) **(If the setup changed significantly) retrain the model**
   - Run: `run_train.bat`
   - Wait for it to finish (it prints the saved model + threshold)

4) **Arm monitoring**
   - Run: `set_mode_MONITOR.bat`
   - Run: `run_monitor.bat`
   - Leave the monitor window open overnight

### Start of day (when returning)
1) **Disarm monitoring**
   - Run: `set_mode_OFF.bat`
   - Close the monitoring window (or press `Ctrl + C` inside it)

2) **Check if any anomalies were detected**
   - Look in: `data/anomalies/`
   - Run: `run_daily_summary.bat` to get a report in `reports/daily_summary/`

---

## Step-by-step instructions

## 1) One-time installation (IT or power user)

1) Open PowerShell in the project folder  
2) Create a virtual environment and install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell execution policy blocks activation, ask IT for help or run scripts using `.venv\Scripts\python.exe` directly.

---

## 2) Configure Slack notifications (recommended)

The Slack webhook **must be stored as an environment variable** (not in the YAML config).

Option A (temporary, for current PowerShell window only):
```powershell
$env:SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"
```

Option B (permanent, recommended):
- Windows search: “Environment Variables”
- Add a **User** environment variable:
  - Name: `SLACK_WEBHOOK_URL`
  - Value: the webhook URL

Then restart any terminal/monitor windows so they can see the new env var.

---

## 3) Run a health check (recommended before first use)

Double-click:
- `run_health_check.bat`

What it checks:
- Disk space
- Ability to write logs
- Camera produces a frame
- Model artifacts (if already trained)
- Slack/email configuration (optional)

A snapshot may be saved to:
- `reports/health_check/`

---

## 4) Capture normal images

Normal images are the “training examples” of how the setup looks without leaks/spills.

1) Make sure the setup is stable and no one is in the frame  
2) Double-click:
   - `set_mode_CAPTURE.bat`
   - `run_capture.bat`

Let it run for 10–30 minutes (or longer for better robustness).

3) Double-click:
   - `set_mode_OFF.bat`

Normal images will be stored in:
- `data/normal/`

---

## 4b) (Optional) Crop the camera feed (ROI) + Top-K scoring

If the camera sees a lot of irrelevant background, you can crop the feed to focus on the tubing/catch area
and make anomalies stand out more. These settings live in `config/config.yaml`.

**ROI crop (recommended when the camera sees unused space):**

```yaml
roi:
  enabled: true
  units: "relative"   # fractions of width/height
  x: 0.10             # left offset (10% of width)
  y: 0.20             # top offset (20% of height)
  width: 0.60         # 60% of width
  height: 0.50        # 50% of height
```

**Top-K scoring (more sensitive to localized anomalies):**

```yaml
scoring:
  method: "topk"
  topk_percent: 1.0       # average top 1% highest-error pixels
  topk_min_pixels: 100    # never average fewer than this
```

When you change **ROI** or **scoring**, you should:
1) Re-capture normal images (if ROI changed)
2) Retrain the model
3) Re-run threshold calibration

---

## 5) Train (or retrain) the model

Retrain whenever:
- Camera moved (even slightly)
- Lighting conditions changed drastically
- Tubing layout / background changed significantly

Double-click:
- `run_train.bat`

Artifacts will be saved to:
- `model/`

If training fails with “not enough images”, capture more normal images first.

---

## 6) Monitor unattended

1) Double-click:
   - `set_mode_MONITOR.bat`
   - `run_monitor.bat`

Keep the monitor window open (minimized is fine).

### What happens on an alert
- A Slack (or email) alert is sent (best effort)
- Files are saved to `data/anomalies/`, typically including:
  - `anomaly_*.jpg` (camera snapshot, cropped if ROI is enabled)
  - `input_resized_*.jpg` (model input resolution)
  - `reconstruction_*.jpg`
  - `error_heatmap_*.jpg`
  - `compare_*.jpg` (side-by-side)

### How to stop monitoring
- Double-click `set_mode_OFF.bat`
- Close the monitor window (or press `Ctrl + C`)

---

## 7) Reconstruction visualization (debugging / trust-building)

If you want to understand what the model is doing, generate reconstruction panels from normal images:

Double-click:
- `run_visualize_reconstruction.bat`

Outputs:
- `reports/reconstruction_viz/`

Each panel is:
- input (resized)
- reconstruction
- error heatmap

**If normal images already show large heatmaps**, the model may be undertrained or the camera/lighting changed.

---

## 8) Threshold calibration plot (error histogram)

This plot helps you see whether the threshold looks reasonable and whether “normal” scores have drifted.

Double-click:
- `run_threshold_calibration.bat`

Outputs:
- `reports/threshold_calibration/`
  - histogram plot `.png`
  - stats `.json`

If the histogram is very wide or shifted higher than past days:
- lighting changed overnight
- camera focus/exposure changed
- setup background changed
- consider retraining

> Advanced: overwriting the threshold  
> The calibration script supports `--write_threshold` to overwrite `model/threshold.json`.  
> Only do this if you understand what you are changing. (A backup is created automatically.)

---

## 9) Daily summary report

Double-click:
- `run_daily_summary.bat`

Outputs:
- `reports/daily_summary/`
  - `summary_YYYY-MM-DD.md`
  - `summary_YYYY-MM-DD.json`
  - optional score plot

---

## Troubleshooting

### “Camera check failed” / monitor says camera feed down
- Unplug/replug USB camera
- Try a different USB port
- Check Windows camera privacy settings (camera access must be enabled)
- Try switching `camera.backend` in `config/config.yaml`:
  - `DSHOW` (recommended), or `MSMF`

### No Slack alerts
- Confirm `SLACK_WEBHOOK_URL` is set in Windows Environment Variables
- Re-open the monitor window after setting the env var
- Run `scripts/health_check.py --notify` (advanced) to send a test alert

### Too many false alerts
Common causes:
- lighting changes (sunrise, lab lights switching)
- reflections/glare
- something left in view (paper towel, glove)
- camera moved slightly

What to do:
1) Check `compare_*.jpg` files in `data/anomalies/`
2) Run `run_threshold_calibration.bat` and compare to older plots
3) Consider capturing more normal images under the same lighting
4) Retrain the model if the setup changed significantly
5) Increase `monitor.consecutive_anomalies` (debounce) or `alert_cooldown_seconds`

---

## Who to contact
If you encounter repeated issues, contact the person/team responsible for maintaining this system and provide:
- `logs/events.jsonl`
- a few example files from `data/anomalies/`
- your `config/config.yaml` (do **not** include Slack webhook; it is stored as an env var)
