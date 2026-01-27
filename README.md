# Redox Flow Battery Leak/Spill Monitor (Autoencoder)

Local Windows-friendly project:
- Capture "normal" images from a static USB camera
- Train a convolutional autoencoder (TensorFlow/Keras)
- Monitor unattended, flag anomalies, save frames, log JSONL, notify via Slack/email

**For chemists / non-coders:** see `USER_GUIDE.md` for a step-by-step walkthrough.

## 1) Setup (Windows PowerShell)

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Configure Slack (recommended)

Set env var (PowerShell):

```powershell
$env:SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"
```

(Do not put secrets in config/config.yaml.)

## 3) Capture normal images

Set the mode:

- Edit `state/mode.txt` and set it to `CAPTURE`

Then run:

```powershell
python .\scripts\capture_images.py
```

Images will be saved to `data/normal/`.

When done capturing, set `state/mode.txt` to `OFF`.

## 4) Train the model

```powershell
python .\scripts\train_autoencoder.py
```

Artifacts saved to `model/`:
- `autoencoder.keras`
- `threshold.json`
- `stats.json`

## 5) Monitor unattended

Set `state/mode.txt` to `MONITOR`, then run:

```powershell
python .\scripts\monitor.py
```

Anomalous frames saved to:
- `data/anomalies/`

Logs:
- `logs/events.jsonl`

## 6) Visualize reconstructions (debug / sanity-check)

```powershell
python .\scripts\visualize_reconstruction.py --num 16
```

Outputs are written to `reports/reconstruction_viz/`.

## 7) Threshold calibration plot (error histogram)

```powershell
python .\scripts\threshold_calibration.py
```

Outputs are written to `reports/threshold_calibration/`.

## 8) Health check

```powershell
python .\scripts\health_check.py --save_snapshot
```

## 9) Daily summary report

```powershell
python .\scripts\daily_summary.py
```

## 10) Set mode (optional helper)

```powershell
python .\scripts\set_mode.py CAPTURE
python .\scripts\set_mode.py MONITOR
python .\scripts\set_mode.py OFF
```

## Notes
- Threshold is set to the 99.5th percentile of validation reconstruction errors (normal-only).
- Debounce: anomaly triggers only after N consecutive frames exceed threshold (configurable).
- Cooldown prevents alert spam.
