# OH HAIL NAH!! Astrophotography — processing workspace

Working repo for Zach's DWARF 3 astrophotography (Sugar Land / Houston, Bortle 8): processing scripts,
unsigned masters, finished signed images, branded vertical shorts, and the in-progress Alhena / Orionid work.

Raw DWARF folders are **not** in this repo (several GB). They live on the external drive
`/Volumes/Untitled/Astronomy/` and in iCloud `Dwarf/`.

## Setup in a new session

```bash
pip install --break-system-packages numpy opencv-python-headless scipy pillow pywavelets
export OHN_WM_KEY=...   # invisible-watermark key; set it as a secret env var, never commit it
```
`ffmpeg` is needed for the shorts. Scripts were written to run from one flat folder
(fonts, `badge.png` and the scripts side by side), so copy `scripts/*.py`, `assets/fonts/*`
and `assets/badge.png` into a working folder before running. Hard-coded paths point at the old
session (`/mnt/user-data/uploads/...`, the scratchpad); repoint them to `work/` and `masters/`.

## Layout

| Path | What it is |
|---|---|
| `scripts/proc.py`, `run_target.py`, `nodebg.py` | Deep-sky pipeline: coverage crop, grid/node background extraction, star colour calibration, NR, arcsinh stretch |
| `scripts/sign.py`, `wm.py`, `embed_var.py`, `wm2.py` | Handwritten signature + target + date stamp, keyed DWT watermark, crop-tolerant detector |
| `scripts/saturn*.py`, `sun*.py`, `jup.py`, `moon2.py` | Planetary / solar lucky imaging |
| `scripts/double.py`, `dc_crop.py` | Two-session Double Cluster registration and combine |
| `scripts/alhena.py`, `alh_fin2.py`, `solve.py`, `annotate.py` | Alhena wide field: crop, background, colour, finish, plate solve, labels |
| `scripts/short*.py`, `audio*.py` | 1080x1920 shorts with theme song (short4 = Orionid short, not rendered yet) |
| `scripts/device/device_commands_log.sh` | Every command run on the Mac for the meteor scan and the raw-frame stack (meteor scan, registration, sigma-clip stack, time-lapse) |
| `assets/` | Fonts, OHN badge, theme song |
| `masters/` | Unsigned processed masters (sign these, don't re-process) |
| `outputs/` | Delivered signed JPGs, shorts, covers |
| `work/alhena/` | Alhena stack (389 clean frames), calibrated linear crop, coverage, plate solution, time-lapse parts, meteor-scan sheets |

See `NOTES.md` for status and next steps.
