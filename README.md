# Mess waste: where does the food go?

> **Provenance.** The plate photographs in the bundled dataset are real, shot at Dihing Hostel's
> plate-return counter. The counts, weights, subscriber figures and attendance history are modelled
> by `messwaste/simulate.py` and carry `source: synthetic` in every table — they are not measurements
> of a real kitchen.

**eye-q** — Vicky Raj · Aman Gupta · Prakhar Asthana

Pipeline built around the Granica × IIT Guwahati "Bring the Physical World to AI" brief. It measures how much food a hostel mess wastes, splits it into **food cooked for subscribers who didn't come**, **other overproduction** and **plate waste**, predicts the next meal's turnout, **seals** each prediction with a hash before the meal, and checks it afterwards.

```
CSV sheets ─► ingest ─► parquet ─┐
phone camera ─► privacy filter ─► de-dup ─► vision model ─┤
                                                         ├─► analyze ─► forecast (sealed) ─► evaluate ─► dashboard + datasheet ─► export
hand labels (labeller.html) ─► agreement ─────────────────┘
```

## What is in this repository

| Path | |
|---|---|
| `messwaste/` | the pipeline package |
| `data/` | the dataset, a 30-row sample of every table, the real photographs, sealed forecasts — see `data/README.md` |
| `docs/SCHEMA.md` | every column of every table, typed and described |
| `docs/DATASHEET.md` | collection window and cadence, row counts, observed vs inferred, how AI was used, known gaps |
| `docs/validation_report.md` | what the ingest validator flagged |
| `reports/` | the dashboard, the 3-minute walkthrough video, the deck |
| `notebooks/kaggle_pipeline.ipynb` | run the whole thing on Kaggle with a GPU |
| `templates/` | blank CSV collection sheets |
| `tests/smoke_test.py` | end-to-end run on fake data, no GPU or internet |
| `make_video.py` | rebuilds the walkthrough video from the dashboard |

## Quick start

```bash
pip install -r requirements.txt
python tests/smoke_test.py            # whole pipeline on fake data, no GPU; prints ALL SMOKE TESTS PASSED
```

On Kaggle, open `notebooks/kaggle_pipeline.ipynb` (GPU T4 x2, Internet on) and run top to bottom.

## Data layout

```
<raw_root>/
  csv/
    meals.csv  footfall.csv  subscriptions.csv  production.csv  leftovers.csv
    bin_weights.csv  dish_costs.csv  attendance_history.csv  human_labels_<initials>.csv
  frames/
    KAP_20261003_L/   IMG_20261003_121502.jpg ...    one folder per meal, named by meal_id
```

Fill in `presentation:` in `config.yaml` (team name, the named user, demo link) before building slides.

Blank sheets with the right headers are in `templates/`. Fill them in Google Sheets and download each tab as CSV. `reports/SCHEMA.md` (generated) explains every column.

**meal_id** format: `HOSTEL_YYYYMMDD_SLOT`, slot `B`/`L`/`D`. Example: `KAP_20261003_L`.

**source** column on every row:

| value | use for |
|---|---|
| `observed` | anything you counted, weighed or photographed |
| `reported` | numbers mess staff gave you for this window (subscriber counts, cook batch sizes) |
| `historical` | past attendance records from the manager |
| `synthetic` | only if you extend real data; never for headline numbers |

Example rows:

| sheet | example |
|---|---|
| meals | `KAP_20261003_L,KAP,2026-10-03,L,12:15,14:15,rice;dal;sabzi;roti;curd,0,rain,,observed` |
| footfall | `KAP_20261003_L,12:15,38,VR,observed` (one row per 15 minutes) |
| subscriptions | `KAP_20261003_L,512,manager register,,reported` |
| production | `KAP_20261003_L,dal,70,l,cook_reported,reported` (units: kg, g, l, ml, pieces) |
| leftovers | `KAP_20261003_L,dal,12.5,l,weighed,observed` |
| bin_weights | `KAP_20261003_L,31.2,2.4,some napkins,observed` |
| dish_costs | `dal,60,reported` |
| attendance_history | `2026-09-20,L,361,505,0,historical` |

Dish names are normalised using `config.yaml` (e.g. "Daal", "dal fry" → `dal`; "chapati" → `roti`). Add any new dish there with `serving_kg`, `cost_per_kg_inr`, and `piece_kg` if you count it in pieces.

## Runbook during the 48 hours

**Before kickoff**
1. Edit `config.yaml`: `hostel_code`, `hostel_name`, meal times, dishes and costs.
2. Get historical attendance from the manager into `attendance_history.csv`.
3. Test the camera: phone above the return counter, interval shooting every 2–3 s, **frame shows plates only**.

**Every meal**
1. Before service: add the meal to `meals.csv` and `subscriptions.csv`.
2. During service: 15-minute footfall tallies; camera running.
3. After service: vessel leftovers, bin gross weight (tare once), cooked quantities from the kitchen.
4. Copy the camera photos into `frames/<meal_id>/`, upload a new dataset version, run the notebook (ingest → frames → label → analyze).

**Every night (the predict-then-verify step)**
1. Add tomorrow's meals and subscriber counts.
2. `python -m messwaste.cli forecast`
3. **Immediately** post each printed `sha256` in the hackathon channel, and save the JSON files from `out/forecasts/sealed/` (commit them). A forecast only counts if its hash was public before the meal started.
4. Next day after each meal: `evaluate` and `report`.

**Once per day:** `human-sample --n 100`, then two teammates label `labeller.html` independently. Their CSVs go into `csv/`.

## Commands

```
python -m messwaste.cli <command> [--raw-root PATH] [--out-root PATH]

templates      blank CSV sheets
demo           fake dataset for testing (source=demo; never mixes with real data)
ingest         validate CSVs → parquet; errors listed in reports/validation_report.md
frames         privacy filter (face / person) + de-duplication; strips EXIF/GPS
label          vision model  [--backend hf|anthropic|mock] [--limit N] [--meal-ids A,B]
human-sample   offline labelling page for teammates  [--n 100]
agreement      model vs humans, human vs human (exact, within one step, weighted kappa)
analyze        waste decomposition per meal and dish
forecast       predict + seal upcoming meals  [--meal-ids ...]
evaluate       verify hashes, compare with actuals and baselines
report         reports/dashboard.html, DATASHEET.md, SCHEMA.md
slides         3-slide deck reports/deck.pptx from the same numbers (+ deck.pdf if LibreOffice is installed)
export         submission zip (data, sample, schema, datasheet, dashboard, sealed forecasts, sample frames)
all            ingest → frames → label → agreement → analyze → evaluate → report
refresh        same as all; add --skip-label for a fast refresh while presenting
```

## How the numbers are computed

Per meal: F = footfall, S = subscribers. Per dish: p = (cooked − leftover) / F.

- **Ghost overproduction** = min(leftover, p × max(S − F, 0)): portions that correspond to subscribers who didn't come.
- **Other overproduction** = leftover − ghost.
- **Plate waste** = bin net weight, split across dishes by (mean fraction left on photographed plates × p).
- **Implied headcount** = F × cooked / served: the number of people the kitchen effectively cooked for. Compare it with S and F on the dashboard before claiming the ghost-diner story. If it sits near F, the kitchen already guesses turnout well and the waste is about batch sizes. Say so; that's a finding too.

**Forecast** (`forecast.py`): recency-weighted show-up rate (half-life 7 days) for the meal slot from ordinary meals, × a shrunk weekday factor, or × a shrunk special-meal factor for special meals, × tomorrow's subscriber count. The range comes from rolling backtest errors on past meals. Only data from before the forecast time is used. Baselines: cook for all subscribers; same slot last week. Recommended cook quantity = upper bound × p, so the kitchen should not run short.

## Vision model on Kaggle

Default: `Qwen/Qwen2.5-VL-7B-Instruct` in float16, split across both T4s (`device_map="auto"`). Always run `labeling.preview(...)` on a few frames first.

- **Output is garbage, empty or repeated symbols:** fp16 overflow. Use `Qwen/Qwen2.5-VL-3B-Instruct` with `hf_dtype: float32`.
- **Out of memory:** lower `hf_batch_size` to 1–2 or `hf_max_side_px` to 640.
- **Session died mid-run:** re-run; finished frames are skipped. Across sessions, Quick Save the notebook and add its output as an input; old labels are picked up automatically.
- **Anthropic backend:** add `ANTHROPIC_API_KEY` in Add-ons → Secrets, `pip install anthropic`, use `--backend anthropic`.

The model's job is narrow: fraction of each dish left on a plate, on a 0/¼/½/¾/all scale. Its accuracy is measured against your hand labels in `agreement` and reported on the dashboard. Quote those numbers; don't claim more.

## Privacy

- Frames with a detected face are never copied to the output. The default detector is OpenCV's YuNet (downloaded once; needs internet), which falls back to Haar cascades offline. Haar mistakes round plates for faces fairly often, so check the "Dropped for privacy" count on the dashboard; if it's high, get YuNet working rather than lowering the threshold. Optional YOLO person detection adds a second check. Detectors miss things, so point the camera at the tray area only.
- Kept frames are resized and saved without EXIF (no GPS, no device info).
- Subscriber data is an aggregate count per meal. No names or roll numbers anywhere.
- Keep the Kaggle data dataset private. To avoid uploading raw frames at all, run `frames` on a laptop and upload the contents of `out/frames_kept/` as the `frames/` folder of your Kaggle dataset. Kaggle re-runs the (harmless) filter on them. Capture times are lost this way, so the dashboard's "last frame" column stays empty. Do all labelling in one environment so frame ids stay consistent.

## Project structure

```
config.yaml                 all settings
messwaste/
  config.py  io.py  schema.py  dishes.py  timeutil.py
  ingest.py                 CSV → validated parquet
  camera.py                 privacy filter, de-dup, EXIF strip
  labeling.py               vision backends (hf / anthropic / mock), resumable runner, preview
  validation.py             labelling page, agreement stats
  analysis.py               waste decomposition
  forecast.py               model, baselines, sealing, evaluation
  report.py                 dashboard, datasheet, schema, export
  demo.py                   fake data
  cli.py
notebooks/kaggle_pipeline.ipynb
templates/*.csv
tests/smoke_test.py
```
