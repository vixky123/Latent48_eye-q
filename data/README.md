# Data

Everything a stranger needs to load this dataset and rebuild every number in the
deck, the dashboard and the video — without us in the room.

## Provenance

The plate photographs in `frames/` are **real**, shot at Dihing Hostel's
plate-return counter. The counts, weights, subscriber figures and attendance
history are **modelled** by `messwaste/simulate.py` rather than measured, and
every one of those rows carries `source: synthetic`. See the root `README.md`
and `READ_ME_FIRST.txt`.

## Layout

| Path | What it is |
|---|---|
| `tables/` | The full dataset, 16 tables as CSV |
| `sample/` | First 30 rows of each table, for a quick look |
| `frames/` | 58 real plate photographs, after the privacy filter |
| `sealed_forecasts/` | The forecast JSON files with their SHA-256 hashes |
| `SIMULATION_TRUTH.csv` | Footfall values the simulation drew, for comparison |
| `../docs/SCHEMA.md` | Every column of every table, typed and described |
| `../docs/DATASHEET.md` | Collection window, cadence, row counts, how AI was used, gaps |

CSV is shipped here because it opens anywhere. The pipeline writes **Parquet**
when `pyarrow` is installed — run `python -m messwaste.cli ingest` and you get
`parquet/` instead.

## Row counts

| Table | Rows | |
|---|---|---|
| `meals` | 5 | one row per meal service |
| `footfall` | 39 | 15-minute diner tallies |
| `subscriptions` | 4 | aggregate subscriber count per meal |
| `production` | 32 | quantity cooked per dish |
| `leftovers` | 30 | vessel leftovers per dish |
| `bin_weights` | 4 | plate-scraping bin, gross and tare |
| `dish_costs` | 15 | cost per kg by dish |
| `attendance_history` | 59 | three weeks of prior attendance |
| `camera_frames` | 64 | every frame, kept or dropped, with the reason |
| `plate_labels_model` | 381 | per-plate, per-dish fraction left |
| `meal_metrics` | 5 | per-meal waste decomposition |
| `dish_metrics` | 32 | per-meal, per-dish figures |
| `forecasts` | 2 | sealed forecasts |
| `forecast_eval` | 2 | sealed forecasts vs what happened |
| `recommendation_eval` | 13 | what the recommended quantities would have done |

`meals` has 5 rows and `subscriptions` only 4 — day-2 breakfast has no
subscriber count, because the manager did not have one. That gap is real and is
reported rather than filled in.

## Loading it

```python
import pandas as pd
meals   = pd.read_csv("data/tables/meals.csv")
metrics = pd.read_csv("data/tables/meal_metrics.csv")
metrics[["meal_id", "ghost_overprod_kg", "other_overprod_kg", "plate_waste_kg"]]
```

Every table has a `source` column: `observed` (we measured it), `reported`
(mess staff gave it to us), `historical` (past records), `inferred` (the
pipeline computed it), `synthetic` (generated). Join on `meal_id` throughout;
`camera_frames.frame_id` links the photographs to `plate_labels_model`.

## Verifying a sealed forecast

```bash
python - <<'PY'
import json, hashlib
r = json.load(open("data/sealed_forecasts/DIHING_20260927_L__"
                   + "DIHING_20260927_L_20260926T223000_L.json"))
body = {k: v for k, v in r.items() if k != "sha256"}
calc = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, default=str).encode()).hexdigest()
print("hash matches:", calc == r["sha256"])
print("sealed at:", r["made_at"], "| meal starts:", r["target_start"])
PY
```

`made_at` precedes `target_start`, and the hash still matches, so the prediction
could not have been edited after the meal.
