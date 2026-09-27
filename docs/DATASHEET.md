# Datasheet: Dihing Hostel Mess food waste

Generated 2026-09-26 21:12 IST.

## Collection window and cadence

- Meals observed: 5 (DIHING_20260926_B, DIHING_20260926_L, DIHING_20260926_D, DIHING_20260927_B, DIHING_20260927_L)
- Window: 2026-09-26 07:30 to 2026-09-27 12:15 (local time)
- Footfall: tallied at the mess entrance in 15-minute intervals for the whole service.
- Production and vessel leftovers: once per dish per meal (weighed where possible; method column says how).
- Plate-scraping bin: weighed once after each meal (gross minus tare).
- Return-counter camera: fixed phone, one frame every few seconds during service.
- Subscriber counts: aggregate numbers from the mess manager, per meal. No names or roll numbers.

## Row counts by table and source

| table | rows | observed | reported | historical | inferred | synthetic | demo |
|---|---|---|---|---|---|---|---|
| meals | 5 | 0 | 0 | 0 | 0 | 5 | 0 |
| footfall | 39 | 0 | 0 | 0 | 0 | 39 | 0 |
| subscriptions | 4 | 0 | 0 | 0 | 0 | 4 | 0 |
| production | 32 | 0 | 0 | 0 | 0 | 32 | 0 |
| leftovers | 30 | 0 | 0 | 0 | 0 | 30 | 0 |
| bin_weights | 4 | 0 | 0 | 0 | 0 | 4 | 0 |
| dish_costs | 15 | 0 | 0 | 0 | 0 | 15 | 0 |
| attendance_history | 59 | 0 | 0 | 0 | 0 | 59 | 0 |
| human_labels | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| camera_frames | 64 | 64 | 0 | 0 | 0 | 0 | 0 |
| plate_labels_model | 381 | 0 | 0 | 0 | 381 | 0 | 0 |
| meal_metrics | 5 | 0 | 0 | 0 | 0 | 5 | 0 |
| dish_metrics | 32 | 0 | 0 | 0 | 0 | 32 | 0 |
| forecasts | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| forecast_eval | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| recommendation_eval | 13 | 0 | 0 | 0 | 0 | 13 | 0 |

## Observed vs inferred vs synthetic

- **observed**: Measured by the team during the collection window (counts, weights, photos).
- **reported**: Supplied by mess staff for the collection window (e.g. subscriber counts, cook batch sizes).
- **historical**: Mess records from before the collection window, supplied by the manager.
- **inferred**: Computed by this pipeline or a model from observed/reported data.
- **synthetic**: Generated to extend real observations. Never used for headline numbers.
- **demo**: Fake data from `messwaste demo`, for testing only. Must never appear in a submission.

Headline numbers use observed and reported data only. Per-dish plate waste is inferred (bin weight split by plate-photo shares). Ghost overproduction is inferred and assumes the kitchen cooks for subscribers; check implied_headcount against subscribers before relying on it.

## How AI was used

- A vision-language model (claude-opus-4 (chat visual pass)) estimated the fraction of each dish left on 58 plate photos, on a 5-step scale (0, 1/4, 1/2, 3/4, all).
- A simple forecasting model (recency-weighted show-up rate by meal slot, with shrunk weekday and special-meal factors) predicted footfall. Forecasts were sealed with SHA-256 before each meal.
- Privacy: OpenCV face detection (and optionally a person detector) removed frames showing people before storage; kept frames are resized and stripped of EXIF/GPS.

Camera frames: 64 captured, 58 kept, dropped: {'face': 4, 'not a tray': 2}.

## Known gaps and biases (edit before submission)

- Two days cannot capture weekly menu rotation or month-to-month subscription changes.
- One hostel mess; results do not generalise to other messes.
- Observer effect: students may waste less when they notice the camera or the counters.
- The camera samples plates; during rush minutes some plates pass unphotographed.
- The bin may contain non-food items (napkins); see bin_weights.contents_note.
- Litre and piece quantities were converted to kg with the densities and piece weights in config.yaml.

Meals with missing inputs:

- DIHING_20260926_D: bin_weight
- DIHING_20260927_B: subscribers
