# Schema

Every table is stored as Parquet under `parquet/`. Every row has a `source` column.

## Sources

- **observed**: Measured by the team during the collection window (counts, weights, photos).
- **reported**: Supplied by mess staff for the collection window (e.g. subscriber counts, cook batch sizes).
- **historical**: Mess records from before the collection window, supplied by the manager.
- **inferred**: Computed by this pipeline or a model from observed/reported data.
- **synthetic**: Generated to extend real observations. Never used for headline numbers.
- **demo**: Fake data from `messwaste demo`, for testing only. Must never appear in a submission.

## `meals`

One row per meal service observed (or planned, for forecasting). *(raw input (CSV filled by the team); key: meal_id)*

| column | type | nullable | meaning |
|---|---|---|---|
| `meal_id` | str | no | Unique meal id, recommended HOSTEL_YYYYMMDD_SLOT, e.g. KAP_20261003_L. |
| `hostel` | str | no | Hostel code. |
| `date` | date | no | Service date (YYYY-MM-DD, local time). |
| `slot` | str | no | B = breakfast, L = lunch, D = dinner. Allowed: B, L, D. |
| `start_time` | time | yes | Service start HH:MM. Empty = default from config. |
| `end_time` | time | yes | Service end HH:MM. Empty = default from config. |
| `menu_items` | str | no | Dishes served, separated by ';' (canonical names from config). |
| `special_flag` | int | no | 1 = special meal (special dinner, paneer day, fest), else 0. Allowed: 0, 1. |
| `weather` | str | yes | Free text, e.g. 'rain', 'clear'. |
| `notes` | str | yes | Anything unusual (exam day, event, power cut). |
| `source` | str | no | Provenance label; see the Sources section. |

## `footfall`

Diners entering, tallied per 15-minute interval. *(raw input (CSV filled by the team); key: meal_id, interval_start)*

| column | type | nullable | meaning |
|---|---|---|---|
| `meal_id` | str | no | Meal id. |
| `interval_start` | time | no | Start of the 15-minute interval, HH:MM. |
| `count` | int | no | Diners who entered during the interval. |
| `counter_person` | str | yes | Initials of whoever counted. |
| `source` | str | no | Provenance label; see the Sources section. |

## `subscriptions`

Diners entitled to eat this meal (aggregate count from the manager). *(raw input (CSV filled by the team); key: meal_id)*

| column | type | nullable | meaning |
|---|---|---|---|
| `meal_id` | str | no | Meal id. |
| `subscriber_count` | int | no | Aggregate subscriber count. Never names or roll numbers. |
| `notes` | str | yes | How the manager arrived at the number. |
| `source` | str | no | Provenance label; see the Sources section. |

## `production`

Quantity cooked per dish. *(raw input (CSV filled by the team); key: meal_id, dish)*

| column | type | nullable | meaning |
|---|---|---|---|
| `meal_id` | str | no | Meal id. |
| `dish` | str | no | Canonical dish name. |
| `qty` | float | no | Quantity cooked. |
| `unit` | str | no | Unit of qty. Allowed: kg, g, l, ml, pieces. |
| `method` | str | no | How the number was obtained. Allowed: weighed, cook_reported, estimated, counted, register, card_swipes. |
| `source` | str | no | Provenance label; see the Sources section. |
| `qty_kg` | float | yes | qty converted to kg. |
| `kg_note` | str | yes | Assumption used for the kg conversion. |

## `leftovers`

Quantity left in serving vessels at the end of service (never served). *(raw input (CSV filled by the team); key: meal_id, dish)*

| column | type | nullable | meaning |
|---|---|---|---|
| `meal_id` | str | no | Meal id. |
| `dish` | str | no | Canonical dish name. |
| `qty` | float | no | Quantity left. |
| `unit` | str | no | Unit of qty. Allowed: kg, g, l, ml, pieces. |
| `method` | str | no | How the number was obtained. Allowed: weighed, cook_reported, estimated, counted, register, card_swipes. |
| `source` | str | no | Provenance label; see the Sources section. |
| `qty_kg` | float | yes | qty converted to kg. |
| `kg_note` | str | yes | Assumption used for the kg conversion. |

## `bin_weights`

Plate-scraping bin weighed after service. *(raw input (CSV filled by the team); key: meal_id)*

| column | type | nullable | meaning |
|---|---|---|---|
| `meal_id` | str | no | Meal id. |
| `gross_kg` | float | no | Bin + contents, kg. |
| `tare_kg` | float | no | Empty bin, kg. |
| `contents_note` | str | yes | Non-food items seen (napkins, bones...). |
| `source` | str | no | Provenance label; see the Sources section. |
| `net_kg` | float | yes | gross_kg - tare_kg. |

## `dish_costs`

Cost per kg of each dish, from the manager. *(raw input (CSV filled by the team); key: dish)*

| column | type | nullable | meaning |
|---|---|---|---|
| `dish` | str | no | Canonical dish name. |
| `cost_per_kg_inr` | float | no | Ingredient + fuel cost per kg cooked, INR. |
| `source` | str | no | Provenance label; see the Sources section. |

## `attendance_history`

Past attendance records from the mess (before the collection window). *(raw input (CSV filled by the team); key: date, slot)*

| column | type | nullable | meaning |
|---|---|---|---|
| `date` | date | no | Date. |
| `slot` | str | no | B/L/D. Allowed: B, L, D. |
| `count` | int | no | Diners who ate. |
| `subscriber_count` | int | yes | Subscribers that day, if known. |
| `special_flag` | int | no | 1 = special meal. Allowed: 0, 1. |
| `source` | str | no | Provenance label; see the Sources section. |

## `human_labels`

Hand labels of plate photos (validation set for the vision model). *(raw input (CSV filled by the team); key: frame_id, labeller, dish)*

| column | type | nullable | meaning |
|---|---|---|---|
| `frame_id` | str | no | Frame id from camera_frames. |
| `labeller` | str | no | Labeller initials. |
| `dish` | str | no | Canonical dish name. |
| `fraction_left` | float | yes | Fraction of this dish left on the plate (0, .25, .5, .75, 1). Empty = can't tell. |
| `plate_present` | int | no | 1 = a single distinct plate is visible. Allowed: 0, 1. |
| `labelled_at` | str | yes | When the label was made (ISO time). |
| `source` | str | yes | Filled automatically: observed (or demo). |

## `camera_frames`

Every frame from the return-counter camera and whether it was kept. *(derived by the pipeline; key: frame_id)*

| column | type | nullable | meaning |
|---|---|---|---|
| `frame_id` | str | no | meal_id + short content hash. |
| `meal_id` | str | no | Meal id (from the folder the frame was in). |
| `taken_at` | datetime | yes | Capture time (EXIF or filename), local time. |
| `file_name` | str | no | Original file name. |
| `kept` | bool | no | True if the frame passed privacy and de-duplication filters. |
| `drop_reason` | str | yes | face | person | duplicate | unreadable | empty string if kept. |
| `n_faces` | int | no | Faces detected. |
| `n_persons` | int | no | People detected (0 if detector off). |
| `dhash` | str | no | 64-bit difference hash (hex). |
| `in_meal_window` | bool | yes | Capture time falls within the meal window. |
| `source` | str | no | Provenance label; see the Sources section. |

## `plate_labels_model`

Vision-model estimate of each dish left on each kept plate. *(derived by the pipeline; key: frame_id, dish, model_id)*

| column | type | nullable | meaning |
|---|---|---|---|
| `frame_id` | str | no | Frame id. |
| `meal_id` | str | no | Meal id. |
| `dish` | str | no | Canonical dish name. |
| `fraction_left` | float | yes | 0, .25, .5, .75, 1; empty = model could not tell. |
| `plate_present` | int | no | 1 if the model saw one distinct plate. |
| `backend` | str | no | hf | anthropic | mock. |
| `model_id` | str | no | Exact model name. |
| `parse_ok` | bool | no | Model output parsed as valid JSON. |
| `labelled_at` | str | no | ISO time of labelling. |
| `source` | str | no | Provenance label; see the Sources section. |

## `meal_metrics`

Per-meal waste decomposition. *(derived by the pipeline; key: meal_id)*

| column | type | nullable | meaning |
|---|---|---|---|
| `meal_id` | str | no | Meal id. |
| `date` | date | no | Date. |
| `slot` | str | no | Slot. |
| `start_dt` | datetime | no | Service start, local time. |
| `footfall` | float | yes | F: diners counted. |
| `subscribers` | float | yes | S: subscribers entitled. |
| `show_up_rate` | float | yes | F / S. |
| `ghost_diners` | float | yes | S - F (never negative). |
| `cooked_kg` | float | yes | Total cooked. |
| `leftover_kg` | float | yes | Total left in vessels. |
| `served_kg` | float | yes | cooked - leftover. |
| `implied_headcount` | float | yes | Headcount the kitchen effectively cooked for = F x cooked/served. |
| `ghost_overprod_kg` | float | yes | Overproduction explained by cooking for absent subscribers. |
| `other_overprod_kg` | float | yes | Remaining vessel leftovers. |
| `plate_waste_kg` | float | yes | Bin net weight. |
| `total_waste_kg` | float | yes | Vessel leftovers + plate waste. |
| `waste_g_per_diner` | float | yes | total_waste_kg / F in grams. |
| `ghost_inr` | float | yes | Cost of ghost overproduction. |
| `other_overprod_inr` | float | yes | Cost of other overproduction. |
| `plate_inr` | float | yes | Cost of plate waste. |
| `total_inr` | float | yes | Total cost of waste. |
| `camera_mean_fraction_left` | float | yes | Mean fraction left across labelled plates (camera index). |
| `camera_plates` | int | no | Plates labelled for this meal. |
| `gaps` | str | yes | Missing inputs for this meal. |
| `source` | str | no | Provenance label; see the Sources section. |

## `dish_metrics`

Per-meal, per-dish waste figures. *(derived by the pipeline; key: meal_id, dish)*

| column | type | nullable | meaning |
|---|---|---|---|
| `meal_id` | str | no | Meal id. |
| `dish` | str | no | Dish. |
| `cooked_kg` | float | yes | Cooked. |
| `leftover_kg` | float | yes | Left in vessels. |
| `served_kg` | float | yes | Served. |
| `p_kg_per_diner` | float | yes | served_kg / footfall. |
| `overprod_rate` | float | yes | leftover / cooked. |
| `implied_headcount` | float | yes | F x cooked/served. |
| `ghost_overprod_kg` | float | yes | min(leftover, p x (S-F)). |
| `other_overprod_kg` | float | yes | leftover - ghost. |
| `plate_kg` | float | yes | Bin weight attributed to this dish via plate photos. |
| `plate_attr_method` | str | yes | How plate_kg was attributed. |
| `mean_fraction_left` | float | yes | Mean fraction left on labelled plates. |
| `cost_per_kg_inr` | float | yes | Cost used. |
| `waste_inr` | float | yes | (leftover + plate_kg) x cost. |
| `source` | str | no | Provenance label; see the Sources section. |

## `forecasts`

Sealed footfall forecasts (one row per sealed file). *(derived by the pipeline; key: forecast_id)*

| column | type | nullable | meaning |
|---|---|---|---|
| `forecast_id` | str | no | Unique id. |
| `meal_id` | str | no | Target meal. |
| `made_at` | datetime | no | When the forecast was sealed (local time). |
| `sha256` | str | no | Hash of the sealed JSON content. |
| `predicted_footfall` | float | no | Point forecast. |
| `lo` | float | no | Lower bound. |
| `hi` | float | no | Upper bound. |
| `mode` | str | no | rate (x subscribers) or count. |
| `baseline_subscribers` | float | yes | Baseline 1: cook for all subscribers. |
| `baseline_last_week` | float | yes | Baseline 2: same slot last week / last seen. |
| `simulated_time` | bool | no | True only for demo runs with a fake clock. |
| `source` | str | no | Provenance label; see the Sources section. |

## `forecast_eval`

Sealed forecasts checked against what happened. *(derived by the pipeline; key: forecast_id)*

| column | type | nullable | meaning |
|---|---|---|---|
| `forecast_id` | str | no | Forecast id. |
| `meal_id` | str | no | Meal id. |
| `hash_ok` | bool | no | Sealed file content still matches its hash. |
| `sealed_before_meal` | bool | no | made_at earlier than the meal start. |
| `actual_footfall` | float | yes | F. |
| `predicted_footfall` | float | no | Forecast. |
| `abs_error` | float | yes | |forecast - actual|. |
| `in_interval` | bool | yes | Actual within [lo, hi]. |
| `baseline_subscribers_abs_error` | float | yes | Baseline 1 error. |
| `baseline_last_week_abs_error` | float | yes | Baseline 2 error. |
| `source` | str | no | Provenance label; see the Sources section. |

## `agreement`

Vision model vs human labels, and human vs human. *(derived by the pipeline; key: pair)*

| column | type | nullable | meaning |
|---|---|---|---|
| `pair` | str | no | Who is compared with whom. |
| `n_frames` | int | yes | Plates compared. |
| `n_items` | int | no | Plate x dish pairs where both gave a value. |
| `exact_agreement` | float | yes | Share of identical labels. |
| `within_one_step` | float | yes | Share within one quarter. |
| `mae_fraction` | float | yes | Mean absolute difference in fraction left. |
| `weighted_kappa` | float | yes | Quadratic-weighted Cohen's kappa. |
| `plate_present_agreement` | float | yes | Agreement on whether a plate is visible. |
| `source` | str | no | Provenance label; see the Sources section. |

## `recommendation_eval`

What would have happened if the kitchen had cooked the recommended amount. *(derived by the pipeline; key: forecast_id, dish)*

| column | type | nullable | meaning |
|---|---|---|---|
| `forecast_id` | str | no | Forecast id. |
| `meal_id` | str | no | Meal id. |
| `dish` | str | no | Dish. |
| `recommended_kg` | float | no | Sealed recommendation. |
| `actual_cooked_kg` | float | yes | What was cooked. |
| `actual_served_kg` | float | yes | What was served. |
| `actual_leftover_kg` | float | yes | What was left in vessels. |
| `would_run_short_kg` | float | yes | served - recommended, if positive. |
| `leftover_if_followed_kg` | float | yes | recommended - served, if positive. |
| `source` | str | no | Provenance label; see the Sources section. |
