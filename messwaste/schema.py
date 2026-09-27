"""Single source of truth for every table: columns, types, allowed values, meaning.

SCHEMA.md and the datasheet are generated from this file, so descriptions here
are what a stranger reads when they open the dataset.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Provenance labels. Every row in every table carries one of these.
SOURCES = {
    "observed": "Measured by the team during the collection window (counts, weights, photos).",
    "reported": "Supplied by mess staff for the collection window (e.g. subscriber counts, cook batch sizes).",
    "historical": "Mess records from before the collection window, supplied by the manager.",
    "inferred": "Computed by this pipeline or a model from observed/reported data.",
    "synthetic": "Generated to extend real observations. Never used for headline numbers.",
    "demo": "Fake data from `messwaste demo`, for testing only. Must never appear in a submission.",
}
SLOTS = ["B", "L", "D"]
UNITS = ["kg", "g", "l", "ml", "pieces"]
METHODS = ["weighed", "cook_reported", "estimated", "counted", "register", "card_swipes"]
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")


@dataclass
class Col:
    name: str
    dtype: str                 # str | int | float | date | time | datetime | bool
    desc: str
    required: bool = True      # column must exist
    nullable: bool = False     # value may be empty
    allowed: list | None = None
    min: float | None = None
    max: float | None = None
    computed: bool = False     # filled by the pipeline during ingest, not typed by the team


@dataclass
class Table:
    name: str
    desc: str
    cols: list[Col]
    key: list[str] = field(default_factory=list)
    raw_input: bool = False    # True = team fills this in as CSV

    def col(self, name: str) -> Col:
        return next(c for c in self.cols if c.name == name)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cols]

    @property
    def input_names(self) -> list[str]:
        return [c.name for c in self.cols if not c.computed]


SRC = Col("source", "str", "Provenance label; see the Sources section.", allowed=list(SOURCES))

TABLES: dict[str, Table] = {}


def _t(t: Table) -> Table:
    TABLES[t.name] = t
    return t


# ----------------------------------------------------------------------------- raw inputs
_t(Table("meals", "One row per meal service observed (or planned, for forecasting).", [
    Col("meal_id", "str", "Unique meal id, recommended HOSTEL_YYYYMMDD_SLOT, e.g. KAP_20261003_L."),
    Col("hostel", "str", "Hostel code."),
    Col("date", "date", "Service date (YYYY-MM-DD, local time)."),
    Col("slot", "str", "B = breakfast, L = lunch, D = dinner.", allowed=SLOTS),
    Col("start_time", "time", "Service start HH:MM. Empty = default from config.", nullable=True),
    Col("end_time", "time", "Service end HH:MM. Empty = default from config.", nullable=True),
    Col("menu_items", "str", "Dishes served, separated by ';' (canonical names from config)."),
    Col("special_flag", "int", "1 = special meal (special dinner, paneer day, fest), else 0.", allowed=[0, 1]),
    Col("weather", "str", "Free text, e.g. 'rain', 'clear'.", required=False, nullable=True),
    Col("notes", "str", "Anything unusual (exam day, event, power cut).", required=False, nullable=True),
    SRC,
], key=["meal_id"], raw_input=True))

_t(Table("footfall", "Diners entering, tallied per 15-minute interval.", [
    Col("meal_id", "str", "Meal id."),
    Col("interval_start", "time", "Start of the 15-minute interval, HH:MM."),
    Col("count", "int", "Diners who entered during the interval.", min=0),
    Col("counter_person", "str", "Initials of whoever counted.", required=False, nullable=True),
    SRC,
], key=["meal_id", "interval_start"], raw_input=True))

_t(Table("subscriptions", "Diners entitled to eat this meal (aggregate count from the manager).", [
    Col("meal_id", "str", "Meal id."),
    Col("subscriber_count", "int", "Aggregate subscriber count. Never names or roll numbers.", min=0),
    Col("notes", "str", "How the manager arrived at the number.", required=False, nullable=True),
    SRC,
], key=["meal_id"], raw_input=True))

_t(Table("production", "Quantity cooked per dish.", [
    Col("meal_id", "str", "Meal id."),
    Col("dish", "str", "Canonical dish name."),
    Col("qty", "float", "Quantity cooked.", min=0),
    Col("unit", "str", "Unit of qty.", allowed=UNITS),
    Col("method", "str", "How the number was obtained.", allowed=METHODS),
    SRC,
    Col("qty_kg", "float", "qty converted to kg.", nullable=True, computed=True),
    Col("kg_note", "str", "Assumption used for the kg conversion.", nullable=True, computed=True),
], key=["meal_id", "dish"], raw_input=True))

_t(Table("leftovers", "Quantity left in serving vessels at the end of service (never served).", [
    Col("meal_id", "str", "Meal id."),
    Col("dish", "str", "Canonical dish name."),
    Col("qty", "float", "Quantity left.", min=0),
    Col("unit", "str", "Unit of qty.", allowed=UNITS),
    Col("method", "str", "How the number was obtained.", allowed=METHODS),
    SRC,
    Col("qty_kg", "float", "qty converted to kg.", nullable=True, computed=True),
    Col("kg_note", "str", "Assumption used for the kg conversion.", nullable=True, computed=True),
], key=["meal_id", "dish"], raw_input=True))

_t(Table("bin_weights", "Plate-scraping bin weighed after service.", [
    Col("meal_id", "str", "Meal id."),
    Col("gross_kg", "float", "Bin + contents, kg.", min=0),
    Col("tare_kg", "float", "Empty bin, kg.", min=0),
    Col("contents_note", "str", "Non-food items seen (napkins, bones...).", required=False, nullable=True),
    SRC,
    Col("net_kg", "float", "gross_kg - tare_kg.", nullable=True, computed=True),
], key=["meal_id"], raw_input=True))

_t(Table("dish_costs", "Cost per kg of each dish, from the manager.", [
    Col("dish", "str", "Canonical dish name."),
    Col("cost_per_kg_inr", "float", "Ingredient + fuel cost per kg cooked, INR.", min=0),
    SRC,
], key=["dish"], raw_input=True))

_t(Table("attendance_history", "Past attendance records from the mess (before the collection window).", [
    Col("date", "date", "Date."),
    Col("slot", "str", "B/L/D.", allowed=SLOTS),
    Col("count", "int", "Diners who ate.", min=0),
    Col("subscriber_count", "int", "Subscribers that day, if known.", nullable=True, min=0),
    Col("special_flag", "int", "1 = special meal.", allowed=[0, 1]),
    SRC,
], key=["date", "slot"], raw_input=True))

_t(Table("human_labels", "Hand labels of plate photos (validation set for the vision model).", [
    Col("frame_id", "str", "Frame id from camera_frames."),
    Col("labeller", "str", "Labeller initials."),
    Col("dish", "str", "Canonical dish name."),
    Col("fraction_left", "float", "Fraction of this dish left on the plate (0, .25, .5, .75, 1). Empty = can't tell.",
        nullable=True, min=0, max=1),
    Col("plate_present", "int", "1 = a single distinct plate is visible.", allowed=[0, 1]),
    Col("labelled_at", "str", "When the label was made (ISO time).", required=False, nullable=True),
    Col("source", "str", "Filled automatically: observed (or demo).", required=False, nullable=True,
        allowed=list(SOURCES)),
], key=["frame_id", "labeller", "dish"], raw_input=True))

# ----------------------------------------------------------------------------- derived
_t(Table("camera_frames", "Every frame from the return-counter camera and whether it was kept.", [
    Col("frame_id", "str", "meal_id + short content hash."),
    Col("meal_id", "str", "Meal id (from the folder the frame was in)."),
    Col("taken_at", "datetime", "Capture time (EXIF or filename), local time.", nullable=True),
    Col("file_name", "str", "Original file name."),
    Col("kept", "bool", "True if the frame passed privacy and de-duplication filters."),
    Col("drop_reason", "str", "face | person | duplicate | unreadable | empty string if kept.", nullable=True),
    Col("n_faces", "int", "Faces detected."),
    Col("n_persons", "int", "People detected (0 if detector off)."),
    Col("dhash", "str", "64-bit difference hash (hex)."),
    Col("in_meal_window", "bool", "Capture time falls within the meal window.", nullable=True),
    SRC,
], key=["frame_id"]))

_t(Table("plate_labels_model", "Vision-model estimate of each dish left on each kept plate.", [
    Col("frame_id", "str", "Frame id."),
    Col("meal_id", "str", "Meal id."),
    Col("dish", "str", "Canonical dish name."),
    Col("fraction_left", "float", "0, .25, .5, .75, 1; empty = model could not tell.", nullable=True),
    Col("plate_present", "int", "1 if the model saw one distinct plate."),
    Col("backend", "str", "hf | anthropic | mock."),
    Col("model_id", "str", "Exact model name."),
    Col("parse_ok", "bool", "Model output parsed as valid JSON."),
    Col("labelled_at", "str", "ISO time of labelling."),
    SRC,
], key=["frame_id", "dish", "model_id"]))

_t(Table("meal_metrics", "Per-meal waste decomposition.", [
    Col("meal_id", "str", "Meal id."),
    Col("date", "date", "Date."), Col("slot", "str", "Slot."),
    Col("start_dt", "datetime", "Service start, local time."),
    Col("footfall", "float", "F: diners counted.", nullable=True),
    Col("subscribers", "float", "S: subscribers entitled.", nullable=True),
    Col("show_up_rate", "float", "F / S.", nullable=True),
    Col("ghost_diners", "float", "S - F (never negative).", nullable=True),
    Col("cooked_kg", "float", "Total cooked.", nullable=True),
    Col("leftover_kg", "float", "Total left in vessels.", nullable=True),
    Col("served_kg", "float", "cooked - leftover.", nullable=True),
    Col("implied_headcount", "float", "Headcount the kitchen effectively cooked for = F x cooked/served.", nullable=True),
    Col("ghost_overprod_kg", "float", "Overproduction explained by cooking for absent subscribers.", nullable=True),
    Col("other_overprod_kg", "float", "Remaining vessel leftovers.", nullable=True),
    Col("plate_waste_kg", "float", "Bin net weight.", nullable=True),
    Col("total_waste_kg", "float", "Vessel leftovers + plate waste.", nullable=True),
    Col("waste_g_per_diner", "float", "total_waste_kg / F in grams.", nullable=True),
    Col("ghost_inr", "float", "Cost of ghost overproduction.", nullable=True),
    Col("other_overprod_inr", "float", "Cost of other overproduction.", nullable=True),
    Col("plate_inr", "float", "Cost of plate waste.", nullable=True),
    Col("total_inr", "float", "Total cost of waste.", nullable=True),
    Col("camera_mean_fraction_left", "float", "Mean fraction left across labelled plates (camera index).", nullable=True),
    Col("camera_plates", "int", "Plates labelled for this meal."),
    Col("gaps", "str", "Missing inputs for this meal.", nullable=True),
    SRC,
], key=["meal_id"]))

_t(Table("dish_metrics", "Per-meal, per-dish waste figures.", [
    Col("meal_id", "str", "Meal id."), Col("dish", "str", "Dish."),
    Col("cooked_kg", "float", "Cooked.", nullable=True),
    Col("leftover_kg", "float", "Left in vessels.", nullable=True),
    Col("served_kg", "float", "Served.", nullable=True),
    Col("p_kg_per_diner", "float", "served_kg / footfall.", nullable=True),
    Col("overprod_rate", "float", "leftover / cooked.", nullable=True),
    Col("implied_headcount", "float", "F x cooked/served.", nullable=True),
    Col("ghost_overprod_kg", "float", "min(leftover, p x (S-F)).", nullable=True),
    Col("other_overprod_kg", "float", "leftover - ghost.", nullable=True),
    Col("plate_kg", "float", "Bin weight attributed to this dish via plate photos.", nullable=True),
    Col("plate_attr_method", "str", "How plate_kg was attributed.", nullable=True),
    Col("mean_fraction_left", "float", "Mean fraction left on labelled plates.", nullable=True),
    Col("cost_per_kg_inr", "float", "Cost used.", nullable=True),
    Col("waste_inr", "float", "(leftover + plate_kg) x cost.", nullable=True),
    SRC,
], key=["meal_id", "dish"]))

_t(Table("forecasts", "Sealed footfall forecasts (one row per sealed file).", [
    Col("forecast_id", "str", "Unique id."), Col("meal_id", "str", "Target meal."),
    Col("made_at", "datetime", "When the forecast was sealed (local time)."),
    Col("sha256", "str", "Hash of the sealed JSON content."),
    Col("predicted_footfall", "float", "Point forecast."),
    Col("lo", "float", "Lower bound."), Col("hi", "float", "Upper bound."),
    Col("mode", "str", "rate (x subscribers) or count."),
    Col("baseline_subscribers", "float", "Baseline 1: cook for all subscribers.", nullable=True),
    Col("baseline_last_week", "float", "Baseline 2: same slot last week / last seen.", nullable=True),
    Col("simulated_time", "bool", "True only for demo runs with a fake clock."),
    SRC,
], key=["forecast_id"]))

_t(Table("forecast_eval", "Sealed forecasts checked against what happened.", [
    Col("forecast_id", "str", "Forecast id."), Col("meal_id", "str", "Meal id."),
    Col("hash_ok", "bool", "Sealed file content still matches its hash."),
    Col("sealed_before_meal", "bool", "made_at earlier than the meal start."),
    Col("actual_footfall", "float", "F.", nullable=True),
    Col("predicted_footfall", "float", "Forecast."),
    Col("abs_error", "float", "|forecast - actual|.", nullable=True),
    Col("in_interval", "bool", "Actual within [lo, hi].", nullable=True),
    Col("baseline_subscribers_abs_error", "float", "Baseline 1 error.", nullable=True),
    Col("baseline_last_week_abs_error", "float", "Baseline 2 error.", nullable=True),
    SRC,
], key=["forecast_id"]))


_t(Table("agreement", "Vision model vs human labels, and human vs human.", [
    Col("pair", "str", "Who is compared with whom."),
    Col("n_frames", "int", "Plates compared.", nullable=True),
    Col("n_items", "int", "Plate x dish pairs where both gave a value."),
    Col("exact_agreement", "float", "Share of identical labels.", nullable=True),
    Col("within_one_step", "float", "Share within one quarter.", nullable=True),
    Col("mae_fraction", "float", "Mean absolute difference in fraction left.", nullable=True),
    Col("weighted_kappa", "float", "Quadratic-weighted Cohen's kappa.", nullable=True),
    Col("plate_present_agreement", "float", "Agreement on whether a plate is visible.", nullable=True),
    SRC,
], key=["pair"]))

_t(Table("recommendation_eval", "What would have happened if the kitchen had cooked the recommended amount.", [
    Col("forecast_id", "str", "Forecast id."), Col("meal_id", "str", "Meal id."), Col("dish", "str", "Dish."),
    Col("recommended_kg", "float", "Sealed recommendation."),
    Col("actual_cooked_kg", "float", "What was cooked.", nullable=True),
    Col("actual_served_kg", "float", "What was served.", nullable=True),
    Col("actual_leftover_kg", "float", "What was left in vessels.", nullable=True),
    Col("would_run_short_kg", "float", "served - recommended, if positive.", nullable=True),
    Col("leftover_if_followed_kg", "float", "recommended - served, if positive.", nullable=True),
    SRC,
], key=["forecast_id", "dish"]))


# ============================================================================= validation
@dataclass
class Issue:
    level: str   # error | warning
    table: str
    row: int | None
    msg: str

    def __str__(self) -> str:
        r = f" row {self.row}" if self.row is not None else ""
        return f"[{self.level.upper()}] {self.table}{r}: {self.msg}"


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "") \
        or v is pd.NA or v is pd.NaT


def coerce_and_validate(df: pd.DataFrame, table: Table) -> tuple[pd.DataFrame, list[Issue]]:
    """Coerce column types in place and report problems. Row numbers are CSV line numbers."""
    issues: list[Issue] = []
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    for c in table.cols:
        if c.computed:
            df[c.name] = None
            continue
        if c.name not in df.columns:
            if c.required:
                issues.append(Issue("error", table.name, None, f"missing column '{c.name}'"))
            df[c.name] = None
    extra = [c for c in df.columns if c not in table.names]
    if extra:
        issues.append(Issue("warning", table.name, None, f"ignoring unknown columns {extra}"))
    df = df[table.names].copy()

    for c in table.cols:
        if c.computed:
            continue
        s = df[c.name]
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.map(lambda v: None if _empty(v) else v)
        empty_mask = s.isna()
        for i in np.where(empty_mask & (not c.nullable))[0]:
            issues.append(Issue("error", table.name, int(i) + 2, f"'{c.name}' is empty"))

        if c.dtype in ("int", "float"):
            num = pd.to_numeric(s, errors="coerce")
            bad = num.isna() & ~empty_mask
            for i in np.where(bad)[0]:
                issues.append(Issue("error", table.name, int(i) + 2, f"'{c.name}' is not a number: {s.iloc[i]!r}"))
            if c.min is not None:
                for i in np.where(num < c.min)[0]:
                    issues.append(Issue("error", table.name, int(i) + 2, f"'{c.name}' below {c.min}"))
            if c.max is not None:
                for i in np.where(num > c.max)[0]:
                    issues.append(Issue("error", table.name, int(i) + 2, f"'{c.name}' above {c.max}"))
            if c.dtype == "int":
                frac = (num.dropna() % 1 != 0)
                for i in frac[frac].index:
                    issues.append(Issue("error", table.name, int(i) + 2, f"'{c.name}' must be a whole number"))
                df[c.name] = num.round().astype("Int64")
            else:
                df[c.name] = num.astype("float64")
        elif c.dtype == "date":
            d = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
            for i in np.where(d.isna() & ~empty_mask)[0]:
                issues.append(Issue("error", table.name, int(i) + 2, f"'{c.name}' must be YYYY-MM-DD: {s.iloc[i]!r}"))
            df[c.name] = d.dt.strftime("%Y-%m-%d").where(d.notna(), None)
        elif c.dtype == "time":
            def _t(v):
                if v is None:
                    return None
                v = str(v)
                if TIME_RE.match(v):
                    h, m = v.split(":")
                    if 0 <= int(h) < 24 and 0 <= int(m) < 60:
                        return f"{int(h):02d}:{int(m):02d}"
                return "__bad__"
            t = s.map(_t)
            for i in np.where(t == "__bad__")[0]:
                issues.append(Issue("error", table.name, int(i) + 2, f"'{c.name}' must be HH:MM: {s.iloc[i]!r}"))
            df[c.name] = t.where(t != "__bad__", None)
        else:
            df[c.name] = s.map(lambda v: None if v is None else str(v))

        if c.allowed is not None:
            col = df[c.name]
            for i, v in enumerate(col):
                if _empty(v):
                    continue
                vv = v.lower() if isinstance(v, str) and c.name != "slot" else v
                if isinstance(vv, str) and c.name == "slot":
                    vv = vv.upper()
                ok = vv in c.allowed or (not isinstance(vv, str) and int(vv) in c.allowed)
                if not ok:
                    issues.append(Issue("error", table.name, i + 2, f"'{c.name}'={v!r} not in {c.allowed}"))
            if c.dtype == "str":
                if c.name == "slot":
                    df[c.name] = df[c.name].map(lambda v: v.upper() if isinstance(v, str) else v)
                else:
                    df[c.name] = df[c.name].map(lambda v: v.lower() if isinstance(v, str) else v)

    if table.key and not df.empty:
        dup = df.duplicated(subset=table.key, keep=False)
        for i in np.where(dup)[0]:
            issues.append(Issue("error", table.name, int(i) + 2,
                                f"duplicate key {tuple(df.iloc[i][table.key])}"))
    return df, issues


def schema_markdown() -> str:
    lines = ["# Schema", "",
             "Every table is stored as Parquet under `parquet/`. Every row has a `source` column.", "",
             "## Sources", ""]
    for k, v in SOURCES.items():
        lines.append(f"- **{k}**: {v}")
    for t in TABLES.values():
        kind = "raw input (CSV filled by the team)" if t.raw_input else "derived by the pipeline"
        lines += ["", f"## `{t.name}`", "", f"{t.desc} *({kind}; key: {', '.join(t.key) or 'none'})*", "",
                  "| column | type | nullable | meaning |", "|---|---|---|---|"]
        for c in t.cols:
            allowed = f" Allowed: {', '.join(map(str, c.allowed))}." if c.allowed and c.name != "source" else ""
            lines.append(f"| `{c.name}` | {c.dtype} | {'yes' if c.nullable else 'no'} | {c.desc}{allowed} |")
    return "\n".join(lines) + "\n"
