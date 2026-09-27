"""Read the team's CSV sheets, validate them, normalise them and write Parquet."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import Config
from .dishes import DishBook
from .io import write_table
from .schema import TABLES, Issue, coerce_and_validate

MEAL_ID_RE = re.compile(r"^[A-Za-z0-9]+_(\d{8})_([BLD])$")
OPTIONAL = {"subscriptions", "production", "leftovers", "bin_weights", "dish_costs",
            "attendance_history", "human_labels", "footfall"}


class IngestError(RuntimeError):
    pass


def _read_csvs(folder: Path, name: str) -> pd.DataFrame | None:
    files = sorted(folder.glob(f"{name}.csv"))
    if name == "human_labels":
        files = sorted(folder.glob("human_labels*.csv"))
    if not files:
        return None
    frames = [pd.read_csv(f, dtype=str, keep_default_na=False, skip_blank_lines=True) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # drop fully-blank rows (common when sheets are exported)
    df = df[~df.apply(lambda r: all(str(v).strip() == "" for v in r), axis=1)].reset_index(drop=True)
    return df


def ingest(cfg: Config, allow_errors: bool = False, verbose: bool = True) -> dict[str, pd.DataFrame]:
    book = DishBook(cfg)
    folder = cfg.paths.raw_csv
    if not folder.exists():
        raise IngestError(f"No csv folder at {folder}. Put the filled sheets in <raw_root>/csv/.")

    issues: list[Issue] = []
    out: dict[str, pd.DataFrame] = {}

    for name, table in TABLES.items():
        if not table.raw_input:
            continue
        raw = _read_csvs(folder, name)
        if raw is None:
            if name not in OPTIONAL:
                issues.append(Issue("error", name, None, f"missing {name}.csv"))
            else:
                issues.append(Issue("warning", name, None, f"no {name}.csv (optional)"))
            out[name] = pd.DataFrame(columns=table.names)
            continue
        df, iss = coerce_and_validate(raw, table)
        issues += iss
        out[name] = df

    meals = out["meals"]
    slots = cfg.meal_slots

    # ---- meals: fill default times, canonical menu, check meal_id format
    if not meals.empty:
        for i, r in meals.iterrows():
            spec = slots.get(r["slot"], {})
            if r["start_time"] is None and spec:
                meals.at[i, "start_time"] = spec.get("start")
            if r["end_time"] is None and spec:
                meals.at[i, "end_time"] = spec.get("end")
            menu = book.parse_menu(r["menu_items"])
            meals.at[i, "menu_items"] = ";".join(menu)
            for d in menu:
                if not book.is_known(d):
                    issues.append(Issue("warning", "meals", i + 2,
                                        f"dish '{d}' not in config dishes; add it (with cost/serving) if real"))
            m = MEAL_ID_RE.match(str(r["meal_id"]))
            if not m:
                issues.append(Issue("warning", "meals", i + 2,
                                    f"meal_id '{r['meal_id']}' does not follow HOSTEL_YYYYMMDD_SLOT"))
            elif r["date"] and (m.group(1) != r["date"].replace("-", "") or m.group(2) != r["slot"]):
                issues.append(Issue("warning", "meals", i + 2,
                                    f"meal_id '{r['meal_id']}' disagrees with date/slot columns"))
        out["meals"] = meals

    meal_ids = set(meals["meal_id"].dropna()) if not meals.empty else set()
    menus = {r["meal_id"]: set(str(r["menu_items"]).split(";")) for _, r in meals.iterrows()}

    # ---- meal_id foreign keys
    for name in ["footfall", "subscriptions", "production", "leftovers", "bin_weights"]:
        df = out[name]
        for i, mid in enumerate(df["meal_id"]):
            if mid is not None and mid not in meal_ids:
                issues.append(Issue("error", name, i + 2, f"meal_id '{mid}' not in meals.csv"))

    # ---- dishes: canonical names + kg conversion
    for name in ["production", "leftovers"]:
        df = out[name]
        if df.empty:
            continue
        df["dish"] = df["dish"].map(book.canonical)
        kgs, notes = [], []
        for i, r in df.iterrows():
            qty = None if pd.isna(r["qty"]) else float(r["qty"])
            kg, note = book.to_kg(r["dish"], qty, r["unit"])
            if kg is None and qty is not None:
                issues.append(Issue("error", name, i + 2, f"cannot convert to kg: {note}"))
            kgs.append(kg)
            notes.append(note or None)
            if r["meal_id"] in menus and r["dish"] not in menus[r["meal_id"]]:
                issues.append(Issue("warning", name, i + 2,
                                    f"dish '{r['dish']}' not on the menu of {r['meal_id']}"))
        df["qty_kg"] = pd.array(kgs, dtype="Float64").astype("float64")
        df["kg_note"] = notes
        if df.duplicated(["meal_id", "dish"]).any():
            issues.append(Issue("error", name, None, "two rows for the same meal and dish after name normalisation"))
        out[name] = df

    lp = out["leftovers"]
    pr = out["production"]
    if not lp.empty and not pr.empty:
        m = lp.merge(pr, on=["meal_id", "dish"], suffixes=("_left", "_cooked"), how="left")
        for _, r in m.iterrows():
            if pd.notna(r["qty_kg_cooked"]) and pd.notna(r["qty_kg_left"]) and r["qty_kg_left"] > r["qty_kg_cooked"] + 1e-9:
                issues.append(Issue("error", "leftovers", None,
                                    f"{r['meal_id']}/{r['dish']}: leftover {r['qty_kg_left']:.2f} kg > cooked {r['qty_kg_cooked']:.2f} kg"))
        missing = m[m["qty_kg_cooked"].isna()]
        for _, r in missing.iterrows():
            issues.append(Issue("warning", "leftovers", None,
                                f"{r['meal_id']}/{r['dish']}: leftover recorded but no production row"))

    for name in ["dish_costs", "human_labels"]:
        if not out[name].empty:
            out[name]["dish"] = out[name]["dish"].map(book.canonical)
    hl = out["human_labels"]
    if not hl.empty:
        demo_meals = not meals.empty and (meals["source"] == "demo").all()
        hl["source"] = hl["source"].fillna("demo" if demo_meals else "observed")
        out["human_labels"] = hl

    bw = out["bin_weights"]
    if not bw.empty:
        bw["net_kg"] = bw["gross_kg"] - bw["tare_kg"]
        for i, v in enumerate(bw["net_kg"]):
            if pd.notna(v) and v < 0:
                issues.append(Issue("error", "bin_weights", i + 2, "tare_kg is larger than gross_kg"))
        out["bin_weights"] = bw

    # ---- demo data must never mix with real data
    srcs: set[str] = set()
    for df in out.values():
        if "source" in df.columns:
            srcs |= set(df["source"].dropna())
    if "demo" in srcs and srcs - {"demo", "inferred"}:
        issues.append(Issue("error", "*", None,
                            f"demo rows mixed with real rows (sources: {sorted(srcs)}). Use separate raw folders."))

    for name, df in out.items():
        write_table(cfg, df, name)

    _write_report(cfg, issues, out)
    errors = [i for i in issues if i.level == "error"]
    if verbose:
        print(f"Ingest: {sum(len(d) for d in out.values())} rows in {len(out)} tables, "
              f"{len(errors)} errors, {len(issues) - len(errors)} warnings")
        for i in issues[:40]:
            print("  ", i)
        if len(issues) > 40:
            print(f"   ... {len(issues) - 40} more in reports/validation_report.md")
    if errors and not allow_errors:
        raise IngestError(f"{len(errors)} validation errors; fix the CSVs (see reports/validation_report.md) "
                          "or rerun with --allow-errors to continue anyway.")
    return out


def _write_report(cfg: Config, issues: list[Issue], out: dict[str, pd.DataFrame]) -> None:
    lines = ["# Validation report", "", "| table | rows |", "|---|---|"]
    lines += [f"| {k} | {len(v)} |" for k, v in out.items()]
    lines += ["", f"## Issues ({len(issues)})", ""]
    lines += [f"- {i}" for i in issues] or ["- none"]
    (cfg.paths.reports / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
