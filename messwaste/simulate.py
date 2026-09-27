"""Build a realistic SYNTHETIC operational dataset around the real plate photos.

The photos are real (source=observed). Everything else this module writes —
subscriber counts, footfall tallies, cooked quantities, vessel leftovers, bin
weights, attendance history — is generated, and every one of those rows carries
source="synthetic", which the schema defines as "generated to extend real
observations". The dashboard and datasheet surface that split; don't strip it.

Structure it produces:
  Day 1 (Sat)  B, L, D  fully measured        -> 80% of the real photos
  Day 2 (Sun)  B, L     forecast + validated  -> 20% of the real photos
Forecasts for the day-2 meals are sealed on day-1 night with a fake clock, then
evaluated, so the predict-then-verify loop runs end to end.

Deliberate imperfections, because real mess records look like this:
  - subscriber count missing for two meals; one is a rounded guess with a note
  - one bin never weighed (the scale was borrowed back)
  - dal recorded in litres, roti in pieces, one dish typed "chapati"
  - a few dishes cook_reported instead of weighed
  - two footfall intervals missed during the rush
  - attendance history has gaps and one implausible outlier
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SYN = "synthetic"
HOSTEL = "DIHING"
DAY1 = date(2026, 9, 26)          # Saturday: the day the real photos were taken
DAY2 = DAY1 + timedelta(days=1)

# Menus chosen to match what is actually visible in the real photographs.
MENUS = {
    "B": ["poha", "pasta", "sprouts", "bread", "banana", "chutney", "egg"],
    "L": ["rice", "dal", "curry", "chapati", "curd", "salad"],   # "chapati" on purpose: alias of roti
    "D": ["rice", "dal", "curry", "roti", "curd", "salad"],
}
SERVING = {"poha": .17, "bread": .07, "sprouts": .06, "egg": .05, "banana": .11, "chutney": .03,
           "rice": .20, "dal": .15, "sabzi": .12, "chapati": .10, "roti": .10, "curd": .08,
           "salad": .04, "curry": .15, "pasta": .15, "dosa": .10}
COSTS = {"poha": 48, "bread": 58, "sprouts": 90, "egg": 130, "banana": 55, "chutney": 70,
         "rice": 44, "dal": 62, "sabzi": 78, "roti": 70, "curd": 68, "salad": 38, "curry": 145,
         "pasta": 85, "dosa": 65}
START = {"B": "07:30", "L": "12:15", "D": "19:45"}
# how many of the subscribed actually turn up
SHOW = {"B": 0.47, "L": 0.71, "D": 0.80}
# how the kitchen sizes batches: fraction of the subscriber list it cooks for
COOK_FOR = {"B": 0.78, "L": 1.02, "D": 1.04}


def _unique_photos(*roots: str) -> list[str]:
    seen, out = set(), []
    for r in roots:
        for f in sorted(glob.glob(os.path.join(r, "**", "*.jpeg"), recursive=True),
                        key=lambda p: os.path.basename(p)):
            h = hashlib.sha1(open(f, "rb").read()).hexdigest()
            if h not in seen:
                seen.add(h)
                out.append(f)
    return out


LUNCH_DISHES = {"rice", "dal", "curry", "chapati", "roti", "sabzi"}


def _photo_slot(content: dict, name: str) -> str:
    """Route each photo to the meal whose menu actually matches what is on the tray.

    Assigning photos at random produced nonsense labels: a tray of pasta and sprouts
    landed in a rice-and-dal lunch, so every dish on that meal's menu read 0%.
    """
    d = set((content.get(name) or {}).get("dishes", {}))
    return "L" if d & LUNCH_DISHES else "B"


def _split_photos(photos: list[str], rng, content: dict) -> dict[str, list[str]]:
    """Group by tray content, then hold 20% of each group back for the day-2 meals."""
    out: dict[str, list[str]] = {f"{HOSTEL}_{DAY1:%Y%m%d}_B": [], f"{HOSTEL}_{DAY1:%Y%m%d}_L": [],
                                 f"{HOSTEL}_{DAY1:%Y%m%d}_D": [], f"{HOSTEL}_{DAY2:%Y%m%d}_B": [],
                                 f"{HOSTEL}_{DAY2:%Y%m%d}_L": []}
    for slot in ("B", "L"):
        group = [p for p in photos if _photo_slot(content, os.path.basename(p).replace(" ", "_")) == slot]
        idx = rng.permutation(len(group))
        n_val = max(2, int(round(0.20 * len(group))))
        val = [group[i] for i in idx[:n_val]]
        train = [group[i] for i in idx[n_val:]]
        out[f"{HOSTEL}_{DAY2:%Y%m%d}_{slot}"] = val
        if slot == "B":
            out[f"{HOSTEL}_{DAY1:%Y%m%d}_B"] = train
        else:                       # split the day-1 lunch-type trays across lunch and dinner
            cut = int(round(0.6 * len(train)))
            out[f"{HOSTEL}_{DAY1:%Y%m%d}_L"] = train[:cut]
            out[f"{HOSTEL}_{DAY1:%Y%m%d}_D"] = train[cut:]
    return out


def generate(root: str | Path, photo_roots: list[str], seed: int = 11,
             content_path: str | Path | None = None) -> Path:
    root = Path(root)
    csv, frames = root / "csv", root / "frames"
    shutil.rmtree(root, ignore_errors=True)
    csv.mkdir(parents=True)
    frames.mkdir(parents=True)
    rng = np.random.default_rng(seed)

    photos = _unique_photos(*photo_roots)
    if not photos:
        raise SystemExit(f"No .jpeg photos found under {photo_roots}")
    content = {}
    if content_path and Path(content_path).exists():
        content = json.loads(Path(content_path).read_text())
    buckets = _split_photos(photos, rng, content)
    for mid, fs in buckets.items():
        d = frames / mid
        d.mkdir()
        for f in fs:
            shutil.copy(f, d / os.path.basename(f).replace(" ", "_"))

    subs_base = 400

    # A shared per-day effect: weather, an exam, a fest, a long weekend. It moves every
    # meal that day together, which is why copying one past day is unreliable while a
    # recency-weighted average over many days is not.
    day_effect: dict[date, float] = {}

    def eff(dt: date) -> float:
        if dt not in day_effect:
            e = rng.normal(0, 0.075)
            if rng.random() < 0.10:            # exam block / fest / long weekend
                e -= abs(rng.normal(0.14, 0.05))
            day_effect[dt] = e
        return day_effect[dt]

    # ---------------------------------------------------------------- history (3 weeks, with gaps)
    hist = []
    for k in range(21, 0, -1):
        dt = DAY1 - timedelta(days=k)
        if rng.random() < 0.13:            # the register simply was not filled that day
            continue
        s = subs_base - (5 if dt < DAY1 - timedelta(days=14) else 0)
        for slot in "BLD":
            if rng.random() < 0.07:
                continue
            r = SHOW[slot] + (0.05 if dt.weekday() == 6 else 0) - (0.04 if dt.weekday() == 5 and slot == "B" else 0)
            c = int(round(s * min(max(r + eff(dt) + rng.normal(0, 0.03), 0.15), 0.95)))
            sp = int(dt.weekday() == 6 and slot == "D")
            if sp:
                c = int(c * 1.09)
            hist.append(dict(date=dt.isoformat(), slot=slot, count=c,
                             subscriber_count=s if rng.random() > 0.25 else "",
                             special_flag=sp, source="synthetic"))
    if hist:                                # one obvious transcription error, left in on purpose
        hist[len(hist) // 3]["count"] = int(hist[len(hist) // 3]["count"] * 10)
    pd.DataFrame(hist).to_csv(csv / "attendance_history.csv", index=False)

    meals, foot, subs, prod, left, bins = [], [], [], [], [], []
    truth = []   # what the simulation used, kept beside the data so nothing is hidden

    for day in (DAY1, DAY2):
        slots = "BLD" if day == DAY1 else "BL"
        for slot in slots:
            mid = f"{HOSTEL}_{day:%Y%m%d}_{slot}"
            menu = MENUS[slot]
            sp = int(day.weekday() == 6 and slot == "D")
            meals.append(dict(meal_id=mid, hostel=HOSTEL, date=day.isoformat(), slot=slot,
                              start_time=START[slot], end_time="", menu_items=";".join(menu),
                              special_flag=sp, weather="rain" if (day == DAY2 and slot == "B") else "clear",
                              notes="scale borrowed back before service ended" if (day == DAY1 and slot == "D") else "",
                              source=SYN))

            S = subs_base
            # the manager does not have a clean per-meal number for everything
            if slot == "D" and day == DAY1:
                subs.append(dict(meal_id=mid, subscriber_count=390, notes="manager's rough figure, rounded",
                                 source=SYN))
            elif not (day == DAY2 and slot == "B"):        # day-2 breakfast: no count at all
                subs.append(dict(meal_id=mid, subscriber_count=S, notes="mess register", source=SYN))

            rate = SHOW[slot] + (0.05 if day.weekday() == 6 else 0) + eff(day)
            if day == DAY2 and slot == "B":
                rate -= 0.05                                # it rained
            F = int(round(S * min(max(rate + rng.normal(0, 0.025), 0.15), 0.95)))
            truth.append(dict(meal_id=mid, true_footfall=F, note="value the simulation drew"))

            # footfall in 15-min buckets, with two intervals missed in the rush
            w = np.exp(-0.5 * ((np.arange(8) - 3.1) / 1.7) ** 2)
            parts = rng.multinomial(F, w / w.sum())
            h0, m0 = map(int, START[slot].split(":"))
            miss = {3} if (day == DAY1 and slot == "L") else set()
            for i, c in enumerate(parts):
                if i in miss:
                    continue
                t = h0 * 60 + m0 + 15 * i
                foot.append(dict(meal_id=mid, interval_start=f"{t // 60:02d}:{t % 60:02d}", count=int(c),
                                 counter_person=rng.choice(["VR", "AB", "SK"]), source=SYN))

            basis = S * COOK_FOR[slot] * rng.normal(1, 0.02)
            bin_kg = 0.0
            for dish in menu:
                cooked = basis * SERVING[dish] * rng.normal(1, 0.04)
                served = min(F * SERVING[dish] * rng.normal(1, 0.06), cooked * 0.97)
                leftover = cooked - served
                plate_share = np.clip(rng.normal(0.30 if dish in ("salad", "sprouts", "curd") else 0.11, 0.05), 0, .6)
                bin_kg += served * plate_share

                unit, qc, ql, method = "kg", cooked, leftover, "weighed"
                if dish == "dal":
                    unit, qc, ql = "l", cooked / 1.05, leftover / 1.05
                elif dish in ("roti", "chapati"):
                    unit, qc, ql, method = "pieces", round(cooked / .035), round(leftover / .035), "cook_reported"
                elif dish in ("bread", "banana", "egg", "dosa"):
                    unit, qc, ql, method = "pieces", round(cooked / SERVING[dish]), round(leftover / SERVING[dish]), "cook_reported"
                if dish in ("chutney", "salad"):
                    method = "estimated"
                prod.append(dict(meal_id=mid, dish=dish, qty=round(qc, 2), unit=unit, method=method, source=SYN))
                # chutney leftovers nobody bothered to measure
                if dish != "chutney":
                    left.append(dict(meal_id=mid, dish=dish, qty=round(ql, 2), unit=unit,
                                     method=method if method != "cook_reported" else "estimated", source=SYN))

            if not (day == DAY1 and slot == "D"):           # that bin was never weighed
                tare = 2.4
                bins.append(dict(meal_id=mid, gross_kg=round(bin_kg + tare, 2), tare_kg=tare,
                                 contents_note="napkins and a few banana peels" if slot == "B" else "some bones/peel",
                                 source=SYN))

    pd.DataFrame(meals).to_csv(csv / "meals.csv", index=False)
    pd.DataFrame(foot).to_csv(csv / "footfall.csv", index=False)
    pd.DataFrame(subs).to_csv(csv / "subscriptions.csv", index=False)
    pd.DataFrame(prod).to_csv(csv / "production.csv", index=False)
    pd.DataFrame(left).to_csv(csv / "leftovers.csv", index=False)
    pd.DataFrame(bins).to_csv(csv / "bin_weights.csv", index=False)
    pd.DataFrame([dict(dish=k, cost_per_kg_inr=v, source=SYN) for k, v in COSTS.items()]).to_csv(
        csv / "dish_costs.csv", index=False)
    pd.DataFrame(truth).to_csv(root / "SIMULATION_TRUTH.csv", index=False)

    (root / "READ_ME_FIRST.txt").write_text(
        "Plate photographs under frames/ are REAL (source=observed once ingested).\n"
        "Everything in csv/ is SYNTHETIC: subscriber counts, footfall, cooked quantities,\n"
        "vessel leftovers, bin weights and attendance history were generated by\n"
        "messwaste/simulate.py, not measured. Every one of those rows carries\n"
        "source=synthetic. SIMULATION_TRUTH.csv lists the footfall values the\n"
        "simulation drew, for comparison with what the forecast predicted.\n"
        "This dataset is for pipeline testing and portfolio use. It is not evidence\n"
        "about any real mess and must not be presented as measurement.\n", encoding="utf-8")

    print(f"Synthetic operational data + {len(photos)} real photos written to {root}")
    print(f"  day 1 {DAY1} B/L/D measured, day 2 {DAY2} B/L for forecast validation")
    for mid, fs in buckets.items():
        print(f"    {mid}: {len(fs)} photos")
    return root
