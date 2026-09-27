"""Generate a complete FAKE dataset (source='demo') to test the pipeline end to end.

Every row is labelled 'demo'. The ingest step refuses to mix demo rows with real rows.
"""
from __future__ import annotations

import hashlib
import io as _io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HOSTEL = "DEMO"
MENUS = {"B": ["poha", "bread", "egg"],
         "L": ["rice", "dal", "sabzi", "roti", "curd"],
         "D": ["rice", "dal", "curry", "roti", "salad"]}
SERVING = {"poha": .18, "bread": .08, "egg": .05, "rice": .20, "dal": .15, "sabzi": .12, "roti": .10,
           "curd": .08, "curry": .15, "salad": .04, "sweet": .06}
SHOW = {"B": .52, "L": .72, "D": .80}
START = {"B": "07:30", "L": "12:15", "D": "19:45"}
COLORS = {"poha": (230, 190, 60), "bread": (205, 160, 100), "egg": (250, 240, 200), "rice": (245, 245, 235),
          "dal": (225, 170, 40), "sabzi": (120, 150, 60), "roti": (200, 150, 90), "curd": (250, 250, 250),
          "curry": (170, 60, 40), "salad": (90, 170, 80), "sweet": (240, 200, 170)}


def _plate_image(menu, fracs, rng) -> Image.Image:
    img = Image.new("RGB", (640, 480), (95, 100, 105))
    d = ImageDraw.Draw(img)
    cx, cy = 320 + rng.integers(-30, 30), 240 + rng.integers(-20, 20)
    d.ellipse([cx - 200, cy - 180, cx + 200, cy + 180], fill=(200, 205, 210), outline=(150, 155, 160), width=6)
    for i, dish in enumerate(menu):
        ang = 2 * np.pi * i / len(menu)
        x, y = cx + 110 * np.cos(ang), cy + 95 * np.sin(ang)
        r = 12 + 45 * np.sqrt(fracs[dish])
        if fracs[dish] > 0:
            d.ellipse([x - r, y - r, x + r, y + r], fill=COLORS.get(dish, (180, 180, 180)))
    return img


def generate_demo(root: str | Path, seed: int = 42, day1: str = "2026-10-03") -> Path:
    root = Path(root)
    csv, frames = root / "csv", root / "frames"
    csv.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d1 = date.fromisoformat(day1)
    S = 510

    def rate(slot, dt, special):
        r = SHOW[slot]
        if dt.weekday() >= 5 and slot == "L":
            r -= 0.1
        if special:
            r += 0.1
        return min(r + rng.normal(0, 0.03), 0.98)

    # ---- history: 28 days before day 1
    hist = []
    for k in range(28, 0, -1):
        dt = d1 - timedelta(days=k)
        subs = 500 if dt.month != d1.month else S
        for slot in "BLD":
            sp = int(slot == "D" and dt.weekday() == 6)
            hist.append(dict(date=dt.isoformat(), slot=slot, count=int(round(subs * rate(slot, dt, sp))),
                             subscriber_count=subs, special_flag=sp, source="demo"))
    pd.DataFrame(hist).to_csv(csv / "attendance_history.csv", index=False)

    meals, foot, subs_rows, prod, left, bins, truth, hl = [], [], [], [], [], [], [], []
    for day in (d1, d1 + timedelta(days=1)):
        for slot in "BLD":
            sp = int(slot == "D" and day == d1 + timedelta(days=1))
            menu = MENUS[slot] + (["sweet"] if sp else [])
            mid = f"{HOSTEL}_{day.strftime('%Y%m%d')}_{slot}"
            F = int(round(S * rate(slot, day, sp)))
            meals.append(dict(meal_id=mid, hostel=HOSTEL, date=day.isoformat(), slot=slot, start_time=START[slot],
                              end_time="", menu_items=";".join(menu), special_flag=sp,
                              weather="clear" if rng.random() > .3 else "rain", notes="", source="demo"))
            subs_rows.append(dict(meal_id=mid, subscriber_count=S, notes="manager register", source="demo"))
            # 15-min footfall, bell shaped over 2 hours
            w = np.exp(-0.5 * ((np.arange(8) - 3.2) / 1.6) ** 2)
            parts = rng.multinomial(F, w / w.sum())
            h0, m0 = map(int, START[slot].split(":"))
            for i, c in enumerate(parts):
                t = h0 * 60 + m0 + 15 * i
                foot.append(dict(meal_id=mid, interval_start=f"{t // 60:02d}:{t % 60:02d}", count=int(c),
                                 counter_person="VR", source="demo"))
            # production: kitchen cooks for ~all subscribers (+5%) at lunch/dinner, 80% at breakfast
            basis = S * (0.8 if slot == "B" else 1.05)
            plate_share = {dd: float(np.clip(rng.normal(0.12 if dd != "salad" else .3, .05), 0, .6)) for dd in menu}
            bin_kg = 0.0
            for dish in menu:
                cooked = basis * SERVING[dish] * rng.normal(1, .04)
                served = min(F * SERVING[dish] * rng.normal(1, .05), cooked * .98)
                leftover = cooked - served
                bin_kg += served * plate_share[dish]
                unit, qc, ql = "kg", cooked, leftover
                if dish == "dal":
                    unit, qc, ql = "l", cooked / 1.05, leftover / 1.05
                if dish == "roti":
                    unit, qc, ql = "pieces", round(cooked / .035), round(leftover / .035)
                prod.append(dict(meal_id=mid, dish=dish, qty=round(qc, 2), unit=unit, method="weighed", source="demo"))
                left.append(dict(meal_id=mid, dish=dish, qty=round(ql, 2), unit=unit, method="weighed", source="demo"))
            bins.append(dict(meal_id=mid, gross_kg=round(bin_kg + 2.4, 2), tare_kg=2.4, contents_note="few napkins",
                             source="demo"))
            # camera frames
            mdir = frames / mid
            mdir.mkdir(exist_ok=True)
            n = 36
            for i in range(n):
                fr = {dd: float(rng.choice([0, .25, .5, .75, 1], p=[.45, .25, .15, .1, .05]) if dd != "salad"
                                else rng.choice([0, .5, 1], p=[.3, .3, .4])) for dd in menu}
                img = _plate_image(menu, fr, rng)
                copies = [img]
                if i % 6 == 0:  # near-duplicate: the same plate caught twice
                    arr = np.asarray(img).astype(np.int16) + rng.integers(-3, 4, (480, 640, 3))
                    copies.append(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
                for j, im in enumerate(copies):
                    t = h0 * 60 + m0 + int(i * 100 / n) + j
                    ts = f"{day.strftime('%Y%m%d')}_{t // 60:02d}{t % 60:02d}{(7 * i + j) % 60:02d}"
                    buf = _io.BytesIO()
                    im.save(buf, "JPEG", quality=90)
                    b = buf.getvalue()
                    (mdir / f"IMG_{ts}.jpg").write_bytes(b)
                    fh = hashlib.sha1(b).hexdigest()[:10]
                    for dish in menu:
                        truth.append(dict(meal_id=mid, frame_hash=fh, dish=dish, fraction_left=fr[dish]))
                    if j == 0 and i % 3 == 0:
                        for who, noise in (("VR", .1), ("AB", .2)):
                            for dish in menu:
                                v = fr[dish] + (rng.choice([-.25, .25]) if rng.random() < noise else 0)
                                hl.append(dict(frame_id=f"{mid}_{fh}", labeller=who, dish=dish,
                                               fraction_left=float(np.clip(v, 0, 1)), plate_present=1,
                                               labelled_at=""))

    pd.DataFrame(meals).to_csv(csv / "meals.csv", index=False)
    pd.DataFrame(foot).to_csv(csv / "footfall.csv", index=False)
    pd.DataFrame(subs_rows).to_csv(csv / "subscriptions.csv", index=False)
    pd.DataFrame(prod).to_csv(csv / "production.csv", index=False)
    pd.DataFrame(left).to_csv(csv / "leftovers.csv", index=False)
    pd.DataFrame(bins).to_csv(csv / "bin_weights.csv", index=False)
    pd.DataFrame([dict(dish=k, cost_per_kg_inr=v, source="demo") for k, v in
                  {"rice": 45, "dal": 60, "roti": 70, "sabzi": 80, "curry": 140, "curd": 70, "salad": 40,
                   "poha": 50, "bread": 60, "egg": 130, "sweet": 120}.items()]).to_csv(csv / "dish_costs.csv", index=False)
    hdf = pd.DataFrame(hl)
    for who, g in hdf.groupby("labeller"):
        g.to_csv(csv / f"human_labels_{who}.csv", index=False)
    pd.DataFrame(truth).to_csv(frames / "_demo_truth.csv", index=False)
    print(f"Demo data written to {root} ({len(meals)} meals, {len(foot)} footfall rows, "
          f"{sum(1 for _ in frames.glob('*/*.jpg'))} frames). Everything is labelled source=demo.")
    return root
