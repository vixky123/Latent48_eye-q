"""Predict-then-verify.

`forecast` predicts footfall for meals that have not happened yet, recommends
cook quantities, and SEALS each forecast: a JSON file whose SHA-256 hash you
post publicly (hackathon channel / commit) before the meal starts.

`evaluate` re-checks each hash, checks it was sealed before the meal started,
and compares the forecast and two baselines with the actual count.

Only data from before the forecast time is used for training (no leakage).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .dishes import DishBook
from .io import read_table, write_table
from .timeutil import local_dt, now_local, parse_local

MODEL_VERSION = "shrunk-rate-v1"


def SYN_OR_INF(rec: dict) -> str:
    """A forecast made on a simulated clock is synthetic, not an inference from measurement."""
    return "synthetic" if rec.get("simulated_time") else "inferred"


# ----------------------------------------------------------------------------- history
def build_history(cfg: Config) -> pd.DataFrame:
    """date, slot, start_dt, count, subscribers, special, source  (one row per past meal)."""
    rows = []
    hist = read_table(cfg, "attendance_history")
    slots = cfg.meal_slots
    for _, r in hist.iterrows():
        rows.append(dict(date=r["date"], slot=r["slot"],
                         start_dt=local_dt(cfg, r["date"], slots.get(r["slot"], {}).get("start", "12:00")),
                         count=float(r["count"]),
                         subscribers=float(r["subscriber_count"]) if pd.notna(r["subscriber_count"]) else np.nan,
                         special=int(r["special_flag"]), source=r["source"]))
    mm = read_table(cfg, "meal_metrics")
    meals = read_table(cfg, "meals")
    if not mm.empty:
        sp = dict(zip(meals["meal_id"], meals["special_flag"]))
        for _, r in mm[mm["footfall"].notna()].iterrows():
            rows.append(dict(date=r["date"], slot=r["slot"], start_dt=pd.Timestamp(r["start_dt"]).tz_convert(cfg.tz),
                             count=float(r["footfall"]), subscribers=r["subscribers"],
                             special=int(sp.get(r["meal_id"], 0)), source=r["source"]))
    h = pd.DataFrame(rows, columns=["date", "slot", "start_dt", "count", "subscribers", "special", "source"])
    if h.empty:
        return h
    # observed rows win over historical rows for the same date+slot
    h["_rank"] = h["source"].map({"observed": 0, "demo": 0, "synthetic": 0,
                                  "reported": 1, "historical": 2}).fillna(3)
    h = h.sort_values("_rank").drop_duplicates(["date", "slot"]).drop(columns="_rank")
    h = h.sort_values("start_dt").reset_index(drop=True)

    # Transcription errors (a digit added, a column shifted) are common in hand-kept
    # registers and wreck both the level and the backtest spread. Drop counts far off
    # the median for their slot and say which ones went.
    # Only drop the physically impossible. A genuinely quiet day (rain, an exam block)
    # is real signal and must survive; a count above the subscriber list, or several
    # times the slot median, is a transcription error.
    keep = pd.Series(True, index=h.index)
    for slot, g in h.groupby("slot"):
        med = g["count"].median()
        if med > 0:
            keep.loc[g.index[g["count"] > 3 * med]] = False
    subs = h["subscribers"]
    keep.loc[h.index[subs.notna() & (h["count"] > 1.5 * subs)]] = False
    dropped = h[~keep]
    if len(dropped):
        for _, r in dropped.iterrows():
            print(f"  ! dropped implausible history: {r['date']} {r['slot']} count={r['count']:.0f} "
                  f"(far from the median for that slot)")
    return h[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------- model
@dataclass
class Pred:
    point: float
    mode: str
    base: float
    f_weekday: float
    f_special: float
    n_train: int


def _shrunk_ratio(sub: pd.Series, allv: pd.Series, w_sub, w_all, k: float) -> float:
    if len(sub) == 0 or len(allv) == 0:
        return 1.0
    m_all = np.average(allv, weights=w_all)
    if m_all <= 0:
        return 1.0
    r = np.average(sub, weights=w_sub) / m_all
    n = len(sub)
    return float((n * r + k * 1.0) / (n + k))


def predict_one(cfg: Config, hist: pd.DataFrame, slot: str, when: pd.Timestamp, subscribers: float | None,
                special: int) -> Pred | None:
    fc = cfg.section("forecast")
    hl = float(fc.get("half_life_days", 7))
    k = float(fc.get("shrink_k", 3))
    h = hist[(hist["slot"] == slot) & (hist["start_dt"] < when)]
    if h.empty:
        return None
    use_rate = subscribers is not None and not np.isnan(subscribers) and h["subscribers"].notna().sum() >= 2
    if use_rate:
        h = h[h["subscribers"].notna() & (h["subscribers"] > 0)]
        y = h["count"] / h["subscribers"]
    else:
        y = h["count"]
    age = (when - h["start_dt"]).dt.total_seconds() / 86400.0
    w = np.power(0.5, age / hl)
    # base level and weekday effect come from ordinary meals; special meals get their own factor
    normal = (h["special"] == 0).values
    if normal.sum() == 0:
        normal = np.ones(len(h), dtype=bool)
    yn, wn = y[normal], w[normal]
    base = float(np.average(yn, weights=wn))
    wd = (h["start_dt"].dt.dayofweek == when.dayofweek).values & normal
    f_wd = _shrunk_ratio(y[wd], yn, w[wd], wn, k)
    f_sp = 1.0
    s1 = (h["special"] == 1).values
    if special and s1.any() and normal.any():
        f_sp = _shrunk_ratio(y[s1], yn, w[s1], wn, k)
        f_wd = 1.0  # special meals are rare; don't stack a weekday effect on top
    val = base * f_wd * f_sp
    point = val * subscribers if use_rate else val
    return Pred(point=float(point), mode="rate" if use_rate else "count", base=base, f_weekday=f_wd,
                f_special=f_sp, n_train=int(len(h)))


def backtest(cfg: Config, hist: pd.DataFrame, slot: str, before: pd.Timestamp) -> dict:
    """Rolling-origin errors of model and baselines on past meals of this slot."""
    fc = cfg.section("forecast")
    min_train = int(fc.get("min_train_points", 3))
    h = hist[(hist["slot"] == slot) & (hist["start_dt"] < before)].reset_index(drop=True)
    rel, am, ab1, ab2 = [], [], [], []
    for i in range(len(h)):
        if i < min_train:
            continue
        r = h.iloc[i]
        p = predict_one(cfg, h.iloc[:i], slot, r["start_dt"],
                        r["subscribers"] if pd.notna(r["subscribers"]) else None, int(r["special"]))
        if p is None or p.point <= 0:
            continue
        rel.append(r["count"] / p.point - 1)
        am.append(abs(r["count"] - p.point))
        if pd.notna(r["subscribers"]):
            ab1.append(abs(r["count"] - r["subscribers"]))
        b2 = _last_week(h.iloc[:i], slot, r["start_dt"])
        if b2 is not None:
            ab2.append(abs(r["count"] - b2))
    return dict(rel_errors=rel, mae_model=float(np.mean(am)) if am else np.nan,
                mae_subscribers=float(np.mean(ab1)) if ab1 else np.nan,
                mae_last_week=float(np.mean(ab2)) if ab2 else np.nan, n=len(am))


def _last_week(hist: pd.DataFrame, slot: str, when: pd.Timestamp) -> float | None:
    h = hist[(hist["slot"] == slot) & (hist["start_dt"] < when)]
    if h.empty:
        return None
    lw = h[(h["start_dt"].dt.dayofweek == when.dayofweek) &
           ((when - h["start_dt"]).dt.days.between(5, 9))]
    return float((lw if len(lw) else h).iloc[-1]["count"])


# ----------------------------------------------------------------------------- sealing
def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def seal(cfg: Config, record: dict) -> tuple[str, str]:
    body = {k: v for k, v in record.items() if k != "sha256"}
    digest = hashlib.sha256(_canon(body).encode()).hexdigest()
    record = dict(body, sha256=digest)
    fname = f"{record['meal_id']}__{record['forecast_id']}.json"
    path = cfg.paths.sealed / fname
    path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path), digest


def verify(record: dict) -> bool:
    body = {k: v for k, v in record.items() if k != "sha256" and not k.startswith("_")}
    return hashlib.sha256(_canon(body).encode()).hexdigest() == record.get("sha256")


def load_sealed(cfg: Config) -> list[dict]:
    seen, out = set(), []
    for d in cfg.paths.sealed_search_dirs():
        for p in sorted(d.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"  ! unreadable sealed file {p}")
                continue
            if rec.get("forecast_id") in seen:
                continue
            seen.add(rec.get("forecast_id"))
            rec["_path"] = str(p)
            out.append(rec)
    return out


# ----------------------------------------------------------------------------- commands
def make_forecasts(cfg: Config, meal_ids: list[str] | None = None, now: str | None = None,
                   verbose: bool = True) -> list[dict]:
    book = DishBook(cfg)
    fc = cfg.section("forecast")
    meals = read_table(cfg, "meals", required=True)
    subs = read_table(cfg, "subscriptions")
    foot = read_table(cfg, "footfall")
    dish = read_table(cfg, "dish_metrics")
    is_demo = meals["source"].isin(["demo", "synthetic"]).any()

    simulated = now is not None
    if simulated and not is_demo:
        raise ValueError("--now (fake clock) is only allowed on demo/synthetic data. "
                         "Real forecasts use the real time.")
    t_now = parse_local(cfg, now) if simulated else now_local(cfg)

    hist_all = build_history(cfg)
    hist = hist_all[hist_all["start_dt"] < t_now] if not hist_all.empty else hist_all
    S_of = subs.set_index("meal_id")["subscriber_count"].to_dict() if not subs.empty else {}
    counted = set(foot["meal_id"]) if not foot.empty else set()

    targets = meals.copy()
    targets["start_dt"] = [local_dt(cfg, d, t) for d, t in zip(targets["date"], targets["start_time"])]
    if meal_ids:
        targets = targets[targets["meal_id"].isin(meal_ids)]
    else:
        targets = targets[targets["start_dt"].map(lambda x: x is not None and x > t_now)]
        if not simulated:
            targets = targets[~targets["meal_id"].isin(counted)]
    if targets.empty:
        print("No upcoming meals to forecast. Add the next meals to meals.csv (and subscriptions.csv).")
        return []

    # consumption per diner per dish, from meals finished before now
    p_of: dict[str, float] = {}
    if not dish.empty and not hist.empty:
        mm = read_table(cfg, "meal_metrics")
        done_ids = set(mm.loc[mm["start_dt"].map(lambda x: pd.Timestamp(x) < t_now), "meal_id"])
        dd = dish[dish["meal_id"].isin(done_ids) & dish["p_kg_per_diner"].notna()]
        p_of = dd.groupby("dish")["p_kg_per_diner"].mean().to_dict()

    q_lo, q_hi = fc.get("interval_quantiles", [0.1, 0.9])
    records = []
    for _, t in targets.iterrows():
        when = t["start_dt"]
        if when <= t_now:
            print(f"  ! {t['meal_id']} already started at {when}; a forecast now would not count as sealed.")
        S = S_of.get(t["meal_id"])
        S = float(S) if S is not None and pd.notna(S) else None
        pred = predict_one(cfg, hist, t["slot"], when, S, int(t["special_flag"]))
        if pred is None:
            print(f"  ! no history for slot {t['slot']} before {when}; cannot forecast {t['meal_id']}")
            continue
        bt = backtest(cfg, hist, t["slot"], when)
        errs = bt["rel_errors"]
        if len(errs) >= int(fc.get("min_backtest_errors", 5)):
            lo_r, hi_r = np.quantile(errs, q_lo), np.quantile(errs, q_hi)
            interval_method = f"backtest quantiles ({len(errs)} past meals)"
        else:
            e = float(fc.get("default_rel_err", 0.15))
            lo_r, hi_r = -e, e
            interval_method = f"default +/-{e:.0%} (only {len(errs)} backtest errors)"
        lo, hi = max(pred.point * (1 + lo_r), 0), pred.point * (1 + hi_r)
        b2 = _last_week(hist, t["slot"], when)

        recs = []
        for d in book.parse_menu(t["menu_items"]):
            p = p_of.get(d)
            p_src = "measured"
            if p is None:
                p = book.spec(d).get("serving_kg")
                p_src = "config serving_kg"
            if p is None:
                continue
            recs.append(dict(dish=d, p_kg_per_diner=round(float(p), 4), p_source=p_src,
                             recommended_kg=round(float(hi * p), 2),
                             cook_for_all_subscribers_kg=round(float(S * p), 2) if S else None))
        fid = f"{t_now.strftime('%Y%m%dT%H%M%S')}_{t['slot']}"
        rec = dict(
            forecast_id=f"{t['meal_id']}_{fid}", meal_id=t["meal_id"], target_start=str(when),
            made_at=str(t_now), simulated_time=simulated, model_version=MODEL_VERSION,
            mode=pred.mode, predicted_footfall=round(pred.point, 1), lo=round(float(lo), 1), hi=round(float(hi), 1),
            interval_method=interval_method, subscribers=S,
            factors=dict(base=round(pred.base, 4), weekday=round(pred.f_weekday, 3), special=round(pred.f_special, 3)),
            train_meals=pred.n_train,
            baselines=dict(subscribers=S, last_week=b2),
            backtest=dict(n=bt["n"], mae_model=_r(bt["mae_model"]), mae_subscribers=_r(bt["mae_subscribers"]),
                          mae_last_week=_r(bt["mae_last_week"])),
            recommendations=recs,
            note="recommended_kg = upper forecast bound x consumption per diner, so the kitchen should not run out.",
        )
        path, digest = seal(cfg, rec)
        rec["sha256"], rec["_path"] = digest, path
        records.append(rec)
        if verbose:
            print(f"\nSEALED {t['meal_id']}  predicted {rec['predicted_footfall']:.0f} diners "
                  f"[{rec['lo']:.0f}-{rec['hi']:.0f}]  ({pred.mode} mode, {interval_method})")
            for r in recs:
                print(f"   cook {r['dish']}: {r['recommended_kg']} kg"
                      + (f"  (all-subscribers would be {r['cook_for_all_subscribers_kg']} kg)"
                         if r['cook_for_all_subscribers_kg'] else ""))
            print(f"   POST THIS PUBLICLY NOW -> {t['meal_id']} sha256:{digest}")
            print(f"   file: {path}  (download it / commit it; /kaggle/working is wiped when the session ends)")
    _write_forecast_table(cfg)
    return records


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def _write_forecast_table(cfg: Config) -> pd.DataFrame:
    rows = []
    for r in load_sealed(cfg):
        rows.append(dict(forecast_id=r["forecast_id"], meal_id=r["meal_id"], made_at=r["made_at"],
                         sha256=r.get("sha256"), predicted_footfall=r["predicted_footfall"], lo=r["lo"], hi=r["hi"],
                         mode=r["mode"], baseline_subscribers=r["baselines"].get("subscribers"),
                         baseline_last_week=r["baselines"].get("last_week"),
                         simulated_time=bool(r.get("simulated_time")),
                         source=SYN_OR_INF(r)))
    df = pd.DataFrame(rows)
    write_table(cfg, df, "forecasts")
    return df


def evaluate(cfg: Config, verbose: bool = True) -> pd.DataFrame:
    _write_forecast_table(cfg)
    mm = read_table(cfg, "meal_metrics")
    dish = read_table(cfg, "dish_metrics")
    F_of = dict(zip(mm["meal_id"], mm["footfall"])) if not mm.empty else {}
    start_of = dict(zip(mm["meal_id"], mm["start_dt"])) if not mm.empty else {}
    rows, rec_rows = [], []
    for r in load_sealed(cfg):
        mid = r["meal_id"]
        ok = verify(r)
        start = pd.Timestamp(r["target_start"]) if r.get("target_start") else (
            pd.Timestamp(start_of[mid]) if mid in start_of else None)
        before = bool(start is not None and pd.Timestamp(r["made_at"]) < start)
        actual = F_of.get(mid)
        actual = None if actual is None or pd.isna(actual) else float(actual)
        pf = float(r["predicted_footfall"])
        b1, b2 = r["baselines"].get("subscribers"), r["baselines"].get("last_week")
        rows.append(dict(
            forecast_id=r["forecast_id"], meal_id=mid, target_start=str(start) if start is not None else None,
            made_at=r["made_at"], hash_ok=ok, sealed_before_meal=before,
            simulated_time=bool(r.get("simulated_time")), actual_footfall=actual, predicted_footfall=pf,
            lo=r["lo"], hi=r["hi"],
            abs_error=abs(pf - actual) if actual is not None else np.nan,
            in_interval=(r["lo"] <= actual <= r["hi"]) if actual is not None else None,
            baseline_subscribers=b1, baseline_last_week=b2,
            baseline_subscribers_abs_error=abs(b1 - actual) if (actual is not None and b1 is not None) else np.nan,
            baseline_last_week_abs_error=abs(b2 - actual) if (actual is not None and b2 is not None) else np.nan,
            source=SYN_OR_INF(r)))
        if actual is not None and not dish.empty:
            dm = dish[dish["meal_id"] == mid].set_index("dish")
            for rc in r.get("recommendations", []):
                if rc["dish"] not in dm.index:
                    continue
                d = dm.loc[rc["dish"]]
                served, cooked, left = d["served_kg"], d["cooked_kg"], d["leftover_kg"]
                rec_rows.append(dict(forecast_id=r["forecast_id"], meal_id=mid, dish=rc["dish"],
                                     recommended_kg=rc["recommended_kg"], actual_cooked_kg=cooked,
                                     actual_served_kg=served, actual_leftover_kg=left,
                                     would_run_short_kg=max(served - rc["recommended_kg"], 0) if pd.notna(served) else np.nan,
                                     leftover_if_followed_kg=max(rc["recommended_kg"] - served, 0) if pd.notna(served) else np.nan,
                                     source=SYN_OR_INF(r)))
    ev = pd.DataFrame(rows)
    # latest valid forecast per meal is the one that counts
    if not ev.empty:
        ev = ev.sort_values("made_at")
        ev["counts"] = False
        valid = ev[ev["hash_ok"] & (ev["sealed_before_meal"] | ev["simulated_time"])]
        ev.loc[valid.groupby("meal_id").tail(1).index, "counts"] = True
        ev = ev.sort_values(["target_start", "made_at"]).reset_index(drop=True)
    write_table(cfg, ev, "forecast_eval")
    write_table(cfg, pd.DataFrame(rec_rows), "recommendation_eval")
    if verbose:
        if ev.empty:
            print("No sealed forecasts found.")
        else:
            cols = ["meal_id", "hash_ok", "sealed_before_meal", "actual_footfall", "predicted_footfall",
                    "abs_error", "baseline_subscribers_abs_error", "baseline_last_week_abs_error", "in_interval", "counts"]
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                print(ev[cols].round(1).to_string(index=False))
            bad = ev[~ev["hash_ok"]]
            if len(bad):
                print(f"  ! {len(bad)} sealed files were modified after sealing; they are excluded.")
    return ev
