"""Waste decomposition.

For each meal:
  F  footfall (sum of 15-min tallies)       S  subscribers (manager aggregate)
  per dish: p = (cooked - leftover) / F      consumption per diner
            ghost overproduction = min(leftover, p * max(S - F, 0))
            other overproduction = leftover - ghost
            plate waste          = bin net weight x dish share from plate photos
  implied headcount = F * cooked / served   (who the kitchen effectively cooked for)

Ghost overproduction assumes the kitchen cooks for subscribers. Compare
implied_headcount with S and F to check that assumption before claiming it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .dishes import DishBook
from .io import read_table, write_table
from .timeutil import local_dt
from .validation import human_consensus


def _nan(x):
    return float("nan") if x is None else x


def plate_fractions(cfg: Config) -> pd.DataFrame:
    """Per frame/dish fraction left, choosing human consensus or model per config."""
    mdl = read_table(cfg, "plate_labels_model")
    how = cfg.section("analysis").get("attribution_labels", "best")
    if not mdl.empty:
        # one model per frame: take the most recently used model_id
        latest = mdl.sort_values("labelled_at").groupby("frame_id")["model_id"].last()
        mdl = mdl[mdl["model_id"] == mdl["frame_id"].map(latest)]
        mdl = mdl[["frame_id", "meal_id", "dish", "fraction_left", "plate_present"]].assign(label_src="model")
    if how == "best":
        h = human_consensus(read_table(cfg, "human_labels"))
        if not h.empty:
            frames = read_table(cfg, "camera_frames")
            h = h.merge(frames[["frame_id", "meal_id"]], on="frame_id", how="left")
            h = h[["frame_id", "meal_id", "dish", "fraction_left", "plate_present"]].assign(label_src="human")
            if not mdl.empty:
                mdl = mdl[~mdl["frame_id"].isin(set(h["frame_id"]))]
                return pd.concat([h, mdl], ignore_index=True)
            return h
    return mdl if not mdl.empty else pd.DataFrame(
        columns=["frame_id", "meal_id", "dish", "fraction_left", "plate_present", "label_src"])


def analyze(cfg: Config, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    book = DishBook(cfg)
    meals = read_table(cfg, "meals", required=True)
    foot = read_table(cfg, "footfall")
    subs = read_table(cfg, "subscriptions")
    prod = read_table(cfg, "production")
    left = read_table(cfg, "leftovers")
    bins = read_table(cfg, "bin_weights")
    costs = read_table(cfg, "dish_costs")
    fr = plate_fractions(cfg)

    cost_of = {d: s.get("cost_per_kg_inr") for d, s in book.dishes.items()}
    if not costs.empty:
        cost_of.update(dict(zip(costs["dish"], costs["cost_per_kg_inr"])))

    F_of = foot.groupby("meal_id")["count"].sum().astype(float).to_dict() if not foot.empty else {}
    S_of = subs.set_index("meal_id")["subscriber_count"].astype(float).to_dict() if not subs.empty else {}
    bin_of = bins.set_index("meal_id")["net_kg"].astype(float).to_dict() if not bins.empty else {}

    dish_rows, meal_rows = [], []
    for _, m in meals.iterrows():
        mid = m["meal_id"]
        src = m["source"] if m["source"] in ("demo", "synthetic") else "inferred"
        menu = book.parse_menu(m["menu_items"])
        F, S = F_of.get(mid), S_of.get(mid)
        gaps = []
        if F is None:
            gaps.append("footfall")
        if S is None:
            gaps.append("subscribers")
        ghosts = max(S - F, 0.0) if (F is not None and S is not None) else None

        p_m = prod[prod["meal_id"] == mid] if not prod.empty else prod
        l_m = left[left["meal_id"] == mid] if not left.empty else left
        cooked = dict(zip(p_m["dish"], p_m["qty_kg"])) if len(p_m) else {}
        lefto = dict(zip(l_m["dish"], l_m["qty_kg"])) if len(l_m) else {}
        if not cooked:
            gaps.append("production")
        if not lefto:
            gaps.append("leftovers")
        plate_total = bin_of.get(mid)
        if plate_total is None:
            gaps.append("bin_weight")

        dishes = list(dict.fromkeys(menu + list(cooked) + list(lefto)))
        frm = fr[fr["meal_id"] == mid] if not fr.empty else fr
        frm_ok = frm[(frm["plate_present"] == 1) & frm["fraction_left"].notna()] if len(frm) else frm
        n_plates = int(frm_ok["frame_id"].nunique()) if len(frm_ok) else 0
        if n_plates == 0:
            gaps.append("plate_labels")

        # per-dish volume/overproduction
        drows = []
        for d in dishes:
            c, l = cooked.get(d), lefto.get(d)
            c = None if c is None or pd.isna(c) else float(c)
            l = None if l is None or pd.isna(l) else float(l)
            served = (c - l) if (c is not None and l is not None) else None
            p = served / F if (served is not None and F) else None
            ghost = min(l, p * ghosts) if (l is not None and p is not None and ghosts is not None) else None
            other = (l - ghost) if (l is not None and ghost is not None) else l
            implied = F * c / served if (F and c is not None and served) else None
            mf = frm_ok.loc[frm_ok["dish"] == d, "fraction_left"] if len(frm_ok) else pd.Series(dtype=float)
            drows.append(dict(meal_id=mid, dish=d, cooked_kg=_nan(c), leftover_kg=_nan(l), served_kg=_nan(served),
                              p_kg_per_diner=_nan(p), overprod_rate=_nan(l / c if (l is not None and c) else None),
                              implied_headcount=_nan(implied), ghost_overprod_kg=_nan(ghost),
                              other_overprod_kg=_nan(other), mean_fraction_left=float(mf.mean()) if len(mf) else np.nan,
                              cost_per_kg_inr=_nan(cost_of.get(d)), source=src))

        # attribute bin weight to dishes: share ~ mean fraction left x serving size
        method = "none"
        if plate_total is not None and drows:
            weights = {}
            for r in drows:
                mf = r["mean_fraction_left"]
                if np.isnan(mf):
                    continue
                serving = r["p_kg_per_diner"]
                if np.isnan(serving):
                    serving = book.spec(r["dish"]).get("serving_kg", np.nan)
                    method = "fraction x config serving"
                else:
                    method = "fraction x measured serving" if method == "none" else method
                if not np.isnan(serving):
                    weights[r["dish"]] = mf * serving
            tot = sum(weights.values())
            for r in drows:
                if tot > 0 and r["dish"] in weights:
                    r["plate_kg"] = plate_total * weights[r["dish"]] / tot
                    r["plate_attr_method"] = method
                else:
                    r["plate_kg"] = np.nan
                    r["plate_attr_method"] = "no labels" if tot == 0 else "not visible"
        else:
            for r in drows:
                r["plate_kg"], r["plate_attr_method"] = np.nan, "no bin weight" if plate_total is None else "no dishes"
        for r in drows:
            lk = 0 if np.isnan(r["leftover_kg"]) else r["leftover_kg"]
            pk = 0 if np.isnan(r["plate_kg"]) else r["plate_kg"]
            r["waste_inr"] = (lk + pk) * r["cost_per_kg_inr"] if not np.isnan(r["cost_per_kg_inr"]) else np.nan
        dish_rows += drows

        # meal level
        dm = pd.DataFrame(drows)

        def s(col):
            v = dm[col] if len(dm) else pd.Series(dtype=float)
            return float(v.sum()) if v.notna().any() else np.nan

        def cost(kgcol):
            if not len(dm) or dm[kgcol].isna().all():
                return np.nan
            return float((dm[kgcol].fillna(0) * dm["cost_per_kg_inr"].fillna(0)).sum())

        cooked_kg, left_kg = s("cooked_kg"), s("leftover_kg")
        served_kg = cooked_kg - left_kg if not (np.isnan(cooked_kg) or np.isnan(left_kg)) else np.nan
        ghost_kg, other_kg = s("ghost_overprod_kg"), s("other_overprod_kg")
        plate_kg = float(plate_total) if plate_total is not None else np.nan
        total = np.nansum([left_kg, plate_kg]) if not (np.isnan(left_kg) and np.isnan(plate_kg)) else np.nan
        mf_all = frm_ok["fraction_left"] if len(frm_ok) else pd.Series(dtype=float)
        plate_inr = np.nan
        if not np.isnan(plate_kg) and len(dm):
            if dm["plate_kg"].notna().any():
                plate_inr = cost("plate_kg")
            elif dm["cost_per_kg_inr"].notna().any():
                # no photo attribution: price the bin at the average dish cost of the meal
                plate_inr = plate_kg * float(dm["cost_per_kg_inr"].mean())
        ghost_inr, other_inr = cost("ghost_overprod_kg"), cost("other_overprod_kg")
        if np.isnan(ghost_inr) and not np.isnan(left_kg):
            other_inr = cost("leftover_kg")
        meal_rows.append(dict(
            meal_id=mid, date=m["date"], slot=m["slot"],
            start_dt=local_dt(cfg, m["date"], m["start_time"]),
            footfall=_nan(F), subscribers=_nan(S),
            show_up_rate=(F / S) if (F is not None and S) else np.nan,
            ghost_diners=_nan(ghosts), cooked_kg=cooked_kg, leftover_kg=left_kg, served_kg=served_kg,
            implied_headcount=(F * cooked_kg / served_kg) if (F and served_kg and not np.isnan(served_kg)) else np.nan,
            ghost_overprod_kg=ghost_kg, other_overprod_kg=other_kg if not np.isnan(other_kg) else left_kg,
            plate_waste_kg=plate_kg, total_waste_kg=total,
            waste_g_per_diner=(1000 * total / F) if (F and not np.isnan(total)) else np.nan,
            ghost_inr=ghost_inr, other_overprod_inr=other_inr, plate_inr=plate_inr,
            total_inr=np.nansum([ghost_inr, other_inr, plate_inr]) if not all(
                np.isnan(x) for x in [ghost_inr, other_inr, plate_inr]) else np.nan,
            camera_mean_fraction_left=float(mf_all.mean()) if len(mf_all) else np.nan,
            camera_plates=n_plates, gaps=",".join(gaps) or None, source=src))

    dish_df = pd.DataFrame(dish_rows)
    meal_df = pd.DataFrame(meal_rows)
    if not meal_df.empty:
        meal_df = meal_df.sort_values("start_dt").reset_index(drop=True)
    write_table(cfg, dish_df, "dish_metrics")
    write_table(cfg, meal_df, "meal_metrics")
    if verbose and not meal_df.empty:
        show = meal_df[["meal_id", "footfall", "subscribers", "implied_headcount", "ghost_overprod_kg",
                        "other_overprod_kg", "plate_waste_kg", "total_inr", "camera_plates", "gaps"]]
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(show.round(1).to_string(index=False))
    return meal_df, dish_df
