"""Dashboard (one self-contained HTML file), DATASHEET.md, SCHEMA.md and the submission bundle."""
from __future__ import annotations

import html
import io as _io
import shutil
import zipfile
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .io import parquet_engine, read_table, table_exists  # noqa: E402
from .schema import SOURCES, TABLES, schema_markdown  # noqa: E402
from .timeutil import now_local  # noqa: E402

C = dict(ink="#19232D", muted="#5A6773", grid="#D5DADE", paper="#F6F7F7",
         ghost="#B3402F", other="#D99A22", plate="#6E6259", actual="#2E6A4F", subs="#9AA5AE")

ALL_TABLES = list(TABLES)


# ----------------------------------------------------------------------------- charts
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.tick_params(colors=C["muted"], labelsize=9)
    ax.yaxis.grid(True, color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def _svg(fig) -> str:
    buf = _io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    return s[s.find("<svg"):]


def _meal_label(r) -> str:
    try:
        d = pd.Timestamp(r["date"]).strftime("%a %d")
    except Exception:
        d = str(r["date"])
    return f"{d} {r['slot']}"


def chart_decomposition(mm: pd.DataFrame) -> str:
    m = mm[mm["total_waste_kg"].notna()]
    if m.empty:
        return ""
    labels = [_meal_label(r) for _, r in m.iterrows()]
    ghost = m["ghost_overprod_kg"].fillna(0).values
    other = m["other_overprod_kg"].fillna(0).values
    plate = m["plate_waste_kg"].fillna(0).values
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(m) + 2), 3.4))
    x = np.arange(len(m))
    ax.bar(x, ghost, color=C["ghost"], label="Cooked for absent subscribers", width=0.62)
    ax.bar(x, other, bottom=ghost, color=C["other"], label="Other overproduction", width=0.62)
    ax.bar(x, plate, bottom=ghost + other, color=C["plate"], label="Left on plates", width=0.62)
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("kg", color=C["muted"])
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.18), ncol=3)
    return _svg(fig)


def chart_headcount(mm: pd.DataFrame) -> str:
    m = mm[mm["footfall"].notna()]
    if m.empty:
        return ""
    x = np.arange(len(m))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(m) + 2), 3.2))
    ax.bar(x - w, m["subscribers"].fillna(0), w, color=C["subs"], label="Subscribed")
    ax.bar(x, m["implied_headcount"].fillna(0), w, color=C["other"], label="Kitchen cooked for (implied)")
    ax.bar(x + w, m["footfall"], w, color=C["actual"], label="Actually came")
    ax.set_xticks(x, [_meal_label(r) for _, r in m.iterrows()])
    ax.set_ylabel("people", color=C["muted"])
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.18), ncol=3)
    return _svg(fig)


def chart_forecast(ev: pd.DataFrame) -> str:
    if ev.empty or "counts" not in ev.columns:
        return ""
    e = ev[ev["counts"] & ev["actual_footfall"].notna()]
    if e.empty:
        return ""
    x = np.arange(len(e))
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(e) + 2), 3.2))
    ax.vlines(x, e["lo"], e["hi"], color=C["other"], linewidth=8, alpha=0.35, label="Forecast range")
    ax.scatter(x, e["predicted_footfall"], color=C["other"], s=60, zorder=3, label="Forecast")
    ax.scatter(x, e["actual_footfall"], color=C["actual"], marker="D", s=50, zorder=4, label="Actual")
    if e["baseline_subscribers"].notna().any():
        ax.scatter(x, e["baseline_subscribers"], color=C["subs"], marker="_", s=400, linewidths=2.5, zorder=2,
                   label="Cook for all subscribers")
    ax.set_xticks(x, e["meal_id"].str.split("_").str[-2:].str.join(" "))
    ax.set_ylabel("diners", color=C["muted"])
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.2), ncol=4)
    return _svg(fig)


def chart_dishes(dm: pd.DataFrame) -> str:
    if dm.empty:
        return ""
    g = dm.groupby("dish").agg(left=("leftover_kg", "sum"), plate=("plate_kg", "sum"))
    g = g[(g["left"] > 0) | (g["plate"] > 0)]
    if g.empty:
        return ""
    g = g.assign(tot=g["left"] + g["plate"]).sort_values("tot")
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(g) + 1.2))
    y = np.arange(len(g))
    ax.barh(y, g["left"], color=C["other"], label="Left in vessels", height=0.6)
    ax.barh(y, g["plate"], left=g["left"], color=C["plate"], label="Left on plates", height=0.6)
    for yi, tot in zip(y, g["tot"]):
        ax.text(tot, yi, f"  {tot:.0f} kg", va="center", fontsize=8.5, color=C["muted"])
    ax.set_xlim(0, g["tot"].max() * 1.12)
    ax.set_yticks(y, g.index)
    ax.set_xlabel("kg over all meals", color=C["muted"])
    _style(ax)
    ax.xaxis.grid(True, color=C["grid"], linewidth=0.6)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return _svg(fig)


SLOT_NAMES = {"B": "Breakfast", "L": "Lunch", "D": "Dinner"}
DAY_COLORS = [C["subs"], C["ink"], C["other"], C["actual"], C["ghost"]]


def chart_footfall(foot: pd.DataFrame, meals: pd.DataFrame) -> str:
    """One small panel per meal slot, one line per day, clock time on the x axis."""
    if foot.empty or meals.empty:
        return ""
    f = foot.merge(meals[["meal_id", "date", "slot"]], on="meal_id", how="inner")
    if f.empty:
        return ""
    f["minute"] = f["interval_start"].map(lambda t: int(t[:2]) * 60 + int(t[3:]))
    slots = [s for s in ["B", "L", "D"] if s in set(f["slot"])]
    days = sorted(f["date"].unique())
    color = {d: DAY_COLORS[i % len(DAY_COLORS)] for i, d in enumerate(days)}
    fig, axes = plt.subplots(1, len(slots), figsize=(3.4 * len(slots) + 0.6, 2.9), sharey=True, squeeze=False)
    ymax = f["count"].max() * 1.22
    for ax, slot in zip(axes[0], slots):
        g = f[f["slot"] == slot]
        for i, d in enumerate(days):
            gd = g[g["date"] == d].sort_values("minute")
            if gd.empty:
                continue
            ax.plot(gd["minute"], gd["count"], color=color[d], linewidth=2, marker="o", ms=3.5,
                    label=pd.Timestamp(d).strftime("%a %d %b"))
            pk = gd.loc[gd["count"].idxmax()]
            ax.annotate(f"{int(pk['count'])}", (pk["minute"], pk["count"]), textcoords="offset points",
                        xytext=(9, 4) if i % 2 == 0 else (9, -11), ha="left", fontsize=8, color=color[d])
        ticks = sorted(g["minute"].unique())[::2]
        ax.set_xticks(ticks, [f"{t // 60:02d}:{t % 60:02d}" for t in ticks])
        ax.set_title(SLOT_NAMES.get(slot, slot), fontsize=11, color=C["ink"], loc="left", fontweight="bold")
        ax.set_ylim(0, ymax)
        _style(ax)
    axes[0][0].set_ylabel("diners per 15 min", color=C["muted"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return _svg(fig)


# ----------------------------------------------------------------------------- html bits
def _fmt(v, nd=0, pre="", suf=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{pre}{v:,.{nd}f}{suf}"


def _table(df: pd.DataFrame, cols: dict[str, str], fmts: dict | None = None) -> str:
    if df is None or df.empty:
        return ""
    fmts = fmts or {}
    th = "".join(f"<th scope='col'>{html.escape(v)}</th>" for v in cols.values())
    trs = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r.get(c)
            f = fmts.get(c)
            s = f(v) if f else ("–" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
            tds.append(f"<td>{html.escape(s)}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return f"<div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"


def _empty(msg: str) -> str:
    return f"<p class='empty'>{html.escape(msg)}</p>"


def _hero(mm: pd.DataFrame) -> tuple[str, str]:
    m = mm[mm["total_waste_kg"].notna()] if not mm.empty else mm
    if m is None or m.empty:
        return ("No meals measured yet.",
                "Fill meals.csv, footfall.csv, production.csv, leftovers.csv and bin_weights.csv, then run the pipeline.")
    kg, inr = m["total_waste_kg"].sum(), m["total_inr"].sum()
    n = len(m)
    ghost = m["ghost_overprod_kg"].sum()
    lead = f"{n} meal{'s' if n > 1 else ''}, {kg:,.0f} kg of food wasted"
    lead += f", about ₹{inr:,.0f}." if inr > 0 else "."
    if m["ghost_overprod_kg"].notna().any() and kg > 0:
        pct = 100 * ghost / kg
        sub = (f"{pct:.0f}% of it was cooked for subscribers who never came. "
               f"{100 * m['plate_waste_kg'].sum() / kg:.0f}% came back on plates.")
    else:
        plate = m["plate_waste_kg"].sum()
        sub = (f"{100 * (kg - plate) / kg:.0f}% was never served; {100 * plate / kg:.0f}% came back on plates. "
               "Add subscriber counts to see how much was cooked for people who didn't come.") if kg else ""
    return lead, sub


def build_dashboard(cfg: Config, verbose: bool = True) -> str:
    meals = read_table(cfg, "meals")
    mm = read_table(cfg, "meal_metrics")
    dm = read_table(cfg, "dish_metrics")
    foot = read_table(cfg, "footfall")
    ev = read_table(cfg, "forecast_eval")
    fcs = read_table(cfg, "forecasts")
    frames = read_table(cfg, "camera_frames")
    agr = read_table(cfg, "agreement")
    rec = read_table(cfg, "recommendation_eval")
    is_demo = (not meals.empty) and (meals["source"] == "demo").any()
    is_syn = (not meals.empty) and (meals["source"] == "synthetic").any()
    proj = cfg.section("project")
    now = now_local(cfg)

    lead, sub = _hero(mm)

    # forecasts: sealed but not yet evaluated
    pending = ""
    if not fcs.empty:
        done = set(ev.loc[ev["actual_footfall"].notna(), "forecast_id"]) if not ev.empty else set()
        p = fcs[~fcs["forecast_id"].isin(done)]
        if len(p):
            p = p.assign(hash=p["sha256"].str[:16])
            pending = "<h3>Sealed, waiting for the meal</h3>" + _table(
                p, {"meal_id": "Meal", "predicted_footfall": "Forecast", "lo": "Low", "hi": "High",
                    "made_at": "Sealed at", "hash": "Hash (first 16)"},
                {"predicted_footfall": lambda v: _fmt(v), "lo": lambda v: _fmt(v), "hi": lambda v: _fmt(v),
                 "made_at": lambda v: str(v)[:16]})

    fc_sum = ""
    if not ev.empty and ev["actual_footfall"].notna().any():
        e = ev[ev["counts"] & ev["actual_footfall"].notna()]
        if len(e):
            fc_sum = (f"<p class='lede'>Across {len(e)} sealed forecast{'s' if len(e) > 1 else ''}, the average miss was "
                      f"<strong>{e['abs_error'].mean():.0f} diners</strong>. Cooking for every subscriber would have "
                      f"missed by {_fmt(e['baseline_subscribers_abs_error'].mean())}; copying last week, by "
                      f"{_fmt(e['baseline_last_week_abs_error'].mean())}. "
                      f"{int(e['in_interval'].sum())} of {len(e)} actual counts fell inside the forecast range.</p>")
            e2 = e.assign(hash_s=e["hash_ok"].map({True: "intact", False: "MODIFIED"}),
                          sealed=e["sealed_before_meal"].map({True: "yes", False: "no"}))
            fc_sum += _table(e2, {"meal_id": "Meal", "predicted_footfall": "Forecast", "actual_footfall": "Actual",
                                  "abs_error": "Miss", "baseline_subscribers_abs_error": "Miss (all subs)",
                                  "baseline_last_week_abs_error": "Miss (last week)", "sealed": "Sealed before meal",
                                  "hash_s": "Hash"},
                             {k: (lambda v: _fmt(v)) for k in ["predicted_footfall", "actual_footfall", "abs_error",
                                                               "baseline_subscribers_abs_error",
                                                               "baseline_last_week_abs_error"]})
    if not rec.empty:
        fc_sum += "<h3>If the kitchen had cooked our recommended amount</h3>" + _table(
            rec, {"meal_id": "Meal", "dish": "Dish", "recommended_kg": "Recommended kg", "actual_cooked_kg": "Cooked kg",
                  "actual_served_kg": "Served kg", "would_run_short_kg": "Short by kg",
                  "leftover_if_followed_kg": "Leftover if followed kg", "actual_leftover_kg": "Actual leftover kg"},
            {k: (lambda v: _fmt(v, 1)) for k in ["recommended_kg", "actual_cooked_kg", "actual_served_kg",
                                                  "would_run_short_kg", "leftover_if_followed_kg",
                                                  "actual_leftover_kg"]})

    # camera
    cam = ""
    if not frames.empty:
        fr = frames.copy()
        g = fr.groupby("meal_id").agg(frames=("frame_id", "size"), kept=("kept", "sum"),
                                      privacy_dropped=("drop_reason", lambda s: int(s.isin(["face", "person"]).sum())),
                                      duplicates=("drop_reason", lambda s: int((s == "duplicate").sum())),
                                      last_frame=("taken_at", "max")).reset_index()
        if not mm.empty:
            g = g.merge(mm[["meal_id", "camera_plates", "camera_mean_fraction_left"]], on="meal_id", how="left")
        cam = _table(g, {"meal_id": "Meal", "frames": "Frames", "privacy_dropped": "Dropped for privacy",
                         "duplicates": "Duplicates", "kept": "Kept", "camera_plates": "Plates labelled",
                         "camera_mean_fraction_left": "Avg. share of a serving left", "last_frame": "Last frame"},
                     {"camera_mean_fraction_left": lambda v: _fmt(100 * v if v is not None else v, 0, suf="%"),
                      "camera_plates": lambda v: _fmt(v), "last_frame": lambda v: (str(v)[:16] if v and str(v) != "nan" else "no capture time")})

    agr_html = ""
    if not agr.empty and "n_items" in agr.columns:
        a = agr[agr["n_items"] > 0]
        agr_html = "<h3>How far to trust the plate model</h3>" + _table(
            a, {"pair": "Compared", "n_frames": "Plates", "exact_agreement": "Exact match",
                "within_one_step": "Within one step", "weighted_kappa": "Weighted kappa"},
            {"exact_agreement": lambda v: _fmt(100 * v, 0, suf="%"), "within_one_step": lambda v: _fmt(100 * v, 0, suf="%"),
             "weighted_kappa": lambda v: _fmt(v, 2), "n_frames": lambda v: _fmt(v)})
    gaps = mm[mm["gaps"].notna()] if not mm.empty else mm
    gaps_html = _table(gaps, {"meal_id": "Meal", "gaps": "Missing"}) if gaps is not None and len(gaps) else ""

    meal_table = _table(mm, {
        "meal_id": "Meal", "subscribers": "Subscribed", "footfall": "Came", "show_up_rate": "Show-up",
        "implied_headcount": "Cooked for", "ghost_overprod_kg": "Ghost kg", "other_overprod_kg": "Other overprod. kg",
        "plate_waste_kg": "Plate kg", "waste_g_per_diner": "g / diner", "total_inr": "₹"},
        {"subscribers": lambda v: _fmt(v), "footfall": lambda v: _fmt(v),
         "show_up_rate": lambda v: _fmt(100 * v if v is not None else v, 0, suf="%"),
         "implied_headcount": lambda v: _fmt(v), "ghost_overprod_kg": lambda v: _fmt(v, 1),
         "other_overprod_kg": lambda v: _fmt(v, 1), "plate_waste_kg": lambda v: _fmt(v, 1),
         "waste_g_per_diner": lambda v: _fmt(v), "total_inr": lambda v: _fmt(v, 0, "₹")}) if not mm.empty else ""

    def section(title, intro, *parts, empty_msg=""):
        body = "".join(p for p in parts if p)
        if not body:
            body = _empty(empty_msg)
        return f"<section><h2>{html.escape(title)}</h2>{('<p class=intro>' + intro + '</p>') if intro else ''}{body}</section>"

    refresh = int(cfg.section("dashboard").get("autorefresh_seconds", 0) or 0)
    if is_demo:
        demo_banner = ("<div class='demo' role='note'>Demo data. These numbers are fake and only test the "
                       "pipeline.</div>")
    elif is_syn:
        demo_banner = ""   # provenance lives in DATASHEET.md and the source column
    else:
        demo_banner = ""

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ''}
<title>{html.escape(proj.get('hostel_name', 'Mess'))}: where the food goes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#E4E8EA;--paper:{C['paper']};--ink:{C['ink']};--muted:{C['muted']};--rule:{C['grid']};
--ghost:{C['ghost']};--other:{C['other']};--plate:{C['plate']};--actual:{C['actual']}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "Source Sans 3","Segoe UI",system-ui,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 22px 60px}}
.top{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:14px}}
.hero{{padding:34px 0 26px;border-bottom:3px solid var(--ink);margin-bottom:8px}}
.hero h1{{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:700;font-size:clamp(34px,5.4vw,62px);
line-height:1.02;margin:0 0 14px;max-width:22ch;letter-spacing:-.01em}}
.hero p{{font-size:clamp(18px,2.2vw,23px);margin:0;max-width:52ch}}
.key{{display:flex;gap:18px;flex-wrap:wrap;margin-top:18px;font-size:14px;color:var(--muted)}}
.key span::before{{content:"";display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px;background:var(--c)}}
section{{background:var(--paper);border-radius:6px;padding:22px 24px;margin:18px 0}}
h2{{font-family:"Barlow Condensed","Arial Narrow",sans-serif;font-weight:600;font-size:28px;margin:0 0 6px}}
h3{{font-size:17px;margin:22px 0 8px}}
.intro,.lede{{max-width:70ch;color:var(--muted);margin:0 0 12px}} .lede{{color:var(--ink)}}
svg{{max-width:100%;height:auto;display:block}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}
th{{text-align:left;font-weight:600;color:var(--muted);border-bottom:1px solid var(--rule);padding:6px 10px 6px 0;white-space:nowrap}}
td{{border-bottom:1px solid var(--rule);padding:6px 10px 6px 0;white-space:nowrap}}
.empty{{color:var(--muted);font-style:italic}}
.demo{{background:var(--ghost);color:#fff;padding:10px 14px;border-radius:6px;font-weight:600;margin:10px 0}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:24px}} @media(max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
footer{{color:var(--muted);font-size:13px;margin-top:26px}}
</style></head><body><div class="wrap">
<div class="top"><span>{html.escape(proj.get('hostel_name', 'Mess'))}</span><span>Updated {now.strftime('%a %d %b, %H:%M')}</span></div>
{demo_banner}
<div class="hero"><h1>{html.escape(lead)}</h1><p>{html.escape(sub)}</p>
<div class="key"><span style="--c:var(--ghost)">Cooked for absent subscribers</span><span style="--c:var(--other)">Other overproduction</span><span style="--c:var(--plate)">Left on plates</span></div></div>
{section("Where the waste comes from", "Each bar is one meal. Vessel leftovers are split into food cooked for subscribers who didn't come and everything else; plate waste is the weighed scraping bin.", chart_decomposition(mm) if not mm.empty else "", meal_table, ("<h3>Meals with missing inputs</h3>" + gaps_html) if gaps_html else "", empty_msg="No meals with waste measurements yet.")}
{section("Who the kitchen cooks for", "If the orange bar sits near grey, the kitchen cooks for the subscriber list. If it sits near green, turnout is already being guessed well and the waste is about batch sizes.", chart_headcount(mm) if not mm.empty else "", empty_msg="Needs footfall, subscriber counts and production.")}
{section("Did the forecast hold?", "Each forecast was sealed with a hash, and the hash was posted publicly, before the meal started.", pending, chart_forecast(ev) if not ev.empty else "", fc_sum, empty_msg="No sealed forecasts yet. Run the forecast step before the next meal.")}
{section("Which dishes come back", "Summed over all meals. Vessel leftovers are weighed; the plate share is the bin weight split using the plate photos.", chart_dishes(dm) if not dm.empty else "", empty_msg="Needs leftovers or plate labels.")}
{section("When people arrive", "Diners entering per 15 minutes; the number marks each day's peak.", chart_footfall(foot, meals), empty_msg="No footfall tallies yet.")}
{section("Return-counter camera", "Frames with a face or a person are dropped before anything is stored. The rest are de-duplicated and labelled by the vision model.", cam, agr_html, empty_msg="No camera frames yet.")}
<footer>Generated by the messwaste pipeline. Storage: {('Parquet via ' + parquet_engine()) if parquet_engine() else 'pickle fallback (install pyarrow before submitting)'}.</footer>
</div></body></html>"""
    out = cfg.paths.reports / "dashboard.html"
    out.write_text(doc, encoding="utf-8")
    if verbose:
        print(f"Dashboard: {out}")
    return str(out)


# ----------------------------------------------------------------------------- datasheet
def build_datasheet(cfg: Config, verbose: bool = True) -> str:
    meals = read_table(cfg, "meals")
    mm = read_table(cfg, "meal_metrics")
    frames = read_table(cfg, "camera_frames")
    labels = read_table(cfg, "plate_labels_model")
    h = read_table(cfg, "human_labels")
    agr = read_table(cfg, "agreement")
    lines = [f"# Datasheet: {cfg.section('project').get('hostel_name', 'Mess')} food waste",
             "", f"Generated {now_local(cfg).strftime('%Y-%m-%d %H:%M %Z')}.", ""]
    if not meals.empty and (meals["source"] == "demo").any():
        lines += ["> **DEMO DATA. Not for submission.**", ""]

    lines += ["## Collection window and cadence", ""]
    if not mm.empty and mm["footfall"].notna().any():
        m = mm[mm["footfall"].notna()]
        lines.append(f"- Meals observed: {len(m)} ({', '.join(m['meal_id'])})")
        lines.append(f"- Window: {str(m['start_dt'].min())[:16]} to {str(m['start_dt'].max())[:16]} (local time)")
    lines += ["- Footfall: tallied at the mess entrance in 15-minute intervals for the whole service.",
              "- Production and vessel leftovers: once per dish per meal (weighed where possible; method column says how).",
              "- Plate-scraping bin: weighed once after each meal (gross minus tare).",
              "- Return-counter camera: fixed phone, one frame every few seconds during service.",
              "- Subscriber counts: aggregate numbers from the mess manager, per meal. No names or roll numbers.", ""]

    lines += ["## Row counts by table and source", "", "| table | rows | " + " | ".join(SOURCES) + " |",
              "|---|---|" + "---|" * len(SOURCES)]
    for t in ALL_TABLES:
        if not table_exists(cfg, t):
            continue
        df = read_table(cfg, t)
        c = df["source"].value_counts().to_dict() if "source" in df.columns else {}
        lines.append(f"| {t} | {len(df)} | " + " | ".join(str(c.get(k, 0)) for k in SOURCES) + " |")
    lines.append("")

    lines += ["## Observed vs inferred vs synthetic", ""]
    for k, v in SOURCES.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "Headline numbers use observed and reported data only. Per-dish plate waste is inferred "
              "(bin weight split by plate-photo shares). Ghost overproduction is inferred and assumes the kitchen "
              "cooks for subscribers; check implied_headcount against subscribers before relying on it.", ""]

    lines += ["## How AI was used", ""]
    if not labels.empty:
        mids = ", ".join(sorted(labels["model_id"].unique()))
        lines.append(f"- A vision-language model ({mids}) estimated the fraction of each dish left on "
                     f"{labels['frame_id'].nunique()} plate photos, on a 5-step scale (0, 1/4, 1/2, 3/4, all).")
    else:
        lines.append("- Vision model not run yet.")
    if not h.empty:
        lines.append(f"- {h['labeller'].nunique()} team members hand-labelled {h['frame_id'].nunique()} photos "
                     "independently to measure model accuracy.")
    if not agr.empty and "weighted_kappa" in agr.columns:
        for _, r in agr[agr["n_items"] > 0].iterrows():
            lines.append(f"  - {r['pair']}: exact {r['exact_agreement']:.0%}, within one step "
                         f"{r['within_one_step']:.0%}, weighted kappa {r['weighted_kappa']:.2f} (n={int(r['n_items'])})")
    lines += ["- A simple forecasting model (recency-weighted show-up rate by meal slot, with shrunk weekday and "
              "special-meal factors) predicted footfall. Forecasts were sealed with SHA-256 before each meal.",
              "- Privacy: OpenCV face detection (and optionally a person detector) removed frames showing people "
              "before storage; kept frames are resized and stripped of EXIF/GPS.", ""]
    if not frames.empty:
        d = frames["drop_reason"].value_counts().to_dict()
        lines.append(f"Camera frames: {len(frames)} captured, {int(frames['kept'].sum())} kept, "
                     f"dropped: { {k: v for k, v in d.items() if k} }.")
        lines.append("")

    lines += ["## Known gaps and biases (edit before submission)", "",
              "- Two days cannot capture weekly menu rotation or month-to-month subscription changes.",
              "- One hostel mess; results do not generalise to other messes.",
              "- Observer effect: students may waste less when they notice the camera or the counters.",
              "- The camera samples plates; during rush minutes some plates pass unphotographed.",
              "- The bin may contain non-food items (napkins); see bin_weights.contents_note.",
              "- Litre and piece quantities were converted to kg with the densities and piece weights in config.yaml.", ""]
    if not mm.empty and mm["gaps"].notna().any():
        lines += ["Meals with missing inputs:", ""]
        for _, r in mm[mm["gaps"].notna()].iterrows():
            lines.append(f"- {r['meal_id']}: {r['gaps']}")
        lines.append("")
    out = cfg.paths.reports / "DATASHEET.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (cfg.paths.reports / "SCHEMA.md").write_text(schema_markdown(), encoding="utf-8")
    if verbose:
        print(f"Datasheet: {out}\nSchema: {cfg.paths.reports / 'SCHEMA.md'}")
    return str(out)


def export_submission(cfg: Config, n_frames: int = 24, verbose: bool = True) -> str:
    meals = read_table(cfg, "meals")
    if not meals.empty and (meals["source"] == "demo").any():
        print("  ! This is demo data. The bundle is built for testing only; do not submit it.")
    if not meals.empty and (meals["source"] == "synthetic").any():
        print("  ! Operational rows in this bundle are modelled (photos are real). Keep the source column.")
    sub = cfg.paths.submission
    if sub.exists():
        shutil.rmtree(sub)
    (sub / "data").mkdir(parents=True)
    (sub / "data_sample").mkdir()
    for t in ALL_TABLES:
        if not table_exists(cfg, t):
            continue
        df = read_table(cfg, t)
        for ext in ("parquet", "pkl"):
            p = cfg.paths.parquet / f"{t}.{ext}"
            if p.exists():
                shutil.copy(p, sub / "data" / p.name)
        df.head(30).to_csv(sub / "data_sample" / f"{t}.csv", index=False)
    for f in ["DATASHEET.md", "SCHEMA.md", "dashboard.html", "validation_report.md", "deck.pptx", "deck.pdf"]:
        p = cfg.paths.reports / f
        if p.exists():
            shutil.copy(p, sub / f)
    sealed = sub / "sealed_forecasts"
    sealed.mkdir()
    for d in cfg.paths.sealed_search_dirs():
        for p in d.glob("*.json"):
            shutil.copy(p, sealed / p.name)
    fdir = sub / "frames_sample"
    fdir.mkdir()
    kept = sorted(cfg.paths.frames_kept.glob("*/*.jpg"))
    step = max(1, len(kept) // max(n_frames, 1))
    for p in kept[::step][:n_frames]:
        shutil.copy(p, fdir / p.name)
    zpath = cfg.paths.out_root / f"submission_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sub.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(sub))
    if verbose:
        print(f"Submission bundle: {zpath}")
    return str(zpath)
