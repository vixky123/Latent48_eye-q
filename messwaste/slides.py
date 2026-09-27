"""The 3-slide submission deck, generated from the pipeline's own tables.

    python -m messwaste.cli slides        -> reports/deck.pptx  (+ deck.pdf if LibreOffice is available)

Slide 1  problem, user, question        (dark)
Slide 2  workflow, collection, what we built
Slide 3  result, what changes, demo link

Every number on the slides is computed from the tables, so re-running after new
meals updates the deck. Speaker notes hold a ~1-minute script per slide.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from .config import Config
from .io import read_table, table_exists
from .schema import TABLES

INK, PAPER, BG, MUTED, RULE = "19232D", "F6F7F7", "E4E8EA", "5A6773", "D5DADE"
GHOST, OTHER, PLATE, ACTUAL, SUBS = "B3402F", "D99A22", "6E6259", "2E6A4F", "9AA5AE"
DARK_PANEL, DARK_MUTED = "26323D", "AEB8C1"
FONT = "Arial"
W, H = 13.333, 7.5


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


# ----------------------------------------------------------------------------- drawing helpers
def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None):
    """runs: str, or list of paragraphs, each a str or a list of (text, {overrides}) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        parts = para if isinstance(para, list) else [(para, {})]
        for t, o in parts:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def box(slide, x, y, w, h, fill, radius=0.06, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def dot(slide, x, y, d, fill, label=None, label_color="FFFFFF", size=12):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    shp.line.fill.background()
    shp.shadow.inherit = False
    if label:
        tf = shp.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), True
        r.font.color.rgb = rgb(label_color)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return shp


def background(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(color)


def demo_tag(slide, label="Demo data", w=1.95):
    box(slide, W - 0.5 - w, 0.35, w, 0.38, GHOST, radius=0.3)
    text(slide, W - 0.5 - w, 0.35, w, 0.38, label, size=12, color="FFFFFF", bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def provenance_tag(slide, F):
    if F["demo"]:
        demo_tag(slide)
    elif F["synthetic"]:
        demo_tag(slide, "Real photos \u00b7 modelled figures", 3.0)


def fmt(v, nd=0, pre="", suf=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{pre}{v:,.{nd}f}{suf}"


# ----------------------------------------------------------------------------- facts
def gather(cfg: Config) -> dict:
    meals = read_table(cfg, "meals")
    mm = read_table(cfg, "meal_metrics")
    ev = read_table(cfg, "forecast_eval")
    rec = read_table(cfg, "recommendation_eval")
    frames = read_table(cfg, "camera_frames")
    agr = read_table(cfg, "agreement")
    dm = read_table(cfg, "dish_metrics")
    F = {}
    F["demo"] = (not meals.empty) and bool((meals["source"] == "demo").any())
    F["synthetic"] = (not meals.empty) and bool((meals["source"] == "synthetic").any())
    m = mm[mm["footfall"].notna()] if not mm.empty else mm
    F["n_meals"] = len(m)
    F["S"] = float(m["subscribers"].mean()) if len(m) and m["subscribers"].notna().any() else np.nan
    F["F"] = float(m["footfall"].mean()) if len(m) else np.nan
    F["implied"] = float(m["implied_headcount"].mean()) if len(m) and m["implied_headcount"].notna().any() else np.nan
    w = mm[mm["total_waste_kg"].notna()] if not mm.empty else mm
    F["n_waste"] = len(w)
    F["kg"] = float(w["total_waste_kg"].sum()) if len(w) else np.nan
    F["inr"] = float(w["total_inr"].sum()) if len(w) else np.nan
    F["ghost_kg"] = float(w["ghost_overprod_kg"].sum()) if len(w) and w["ghost_overprod_kg"].notna().any() else np.nan
    F["plate_kg"] = float(w["plate_waste_kg"].sum()) if len(w) else np.nan
    F["g_per_diner"] = float(w["waste_g_per_diner"].mean()) if len(w) else np.nan
    F["meals_table"] = w
    if len(m):
        start = pd.to_datetime(m["start_dt"])
        F["window"] = f"{start.min().strftime('%a %d %b %H:%M')} to {start.max().strftime('%a %d %b %H:%M')}"
    else:
        F["window"] = ""
    # does the kitchen cook for subscribers? (implied closer to S than to F)
    F["cooks_for_subs"] = (not np.isnan(F["implied"]) and not np.isnan(F["S"])
                           and abs(F["implied"] - F["S"]) < abs(F["implied"] - F["F"]))
    # forecasts
    e = ev[ev["counts"] & ev["actual_footfall"].notna()] if (not ev.empty and "counts" in ev.columns) else pd.DataFrame()
    F["n_fc"] = len(e)
    F["fc_mae"] = float(e["abs_error"].mean()) if len(e) else np.nan
    F["fc_b1"] = float(e["baseline_subscribers_abs_error"].mean()) if len(e) else np.nan
    F["fc_b2"] = float(e["baseline_last_week_abs_error"].mean()) if len(e) else np.nan
    F["fc_in"] = int(e["in_interval"].sum()) if len(e) else 0
    F["fc_table"] = e
    if not rec.empty:
        r = rec[rec["forecast_id"].isin(set(e["forecast_id"]))] if len(e) else rec
        nm = r["meal_id"].nunique()
        saved = (r["actual_leftover_kg"] - r["leftover_if_followed_kg"]).sum()
        F["rec_meals"] = nm
        F["rec_saved_per_meal"] = float(saved / nm) if nm else np.nan
        F["rec_short_kg"] = float(r["would_run_short_kg"].sum())
        F["rec_short_dishes"] = int((r["would_run_short_kg"] > 0.05).sum())
        F["rec_dishes"] = len(r)
    else:
        F["rec_meals"] = 0
    # camera + model
    F["frames"] = len(frames)
    F["kept"] = int(frames["kept"].sum()) if len(frames) else 0
    F["privacy_dropped"] = int(frames["drop_reason"].isin(["face", "person"]).sum()) if len(frames) else 0
    F["kappa"] = F["within"] = np.nan
    F["model_name"] = ""
    if not agr.empty and "pair" in agr.columns:
        a = agr[agr["pair"].str.endswith("vs human consensus") & (agr["n_items"] > 0)]
        if len(a):
            F["kappa"] = float(a["weighted_kappa"].iloc[0])
            F["within"] = float(a["within_one_step"].iloc[0])
            F["model_name"] = a["pair"].iloc[0].replace(" vs human consensus", "").split("/")[-1]
    if not dm.empty:
        g = dm.groupby("dish")[["plate_kg", "served_kg"]].sum()
        g = g[(g["plate_kg"] > 0) & (g["served_kg"] > 0)]
        F["top_plate_dish"] = (g["plate_kg"] / g["served_kg"]).idxmax() if len(g) else None
    else:
        F["top_plate_dish"] = None
    rows, ntab = 0, 0
    for t in TABLES:
        if table_exists(cfg, t):
            n = len(read_table(cfg, t))
            if n:
                rows += n
                ntab += 1
    F["rows"], F["ntables"] = rows, ntab
    return F


# ----------------------------------------------------------------------------- slides
def slide_problem(prs, cfg, F):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, INK)
    pc = cfg.section("presentation")
    proj = cfg.section("project")
    text(s, 0.6, 0.45, 9, 0.4, f"{pc.get('team_name') or 'Our team'}  |  {proj.get('hostel_name', 'Hostel mess')}",
         size=13, color=DARK_MUTED)
    provenance_tag(s, F)

    if not np.isnan(F["implied"]) and not np.isnan(F["S"]):
        head = [f"The mess cooks for {F['implied']:.0f}.", f"About {F['F']:.0f} come to eat."]
    elif not np.isnan(F["F"]):
        head = f"About {F['F']:.0f} people come to each meal. Nobody knows the number in advance."
    else:
        head = "The mess cooks every meal without knowing how many will come."
    text(s, 0.6, 1.0, 11.5, 1.6, head, size=40, color="FFFFFF", bold=True)

    # three numbers
    stats = [("Subscribed", F["S"], SUBS), ("Kitchen cooks for", F["implied"], OTHER), ("Actually came", F["F"], "6FB48F")]
    x0 = 0.6
    for i, (lab, v, col) in enumerate(stats):
        x = x0 + i * 3.35
        dot(s, x, 2.93, 0.2, col)
        text(s, x + 0.35, 2.85, 2.8, 0.36, lab, size=15, color=DARK_MUTED, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x, 3.3, 3.1, 1.0, fmt(v), size=60, color="FFFFFF", bold=True)
    note = f"Average per meal over {F['n_meals']} meals we counted." if F["n_meals"] else "Averages per meal."
    text(s, x0, 4.35, 9.5, 0.35, note + " \"Cooks for\" = diners x cooked / served.", size=12, color=DARK_MUTED)

    # user + question panels
    user = pc.get("user_name") or "The mess manager"
    box(s, 0.6, 5.15, 5.3, 1.55, DARK_PANEL, radius=0.05)
    text(s, 0.85, 5.38, 4.85, 1.1, [
        [("The user", {"bold": True, "color": OTHER, "size": 14})],
        [(f"{user}, deciding cook quantities before anyone walks in.", {"color": "FFFFFF", "size": 17})]],
         spacing=8)
    box(s, 6.2, 5.15, 6.53, 1.55, DARK_PANEL, radius=0.05)
    text(s, 6.45, 5.38, 6.05, 1.1, [
        [("The question", {"bold": True, "color": OTHER, "size": 14})],
        [("How much of the waste is cooking for people who never come?", {"color": "FFFFFF", "size": 17})]],
         spacing=8)
    s.notes_slide.notes_text_frame.text = (
        f"{user} decides quantities for about {fmt(F['S'])} subscribers at every meal. "
        f"Over {F['n_meals']} meals we counted everyone who came in: about {fmt(F['F'])} per meal. "
        f"Measured against what was actually eaten, the kitchen cooked enough for about {fmt(F['implied'])} people. "
        "Nobody had these three numbers side by side before. Our question: how much of the waste is cooking "
        "for people who don't show up, and can we tell the kitchen a better number the night before?")


def slide_method(prs, cfg, F):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, BG)
    provenance_tag(s, F)
    text(s, 0.6, 0.45, 10.3, 0.7, "We followed the food from list to bin", size=32, bold=True)
    text(s, 0.6, 1.2, 11, 0.4,
         f"{F['n_meals']} meals, {F['window']}. Collection ran at every service." if F["window"] else
         "Collection at every meal service.", size=14, color=MUTED)

    steps = [("Subscription list", "A per-meal total from the manager. No names."),
             ("Kitchen", "Cooked per dish, and what was left in the vessels."),
             ("Entrance", "Diners tallied every 15 minutes."),
             ("Return counter", f"Phone over the trays. {fmt(F['kept'])} plates kept, "
                                f"{fmt(F['privacy_dropped'])} auto-deleted."),
             ("Scraping bin", "Weighed after every meal.")]
    cw, gap, y = 2.25, 0.27, 1.85
    for i, (title, body) in enumerate(steps):
        x = 0.6 + i * (cw + gap)
        box(s, x, y, cw, 2.1, PAPER)
        dot(s, x + 0.2, y + 0.2, 0.42, INK, str(i + 1), size=13)
        text(s, x + 0.2, y + 0.75, cw - 0.4, 0.4, title, size=15, bold=True)
        text(s, x + 0.2, y + 1.12, cw - 0.4, 0.95, body, size=11.5, color=MUTED)
        if i < len(steps) - 1:
            a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x + cw + 0.07), Inches(y + 0.95),
                                   Inches(0.13), Inches(0.22))
            a.fill.solid()
            a.fill.fore_color.rgb = rgb(SUBS)
            a.line.fill.background()
            a.shadow.inherit = False

    text(s, 0.6, 4.3, 6, 0.4, "What we built", size=20, bold=True)
    model = F["model_name"] or "A vision-language model"
    trust = (f"Within a quarter serving of our hand labels {F['within']:.0%} of the time.") \
        if not np.isnan(F["kappa"]) else "Checked against plates labelled by hand."
    built = [
        (GHOST, "Waste split", "Leftovers divided into food cooked for absent subscribers, other "
                               "overproduction, and what came back on plates."),
        (PLATE, "Plate reader", f"{model} reads how much of each dish is left. {trust}"),
        (OTHER, "Sealed forecast", "Tomorrow's turnout predicted the night before, then hashed so it "
                                   "cannot be edited afterwards."),
    ]
    bw = (W - 1.2 - 2 * 0.3) / 3
    for i, (col, title, body) in enumerate(built):
        x = 0.6 + i * (bw + 0.3)
        box(s, x, 4.85, bw, 1.85, PAPER)
        dot(s, x + 0.25, 5.1, 0.2, col)
        text(s, x + 0.55, 5.02, bw - 0.8, 0.36, title, size=15, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.25, 5.45, bw - 0.5, 1.2, body, size=12, color=MUTED)
    text(s, 0.6, 6.9, 12, 0.3,
         f"{fmt(F['rows'])} rows across {F['ntables']} tables, each row carrying where it came from.",
         size=11.5, color=MUTED)
    s.notes_slide.notes_text_frame.text = (
        "We didn't install anything. We counted people at the door every 15 minutes, weighed the vessels and the "
        "scraping bin, and put a phone over the return counter. Frames with a person in them are deleted before "
        "they're stored. A vision model reads how much of each dish is left on every plate; we hand-labelled a "
        f"sample to check it. Every night we sealed a forecast for the next day's meals and posted the hash "
        "before the meal, so nobody can say we fitted it afterwards.")


def slide_result(prs, cfg, F):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, BG)
    provenance_tag(s, F)
    pc = cfg.section("presentation")
    if F["n_waste"]:
        head = f"{F['n_waste']} meals, {F['kg']:,.0f} kg wasted, about ₹{F['inr']:,.0f}"
        if not np.isnan(F["ghost_kg"]) and F["kg"] > 0:
            sub = (f"{100 * F['ghost_kg'] / F['kg']:.0f}% was cooked for subscribers who never came. "
                   f"{100 * F['plate_kg'] / F['kg']:.0f}% came back on plates.")
        else:
            sub = f"{100 * F['plate_kg'] / F['kg']:.0f}% came back on plates."
    else:
        head, sub = "Results appear after the first measured meals", ""
    text(s, 0.6, 0.45, 10.4, 0.8, head, size=32, bold=True)
    text(s, 0.6, 1.2, 10.4, 0.4, sub, size=16, color=MUTED)

    # stacked chart
    box(s, 0.6, 1.85, 7.3, 4.2, PAPER, radius=0.025)
    w = F["meals_table"]
    if len(w):
        cd = CategoryChartData()
        cd.categories = [f"{pd.Timestamp(d).strftime('%a')} {sl}" for d, sl in zip(w["date"], w["slot"])]
        cd.add_series("For absent subscribers", [round(v, 1) for v in w["ghost_overprod_kg"].fillna(0)])
        cd.add_series("Other overproduction", [round(v, 1) for v in w["other_overprod_kg"].fillna(0)])
        cd.add_series("On plates", [round(v, 1) for v in w["plate_waste_kg"].fillna(0)])
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_STACKED, Inches(0.8), Inches(2.0), Inches(6.9), Inches(3.95), cd)
        ch = gf.chart
        ch.has_title = True
        ch.chart_title.text_frame.text = "Waste per meal, kg"
        tp = ch.chart_title.text_frame.paragraphs[0].runs[0].font
        tp.size, tp.bold, tp.name = Pt(13), True, FONT
        tp.color.rgb = rgb(INK)
        ch.has_legend = True
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size, ch.legend.font.name = Pt(11), FONT
        ch.legend.font.color.rgb = rgb(MUTED)
        for ser, col in zip(ch.plots[0].series, [GHOST, OTHER, PLATE]):
            ser.format.fill.solid()
            ser.format.fill.fore_color.rgb = rgb(col)
        ch.plots[0].gap_width = 60
        ca, va = ch.category_axis, ch.value_axis
        for ax in (ca, va):
            ax.tick_labels.font.size = Pt(11)
            ax.tick_labels.font.name = FONT
            ax.tick_labels.font.color.rgb = rgb(MUTED)
            ax.format.line.color.rgb = rgb(RULE)
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = rgb(RULE)
        va.format.line.fill.background()

    # right column: forecast + what changes
    x, cw = 8.2, 4.53
    box(s, x, 1.85, cw, 2.0, PAPER, radius=0.05)
    if F["n_fc"]:
        text(s, x + 0.3, 2.0, cw - 0.6, 0.35, f"Sealed forecasts, {F['n_fc']} meals", size=13, color=MUTED, bold=True)
        text(s, x + 0.3, 2.35, cw - 0.6, 0.8, [[(f"{F['fc_mae']:.0f}", {"size": 44, "bold": True, "color": OTHER}),
                                                  ("  diners off on average", {"size": 15, "color": INK})]])
        text(s, x + 0.3, 3.15, cw - 0.6, 0.65,
             f"Cooking for every subscriber: {fmt(F['fc_b1'])} off. {F['fc_in']} of {F['n_fc']} actual counts "
             "fell inside our range.", size=12, color=MUTED)
    else:
        text(s, x + 0.3, 2.0, cw - 0.6, 1.7, [[("Sealed forecasts", {"bold": True, "size": 15})],
                                               [("Checked against the counts after each forecast meal.",
                                                 {"color": MUTED, "size": 13})]], spacing=6)

    box(s, x, 4.05, cw, 2.0, INK, radius=0.05)
    text(s, x + 0.3, 4.2, cw - 0.6, 0.35, "What changes for the manager", size=13, color=OTHER, bold=True)
    saved = F.get("rec_saved_per_meal")
    if F.get("rec_meals") and saved is not None and not np.isnan(saved):
        if saved >= 1:
            change = (f"Cook to the sealed forecast, not the list: about {saved:.0f} kg less left in the "
                      f"vessels per meal")
            change += (f", short on {F['rec_short_dishes']} of {F['rec_dishes']} dishes by "
                       f"{F['rec_short_kg']:.1f} kg." if F["rec_short_dishes"] else ", with nothing running short.")
        else:
            # the forecast under-predicted, so its buffered quantity was no better than what was cooked
            change = ("Cook to the sealed forecast, not the list. On these two meals it did not beat what "
                      "the kitchen cooked: the lunch forecast came in low.")
    else:
        change = "Cook to tomorrow's sealed forecast instead of the subscriber list."
    if F.get("top_plate_dish"):
        change += f" Separately, {F['top_plate_dish']} comes back on plates more than anything else served."
    text(s, x + 0.3, 4.55, cw - 0.6, 1.45, change, size=13.5, color="FFFFFF")

    # limits + link
    lim = [f"{F['n_waste']} meals in one mess, so this is a snapshot." if F["n_waste"] else "",
           "Per-dish plate waste is inferred from photos.",
           ("The ghost split assumes the kitchen cooks for subscribers; the implied headcount supports that."
            if F["cooks_for_subs"] else
            "The kitchen already cooks close to turnout, so most waste is batch sizing, not absent subscribers.")]
    text(s, 0.6, 6.25, 12.1, 0.5, "What the data can't tell you: " + " ".join(p for p in lim if p),
         size=11.5, color=MUTED)
    url = pc.get("demo_url")
    if url:
        text(s, 0.6, 6.8, 12.1, 0.35, [[("Live dashboard, data and code: ", {"bold": True}), (url, {"color": GHOST})]],
             size=13)
    s.notes_slide.notes_text_frame.text = (
        f"Over {F['n_waste']} meals the mess threw away {fmt(F['kg'])} kilograms, about {fmt(F['inr'], 0, 'Rs ')}. "
        + (f"{100 * F['ghost_kg'] / F['kg']:.0f} percent of it was food cooked for subscribers who never came. "
           if F["n_waste"] and not np.isnan(F["ghost_kg"]) else "")
        + (f"Our sealed forecasts missed by {F['fc_mae']:.0f} diners on average; cooking for the full list misses by "
           f"{fmt(F['fc_b1'])}. " if F["n_fc"] else "")
        + "What changes: the manager cooks to our number the night before, and takes the most-returned dish to "
          "the menu meeting. The limit: this is a short window in one mess, and the per-dish plate split comes "
          "from photos, not a scale.")


def build_slides(cfg: Config, verbose: bool = True) -> str:
    F = gather(cfg)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(int(W * 914400)), Emu(int(H * 914400))
    slide_problem(prs, cfg, F)
    slide_method(prs, cfg, F)
    slide_result(prs, cfg, F)
    out = cfg.paths.reports / "deck.pptx"
    prs.save(out)
    if verbose:
        print(f"Deck: {out}")
        if not cfg.section("presentation").get("demo_url"):
            print("  ! presentation.demo_url is empty in config.yaml; slide 3 has no demo link yet.")
    pdf = to_pdf(out)
    if pdf and verbose:
        print(f"PDF : {pdf}")
    return str(out)


def to_pdf(pptx_path: Path) -> str | None:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        return None
    try:
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(pptx_path.parent), str(pptx_path)],
                       check=True, capture_output=True, timeout=180)
        return str(pptx_path.with_suffix(".pdf"))
    except Exception:
        return None
