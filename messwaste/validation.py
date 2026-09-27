"""Human validation of the vision model.

1. `human-sample` picks a stratified sample of kept frames and writes an offline
   labelling page (labeller.html). Each teammate opens their own copy in a
   browser, labels independently, and clicks "Download CSV".
2. Put the downloaded human_labels_<initials>.csv files in <raw_root>/csv/ and rerun ingest.
3. `agreement` compares model vs each human and human vs human.
"""
from __future__ import annotations

import base64
import io as _io
import json
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config
from .dishes import DishBook
from .io import read_table, write_table


def make_sample(cfg: Config, n: int = 100, seed: int = 7) -> pd.DataFrame:
    frames = read_table(cfg, "camera_frames", required=True)
    kept = frames[frames["kept"]].copy()
    if kept.empty:
        raise RuntimeError("No kept frames yet. Run the frames step first.")
    per_meal = kept.groupby("meal_id")["frame_id"].count()
    share = (per_meal / per_meal.sum() * n).round().clip(lower=1).astype(int)
    parts = []
    for mid, k in share.items():
        g = kept[kept["meal_id"] == mid]
        parts.append(g.sample(n=min(k, len(g)), random_state=seed))
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sample[["frame_id", "meal_id"]]


def build_labeller(cfg: Config, n: int = 100, seed: int = 7) -> str:
    book = DishBook(cfg)
    meals = read_table(cfg, "meals", required=True)
    menu_of = {r["meal_id"]: book.parse_menu(r["menu_items"]) for _, r in meals.iterrows()}
    sample = make_sample(cfg, n, seed)
    items = []
    for _, r in sample.iterrows():
        p = cfg.paths.frames_kept / r["meal_id"] / f"{r['frame_id']}.jpg"
        im = Image.open(p).convert("RGB")
        im.thumbnail((720, 720))
        buf = _io.BytesIO()
        im.save(buf, "JPEG", quality=82)
        items.append({"frame_id": r["frame_id"], "meal_id": r["meal_id"], "dishes": menu_of.get(r["meal_id"], []),
                      "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
    batch_id = f"batch_{seed}_{len(items)}"
    html = LABELLER_HTML.replace("__DATA__", json.dumps(items)).replace("__BATCH__", batch_id)
    out = cfg.paths.labelling / "labeller.html"
    out.write_text(html, encoding="utf-8")
    sample.to_csv(cfg.paths.labelling / "sample_manifest.csv", index=False)
    print(f"Wrote {out} with {len(items)} frames. Each labeller opens their own copy, labels, "
          "then downloads human_labels_<initials>.csv into <raw_root>/csv/.")
    return str(out)


# ----------------------------------------------------------------------------- agreement
def _cat(x: pd.Series) -> pd.Series:
    return (x * 4).round().astype(int)


def _kappa(a: pd.Series, b: pd.Series) -> float:
    from sklearn.metrics import cohen_kappa_score
    if a.nunique() < 2 and b.nunique() < 2:
        return float("nan")
    try:
        return float(cohen_kappa_score(_cat(a), _cat(b), weights="quadratic", labels=[0, 1, 2, 3, 4]))
    except Exception:
        return float("nan")


def _compare(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    m = a.merge(b, on=["frame_id", "dish"], suffixes=("_a", "_b"))
    both = m.dropna(subset=["fraction_left_a", "fraction_left_b"])
    pp = a.drop_duplicates("frame_id")[["frame_id", "plate_present"]].merge(
        b.drop_duplicates("frame_id")[["frame_id", "plate_present"]], on="frame_id", suffixes=("_a", "_b"))
    if both.empty:
        return dict(pair=f"{name_a} vs {name_b}", n_items=0)
    diff = (both["fraction_left_a"] - both["fraction_left_b"]).abs()
    return dict(pair=f"{name_a} vs {name_b}", n_frames=int(both["frame_id"].nunique()), n_items=int(len(both)),
                exact_agreement=float((diff < 1e-9).mean()),
                within_one_step=float((diff <= 0.25 + 1e-9).mean()),
                mae_fraction=float(diff.mean()),
                weighted_kappa=_kappa(both["fraction_left_a"], both["fraction_left_b"]),
                plate_present_agreement=float((pp["plate_present_a"] == pp["plate_present_b"]).mean()) if len(pp) else np.nan)


def human_consensus(h: pd.DataFrame) -> pd.DataFrame:
    """Mean of human labels per frame/dish, snapped to the grid."""
    if h.empty:
        return h
    g = h.groupby(["frame_id", "dish"], as_index=False).agg(
        fraction_left=("fraction_left", "mean"), plate_present=("plate_present", "max"),
        n_labellers=("labeller", "nunique"))
    g["fraction_left"] = (g["fraction_left"] * 4).round() / 4
    return g


def agreement(cfg: Config, verbose: bool = True) -> pd.DataFrame:
    h = read_table(cfg, "human_labels")
    mdl = read_table(cfg, "plate_labels_model")
    if h.empty:
        print("No human labels yet (human_labels*.csv in raw csv folder). Skipping agreement.")
        write_table(cfg, pd.DataFrame(), "agreement")
        return pd.DataFrame()
    rows = []
    labellers = sorted(h["labeller"].dropna().unique())
    for a, b in combinations(labellers, 2):
        rows.append(_compare(h[h["labeller"] == a], h[h["labeller"] == b], a, b))
    if not mdl.empty:
        for mid in sorted(mdl["model_id"].unique()):
            md = mdl[mdl["model_id"] == mid]
            for a in labellers:
                rows.append(_compare(md, h[h["labeller"] == a], mid, a))
            rows.append(_compare(md, human_consensus(h), mid, "human consensus"))
    df = pd.DataFrame(rows)
    df["source"] = "demo" if (h.get("source") == "demo").all() else "inferred"
    write_table(cfg, df, "agreement")
    if verbose:
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(df.drop(columns=["source"]).round(3).to_string(index=False))
    return df


# ----------------------------------------------------------------------------- HTML tool
LABELLER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plate labelling</title>
<style>
:root{--steel:#e6e9eb;--plate:#f7f8f8;--ink:#1d2731;--muted:#5d6975;--turmeric:#c98a0b;--chili:#a8392b;--line:#c9cfd4}
*{box-sizing:border-box}
body{margin:0;background:var(--steel);color:var(--ink);font:16px/1.5 "Source Sans 3","Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--ink);color:#fff;padding:12px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:18px;margin:0 12px 0 0;font-weight:600}
header input{padding:6px 10px;border-radius:6px;border:0;font:inherit;width:120px}
.bar{flex:1;min-width:160px;height:8px;background:#3a4652;border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--turmeric);width:0}
button{font:inherit;cursor:pointer}
.dl{background:var(--turmeric);color:var(--ink);border:0;border-radius:6px;padding:7px 14px;font-weight:600}
main{max-width:980px;margin:0 auto;padding:20px}
.help{color:var(--muted);max-width:70ch}
.card{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:20px;background:var(--plate);border-radius:14px;padding:16px;margin:18px 0;border:2px solid transparent}
.card.done{border-color:#7fa58e}
.card img{width:100%;border-radius:10px;display:block;background:#ccc}
.meta{color:var(--muted);font-size:13px;margin-bottom:8px}
.row{margin:10px 0}
.row b{display:block;font-weight:600;margin-bottom:4px}
.row b.dish{text-transform:capitalize}
.opts{display:flex;flex-wrap:wrap;gap:6px}
.opts button{border:1px solid var(--line);background:#fff;border-radius:18px;padding:4px 11px;min-width:44px}
.opts button[aria-pressed=true]{background:var(--ink);color:#fff;border-color:var(--ink)}
.opts button.na[aria-pressed=true]{background:var(--chili);border-color:var(--chili)}
button:focus-visible{outline:3px solid var(--turmeric);outline-offset:2px}
@media (max-width:700px){.card{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Plate labelling</h1>
<label>Your initials <input id="who" maxlength="6" placeholder="e.g. VR"></label>
<div class="bar" aria-hidden="true"><i id="prog"></i></div><span id="count"></span>
<button class="dl" id="dl">Download CSV</button></header>
<main>
<p class="help">For each plate, choose how much of each dish is left compared with one normal serving. Pick 0 if the dish is gone or was never taken, and "Can't tell" only if that part of the plate is hidden. If the photo shows no plate, or plates you can't separate, set "Plate visible" to No. Label on your own; don't compare answers with your teammate until both are downloaded. Your progress is saved in this browser.</p>
<div id="list"></div></main>
<script>
const DATA = __DATA__; const BATCH = "__BATCH__";
const OPTS = [["0","0"],["0.25","¼"],["0.5","½"],["0.75","¾"],["1","All"],["","Can't tell"]];
let state = {}; try { state = JSON.parse(localStorage.getItem("lbl_"+BATCH)||"{}"); } catch(e) {}
const who = document.getElementById("who");
try { who.value = localStorage.getItem("lbl_who")||""; } catch(e) {}
who.addEventListener("input",()=>{try{localStorage.setItem("lbl_who",who.value)}catch(e){}});
function save(){ try{localStorage.setItem("lbl_"+BATCH, JSON.stringify(state))}catch(e){} progress(); }
function complete(it){ const s=state[it.frame_id]; if(!s||s.plate===undefined) return false; if(s.plate===0) return true;
  return it.dishes.every(d=>s.d&&s.d[d]!==undefined); }
function progress(){ const n=DATA.filter(complete).length; document.getElementById("prog").style.width=(100*n/DATA.length)+"%";
  document.getElementById("count").textContent=n+" of "+DATA.length+" done";
  DATA.forEach(it=>document.getElementById("c_"+it.frame_id).classList.toggle("done",complete(it))); }
function group(label, opts, cur, onpick, isDish){ const row=document.createElement("div"); row.className="row";
  const lb=document.createElement("b"); lb.textContent=label; if(isDish) lb.className="dish"; row.appendChild(lb); const o=document.createElement("div"); o.className="opts"; o.setAttribute("role","group");
  opts.forEach(([v,t])=>{ const b=document.createElement("button"); b.type="button"; b.textContent=t; if(v==="") b.className="na";
    b.setAttribute("aria-pressed", String(cur===v)); b.onclick=()=>{ onpick(v); o.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed","false")); b.setAttribute("aria-pressed","true"); save(); }; o.appendChild(b); });
  row.appendChild(o); return row; }
const list=document.getElementById("list");
DATA.forEach((it,i)=>{ const s=state[it.frame_id]=state[it.frame_id]||{d:{}};
  const c=document.createElement("section"); c.className="card"; c.id="c_"+it.frame_id;
  c.innerHTML='<img loading="lazy" alt="Plate photo '+(i+1)+'" src="'+it.img+'"><div class="side"><div class="meta">'+"Plate "+(i+1)+" of "+DATA.length+", meal "+it.meal_id+"</div></div>";
  const side=c.querySelector(".side");
  side.appendChild(group("Plate visible", [["1","Yes"],["0","No"]], s.plate===undefined?null:String(s.plate), v=>{s.plate=+v;}));
  it.dishes.forEach(d=>side.appendChild(group(d, OPTS, s.d[d]===undefined?null:(s.d[d]===null?"":String(s.d[d])), v=>{s.d[d]= v===""?null:+v;}, true)));
  list.appendChild(c); });
progress();
document.getElementById("dl").onclick=()=>{ const w=who.value.trim(); if(!w){ alert("Enter your initials first so labels from different people stay separate."); who.focus(); return; }
  const rows=[["frame_id","labeller","dish","fraction_left","plate_present","labelled_at"]]; const t=new Date().toISOString();
  DATA.forEach(it=>{ const s=state[it.frame_id]; if(!complete(it)) return;
    it.dishes.forEach(d=>{ const v = s.plate===0? "" : (s.d[d]===null? "" : s.d[d]); rows.push([it.frame_id,w,d,v,s.plate,t]); }); });
  const blob=new Blob([rows.map(r=>r.join(",")).join("\n")+"\n"],{type:"text/csv"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="human_labels_"+w+".csv"; a.click(); };
</script></body></html>
"""
