"""Estimate how much of each dish is left on each plate photo.

Backends
  hf        : local open vision-language model (default Qwen2.5-VL-7B) spread over both Kaggle T4s
  anthropic : Claude via API (key from Kaggle Secrets or ANTHROPIC_API_KEY)
  mock      : no model; for testing the pipeline

The run is resumable: frames already labelled by the same model_id are skipped,
and partial results are saved every few batches.
"""
from __future__ import annotations

import base64
import hashlib
import io as _io
import json
import os
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config
from .io import read_table, write_table

PROMPT = """You are auditing food waste in an Indian hostel mess. The photo shows the plate-return counter from above.

Today's menu: {menu}

Look for ONE steel plate or thali that has just been returned.
For each menu dish, estimate how much of it is LEFT on that plate, as a fraction of a normal single serving:
0 = none left (eaten, or never taken), 0.25 = a quarter left, 0.5 = half, 0.75 = most of it, 1 = a full untouched serving.
Use null only if you genuinely cannot see the part of the plate where that dish would be.
If there is no plate, or several overlapping plates you cannot separate, set plate_present to false.

Reply with JSON only, no other text:
{{"plate_present": true, "dishes": {{{example}}}}}"""


def build_prompt(menu: list[str]) -> str:
    example = ", ".join(f'"{d}": 0.25' for d in menu)
    return PROMPT.format(menu=", ".join(menu), example=example)


def snap(v, grid: list[float]):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(x):
        return None
    x = min(max(x, 0.0), 1.0)
    return float(min(grid, key=lambda g: abs(g - x)))


def parse_response(text: str, menu: list[str], grid: list[float], canon) -> tuple[bool, int, dict]:
    """Returns (parse_ok, plate_present, {dish: fraction|None})."""
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return False, 0, {d: None for d in menu}
    blob = m.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        try:
            data = json.loads(blob.replace("'", '"').replace("None", "null").replace("True", "true")
                              .replace("False", "false"))
        except json.JSONDecodeError:
            return False, 0, {d: None for d in menu}
    present = bool(data.get("plate_present", False))
    dishes = data.get("dishes", {}) or {}
    norm = {canon(k): v for k, v in dishes.items()} if isinstance(dishes, dict) else {}
    return True, int(present), {d: (snap(norm.get(d), grid) if present else None) for d in menu}


# ----------------------------------------------------------------------------- backends
def _resize(img: Image.Image, max_side: int) -> Image.Image:
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


class MockBackend:
    """Deterministic fake labels. On demo data it reads the hidden truth and adds noise."""
    model_id = "mock-v1"

    def __init__(self, cfg: Config):
        self.grid = cfg.section("labeling").get("fraction_grid", [0, .25, .5, .75, 1])
        truth_path = cfg.paths.raw_frames / "_demo_truth.csv"
        self.truth = pd.read_csv(truth_path) if truth_path.exists() else None

    def generate(self, images, prompts, frame_ids=None, menus=None):
        outs = []
        for fid, menu in zip(frame_ids, menus):
            rng = np.random.default_rng(int(hashlib.md5(fid.encode()).hexdigest()[:8], 16))
            d = {}
            for dish in menu:
                base = None
                if self.truth is not None:
                    orig = fid.rsplit("_", 1)[0]
                    t = self.truth[(self.truth["meal_id"] == orig) & (self.truth["dish"] == dish)
                                   & (self.truth["frame_hash"] == fid.rsplit("_", 1)[1])]
                    if len(t):
                        base = float(t["fraction_left"].iloc[0])
                if base is None:
                    base = float(rng.choice(self.grid))
                noisy = base + rng.choice([-0.25, 0, 0, 0, 0.25])
                d[dish] = float(min(max(noisy, 0), 1))
            outs.append(json.dumps({"plate_present": True, "dishes": d}))
        return outs


class HFBackend:
    def __init__(self, cfg: Config, model_id: str | None = None):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoVLM
        lab = cfg.section("labeling")
        self.model_id = model_id or lab.get("hf_model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.max_side = int(lab.get("hf_max_side_px", 768))
        self.max_new = int(lab.get("hf_max_new_tokens", 200))
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[lab.get("hf_dtype", "float16")]
        n = torch.cuda.device_count()
        print(f"Loading {self.model_id} ({lab.get('hf_dtype')}) on {n} GPU(s)...")
        if n == 0:
            print("  ! no GPU visible. In Kaggle: Settings -> Accelerator -> GPU T4 x2.")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, min_pixels=128 * 28 * 28, max_pixels=self.max_side * self.max_side)
        except (TypeError, ValueError):
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"
        self.model = AutoVLM.from_pretrained(self.model_id, torch_dtype=dtype,
                                             device_map="auto" if n else None)
        self.model.eval()
        self.torch = torch

    def generate(self, images, prompts, frame_ids=None, menus=None):
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}] for p in prompts]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        imgs = [_resize(i, self.max_side) for i in images]
        inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)


class AnthropicBackend:
    def __init__(self, cfg: Config, model_id: str | None = None):
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            try:
                from kaggle_secrets import UserSecretsClient
                key = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
            except Exception:
                pass
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY (Kaggle: Add-ons -> Secrets).")
        self.client = anthropic.Anthropic(api_key=key)
        self.model_id = model_id or cfg.section("labeling").get("anthropic_model", "claude-haiku-4-5-20251001")
        self.max_side = int(cfg.section("labeling").get("hf_max_side_px", 768))

    def _one(self, img, prompt):
        buf = _io.BytesIO()
        _resize(img, self.max_side).save(buf, "JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()
        for attempt in range(5):
            try:
                r = self.client.messages.create(
                    model=self.model_id, max_tokens=300,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": prompt}]}])
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            except Exception as e:  # rate limits, transient errors
                wait = 2 ** attempt
                print(f"  api error ({e.__class__.__name__}); retrying in {wait}s")
                time.sleep(wait)
        return ""

    def generate(self, images, prompts, frame_ids=None, menus=None):
        return [self._one(i, p) for i, p in zip(images, prompts)]


def get_backend(cfg: Config, name: str | None = None, model_id: str | None = None):
    name = name or cfg.section("labeling").get("backend", "hf")
    if name == "mock":
        return MockBackend(cfg)
    if name == "hf":
        return HFBackend(cfg, model_id)
    if name == "anthropic":
        return AnthropicBackend(cfg, model_id)
    raise ValueError(f"unknown backend {name}")


# ----------------------------------------------------------------------------- runner
def _prior_labels(cfg: Config) -> pd.DataFrame:
    """Labels saved by an earlier Kaggle session (added back as an input dataset)."""
    import glob
    pat = cfg.section("labeling").get("prior_labels_glob", "/kaggle/input/**/plate_labels_model.parquet")
    parts = []
    for f in glob.glob(pat, recursive=True):
        try:
            parts.append(pd.read_parquet(f))
            print(f"  using earlier labels from {f}")
        except Exception as e:
            print(f"  ! could not read {f}: {e}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run_labeling(cfg: Config, backend: str | None = None, model_id: str | None = None,
                 limit: int | None = None, meal_ids: list[str] | None = None, verbose: bool = True,
                 backend_obj=None) -> pd.DataFrame:
    """backend_obj: pass an already-loaded backend (notebook) to avoid reloading the model."""
    from .dishes import DishBook
    book = DishBook(cfg)
    lab = cfg.section("labeling")
    grid = [float(g) for g in lab.get("fraction_grid", [0, .25, .5, .75, 1])]

    frames = read_table(cfg, "camera_frames", required=True)
    meals = read_table(cfg, "meals", required=True)
    if frames.empty or not frames["kept"].any():
        print("No kept frames to label.")
        return read_table(cfg, "plate_labels_model")
    menu_of = {r["meal_id"]: book.parse_menu(r["menu_items"]) for _, r in meals.iterrows()}
    src_of = {r["meal_id"]: (r["source"] if r["source"] in ("demo", "synthetic") else "inferred")
              for _, r in meals.iterrows()}

    todo = frames[frames["kept"]].copy()
    if meal_ids:
        todo = todo[todo["meal_id"].isin(meal_ids)]

    be = backend_obj or get_backend(cfg, backend, model_id)
    existing = read_table(cfg, "plate_labels_model")
    prior = _prior_labels(cfg)
    if not prior.empty:
        existing = pd.concat([existing, prior], ignore_index=True).drop_duplicates(
            ["frame_id", "dish", "model_id"], keep="first")
        existing = existing[existing["frame_id"].isin(set(frames["frame_id"]))]
    done = set()
    if not existing.empty:
        done = set(existing.loc[existing["model_id"] == be.model_id, "frame_id"])
    todo = todo[~todo["frame_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    if verbose:
        print(f"Labelling {len(todo)} frames with {be.model_id} ({len(done)} already done).")

    bs = int(lab.get("hf_batch_size", 4)) if isinstance(be, HFBackend) else 8
    every = int(lab.get("checkpoint_every", 20))
    new_rows: list[dict] = []
    ids = todo["frame_id"].tolist()
    mids = todo["meal_id"].tolist()
    t0 = time.time()

    def save():
        allr = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else existing
        write_table(cfg, allr, "plate_labels_model")
        return allr

    for bi, start in enumerate(range(0, len(ids), bs)):
        b_ids, b_mids = ids[start:start + bs], mids[start:start + bs]
        imgs = [Image.open(cfg.paths.frames_kept / m / f"{f}.jpg") for f, m in zip(b_ids, b_mids)]
        menus = [menu_of.get(m, []) for m in b_mids]
        prompts = [build_prompt(mn) for mn in menus]
        try:
            texts = be.generate(imgs, prompts, frame_ids=b_ids, menus=menus)
        except Exception as e:
            print(f"  batch {bi} failed: {e}. Saving progress.")
            save()
            raise
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for fid, mid, menu, text in zip(b_ids, b_mids, menus, texts):
            ok, present, fr = parse_response(text, menu, grid, book.canonical)
            for dish in menu:
                new_rows.append(dict(frame_id=fid, meal_id=mid, dish=dish, fraction_left=fr[dish],
                                     plate_present=present, backend=type(be).__name__.replace("Backend", "").lower(),
                                     model_id=be.model_id, parse_ok=ok, labelled_at=now,
                                     source=src_of.get(mid, "inferred")))
        if verbose and (bi % 5 == 0):
            n = min(start + bs, len(ids))
            rate = n / max(time.time() - t0, 1e-6)
            print(f"  {n}/{len(ids)} frames  ({rate:.2f} frames/s)")
        if (bi + 1) % every == 0:
            save()
    allr = save()
    if verbose and new_rows:
        nd = pd.DataFrame(new_rows)
        print(f"Done. parse_ok={nd.drop_duplicates('frame_id')['parse_ok'].mean():.0%}, "
              f"plate_present={nd.drop_duplicates('frame_id')['plate_present'].mean():.0%}")
    return allr


def preview(cfg: Config, n: int = 6, backend_obj=None, backend: str | None = None):
    """Label n kept frames WITHOUT saving and show image, raw model text and parsed result.
    Use this on Kaggle before the full run to check the model's output makes sense."""
    from .dishes import DishBook
    book = DishBook(cfg)
    grid = [float(g) for g in cfg.section("labeling").get("fraction_grid", [0, .25, .5, .75, 1])]
    frames = read_table(cfg, "camera_frames", required=True)
    meals = read_table(cfg, "meals", required=True)
    menu_of = {r["meal_id"]: book.parse_menu(r["menu_items"]) for _, r in meals.iterrows()}
    kept = frames[frames["kept"]].sample(n=min(n, int(frames["kept"].sum())), random_state=1)
    be = backend_obj or get_backend(cfg, backend)
    ids, mids = kept["frame_id"].tolist(), kept["meal_id"].tolist()
    imgs = [Image.open(cfg.paths.frames_kept / m / f"{f}.jpg") for f, m in zip(ids, mids)]
    menus = [menu_of.get(m, []) for m in mids]
    t0 = time.time()
    texts = be.generate(imgs, [build_prompt(mn) for mn in menus], frame_ids=ids, menus=menus)
    print(f"{len(ids)} frames in {time.time() - t0:.1f}s")
    out = []
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(ids), figsize=(4 * len(ids), 4))
        axes = np.atleast_1d(axes)
    except Exception:
        axes = [None] * len(ids)
    for ax, fid, img, menu, text in zip(axes, ids, imgs, menus, texts):
        ok, present, fr = parse_response(text, menu, grid, book.canonical)
        out.append(dict(frame_id=fid, parse_ok=ok, plate_present=present, **fr, raw=text[:200]))
        if ax is not None:
            ax.imshow(img)
            ax.set_title("\n".join(f"{k}: {v}" for k, v in fr.items()), fontsize=8)
            ax.axis("off")
    return pd.DataFrame(out)
