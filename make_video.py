"""Build the MP4 walkthrough.

  python make_video.py frames     # render the distinct stills + a concat plan
  python make_video.py encode     # ffmpeg the plan into an MP4

The middle of the film is a real screen recording: the dashboard is rendered once
at full height by a headless browser and this script pans a 1920x1080 window down
it. The opening and closing sequences are animated here, built from the project's
own plate photographs and numbers.
"""
from __future__ import annotations

import glob
import json
import math
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 24
CARD_FPS = 16
OUT = Path("/home/claude/frames")
PLAN = OUT / "plan.json"

INK, BG, PAPER = (25, 35, 45), (228, 232, 234), (246, 247, 247)
MUTED, DIM = (150, 162, 172), (90, 103, 115)
ACCENT, GHOST, GOOD, STEEL = (217, 154, 34), (179, 64, 47), (110, 165, 142), (154, 165, 174)
CAP_H = 168
VIEW_H = H - CAP_H


def font(sz, bold=False):
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
    try:
        return ImageFont.truetype(f, sz)
    except OSError:
        return ImageFont.load_default()


F_CAP, F_LBL = font(31), font(20, True)


def wrap(d, text, fnt, maxw):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=fnt) <= maxw:
            cur = t
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def blend(a, b, t):
    t = min(max(t, 0.0), 1.0)
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


# ----------------------------------------------------------------- plate mosaic
def build_mosaic():
    files = sorted(glob.glob("/home/claude/sim_out/frames_kept/*/*.jpg"))
    random.Random(4).shuffle(files)
    cols, rows = 8, 5
    tw, th = W // cols, H // rows + 1
    m = Image.new("RGB", (W, H), INK)
    for i in range(cols * rows):
        im = Image.open(files[i % len(files)]).convert("RGB")
        s = max(tw / im.width, th / im.height)
        im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.LANCZOS)
        left, top = (im.width - tw) // 2, (im.height - th) // 2
        m.paste(im.crop((left, top, left + tw, top + th)), ((i % cols) * tw, (i // cols) * th))
    return m, tw, th


MOSAIC, TW, TH = build_mosaic()
TILE_ORDER = list(range(40))
random.Random(9).shuffle(TILE_ORDER)


def mosaic_frame(t):
    f = Image.new("RGB", (W, H), INK)
    shown = int(min(t / 0.55, 1.0) * len(TILE_ORDER))
    for k in TILE_ORDER[:shown]:
        x, y = (k % 8) * TW, (k // 8) * TH
        f.paste(MOSAIC.crop((x, y, x + TW, y + TH)), (x, y))
    a = 0.10 + 0.80 * ease(max(0.0, (t - 0.45) / 0.35))
    return Image.blend(f, Image.new("RGB", (W, H), INK), a)


# ----------------------------------------------------------------- animated cards
def title_card(t):
    f = mosaic_frame(t)
    d = ImageDraw.Draw(f)
    ta = ease_out(max(0.0, (t - 0.42) / 0.30))
    if ta > 0.01:
        dy = int(34 * (1 - ta))
        col = blend(INK, (255, 255, 255), ta)
        d.text((120, 372 + dy), "Where the mess", font=font(96, True), fill=col)
        d.text((120, 480 + dy), "food goes", font=font(96, True), fill=col)
        bw = int(ease_out(max(0.0, (t - 0.55) / 0.30)) * 420)
        d.rectangle([120, 618 + dy, 120 + bw, 624 + dy], fill=ACCENT)
    sa = ease_out(max(0.0, (t - 0.62) / 0.30))
    if sa > 0.01:
        d.text((120, 664), "Dihing Hostel mess \u00b7 five meals \u00b7 one forecast sealed before the meal",
               font=font(33), fill=blend(INK, MUTED, sa))
    return f


def note_card(t):
    f = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(f)
    a = ease_out(min(t / 0.30, 1.0))
    cy = 250 + int(30 * (1 - a))
    d.rounded_rectangle([160, cy, W - 160, cy + 540], 18, fill=blend(BG, PAPER, a))
    d.text((228, cy + 58), "About this data", font=font(56, True), fill=blend(BG, INK, a))
    rows = [(GOOD, "Plate photographs", "real, shot at Dihing Hostel's own return counter"),
            (ACCENT, "Counts, weights, subscribers", "modelled, not measured")]
    for i, (col, head, sub) in enumerate(rows):
        ra = ease_out(max(0.0, (t - 0.32 - i * 0.18) / 0.30))
        if ra < 0.01:
            continue
        y = cy + 190 + i * 150
        dx = int(26 * (1 - ra))
        d.ellipse([228 - dx, y + 12, 250 - dx, y + 34], fill=blend(BG, col, ra))
        d.text((282 - dx, y), head, font=font(38, True), fill=blend(BG, INK, ra))
        d.text((282 - dx, y + 54), sub, font=font(30), fill=blend(BG, DIM, ra))
    return f


def change_card(t):
    f = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(f)
    a = ease_out(min(t / 0.22, 1.0))
    d.text((120, 120), "What changes tomorrow", font=font(70, True), fill=blend(INK, (255, 255, 255), a))
    for i, (col, txt) in enumerate([(ACCENT, "Cook to one number, decided the night before"),
                                    (GHOST, "Take the most-returned dish to the menu meeting"),
                                    (STEEL, "Keep weighing the bin \u2014 the cheapest instrument here")]):
        ba = ease_out(max(0.0, (t - 0.20 - i * 0.11) / 0.24))
        if ba < 0.01:
            continue
        y = 262 + i * 70
        dx = int(30 * (1 - ba))
        d.ellipse([120 - dx, y + 12, 140 - dx, y + 32], fill=blend(INK, col, ba))
        d.text((170 - dx, y), txt, font=font(35), fill=blend(INK, (238, 242, 245), ba))

    ca = max(0.0, (t - 0.42) / 0.32)
    if ca > 0.01:
        e = ease_out(min(ca, 1.0))
        d.text((120, 520), str(int(round(17 * e))), font=font(140, True), fill=ACCENT)
        d.text((320, 588), "diners off, on average", font=font(36), fill=blend(INK, MUTED, e))
    b2 = max(0.0, (t - 0.58) / 0.38)
    if b2 > 0.01:
        e = ease_out(min(b2, 1.0))
        for i, (lab, v, col) in enumerate([("our sealed forecast", 17, ACCENT),
                                           ("copying last week", 14, STEEL),
                                           ("cooking for the full list", 98, GHOST)]):
            y = 748 + i * 78
            d.text((120, y - 2), lab, font=font(28), fill=MUTED)
            bx, bw = 560, int(1100 * (v / 98) * e)
            d.rectangle([bx, y, bx + bw, y + 36], fill=col)
            d.text((bx + bw + 18, y + 2), str(v), font=font(30, True), fill=blend(INK, (238, 242, 245), e))
    return f


def limits_card(t):
    f = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(f)
    a = ease_out(min(t / 0.24, 1.0))
    d.text((120, 218), "What this cannot tell you", font=font(64, True), fill=blend(BG, INK, a))
    lines = ["Five meals in one mess is a snapshot, not a season.",
             "Per-dish plate waste is inferred from photographs, not weighed dish by dish.",
             "Saturday dinner has no bin weight: the scale had been borrowed back."]
    for i, ln in enumerate(lines):
        la = ease_out(max(0.0, (t - 0.26 - i * 0.14) / 0.28))
        if la < 0.01:
            continue
        y = 386 + i * 80
        d.rectangle([120, y + 16, 120 + int(26 * la), y + 20], fill=ACCENT)
        d.text((170, y), ln, font=font(34), fill=blend(BG, DIM, la))
    ea = ease_out(max(0.0, (t - 0.70) / 0.30))
    if ea > 0.01:
        d.text((120, 706), "Every figure on the dashboard carries where it came from.",
               font=font(30), fill=blend(BG, MUTED, ea))
    return f


# ----------------------------------------------------------------- page pan
PAGE = Image.open("/home/claude/dash_full.png").convert("RGB")
SCALE = W / PAGE.size[0]
SPAGE = PAGE.resize((W, int(PAGE.size[1] * SCALE)), Image.LANCZOS)
SPH = SPAGE.size[1]


def page_frame(scroll, text, label, progress):
    f = Image.new("RGB", (W, H), BG)
    top = int(max(0, min(scroll, SPH - VIEW_H)))
    f.paste(SPAGE.crop((0, top, W, top + VIEW_H)), (0, 0))
    d = ImageDraw.Draw(f)
    d.rectangle([0, H - CAP_H, W, H], fill=INK)
    d.rectangle([0, H - CAP_H, int(W * progress), H - CAP_H + 4], fill=ACCENT)
    if label:
        d.text((72, H - CAP_H + 24), label.upper(), font=F_LBL, fill=ACCENT)
    y = H - CAP_H + 58
    for ln in wrap(d, text, F_CAP, W - 144)[:3]:
        d.text((72, y), ln, font=F_CAP, fill=(240, 244, 246))
        y += 38
    return f


# ----------------------------------------------------------------- assembly
def render():
    OUT.mkdir(exist_ok=True)
    for p in OUT.glob("*.png"):
        p.unlink()
    plan, n = [], 0

    def put(img, secs):
        nonlocal n
        p = OUT / f"f{n:05d}.png"
        img.save(p, compress_level=1)
        plan.append((str(p), secs))
        n += 1

    def animate(fn, secs, fps=CARD_FPS):
        total = int(secs * fps)
        for i in range(total):
            put(fn(i / max(total - 1, 1)), 1.0 / fps)

    script = {s["id"]: s for s in json.load(open("/home/claude/narration.json"))}
    S = {k: int(v * SCALE) for k, v in
         dict(hero=0, waste=382, cooks=1254, forecast=1741, dishes=2964, arrive=3630).items()}
    beats = [("01", S["hero"]), ("04", S["hero"]), ("05", S["waste"]), ("06", S["cooks"]),
             ("07", S["forecast"]), ("08", S["forecast"] + 380), ("09", S["forecast"] + 700),
             ("10", S["dishes"]), ("02", S["arrive"])]

    animate(title_card, 5.0)

    prev = None
    for i, (sid, target) in enumerate(beats):
        seg = script[sid]
        if prev is not None and target != prev:
            total = int(0.85 * FPS)
            for k in range(total):
                s = prev + (target - prev) * ease(k / max(total - 1, 1))
                put(page_frame(s, "", "", i / len(beats)), 1.0 / FPS)
        for k in range(3):
            pr = (i + (k + 1) / 3) / len(beats)
            put(page_frame(target, seg["text"], f"{i+1} of {len(beats)}", pr), seg["secs"] / 3)
        prev = target


    json.dump(plan, open(PLAN, "w"))
    total = sum(d for _, d in plan)
    print(f"{n} stills, {total:.1f}s ({int(total//60)}:{int(total%60):02d})")


def encode():
    plan = json.load(open(PLAN))
    lst = OUT / "concat.txt"
    with open(lst, "w") as fh:
        for path, dur in plan:
            fh.write(f"file '{path}'\nduration {dur:.5f}\n")
        fh.write(f"file '{plan[-1][0]}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-vsync", "cfr", "-r", str(FPS), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    "-metadata", "title=Where the mess food goes \u2014 Dihing Hostel "
                                 "(plate photographs real, figures modelled)",
                    "-metadata", "comment=Plate photographs are real, shot at Dihing Hostel's plate-return "
                                 "counter. Counts, weights, subscriber figures and attendance history are "
                                 "modelled by messwaste/simulate.py, not measured.",
                    "-metadata", "description=Mess waste pipeline walkthrough. See README.md for provenance.",
                    "/mnt/user-data/outputs/mess_waste_demo.mp4"], check=True)
    print("wrote /mnt/user-data/outputs/mess_waste_demo.mp4")


if __name__ == "__main__":
    (encode if sys.argv[1:2] == ["encode"] else render)()
