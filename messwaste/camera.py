"""Return-counter camera frames: privacy filter -> de-duplicate -> keep.

Input : <raw_root>/frames/<meal_id>/*.jpg  (one folder per meal)
Output: <out_root>/frames_kept/<meal_id>/<frame_id>.jpg  (resized, EXIF and GPS stripped)
        parquet/camera_frames  (every frame, kept or not, with the reason)

Frames with a detected face or person are never copied. Run this step on your
laptop before uploading if you do not want raw frames on Kaggle at all.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .config import Config
from .io import read_table, write_table
from .timeutil import local_dt

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
FNAME_TS = re.compile(r"(20\d{6})[_\-T ]?(\d{6})")


# ----------------------------------------------------------------------------- helpers
def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def small_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize((64, 48), Image.BILINEAR), dtype=np.float32)


def exif_time(img: Image.Image) -> str | None:
    try:
        ex = img.getexif()
        v = None
        try:
            v = ex.get_ifd(0x8769).get(36867)  # DateTimeOriginal
        except Exception:
            pass
        v = v or ex.get(306)                     # DateTime
        if v:
            return pd.Timestamp(str(v).replace(":", "-", 2)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    return None


def filename_time(name: str) -> str | None:
    m = FNAME_TS.search(name)
    if not m:
        return None
    d, t = m.groups()
    try:
        return pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


def _yunet_path() -> Path | None:
    p = Path.home() / ".cache" / "messwaste" / "face_detection_yunet_2023mar.onnx"
    if p.exists() and p.stat().st_size > 10_000:
        return p
    try:
        import urllib.request
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, p)
        return p if p.stat().st_size > 10_000 else None
    except Exception as e:
        print(f"  ! could not download the YuNet face model ({e}); falling back to Haar cascades.")
        return None


class Detectors:
    """Face detection: YuNet (small DNN, far fewer false alarms on round plates) or Haar cascades."""

    def __init__(self, cfg: Config):
        import cv2
        cam = cfg.section("camera")
        self.face = None
        self.yunet = None
        self.person = None
        kind = cam.get("face_detector", "yunet")
        self.min_neighbors = int(cam.get("haar_min_neighbors", 8))
        self.min_size = int(cam.get("haar_min_size_px", 60))
        self.yunet_score = float(cam.get("yunet_score", 0.8))
        self.min_steel_fraction = float(cam.get("min_steel_fraction", 0.35))
        if kind == "yunet":
            path = _yunet_path()
            if path is not None:
                self.yunet = cv2.FaceDetectorYN.create(str(path), "", (320, 320), self.yunet_score, 0.3, 5000)
            else:
                kind = "haar"
        if kind == "haar":
            base = Path(cv2.data.haarcascades)
            self.face = [cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml")),
                         cv2.CascadeClassifier(str(base / "haarcascade_profileface.xml"))]
        self.kind = kind
        if cam.get("person_detector", "none") == "yolo":
            try:
                from ultralytics import YOLO
                self.person = YOLO(cam.get("yolo_model", "yolov8n.pt"))
                self.person_conf = float(cam.get("yolo_conf", 0.4))
            except Exception as e:  # pragma: no cover
                raise RuntimeError("person_detector=yolo needs `pip install ultralytics` and internet "
                                   f"for the weights ({e}). Set it to 'none' to skip.") from e

    def looks_like_a_tray(self, img: Image.Image) -> bool:
        """Cheap guard for frames that are not a plate at all (a room, a queue, a wall).

        A returned tray fills the frame with bright steel, so most of the image is
        light and low-saturation. A photo of people or a room is neither.
        """
        import cv2
        a = np.asarray(img.convert("RGB"))
        scale = 320 / max(a.shape[:2])
        if scale < 1:
            a = cv2.resize(a, None, fx=scale, fy=scale)
        hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
        sat, val = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
        steel = float(((sat < 0.35) & (val > 0.45)).mean())
        return steel >= self.min_steel_fraction

    def count_faces(self, img: Image.Image) -> int:
        import cv2
        if self.yunet is not None:
            a = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            scale = 960 / max(a.shape[:2])
            if scale < 1:
                a = cv2.resize(a, None, fx=scale, fy=scale)
            self.yunet.setInputSize((a.shape[1], a.shape[0]))
            _, faces = self.yunet.detect(a)
            return 0 if faces is None else len(faces)
        if not self.face:
            return 0
        g = np.asarray(img.convert("L"))
        scale = 800 / max(g.shape)
        if scale < 1:
            g = cv2.resize(g, None, fx=scale, fy=scale)
        n = 0
        for c in self.face:
            n += len(c.detectMultiScale(g, scaleFactor=1.1, minNeighbors=self.min_neighbors,
                                        minSize=(self.min_size, self.min_size)))
        return n

    def count_persons(self, img: Image.Image) -> int:
        if self.person is None:
            return 0
        res = self.person.predict(np.asarray(img.convert("RGB")), classes=[0], conf=self.person_conf, verbose=False)
        return int(sum(len(r.boxes) for r in res))


# ----------------------------------------------------------------------------- main step
def process_frames(cfg: Config, verbose: bool = True) -> pd.DataFrame:
    cam = cfg.section("camera")
    max_side = int(cam.get("max_side_px", 1024))
    thr = float(cam.get("dedup_mean_abs_diff", 3.0))
    root = cfg.paths.raw_frames
    meals = read_table(cfg, "meals")
    windows = {}
    for _, r in meals.iterrows():
        s, e = local_dt(cfg, r["date"], r["start_time"]), local_dt(cfg, r["date"], r["end_time"])
        windows[r["meal_id"]] = (s, e, r["source"])

    if not root.exists():
        if verbose:
            print(f"No frames folder at {root}; skipping camera step.")
        df = pd.DataFrame(columns=["frame_id", "meal_id", "taken_at", "file_name", "kept", "drop_reason",
                                   "n_faces", "n_persons", "dhash", "in_meal_window", "source"])
        write_table(cfg, df, "camera_frames")
        return df

    det = Detectors(cfg)
    if verbose:
        print(f"Face detector: {det.kind}")
    rows = []
    for meal_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        meal_id = meal_dir.name
        if meal_id not in windows:
            print(f"  ! frames folder '{meal_id}' has no row in meals.csv; skipped")
            continue
        start, end, meal_src = windows[meal_id]
        src = "demo" if meal_src == "demo" else "observed"   # the photograph itself is always real
        files = [f for f in meal_dir.iterdir() if f.suffix.lower() in IMG_EXT]
        recs = []
        for f in files:
            try:
                img = Image.open(f)
                ts = exif_time(img) or filename_time(f.name)
                img = ImageOps.exif_transpose(img).convert("RGB")
            except Exception:
                rows.append(dict(frame_id=f"{meal_id}_{f.stem}", meal_id=meal_id, taken_at=None, file_name=f.name,
                                 kept=False, drop_reason="unreadable", n_faces=0, n_persons=0, dhash="",
                                 in_meal_window=None, source=src))
                continue
            recs.append((ts, f, img))
        recs.sort(key=lambda x: (x[0] or "", x[1].name))

        out_dir = cfg.paths.frames_kept / meal_id
        out_dir.mkdir(parents=True, exist_ok=True)
        last_small = None
        for ts, f, img in recs:
            h = hashlib.sha1(f.read_bytes()).hexdigest()[:10]
            fid = f"{meal_id}_{h}"
            dh = dhash(img)
            sm = small_gray(img)
            nf = det.count_faces(img)
            npers = det.count_persons(img) if nf == 0 else 0
            reason = ""
            if nf > 0:
                reason = "face"
            elif npers > 0:
                reason = "person"
            elif not det.looks_like_a_tray(img):
                reason = "not a tray"
            elif last_small is not None and float(np.abs(sm - last_small).mean()) <= thr:
                reason = "duplicate"   # same scene as the previous kept frame (plate still sitting there)
            in_win = None
            if ts and start is not None and end is not None:
                t = pd.Timestamp(ts).tz_localize(cfg.tz)
                in_win = bool(start - pd.Timedelta(minutes=15) <= t <= end + pd.Timedelta(minutes=30))
            if not reason:
                last_small = sm
                im = img.copy()
                im.thumbnail((max_side, max_side))
                im.save(out_dir / f"{fid}.jpg", "JPEG", quality=90)  # saved without EXIF
            rows.append(dict(frame_id=fid, meal_id=meal_id, taken_at=ts, file_name=f.name, kept=not reason,
                             drop_reason=reason, n_faces=nf, n_persons=npers, dhash=f"{dh:016x}",
                             in_meal_window=in_win, source=src))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("frame_id")
        # delete any stale kept files that are no longer kept (e.g. after a config change)
        kept_ids = set(df.loc[df["kept"], "frame_id"])
        for p in cfg.paths.frames_kept.glob("*/*.jpg"):
            if p.stem not in kept_ids:
                p.unlink()
    write_table(cfg, df, "camera_frames")
    if verbose and not df.empty:
        summ = df.groupby("meal_id").agg(frames=("frame_id", "size"), kept=("kept", "sum"))
        dropped = df["drop_reason"].value_counts().to_dict()
        print(f"Camera: {len(df)} frames, {int(df['kept'].sum())} kept. Dropped: "
              f"{ {k: v for k, v in dropped.items() if k} }")
        print(summ.to_string())
        if cam.get("warn_if_outside_meal_window", True):
            out = df[df["in_meal_window"] == False]  # noqa: E712
            if len(out):
                print(f"  ! {len(out)} frames were taken outside their meal window; check the folder names.")
    return df
