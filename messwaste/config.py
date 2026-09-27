"""Config loading and path resolution (Kaggle-aware)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def _find_raw_root_on_kaggle() -> Path | None:
    base = Path("/kaggle/input")
    if not base.exists():
        return None
    # look up to 4 levels deep for csv/meals.csv
    for meals in sorted(base.glob("**/csv/meals.csv")):
        if len(meals.relative_to(base).parts) > 6:
            continue
        # skip copies of this repo (e.g. demo data inside an uploaded code dataset)
        if any((a / "messwaste" / "cli.py").exists() for a in meals.parents):
            continue
        return meals.parent.parent
    return None


@dataclass
class Paths:
    raw_root: Path
    out_root: Path

    # ---- raw (may be read-only on Kaggle) ----
    @property
    def raw_csv(self) -> Path:
        return self.raw_root / "csv"

    @property
    def raw_frames(self) -> Path:
        return self.raw_root / "frames"

    # ---- outputs ----
    @property
    def parquet(self) -> Path:
        return self.out_root / "parquet"

    @property
    def frames_kept(self) -> Path:
        return self.out_root / "frames_kept"

    @property
    def sealed(self) -> Path:
        return self.out_root / "forecasts" / "sealed"

    @property
    def reports(self) -> Path:
        return self.out_root / "reports"

    @property
    def labelling(self) -> Path:
        return self.out_root / "labelling"

    @property
    def submission(self) -> Path:
        return self.out_root / "submission"

    def sealed_search_dirs(self) -> list[Path]:
        """Sealed forecasts may live in out_root (just made) or be re-uploaded under raw_root."""
        dirs = [self.sealed, self.raw_root / "forecasts" / "sealed", self.raw_root / "sealed"]
        return [d for d in dirs if d.exists()]

    def ensure(self) -> None:
        for p in [self.parquet, self.frames_kept, self.sealed, self.reports,
                  self.labelling, self.submission]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    raw: dict[str, Any]
    paths: Paths
    source_file: Path
    tz: ZoneInfo = field(init=False)

    def __post_init__(self):
        self.tz = ZoneInfo(self.raw.get("project", {}).get("timezone", "Asia/Kolkata"))

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {}) or {}

    @property
    def dishes(self) -> dict[str, dict[str, Any]]:
        return {k.strip().lower(): (v or {}) for k, v in (self.raw.get("dishes") or {}).items()}

    @property
    def meal_slots(self) -> dict[str, dict[str, Any]]:
        return self.raw.get("meal_slots") or {}


def load_config(path: str | os.PathLike | None = None,
                raw_root: str | None = None,
                out_root: str | None = None) -> Config:
    cfg_path = Path(path) if path else Path(os.environ.get("MW_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    p = raw.get("paths", {}) or {}
    rr = raw_root or os.environ.get("MW_RAW_ROOT") or p.get("raw_root") or ""
    orr = out_root or os.environ.get("MW_OUT_ROOT") or p.get("out_root") or ""

    if not rr:
        if is_kaggle():
            found = _find_raw_root_on_kaggle()
            rr = str(found) if found else "/kaggle/working/raw"
        else:
            rr = str(PROJECT_DIR / "data" / "raw")
    if not orr:
        orr = "/kaggle/working/out" if is_kaggle() else str(PROJECT_DIR / "data" / "out")

    paths = Paths(raw_root=Path(rr).expanduser().resolve(), out_root=Path(orr).expanduser().resolve())
    paths.ensure()
    return Config(raw=raw, paths=paths, source_file=cfg_path)
