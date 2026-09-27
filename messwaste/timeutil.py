from __future__ import annotations

from datetime import datetime

import pandas as pd

from .config import Config


def local_dt(cfg: Config, date: str | None, hhmm: str | None) -> pd.Timestamp | None:
    if not date or not hhmm or pd.isna(date) or pd.isna(hhmm):
        return None
    return pd.Timestamp(f"{date} {hhmm}").tz_localize(cfg.tz)


def now_local(cfg: Config) -> pd.Timestamp:
    return pd.Timestamp(datetime.now(cfg.tz))


def parse_local(cfg: Config, s: str) -> pd.Timestamp:
    """Parse 'YYYY-MM-DD HH:MM' (local) or any ISO string with offset."""
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize(cfg.tz)
    return ts.tz_convert(cfg.tz)
