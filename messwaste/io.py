"""Reading and writing tables. Parquet is the primary format.

If no parquet engine is installed (rare; Kaggle ships pyarrow) we fall back to
pickle (keeps dtypes exactly) and warn once, so the pipeline still runs on a bare
laptop. Install pyarrow before producing the submission.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from .config import Config

_ENGINE: str | None = None
_WARNED = False


def parquet_engine() -> str | None:
    global _ENGINE
    if _ENGINE is None:
        for eng in ("pyarrow", "fastparquet"):
            try:
                __import__(eng)
                _ENGINE = eng
                break
            except ImportError:
                continue
        else:
            _ENGINE = ""
    return _ENGINE or None


def table_path(cfg: Config, name: str) -> Path:
    ext = "parquet" if parquet_engine() else "pkl"
    return cfg.paths.parquet / f"{name}.{ext}"


def write_table(cfg: Config, df: pd.DataFrame, name: str) -> Path:
    path = table_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    global _WARNED
    if parquet_engine():
        df.to_parquet(path, index=False, engine=parquet_engine())
    else:
        if not _WARNED:
            warnings.warn("No parquet engine (pyarrow) found; tables are saved as .pkl. "
                          "`pip install pyarrow` before building the submission.", stacklevel=2)
            _WARNED = True
        df.to_pickle(path)
    return path


def read_table(cfg: Config, name: str, required: bool = False) -> pd.DataFrame:
    for ext in ("parquet", "pkl"):
        path = cfg.paths.parquet / f"{name}.{ext}"
        if path.exists():
            if ext == "parquet":
                return pd.read_parquet(path)
            return pd.read_pickle(path)
    if required:
        raise FileNotFoundError(
            f"Table '{name}' not found in {cfg.paths.parquet}. Run the earlier pipeline step first.")
    return pd.DataFrame()


def table_exists(cfg: Config, name: str) -> bool:
    return any((cfg.paths.parquet / f"{name}.{e}").exists() for e in ("parquet", "pkl"))
