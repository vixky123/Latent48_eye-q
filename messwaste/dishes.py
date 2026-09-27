"""Dish-name normalisation and unit conversion to kg."""
from __future__ import annotations

import re

from .config import Config


class DishBook:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dishes = cfg.dishes
        self.alias: dict[str, str] = {}
        for name, spec in self.dishes.items():
            self.alias[self._norm(name)] = name
            for a in spec.get("aliases", []) or []:
                self.alias[self._norm(a)] = name

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

    def canonical(self, name: str | None) -> str | None:
        """Map a typed dish name to its canonical name. Unknown names are kept (normalised)."""
        if name is None:
            return None
        n = self._norm(name)
        return self.alias.get(n, n.replace(" ", "_"))

    def is_known(self, name: str) -> bool:
        return name in self.dishes

    def parse_menu(self, menu: str | None) -> list[str]:
        if not menu:
            return []
        out: list[str] = []
        for part in re.split(r"[;,|]", str(menu)):
            if part.strip():
                d = self.canonical(part)
                if d and d not in out:
                    out.append(d)
        return out

    def spec(self, dish: str) -> dict:
        return self.dishes.get(dish, {})

    def to_kg(self, dish: str, qty: float | None, unit: str | None) -> tuple[float | None, str]:
        """Return (kg, note). note explains any assumption used."""
        if qty is None or unit is None:
            return None, "missing"
        unit = unit.lower()
        s = self.spec(dish)
        if unit == "kg":
            return float(qty), ""
        if unit == "g":
            return float(qty) / 1000.0, ""
        if unit in ("l", "ml"):
            dens = float(s.get("density_kg_per_l", 1.0))
            litres = float(qty) / (1000.0 if unit == "ml" else 1.0)
            return litres * dens, f"litres x {dens} kg/l"
        if unit == "pieces":
            pk = s.get("piece_kg")
            if pk is None:
                return None, f"no piece_kg for '{dish}' in config"
            return float(qty) * float(pk), f"pieces x {pk} kg"
        return None, f"unknown unit {unit}"
