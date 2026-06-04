"""
UOM canonical synonyms for duplicate detection.

Each key is a canonical unit code; values are known spellings/abbreviations from the
current Style Textile UOM list plus common variants (kg/kgs, m/mtr, pkt/packet, etc.).
"""

from __future__ import annotations

import re

# canonical_code -> all equivalent labels (including the canonical display form)
UOM_CANONICAL_GROUPS: dict[str, list[str]] = {
    "PLY": ["ply"],
    "TROLLEY": ["trolley"],
    "SQUARE_METER": ["sq meters", "sq meter", "sqm", "square meter", "square meters"],
    "LITER": ["liters", "liter", "ltr", "litre", "litres"],
    "KILOGRAM": ["kg", "kgs", "kilogram", "kilograms"],
    "MILLILITER": ["ml", "milliliter", "milliliters", "millilitre"],
    "FEET": ["feet", "foot", "ft"],
    "PIECE": ["pieces", "piece", "pcs", "pc", "nos", "no", "numbers", "number"],
    "DOZEN": ["dozen", "dozens", "dz"],
    "ROLL": ["rolls", "roll"],
    "BAG": ["bags", "bag"],
    "METER": ["meters", "meter", "m", "mtr", "mtrs", "metre", "metres"],
    "YARD": ["yards", "yard", "yd", "yds"],
    "LENGTH": ["length"],
    "COIL": ["coil", "coils"],
    "INCH": ["inches", "inch", "in"],
    "SQUARE_FEET": ["sft", "sq feet", "sq ft", "sqft", "square feet", "square foot"],
    "PAIR": ["pair", "pairs", "pr"],
    "PACKET": ["pkt", "packet", "packets", "pack"],
    "CARTON": ["cartons", "carton", "ctn"],
    "GROSS": ["grs", "gross"],
    "SET": ["set", "sets"],
    "CONE": ["cones", "cone"],
    "PANEL": ["pannels", "panel", "panels"],
    "BUCKET": ["bucket", "buckets"],
    "PAIL": ["pail", "pails"],
    "DRUM": ["drum", "drums"],
    "POUND": ["lbs", "lb", "pound", "pounds"],
    "GALLON": ["galon", "gallon", "gallons", "gal"],
    "GRAM": ["grms", "gram", "grams", "gm", "gms", "g"],
    "BOTTLE": ["btl", "bottle", "bottles"],
    "BOX": ["box", "boxes"],
    "TON": ["ton", "tons", "tonne", "tonnes"],
    "COUNTER": ["counter", "counters"],
    "CENTIMETER": ["cm", "centimeter", "centimeters", "centimetre"],
    "MILLIMETER": ["mm", "millimeter", "millimeters", "millimetre"],
    "DUMPER": ["dumper"],
    "LOT": ["lots", "lot"],
    "QUARTER": ["quarter", "quarters"],
    "RUNNING_FEET": ["rft", "running feet", "running foot"],
    "CUBIC_METER": ["cbm", "cubic meter", "cubic metres", "cubic metre"],
    "DOCUMENT": ["document", "documents", "doc"],
    "AT_ACTUAL": ["at actual", "actual"],
    "CUBIC_FEET": ["cft", "cubic feet", "cubic foot"],
    "PCT": ["pct", "percent", "percentage"],
    "CONTAINER": ["container", "containers"],
    "HOUR": ["hour", "hours", "hr", "hrs"],
    "KIT": ["kit", "kits"],
}


def _normalize_uom_key(text: str) -> str:
    """Lowercase alphanumeric key for alias lookup."""
    return re.sub(r"[^a-z0-9]", "", (text or "").strip().casefold())


def _build_alias_to_canonical() -> dict[str, str]:
    out: dict[str, str] = {}
    for canonical, labels in UOM_CANONICAL_GROUPS.items():
        for label in labels:
            key = _normalize_uom_key(label)
            if key:
                out[key] = canonical
        key = _normalize_uom_key(canonical)
        if key:
            out[key] = canonical
    return out


ALIAS_TO_CANONICAL: dict[str, str] = _build_alias_to_canonical()


def canonical_uom(text: str) -> str:
    """
    Map a UOM label to its canonical code.

    Unknown labels use an uppercased normalized token so new valid units still compare
    consistently (exact normalized match) until aliases are added to UOM_CANONICAL_GROUPS.
    """
    key = _normalize_uom_key(text)
    if not key:
        return ""
    return ALIAS_TO_CANONICAL.get(key, key.upper())
