"""
UOM canonical synonyms for duplicate detection.

Each key is a canonical unit code; values are known spellings/abbreviations from the
Style Textile UOM catalog, common variants (kg/kgs, m/mtr, pkt/packet), and textile-
industry units (denier, gsm, bale, skein, linear meter, square yard, etc.).
"""

from __future__ import annotations

import re

# canonical_code -> all equivalent labels (including the canonical display form)
UOM_CANONICAL_GROUPS: dict[str, list[str]] = {
    "PLY": ["ply", "plies"],
    "TROLLEY": ["trolley", "trolleys"],
    "SQUARE_METER": [
        "sq meters", "sq meter", "sqm", "square meter", "square meters",
        "sq metres", "sq metre", "square metre", "square metres", "m2", "m²",
    ],
    "LITER": ["liters", "liter", "ltr", "litre", "litres", "l"],
    "KILOGRAM": ["kg", "kgs", "kilogram", "kilograms", "kilo", "kilos"],
    "MILLILITER": ["ml", "milliliter", "milliliters", "millilitre", "millilitres"],
    "FEET": ["feet", "foot", "ft", "fts"],
    "PIECE": [
        "pieces", "piece", "pcs", "pc", "nos", "no", "numbers", "number",
        "ea", "each", "unit", "units", "item", "items",
    ],
    "DOZEN": ["dozen", "dozens", "dz", "doz"],
    "ROLL": ["rolls", "roll", "rl", "rls"],
    "BAG": ["bags", "bag", "sack", "sacks"],
    "METER": [
        "meters", "meter", "m", "mtr", "mtrs", "metre", "metres",
        "rm", "rmt", "running meter", "running meters", "running metre",
        "running metres", "linear meter", "linear meters", "linear metre",
        "linear metres", "lm", "lmt", "rmtr",
    ],
    "YARD": ["yards", "yard", "yd", "yds", "yardage"],
    "LENGTH": ["length", "len"],
    "COIL": ["coil", "coils"],
    "INCH": ["inches", "inch", "in", "ins"],
    "SQUARE_FEET": [
        "sft", "sq feet", "sq ft", "sqft", "square feet", "square foot",
        "sq foot", "sf",
    ],
    "SQUARE_YARD": [
        "sq yard", "sq yards", "sq yd", "sq yds", "square yard", "square yards",
        "sy", "yd2", "yd²",
    ],
    "PAIR": ["pair", "pairs", "pr", "prs"],
    "PACKET": ["pkt", "packet", "packets", "pack", "packs", "pkts"],
    "CARTON": ["cartons", "carton", "ctn", "ctns"],
    "GROSS": ["grs", "gross", "gro"],
    "SET": ["set", "sets"],
    "CONE": ["cones", "cone", "cn"],
    "PANEL": ["pannels", "panel", "panels", "pnl", "pnls"],
    "BUCKET": ["bucket", "buckets"],
    "PAIL": ["pail", "pails"],
    "DRUM": ["drum", "drums"],
    "POUND": ["lbs", "lb", "pound", "pounds", "ps"],
    "GALLON": ["galon", "gallon", "gallons", "gal", "gals"],
    "GRAM": ["grms", "gram", "grams", "gm", "gms", "g", "gr", "grm"],
    "BOTTLE": ["btl", "bottle", "bottles", "btls"],
    "BOX": ["box", "boxes", "bx"],
    "TON": ["ton", "tons", "tonne", "tonnes", "t", "mt", "metric ton"],
    "COUNTER": ["counter", "counters"],
    "CENTIMETER": ["cm", "centimeter", "centimeters", "centimetre", "centimetres", "cms"],
    "MILLIMETER": ["mm", "millimeter", "millimeters", "millimetre", "millimetres", "mms"],
    "DUMPER": ["dumper", "dumpers"],
    "LOT": ["lots", "lot"],
    "QUARTER": ["quarter", "quarters", "qtr", "qtrs"],
    "RUNNING_FEET": ["rft", "running feet", "running foot", "rf"],
    "CUBIC_METER": ["cbm", "cubic meter", "cubic meters", "cubic metres", "cubic metre"],
    "DOCUMENT": ["document", "documents", "doc", "docs"],
    "AT_ACTUAL": ["at actual", "actual", "atactual"],
    "CUBIC_FEET": ["cft", "cubic feet", "cubic foot", "cu ft"],
    "PCT": ["pct", "percent", "percentage", "%", "per cent"],
    "CONTAINER": ["container", "containers", "cont"],
    "HOUR": ["hour", "hours", "hr", "hrs"],
    "KIT": ["kit", "kits"],
    # Textile / fabric industry units
    "BALE": ["bale", "bales"],
    "SKEIN": ["skein", "skeins"],
    "HANK": ["hank", "hanks"],
    "SPOOL": ["spool", "spools"],
    "REEL": ["reel", "reels"],
    "BUNDLE": ["bundle", "bundles", "bdl", "bdls"],
    "CASE": ["case", "cases", "cs"],
    "PALLET": ["pallet", "pallets", "plt", "plts"],
    "SKID": ["skid", "skids"],
    "CARD": ["card", "cards"],
    "STRIP": ["strip", "strips"],
    "REAM": ["ream", "reams"],
    "GSM": ["gsm", "g/m2", "g/m²", "grams per square meter", "grams per sq meter"],
    "DENIER": ["denier", "den", "deniers"],
    "DTEX": ["dtex", "d tex"],
    "TEX": ["tex"],
    "OUNCE": ["ounce", "ounces", "oz", "ozs"],
    "MICRON": ["micron", "microns", "um", "µm"],
    "WARP": ["warp", "warps", "warp end", "warp ends"],
    "WEFT": ["weft", "wefts", "weft pick", "weft picks", "pick", "picks"],
    "NE_COUNT": ["ne", "n e", "english count", "cotton count"],
    "NM_COUNT": ["nm", "n m", "metric count"],
    "OZ_PER_SQ_YD": [
        "oz/yd", "oz per sq yd", "oz per square yard", "ounce per square yard",
        "oz/sq yd", "oz sy",
    ],
    "WIDTH": ["width", "wd", "w"],
    "WEIGHT": ["weight", "wgt", "wt"],
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
