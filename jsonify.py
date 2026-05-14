"""
JSON/schema + regex minimization utilities for Item Master.
Kept separate to keep `Item_Master_Duplicate_Engine.py` small and focused.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# ─────────────────────────────────────────────
#  Minimal JSON schema (3 hierarchy + 2 keys from ITEMDESC)
# ─────────────────────────────────────────────

REGEX_SCHEMA_KEYS: tuple[str, ...] = (
    "item_type",
    "main_group",
    "sub_group",
    # Two-key normalization to make word-order duplicates converge
    "text",
    "numeric",
)

REGEX_EXTRACTION_KEYS: tuple[str, ...] = ("text", "numeric")


def clean_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v)
    if s.lower() in {"nan", "none"}:
        return ""
    return s.strip()


# ─────────────────────────────────────────────
#  REGEX / RULE-BASED EXTRACTION (order-invariant)
#  Keys: text, numeric
# ─────────────────────────────────────────────

MATERIALS = {
    "MS",
    "SS",
    "GI",
    "PVC",
    "PPR",
    "BRASS",
    "COPPER",
    "ALUMINUM",
    "ALUMINIUM",
    "STEEL",
    "IRON",
    "RUBBER",
    "PLASTIC",
    "NYLON",
    "PE",
    "HDPE",
    "LDPE",
    "POLYESTER",
    "COTTON",
    "NEOPRENE",
    "FORGED",
    "FORGE",
    "CAST",
    "STAINLESS",
    "GALVANIZED",
    "CHROME",
    "ZINC",
    "BRONZE",
    "CERAMIC",
    "CARBON",
    "FIBER",
    "FIBRE",
    "ACRYLIC",
    "LEATHER",
    "FOAM",
    "SPONGE",
    "WOOD",
    "GLASS",
    "SILICONE",
    "TEFLON",
    "MILD",
    "GALV",
    "CI",
    "WI",
    "MONEL",
    "INCONEL",
    "TITANIUM",
    "ALLOY",
    "COMPOSITE",
    "SYNTHETIC",
    "ABS",
    "PU",
}

COLORS = {
    "RED",
    "BLUE",
    "GREEN",
    "WHITE",
    "BLACK",
    "YELLOW",
    "GREY",
    "GRAY",
    "BROWN",
    "ORANGE",
    "PURPLE",
    "PINK",
    "SILVER",
    "GOLD",
    "VIOLET",
    "BEIGE",
    "NAVY",
    "MAROON",
    "IVORY",
    "CREAM",
    "TEAL",
    "CYAN",
    "MAGENTA",
    "LIME",
    "OLIVE",
    "INDIGO",
    "TURQUOISE",
    "CORAL",
    "SALMON",
    "KHAKI",
    "TRANSPARENT",
    "CLEAR",
    "GLOSSY",
    "MATTE",
    "SATIN",
    "METALLIC",
}

DIM_LABELS = {
    "THICKNESS",
    "LENGTH",
    "WIDTH",
    "HEIGHT",
    "DIA",
    "DIAMETER",
    "OD",
    "ID",
    "SIZE",
    "PITCH",
    "BORE",
    "GAUGE",
    "SCHEDULE",
    "SCH",
    "WEIGHT",
    "CAPACITY",
    "RATING",
    "PRESSURE",
    "TEMP",
    "TEMPERATURE",
    "VOLTAGE",
    "CURRENT",
    "POWER",
    "FREQ",
    "FREQUENCY",
    "SPEED",
    "FLOW",
    "STROKE",
    "TRAVEL",
    "RANGE",
    "SPAN",
    "W",
    "L",
    "H",
    "D",
    "T",
    "R",
}

KNOWN_PREFIXES = {
    "SEWING THREAD",
    "EMB THREAD",
    "BOBBIN THREAD",
    "JUNCTION BOX",
    "ALLEN KEY",
    "PRESSURE GAUGE",
    "FLOW METER",
    "ENERGY METER",
    "ELECTRIC MOTOR",
    "U BOLT",
    "J BOLT",
    "EYE BOLT",
    "STUD BOLT",
    "ANCHOR BOLT",
    "HEX BOLT",
    "ALLEN BOLT",
    "H BEAM",
    "I BEAM",
    "HANDLE TAP",
    "VALVE BUTTERFLY",
    "VALVE STEAM",
    "VALVE GLOBE",
    "VALVE GATE",
    "VALVE NRV",
    "VALVE BALL",
    "VALVE CHECK",
    "VALVE SAFETY",
    "BEARING SELF",
    "BEARING DEEP",
    "BEARING ROLLER",
    "BEARING THRUST",
    "CABLE POWER",
    "CABLE CONTROL",
    "CABLE SIGNAL",
}

_UNIT = r"(?:MM|CM|M\b|FT|INCH|IN\b|UF|MFD|NF|PF|VDC|VAC|VCA|MA\b|TEX|V\b|W\b|KW|MW|HP|BAR|PSI|KGF?|GM|G\b|LB|OZ|RPM|NOS|MTR|YDS|\"|')"
_DIM1 = re.compile(rf"^[\d.,/~\-]+\s*{_UNIT}$", re.IGNORECASE)
_DIM2 = re.compile(r'^[\d.]+[Xx\*][\d.]+(?:[Xx\*][\d.]+)?(?:MM|CM|"|FT|IN)?$', re.IGNORECASE)
_DIM3 = re.compile(r'^[\d.]+["\'](?:[Xx\*][\d.]+["\'])+$', re.IGNORECASE)
_NUM = re.compile(r"^[\d.,/~\-]+$")
_UNIT_W = re.compile(rf"^{_UNIT}$", re.IGNORECASE)


def _is_dim(tok: str) -> bool:
    return bool(_DIM1.match(tok) or _DIM2.match(tok) or _DIM3.match(tok) or _NUM.match(tok) or re.match(r'^[\d.,/~\-]+["\']$', tok))


def _is_unit(tok: str) -> bool:
    return bool(_UNIT_W.match(tok))


def _clean_tok(tok: str) -> str:
    # Remove punctuation including ':' so W: becomes W and matches DIM_LABELS
    return re.sub(r"[().#,:]", "", tok).upper()


def _strip_wrapping_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1].strip()
    return s


def regex_extract_attributes(desc: str) -> dict[str, str]:
    """
    Two-key normalization (order-invariant):
      1) text    : all non-numeric tokens (words, categories, etc.)
      2) numeric : any token containing a digit (codes, sizes, dimensions, etc.)
    """
    d = clean_str(desc)
    out = {k: "" for k in REGEX_EXTRACTION_KEYS}
    if not d:
        return out

    tokens = re.findall(r"\([^)]*\)|\"[^\"]*\"|\S+", d)
    text_parts: list[str] = []
    num_parts: list[str] = []

    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if re.search(r"\d", t):
            num_parts.append(t)
        else:
            text_parts.append(t)

    out["text"] = clean_str(" ".join(text_parts))
    out["numeric"] = clean_str(" ".join(num_parts))
    return out


def row_to_schema_json(
    *,
    item_type: Any,
    main_group: Any,
    sub_group: Any,
    item_description: Any,
) -> dict[str, str]:
    """Convert one row into a base JSON object (ITEMDESC stored temporarily as `_item_description`)."""
    out: dict[str, Any] = {k: None for k in REGEX_SCHEMA_KEYS}
    out["item_type"] = clean_str(item_type)
    out["main_group"] = clean_str(main_group)
    out["sub_group"] = clean_str(sub_group)
    out["_item_description"] = clean_str(item_description)
    return out


def dataframe_to_schema_jsons(
    df: pd.DataFrame,
    *,
    col_item_type: str = "ITEM_TYPE",
    col_main_group: str = "MAINGROUP",
    col_sub_group: str = "SUBGROUP",
    col_item_description: str = "ITEMDESC",
) -> list[dict[str, str]]:
    missing = [c for c in [col_item_type, col_main_group, col_sub_group, col_item_description] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}. Found columns: {list(df.columns)}")

    jsons: list[dict[str, str]] = []
    for _, r in df.iterrows():
        jsons.append(
            row_to_schema_json(
                item_type=r.get(col_item_type),
                main_group=r.get(col_main_group),
                sub_group=r.get(col_sub_group),
                item_description=r.get(col_item_description),
            )
        )
    return jsons


def excel_to_schema_jsons(
    excel_path: str | Path,
    *,
    sheet_name: int | str = 0,
    rows: int | None = None,
    col_item_type: str = "ITEM_TYPE",
    col_main_group: str = "MAINGROUP",
    col_sub_group: str = "SUBGROUP",
    col_item_description: str = "ITEMDESC",
) -> list[dict[str, str]]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str, header=0)
    df = df.dropna(how="all").reset_index(drop=True)
    if rows is not None:
        df = df.head(int(rows)).copy()
    return dataframe_to_schema_jsons(
        df,
        col_item_type=col_item_type,
        col_main_group=col_main_group,
        col_sub_group=col_sub_group,
        col_item_description=col_item_description,
    )


def write_jsonl(rows: list[dict[str, str]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in rows:
            obj = dict(obj)
            obj.pop("_item_description", None)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out_path


def write_json(rows: list[dict[str, str]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned: list[dict[str, Any]] = []
    for obj in rows:
        o = dict(obj)
        o.pop("_item_description", None)
        cleaned.append(o)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    return out_path


def schema_records_to_minimized(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert schema rows to minimized rows (regex extraction for text/numeric)."""
    minimized: list[dict[str, Any]] = []
    for rec in records:
        desc = (rec.get("_item_description") or "").strip()
        extracted = regex_extract_attributes(desc)
        out_rec: dict[str, Any] = {
            "item_type": rec.get("item_type", ""),
            "main_group": rec.get("main_group", ""),
            "sub_group": rec.get("sub_group", ""),
            "text": clean_str(extracted.get("text", "")) or None,
            "numeric": clean_str(extracted.get("numeric", "")) or None,
            "_item_description": rec.get("_item_description", ""),
        }
        minimized.append(out_rec)
    return minimized

