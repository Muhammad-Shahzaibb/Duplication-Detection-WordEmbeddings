"""
JSON/schema + regex minimization utilities for Item Master.
Kept separate to keep `Item_Master_Duplicate_Engine.py` small and focused.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

# ─────────────────────────────────────────────
#  Minimal JSON schema — ITEMDESC only (order-invariant text + numeric)
# ─────────────────────────────────────────────

REGEX_SCHEMA_KEYS: tuple[str, ...] = ("text", "numeric")

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

_UNIT = r"(?:MM|CM|M\b|FT|INCH|IN\b|UF|MFD|NF|PF|VDC|VAC|VCA|MA\b|TEX|GSM|V\b|W\b|KW|MW|HP|BAR|PSI|KGF?|GM|G\b|LB|OZ|RPM|NOS|MTR|YDS|\"|')"
_DIM1 = re.compile(rf"^[\d.,/~\-]+\s*{_UNIT}$", re.IGNORECASE)

# Pre-tokenisation unit-split pattern: inserts a space between a digit and an
# immediately adjacent known unit so that "20mm" and "20 mm" tokenise identically.
# Longer unit strings are listed first to avoid partial matches (e.g. KGF before KG).
# The negative lookbehind (?<![/~\-]) prevents splitting fraction/range numerics
# such as "1/2IN" (the '2' is preceded by '/', so it is left untouched).
_UNIT_STUCK = re.compile(
    r"(?<![/~\-])(\d)"
    r"(INCH|MTR|MFD|KGF|GSM|VDC|VAC|VCA|NOS|YDS|RPM|PSI|BAR|KW|MW|HP|CM|MM|IN|GM|LB|OZ|TEX|MA|UF|NF|PF|KG|FT|M|W|G|V)"
    r"(?=[^a-zA-Z]|$)",
    re.IGNORECASE,
)
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


def _apply_itemdesc_layout_rules(desc: str) -> str:
    """
    Domain layout rules before tokenisation (hash codes, units, decimals, etc.).

    Derived from real patterns in ``vw_item_master_view2`` so variants like
    ``IM#445599``, ``IM # 445599``, ``20mm``, and ``20 mm`` tokenise the same way.
    """
    d = desc
    if not d:
        return d

    d = re.sub(r"\s+", " ", d)

    # Word-attached hash: IM# / RMS# / REF# → IM # / RMS # / REF #
    d = re.sub(r"([A-Za-z]+)#", r"\1 #", d)
    # Hash-colon article codes: IM#:445681 → IM # : 445681
    d = re.sub(r"#\s*:", "# :", d)
    # Symbol before digits: #445599 → # 445599, @120 → @ 120
    d = re.sub(r"([#@])\s*(\d)", r"\1 \2", d)
    # Colon before digits when not a ratio like 1:2 (PRONG:100 → PRONG: 100)
    d = re.sub(r"(?<!\d):\s*(\d)", r": \1", d)
    # European decimal comma: 1,5 → 1.5 (does not touch thousands like 1,000)
    d = re.sub(r"(\d),(\d{1,2})(?=[^\d]|$)", r"\1.\2", d)
    # Digit glued to parenthetical spec: 1015(186GSM) → 1015 (186GSM)
    d = re.sub(r"(\d)(\()", r"\1 \2", d)
    # Digit glued to underscore code: 80023220_120C → 80023220 _120C
    d = re.sub(r"(\d)(_[A-Za-z0-9])", r"\1 \2", d)
    # Split stuck units: 20mm → 20 mm, 170GSM → 170 GSM, 120TEX → 120 TEX
    d = _UNIT_STUCK.sub(r"\1 \2", d)

    return d.strip()


def normalize_item_description(desc: Any) -> str:
    """
    Canonical raw ITEMDESC before schema JSON, regex split, embedding, or cache.

    Step 1 in the Item Master pipeline (cleansing engine, variant check, bulk check,
    and embedding refresh). Unifies quotes, punctuation, spacing, and layout variants.
    """
    s = clean_str(desc)
    if not s:
        return s

    s = unicodedata.normalize("NFKC", s)
    # Unify curly/smart quotes to ASCII, then strip decorative quote characters.
    s = s.translate(str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u00b4": "'", "`": "'", "´": "'",
    }))
    # 40's / 40's → 40s
    s = re.sub(r"(?<=\w)['\u2019](?=\w)", "", s)
    s = re.sub(r"['\"`´]", " ", s)

    # Broad punctuation → space (keep . , / - ~ _ # @ : for codes, fractions, dimensions).
    s = re.sub(r"[?!;*&|^\\<>~\[\]{}()]+", " ", s)
    # Isolated colon not part of a code fragment → space
    s = re.sub(r"(?<![\w#@])\s*:\s*(?![\d])", " ", s)
    # Normalize slash/hyphen surrounded by spaces (keep 120/2, 10-12, 1.5mm intact)
    s = re.sub(r"(?<!\d)\s*/\s*(?!\d)", " ", s)
    s = re.sub(r"(?<![\d/])\s*-\s*(?![\d])", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    s = _strip_wrapping_quotes(s)
    s = _apply_itemdesc_layout_rules(s)
    return re.sub(r"\s+", " ", s).strip()


def regex_extract_attributes(desc: str) -> dict[str, str]:
    """
    Two-key normalization (order-invariant):
      1) text    : all non-numeric tokens (words, categories, etc.)
      2) numeric : any token containing a digit (codes, sizes, dimensions, etc.)
    """
    d = normalize_item_description(desc)
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
        # Strip trailing commas/semicolons that may be copy-paste artefacts
        # (e.g. "120/2," from an Excel cell → "120/2").
        # Only strip from the trailing edge; never touch internal characters.
        t = t.rstrip(",;")
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
    item_description: Any,
    item_type: Any = "",
    main_group: Any = "",
    sub_group: Any = "",
    item_code: Any | None = None,
    uom: Any | None = None,
    supplier: Any | None = None,
    doc_no: Any | None = None,
) -> dict[str, str]:
    """
    Convert one row into a base record for the pipeline.

    Only ITEMDESC is used for JSON minimization and embeddings (``text`` / ``numeric``).
    Hierarchy columns are kept as ``_item_*`` for duplicate-engine output only.
    """
    out: dict[str, str] = {
        "_item_description": normalize_item_description(item_description),
        "_item_type": clean_str(item_type),
        "_main_group": clean_str(main_group),
        "_sub_group": clean_str(sub_group),
    }
    if item_code is not None:
        out["_item_code"] = clean_str(item_code)
    if uom is not None:
        out["_uom"] = clean_str(uom)
    if item_code is not None or uom is not None:
        out["_supplier"] = clean_str(supplier) if supplier is not None else ""
    if doc_no is not None:
        out["_doc_no"] = clean_str(doc_no)
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
            for k in ("_item_description", "_item_type", "_main_group", "_sub_group"):
                obj.pop(k, None)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out_path


def write_json(rows: list[dict[str, str]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned: list[dict[str, Any]] = []
    for obj in rows:
        o = dict(obj)
        for k in ("_item_description", "_item_type", "_main_group", "_sub_group"):
            o.pop(k, None)
        cleaned.append(o)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    return out_path


def minimized_row_to_cache_payload(row: dict[str, Any]) -> dict[str, Any]:
    """One index-aligned cache row: embedding inputs + display fields for duplicate-engine output."""
    out: dict[str, Any] = {
        "text": row.get("text"),
        "numeric": row.get("numeric"),
        "ITEM_TYPE": clean_str(row.get("_item_type", "")),
        "MAINGROUP": clean_str(row.get("_main_group", "")),
        "SUBGROUP": clean_str(row.get("_sub_group", "")),
        "ITEMDESC": clean_str(row.get("_item_description", "")),
    }
    if "_item_code" in row:
        out["ITEM_CODE"] = clean_str(row.get("_item_code", ""))
        out["UOM"] = clean_str(row.get("_uom", ""))
        out["Supplier"] = clean_str(row.get("_supplier", ""))
    elif "_uom" in row:
        out["UOM"] = clean_str(row.get("_uom", ""))
    if "_doc_no" in row:
        out["DocNo"] = clean_str(row.get("_doc_no", ""))
    return out


def write_minimized_embedding_input_json(
    minimized: list[dict[str, Any]],
    *,
    jsonl_path: str | Path,
    json_path: str | Path,
) -> tuple[Path, Path]:
    """
    Persist the full Item Master row cache (text, numeric, and display columns),
    index-aligned with ``embeddings_cache.npy``, as JSONL + pretty JSON.
    """
    jsonl_path = Path(jsonl_path)
    json_path = Path(json_path)
    payload: list[dict[str, Any]] = [minimized_row_to_cache_payload(r) for r in minimized]
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in payload:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return jsonl_path, json_path


def schema_records_to_minimized(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert schema rows to minimized rows (regex extraction on ITEMDESC → text/numeric only)."""
    minimized: list[dict[str, Any]] = []
    for rec in records:
        desc = normalize_item_description(rec.get("_item_description") or "")
        extracted = regex_extract_attributes(desc)
        out_rec: dict[str, Any] = {
            "text": clean_str(extracted.get("text", "")) or None,
            "numeric": clean_str(extracted.get("numeric", "")) or None,
            "_item_description": desc,
            "_item_type": rec.get("_item_type", ""),
            "_main_group": rec.get("_main_group", ""),
            "_sub_group": rec.get("_sub_group", ""),
        }
        if "_item_code" in rec:
            out_rec["_item_code"] = rec.get("_item_code", "")
            out_rec["_uom"] = rec.get("_uom", "")
            out_rec["_supplier"] = rec.get("_supplier", "")
        elif "_uom" in rec:
            out_rec["_uom"] = rec.get("_uom", "")
        if "_doc_no" in rec:
            out_rec["_doc_no"] = rec.get("_doc_no", "")
        minimized.append(out_rec)
    return minimized

