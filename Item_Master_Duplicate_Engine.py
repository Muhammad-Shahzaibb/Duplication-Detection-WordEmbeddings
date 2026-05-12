import csv
import json
import hashlib
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from Config import EMBED_CACHE_FILE as _DEFAULT_EMBED_CACHE_PATH



# ─────────────────────────────────────────────
#  STEP 2 CONFIG — EMBEDDINGS + FAISS
# ─────────────────────────────────────────────

EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model id (downloads once)
EMBED_BATCH    = 256     # encoding batch size (CPU)
EMBED_NEAR_THRESHOLD  = 0.998  # cosine >= 0.995 and < 1.0    → near-duplicate
EMBED_EXACT_THRESHOLD = 1.0    # cosine == 1.0                 → exact duplicate
EMBED_EXACT_EPS       = 1e-7  # float tolerance for cosine==1
EMBED_DUP_THRESHOLD   = EMBED_NEAR_THRESHOLD  # minimum score to appear in output
EMBED_CACHE_FILE      = str(_DEFAULT_EMBED_CACHE_PATH)

# Global singleton embedder (loaded once per process)
_EMBEDDER: SentenceTransformer | None = None


def get_embedder(model_id: str = EMBED_MODEL) -> SentenceTransformer:
    """
    Loads the HF embedding model once and reuses it.
    On first run, this downloads the model to the HuggingFace cache on disk.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        # Tune CPU threading for typical on-prem servers (adjust if needed).
        try:
            import torch
            torch.set_num_threads(6)
            torch.set_num_interop_threads(2)
        except Exception:
            pass
        _EMBEDDER = SentenceTransformer(model_id, device="cpu")
    return _EMBEDDER


# ─────────────────────────────────────────────
#  Minimal JSON schema (3 hierarchy + 2 keys from ITEMDESC)
REGEX_SCHEMA_KEYS: tuple[str, ...] = (
    "item_type",
    "main_group",
    "sub_group",
    # Two-key normalization to make word-order duplicates converge
    "text",
    "numeric",
)

REGEX_EXTRACTION_KEYS: tuple[str, ...] = ("text", "numeric")


def _clean_str(v: Any) -> str:
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

# Known multi-word prefixes to keep as item_name
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
    return bool(
        _DIM1.match(tok)
        or _DIM2.match(tok)
        or _DIM3.match(tok)
        or _NUM.match(tok)
        or re.match(r'^[\d.,/~\-]+["\']$', tok)
    )


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


def _parse_desc_regex(raw: str) -> dict[str, str]:
    """
    Token-based rule parsing into:
    - item_name: leading noun phrase/prefix
    - qualifier: middle non-dimension/non-attribute tokens
    - dimension: all numeric+unit tokens and label/value pairs
    - attribute: material/color/grade tokens
    """
    desc = _strip_wrapping_quotes(raw)
    if not desc:
        return {"item_name": "", "qualifier": "", "dimension": "", "attribute": ""}

    tokens = re.findall(r"\([^)]*\)|\"[^\"]*\"|\S+", desc)
    if not tokens:
        return {"item_name": "", "qualifier": "", "dimension": "", "attribute": ""}

    # Choose item_name using known prefixes, else first 1-2 tokens heuristics
    three = " ".join(t.upper() for t in tokens[:3]) if len(tokens) >= 3 else ""
    two = " ".join(t.upper() for t in tokens[:2]) if len(tokens) >= 2 else ""
    if three in KNOWN_PREFIXES:
        item_name = " ".join(tokens[:3])
        name_end = 3
    elif two in KNOWN_PREFIXES:
        item_name = " ".join(tokens[:2])
        name_end = 2
    else:
        item_name = tokens[0]
        name_end = 1
        if len(tokens) > 1:
            t1 = _clean_tok(tokens[1])
            if (
                t1 not in MATERIALS
                and t1 not in COLORS
                and not _is_dim(tokens[1])
                and not _is_unit(tokens[1])
                and t1 not in DIM_LABELS
                and re.match(r"^[A-Z\-]{2,}$", t1)
            ):
                item_name += " " + tokens[1]
                name_end = 2

    qualifiers: list[str] = []
    dimensions: list[str] = []
    attributes: list[str] = []

    i = name_end
    while i < len(tokens):
        tok = tokens[i]
        tup = _clean_tok(tok)

        # Range pattern: N TO M [unit]
        if (
            _is_dim(tok)
            and i + 2 < len(tokens)
            and tokens[i + 1].upper() in {"TO", "~", "-"}
            and _is_dim(tokens[i + 2])
        ):
            combo = tok + " " + tokens[i + 1] + " " + tokens[i + 2]
            i += 3
            if i < len(tokens) and _is_unit(tokens[i]):
                combo += " " + tokens[i]
                i += 1
            dimensions.append(combo)
            continue

        # Dim label (optional colon already handled in _clean_tok) + value
        if tup in DIM_LABELS:
            if i + 1 < len(tokens) and (_is_dim(tokens[i + 1]) or _is_unit(tokens[i + 1])):
                combo = tok + " " + tokens[i + 1]
                i += 2
                if i < len(tokens) and _is_unit(tokens[i]):
                    combo += " " + tokens[i]
                    i += 1
                dimensions.append(combo)
            else:
                qualifiers.append(tok)
                i += 1
            continue

        # Standalone dimension
        if _is_dim(tok):
            dtok = tok
            i += 1
            if i < len(tokens) and _is_unit(tokens[i]):
                dtok += " " + tokens[i]
                i += 1
            dimensions.append(dtok)
            continue

        # Material / color (fixed vocab)
        if tup in MATERIALS or tup in COLORS:
            if i + 1 < len(tokens):
                nxt = _clean_tok(tokens[i + 1])
                if nxt in MATERIALS or nxt in COLORS:
                    attributes.append(tok + " " + tokens[i + 1])
                    i += 2
                    continue
            attributes.append(tok)
            i += 1
            continue

        # Parenthetical → qualifier
        if re.match(r"^\(.*\)$", tok):
            qualifiers.append(tok)
            i += 1
            continue

        qualifiers.append(tok)
        i += 1

    return {
        "item_name": _clean_str(item_name),
        "qualifier": _clean_str(" ".join(qualifiers)),
        "dimension": _clean_str(",  ".join(dimensions)),
        "attribute": _clean_str(",  ".join(attributes)),
    }


def _coverage_missing_tokens(raw: str, parsed: dict[str, str]) -> set[str]:
    """
    Coverage guard: ensure tokens from raw appear somewhere in parsed fields.
    Returns the set of tokens in raw missing from the concatenated parsed text.
    """
    raw_u = (raw or "").upper()
    got_u = " ".join(
        x for x in [parsed.get("item_name", ""), parsed.get("qualifier", ""), parsed.get("dimension", ""), parsed.get("attribute", "")]
        if x
    ).upper()
    orig = set(re.findall(r"[A-Z0-9/.\"'\-]+", raw_u))
    rec = set(re.findall(r"[A-Z0-9/.\"'\-]+", got_u))
    return orig - rec


def regex_extract_attributes(desc: str) -> dict[str, str]:
    """
    Two-key normalization (order-invariant):
      1) text    : all non-numeric tokens (words, categories, etc.)
      2) numeric : any token containing a digit (codes, sizes, dimensions, etc.)

    This makes these two descriptions converge to the same JSON:
      - "6A0250091009 BRACKET ASSY"
      - "BRACKET ASSY 6A0250091009"
    """
    d = _clean_str(desc)
    out = {k: "" for k in REGEX_EXTRACTION_KEYS}
    if not d:
        return out

    # Tokenize preserving quoted parts / parenthesis groups, but keep it simple.
    tokens = re.findall(r"\([^)]*\)|\"[^\"]*\"|\S+", d)
    text_parts: list[str] = []
    num_parts: list[str] = []

    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        # Any digit anywhere → numeric bucket (covers codes like 6A0250091009, 1/2", 12MMX30MM, etc.)
        if re.search(r"\d", t):
            num_parts.append(t)
        else:
            text_parts.append(t)

    # Normalize spacing and make representation stable.
    out["text"] = _clean_str(" ".join(text_parts))
    out["numeric"] = _clean_str(" ".join(num_parts))
    return out


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Robustly parse a JSON object from LLM output.
    - Accepts raw JSON or JSON fenced in markdown.
    - If there is extra text, extracts the first {...} block.
    """
    if not text:
        return None

    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()

    # Fast path
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Fallback: find first JSON object span
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def row_to_schema_json(
    *,
    item_type: Any,
    main_group: Any,
    sub_group: Any,
    item_description: Any,
) -> dict[str, str]:
    """Phase 1: Convert one Excel row into a base JSON object."""
    # Regex-only: output uses REGEX_SCHEMA_KEYS; ITEMDESC stored temporarily as _item_description (removed on write)
    out: dict[str, Any] = {k: None for k in REGEX_SCHEMA_KEYS}
    out["item_type"] = _clean_str(item_type)
    out["main_group"] = _clean_str(main_group)
    out["sub_group"] = _clean_str(sub_group)
    # Temporarily store description for extraction (removed before writing)
    out["_item_description"] = _clean_str(item_description)
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


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — EMBEDDINGS + FAISS DUPLICATE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def build_embedding_text(row: dict[str, Any]) -> str:
    """
    Flatten one JSON row into a single string for embedding.
    Only the 4 structured keys are used (hierarchy adds context).
    item_description is intentionally excluded — it was already parsed into keys.
    """
    parts = [
        row.get("item_type") or "",
        row.get("main_group") or "",
        row.get("sub_group") or "",
        row.get("text") or "",
        row.get("numeric") or "",
    ]
    return " ".join(p.strip() for p in parts if p.strip())


def _texts_digest(texts: list[str]) -> str:
    """Stable hash of embedding input texts for cache validation."""
    h = hashlib.sha256()
    for t in texts:
        b = (t or "").encode("utf-8", errors="ignore")
        h.update(len(b).to_bytes(8, byteorder="little", signed=False))
        h.update(b)
    return h.hexdigest()


def _texts_digest_records(records: list[dict[str, Any]]) -> str:
    """Stable hash of embedding input texts for cache validation (streaming, low-RAM)."""
    h = hashlib.sha256()
    for r in records:
        t = build_embedding_text(r)
        b = (t or "").encode("utf-8", errors="ignore")
        h.update(len(b).to_bytes(8, byteorder="little", signed=False))
        h.update(b)
    return h.hexdigest()


def embed_texts_local(
    texts: list[str],
    *,
    model_id: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH,
) -> np.ndarray:
    """
    Compute embeddings locally (CPU) using a HuggingFace SentenceTransformer model.

    - Model is downloaded once to HF cache.
    - normalize_embeddings=True makes cosine similarity easy (dot product == cosine).
    - Returns float32 matrix of shape (N, D).
    """
    embedder = get_embedder(model_id)
    try:
        vecs = embedder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
    except Exception as e:
        # On constrained machines, a large batch may blow up memory.
        msg = str(e).lower()
        oomish = any(s in msg for s in ("out of memory", "cannot allocate memory", "alloc", "memoryerror"))
        if oomish and batch_size > 1 and len(texts) > 1:
            new_bs = max(1, batch_size // 2)
            mid = len(texts) // 2
            left = embed_texts_local(texts[:mid], model_id=model_id, batch_size=new_bs)
            right = embed_texts_local(texts[mid:], model_id=model_id, batch_size=new_bs)
            return np.vstack([left, right]).astype(np.float32, copy=False)
        raise


def build_faiss_index(
    records: list[dict[str, Any]],
    *,
    model: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH,
    cache_path: str | Path | None = EMBED_CACHE_FILE,
    reuse_only: bool = False,
) -> tuple[Any, np.ndarray]:
    """
    Embed all records IN PARALLEL and build a FAISS flat inner-product index.

    Note:
      - Embeddings are computed locally via `SentenceTransformer.encode()`.
      - Torch uses CPU threads internally; we batch for throughput and to
        limit memory use.

    Why inner-product on normalized vectors = cosine similarity:
      - We L2-normalize every vector so ||v|| = 1.
      - Then dot(v_i, v_j) = cos(angle) since norms cancel.
      - cosine = 1.0 → identical; cosine ~ 0 → unrelated.
    """
    try:
        import faiss  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu")

    total = len(records)
    digest = _texts_digest_records(records)

    cache_npy: Path | None = Path(cache_path) if cache_path else None
    cache_meta: Path | None = None
    if cache_npy is not None:
        cache_meta = cache_npy.with_suffix(cache_npy.suffix + ".meta.json")
        if cache_npy.exists() and cache_meta.exists():
            try:
                meta = json.loads(cache_meta.read_text(encoding="utf-8"))
                mat_cached = np.load(cache_npy)
                ok = (
                    isinstance(meta, dict)
                    and meta.get("model") == model
                    and int(meta.get("rows", -1)) == total
                    and str(meta.get("text_digest", "")) == digest
                    and mat_cached.ndim == 2
                    and int(mat_cached.shape[0]) == total
                )
                if ok:
                    print(f"  Reusing cached embeddings: {cache_npy}")
                    mat_cached = mat_cached.astype(np.float32)
                    dim = mat_cached.shape[1]
                    index = faiss.IndexFlatIP(dim)
                    index.add(mat_cached)
                    return index, mat_cached
                if reuse_only:
                    raise RuntimeError("Embedding cache present but does not match current rows/model/content. Refusing to recompute (reuse_only).")
                print("  Cache present but does not match rows/model/content. Recomputing embeddings...")
            except Exception as e:
                # Preserve intentional reuse_only failures
                if reuse_only and isinstance(e, RuntimeError):
                    raise
                if reuse_only:
                    raise RuntimeError("Embedding cache unreadable. Refusing to recompute (reuse_only).") from e
                print("  Cache unreadable. Recomputing embeddings...")
        elif reuse_only:
            raise RuntimeError("Embedding cache not found. Refusing to compute embeddings (reuse_only).")

    # Local embedding in batches, written directly to cache (memory-mapped) to avoid high RAM usage.
    if total == 0:
        mat = np.zeros((0, 0), dtype=np.float32)
    else:
        cache_npy_out: Path | None = cache_npy
        # If caching disabled, still avoid large RAM by using a temp memmap in the cache location.
        if cache_npy_out is None:
            cache_npy_out = Path("embeddings_tmp.npy")

        # Embed first batch to discover vector dimension, then create a memmap .npy and fill it.
        first_end = min(batch_size, total)
        first_texts = [build_embedding_text(r) for r in records[:first_end]]
        first_vecs = embed_texts_local(first_texts, model_id=model, batch_size=min(first_end, batch_size))
        dim = int(first_vecs.shape[1])

        mm = np.lib.format.open_memmap(
            cache_npy_out,
            mode="w+",
            dtype=np.float32,
            shape=(total, dim),
        )
        mm[:first_end] = first_vecs
        print(f"  Embedded {first_end}/{total} rows...")

        for start in range(first_end, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = [build_embedding_text(r) for r in records[start:end]]
            vecs = embed_texts_local(batch_texts, model_id=model, batch_size=min(len(batch_texts), batch_size))
            mm[start:end] = vecs
            print(f"  Embedded {end}/{total} rows...")

        # Flush to disk and load as a normal ndarray (still backed by OS paging).
        mm.flush()
        # Keep as memmap-backed array to minimize RAM.
        mat = mm

    dim = int(mat.shape[1])
    index = faiss.IndexFlatIP(dim)   # IP = Inner Product
    index.add(np.asarray(mat, dtype=np.float32))

    if cache_npy is not None:
        try:
            cache_npy.parent.mkdir(parents=True, exist_ok=True)
            # If we already wrote directly to cache_npy via memmap, avoid rewriting.
            # (When caching is enabled, cache_npy_out == cache_npy.)
            if cache_npy_out is not None and Path(cache_npy_out).resolve() != Path(cache_npy).resolve():
                np.save(cache_npy, np.asarray(mat, dtype=np.float32))
            if cache_meta is not None:
                cache_meta.write_text(
                    json.dumps(
                        {
                            "model": model,
                            "rows": int(total),
                            "dim": int(dim),
                            "text_digest": digest,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            print(f"  Saved embedding cache: {cache_npy}")
        except Exception as e:
            print(f"  Warning: could not save cache: {e}")

    return index, mat


def find_exact_duplicate_groups(
    mat: np.ndarray,
    *,
    round_decimals: int = 6,
) -> list[list[int]]:
    """
    Find exact-duplicate groups by hashing embedding vectors (no FAISS top_k limits).

    Why this fixes the drop you saw:
      - The previous approach relied on FAISS nearest-neighbor search with a fixed top_k.
        If an \"exact\" cluster had more members than top_k, some links were never retrieved,
        fragmenting clusters and reducing counted duplicates.

    Here we group by a stable signature of each embedding vector:
      - We round to `round_decimals` to match the common \"cosine shown as 1.000000\" notion.
      - Rows whose vectors are identical after rounding will be grouped together.
    """
    if mat.size == 0:
        return []

    # Round for stability (float noise) then hash each row
    rounded = np.round(mat, decimals=round_decimals).astype(np.float32, copy=False)

    from collections import defaultdict

    buckets: dict[bytes, list[int]] = defaultdict(list)
    for i in range(rounded.shape[0]):
        buckets[rounded[i].tobytes()].append(i)

    groups = [members for members in buckets.values() if len(members) >= 2]
    groups = [sorted(g) for g in groups]
    groups.sort(key=lambda g: g[0])
    return groups


def write_duplicate_groups_csv(groups: list[list[int]], records: list[dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not groups:
        out_path.write_text("No duplicates found.\n", encoding="utf-8")
        return out_path
    fieldnames = ["ID", "ROW #", "ITEM TYPE", "MAIN GROUP", "SUB GROUP", "ITEM DESC"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, members in enumerate(groups, 1):
            dup_id = f"DUP_{idx}"
            for m in members:
                writer.writerow(
                    {
                        "ID": dup_id,
                        "ROW #": m + 2,
                        "ITEM TYPE": records[m].get("item_type", ""),
                        "MAIN GROUP": records[m].get("main_group", ""),
                        "SUB GROUP": records[m].get("sub_group", ""),
                        "ITEM DESC": records[m].get("_item_description", ""),
                    }
                )
    return out_path


def _schema_records_to_minimized(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Step 1: same regex extraction as the original CLI pipeline."""
    minimized: list[dict[str, Any]] = []
    for rec in records:
        desc = (rec.get("_item_description") or "").strip()
        extracted = regex_extract_attributes(desc)
        out_rec: dict[str, Any] = {
            "item_type": rec.get("item_type", ""),
            "main_group": rec.get("main_group", ""),
            "sub_group": rec.get("sub_group", ""),
            "text": _clean_str(extracted.get("text", "")) or None,
            "numeric": _clean_str(extracted.get("numeric", "")) or None,
            "_item_description": rec.get("_item_description", ""),
        }
        minimized.append(out_rec)
    return minimized


def run_item_master_duplicate_engine(
    records: list[dict[str, str]],
    *,
    embed_model: str | None = None,
    embed_batch: int | None = None,
    cache_path: str | Path | None = None,
    reuse_only: bool = False,
) -> dict[str, Any]:
    """
    Full pipeline: Step 1 regex minimization → Step 2 embeddings + exact duplicate groups.
    Same logic as the former duplicate_detector_v2.py embed path.

    Returns a plain dict suitable for ItemMasterDuplicateEngineResponse.
    """
    model = embed_model if embed_model is not None else EMBED_MODEL
    batch = embed_batch if embed_batch is not None else EMBED_BATCH
    cache = cache_path if cache_path is not None else EMBED_CACHE_FILE

    minimized = _schema_records_to_minimized(records)
    total = len(minimized)
    if total == 0:
        return {
            "total_records": 0,
            "valid_records": 0,
            "duplicate_records": 0,
            "Data_quality_score": 0.0,
            "duplicates": {},
        }

    print(f"\n[Step 2] Embedding {total} rows locally (HF model)...")
    print(f"         Model    : {model}")
    print(f"         Batch    : {batch}")
    print("         Exact duplicates only (identical embeddings)")
    print(f"         Cache    : {cache}")
    if reuse_only:
        print("         Cache mode: reuse-only (no recompute)")

    _index, mat = build_faiss_index(
        minimized,
        model=model,
        batch_size=batch,
        cache_path=cache,
        reuse_only=reuse_only,
    )
    print(f"[Step 2] FAISS index built — {mat.shape[0]} vectors of dim {mat.shape[1]}")
    print("[Step 2] Searching for duplicate groups...")

    groups = find_exact_duplicate_groups(mat)
    print(f"[Step 2] Found {len(groups)} duplicate group(s)")

    duplicate_record_count = sum(max(0, len(g) - 1) for g in groups)
    valid_records = total - duplicate_record_count
    # Share of rows that are not "extra" duplicates (vs total pulled): 100 = no duplicate rows.
    data_quality_score = round(100.0 * float(valid_records) / float(total), 2)

    duplicates: dict[str, dict[str, Any]] = {}
    for idx, members in enumerate(groups, 1):
        dup_id = f"DUP_{idx}"
        rows_out: list[dict[str, Any]] = []
        for m in members:
            rec = minimized[m]
            rows_out.append(
                {
                    "row#": m + 1,
                    "ITEM_TYPE": _clean_str(rec.get("item_type", "")),
                    "MAINGROUP": _clean_str(rec.get("main_group", "")),
                    "SUBGROUP": _clean_str(rec.get("sub_group", "")),
                    "ITEMDESC": _clean_str(rec.get("_item_description", "")),
                }
            )
        # Fixed: variation within a duplicate group is evaluated on ITEMDESC vs hierarchy keys.
        duplicates[dup_id] = {"status": "ITEMDESC", "records": rows_out}

    return {
        "total_records": total,
        "valid_records": valid_records,
        "duplicate_records": duplicate_record_count,
        "Data_quality_score": data_quality_score,
        "duplicates": duplicates,
    }

