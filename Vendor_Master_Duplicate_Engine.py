"""
Vendor Master duplicate detection engine.

Six fields are checked **independently** (no combination logic):

  - Name       : embedding cosine similarity >= VENDOR_NAME_TEXT_THRESHOLD
  - CNIC       : normalized exact match (strip specials/spaces, strip leading zeros)
  - NTN        : normalized exact match
  - STRN       : normalized exact match
  - Account No : normalized exact match
  - IBAN       : normalized exact match (preserves letter prefix, e.g. PK36...)

Tuple layout from fetch_vendor_master_rows_from_view:
  index 0 = id
  index 1 = Name
  index 2 = CNIC
  index 3 = NTN
  index 4 = STRN
  index 5 = Account No
  index 6 = IBAN
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from Config import (
    EMBED_VENDOR_CACHE_FILE,
    VENDOR_NAME_TEXT_THRESHOLD,
    VENDOR_VARIANT_CHECK_NAME_THRESHOLD,
)
from embeddings import (
    EMBED_BATCH,
    EMBED_MODEL,
    _embedding_cache_can_reuse,
    _embedding_cache_mismatch_reasons,
    build_faiss_index,
    describe_embedding_cache_action,
    embed_texts_local,
    load_embedding_cache,
)
from logging_setup import get_logger

logger = get_logger("style_textile.vendor_engine")

# ── Tuple column indices ───────────────────────────────────────────────────────
IDX_ID = 0
IDX_NAME = 1
IDX_CNIC = 2
IDX_NTN = 3
IDX_STRN = 4
IDX_ACCOUNT_NO = 5
IDX_IBAN = 6

NUMERIC_FIELDS: list[tuple[str, int]] = [
    ("CNIC", IDX_CNIC),
    ("NTN", IDX_NTN),
    ("STRN", IDX_STRN),
    ("Account No", IDX_ACCOUNT_NO),
    ("IBAN", IDX_IBAN),
]


# ── Numeric normalization ──────────────────────────────────────────────────────

def normalize_numeric_field(value: Any) -> str:
    """
    Normalize a numeric-like identifier for exact-match comparison:
      1. Strip all spaces and special characters (keep alphanumeric only).
      2. If the result is purely numeric, strip leading zeros (so "0786" == "786").
      3. Uppercase (for any alphabetic prefix, e.g. IBAN country code).

    Examples:
      "12345-1234567-1"  →  "1234512345671"
      "0786"             →  "786"
      "PK36SCBL0000001123456702"  →  "PK36SCBL0000001123456702"  (not pure digits)
    """
    if value is None:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", str(value).strip())
    if not cleaned:
        return ""
    if cleaned.isdigit():
        stripped = cleaned.lstrip("0")
        cleaned = stripped if stripped else "0"
    return cleaned.upper()


# ── Numeric duplicate groups ───────────────────────────────────────────────────

def find_numeric_duplicate_groups(
    rows: list[tuple[Any, ...]],
    *,
    col_idx: int,
) -> list[list[int]]:
    """
    Return groups of row indices whose normalized field value is identical.
    Empty / null values are skipped (not grouped).
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        raw = row[col_idx] if col_idx < len(row) else None
        val = normalize_numeric_field(raw)
        if val:
            buckets[val].append(i)

    groups = [sorted(members) for members in buckets.values() if len(members) >= 2]
    groups.sort(key=lambda g: g[0])
    return groups


# ── Name embedding + similarity ───────────────────────────────────────────────

def _embed_vendor_names(
    names: list[str],
    *,
    model: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH,
    cache_path: str | Path | None = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Embed a list of vendor names.  Reuses on-disk cache when row count matches.
    Returns a float32 matrix of shape (N, D), L2-normalised.
    """
    cache = Path(cache_path) if cache_path else Path(EMBED_VENDOR_CACHE_FILE)
    cache_meta = cache.with_suffix(cache.suffix + ".meta.json")
    total = len(names)

    if not force_recompute and cache.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            mat_cached = np.load(cache, mmap_mode="r")
            if _embedding_cache_can_reuse(meta, mat_cached, total=total, model=model):
                logger.info(
                    "Vendor name embeddings: REUSE cache %s (%s rows)", cache, total
                )
                return np.asarray(mat_cached, dtype=np.float32)
            reasons = _embedding_cache_mismatch_reasons(
                [{"text": n} for n in names], cache_path=cache, model=model
            )
            logger.info(
                "Vendor name embeddings: STALE — will COMPUTE %s (%s rows) | %s",
                cache,
                total,
                "; ".join(reasons) if reasons else "unknown mismatch",
            )
        except Exception:
            logger.warning("Vendor name embeddings: cache unreadable — will COMPUTE %s", cache)
    elif not cache.exists():
        logger.info("Vendor name embeddings: MISSING — will COMPUTE %s (%s rows)", cache, total)

    mat = embed_texts_local(names, model_id=model, batch_size=batch_size)
    mat = np.asarray(mat, dtype=np.float32)

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, mat)
        import hashlib
        h = hashlib.sha256()
        for n in names:
            b = n.encode("utf-8", errors="ignore")
            h.update(len(b).to_bytes(8, "little", signed=False))
            h.update(b)
        cache_meta.write_text(
            json.dumps(
                {"model": model, "rows": total, "dim": int(mat.shape[1]), "text_digest": h.hexdigest()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Vendor name embeddings: SAVED %s (%s rows, dim=%s)", cache, total, mat.shape[1])
    except Exception as e:
        logger.warning("Vendor name embeddings: could not save cache %s: %s", cache, e)

    return mat


def find_name_duplicate_groups(
    mat: np.ndarray,
    *,
    text_threshold: float = VENDOR_NAME_TEXT_THRESHOLD,
) -> list[list[int]]:
    """
    Group vendor name indices where pairwise cosine similarity >= text_threshold.
    Uses union-find for transitive closure (A≈B and B≈C → one group).
    O(n²) — acceptable for typical vendor master sizes (< 10 000 rows).
    """
    n = mat.shape[0]
    if n == 0:
        return []

    sims = mat @ mat.T  # (n, n) pairwise cosine (normalised vectors)
    parent = list(range(n))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if float(sims[i, j]) >= text_threshold:
                pi, pj = root(i), root(j)
                if pi != pj:
                    parent[pi] = pj

    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comp[root(i)].append(i)

    groups = [sorted(members) for members in comp.values() if len(members) >= 2]
    groups.sort(key=lambda g: g[0])
    return groups


# ── Result builders ────────────────────────────────────────────────────────────

def _build_field_result(
    rows: list[tuple[Any, ...]],
    groups: list[list[int]],
    *,
    field_col_idx: int | None = None,
    field_label: str | None = None,
) -> dict[str, Any]:
    """
    Produce the dict for one field's duplicate result:
      { "duplicate_groups": N, "duplicate_records": M, "groups": { "DUP_1": {...}, ... } }

    Per group, one row is treated as unique; the rest count toward duplicate_records
    (same rule as Item Master duplicate engine: sum of max(0, group_size - 1)).
    """
    dup_groups: dict[str, Any] = {}
    duplicate_records = 0
    for idx, members in enumerate(groups, 1):
        records: list[dict[str, Any]] = []
        for m in members:
            row = rows[m]
            rec: dict[str, Any] = {
                "id": row[IDX_ID],
                "Name": row[IDX_NAME] if row[IDX_NAME] is not None else "",
            }
            if field_col_idx is not None and field_label is not None:
                rec[field_label] = row[field_col_idx] if row[field_col_idx] is not None else ""
            records.append(rec)
        dup_groups[f"DUP_{idx}"] = {"records": records}
        duplicate_records += max(0, len(members) - 1)

    return {
        "duplicate_groups": len(groups),
        "duplicate_records": duplicate_records,
        "groups": dup_groups,
    }


# ── Variant check helpers ─────────────────────────────────────────────────────

def load_vendor_main_embeddings_reuse_if_present() -> tuple[np.ndarray, list[tuple[Any, ...]]]:
    """
    Load vendor main DB name embeddings from disk cache, reusing regardless of row-count
    or model mismatch (same policy as Item Master duplicate engine).

    Aligns to min(cache_rows, view_rows) prefix.
    Raises RuntimeError if cache files are missing — call /Vendor-Master-update-embeddings first.
    """
    from Db_View import fetch_vendor_master_rows_from_view
    rows = fetch_vendor_master_rows_from_view()
    cache = Path(EMBED_VENDOR_CACHE_FILE)
    cache_meta = cache.with_suffix(cache.suffix + ".meta.json")

    if not cache.exists() or not cache_meta.exists():
        raise RuntimeError(
            f"Vendor name embedding cache not found at {cache}. "
            "Run /Vendor-Master-update-embeddings first."
        )
    try:
        mat = np.load(cache).astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            f"Vendor name embedding cache unreadable at {cache}. "
            "Run /Vendor-Master-update-embeddings to rebuild."
        ) from e

    n = min(int(mat.shape[0]), len(rows))
    if int(mat.shape[0]) != len(rows):
        logger.warning(
            "Vendor name embeddings: cache/view row mismatch (cache=%s view=%s). "
            "Using aligned prefix of %s rows.",
            int(mat.shape[0]), len(rows), n,
        )
    return np.asarray(mat[:n], dtype=np.float32), rows[:n]


def embed_vendor_approval_names_at_runtime() -> tuple[np.ndarray, list[tuple[Any, ...]]]:
    """
    Fetch the vendor approval view and embed vendor Names in memory (not saved to disk).
    Returns (matrix, view_tuples). Matrix has shape (N, D); empty array when 0 rows.
    """
    from Db_View import fetch_vendor_master_rows_from_approval_view
    rows = fetch_vendor_master_rows_from_approval_view()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []

    names = [str(r[IDX_NAME]) if r[IDX_NAME] is not None else "" for r in rows]
    logger.info("Vendor approval view: computing %s runtime name embeddings (not cached)", len(names))
    mat = embed_texts_local(names, model_id=EMBED_MODEL)
    return np.asarray(mat, dtype=np.float32), rows


def match_vendor_name_variant(
    candidate_name: str,
    db_rows: list[tuple[Any, ...]],
    db_mat: np.ndarray,
    approval_rows: list[tuple[Any, ...]],
    ap_mat: np.ndarray,
    *,
    threshold: float = VENDOR_VARIANT_CHECK_NAME_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Check a candidate vendor Name against main DB (cached) and approval (runtime) embeddings.
    Returns match dicts: {id, Name, field_value, location, row}.
    """
    cand_vec = embed_texts_local([candidate_name], model_id=EMBED_MODEL, batch_size=1)
    if cand_vec.ndim != 2 or cand_vec.shape[0] != 1:
        raise RuntimeError("Unexpected embedding shape for candidate name")
    cand_vec = cand_vec[0]

    matches: list[dict[str, Any]] = []
    for mat, rows, location in [
        (db_mat, db_rows, "db"),
        (ap_mat, approval_rows, "approval"),
    ]:
        if mat.size == 0 or not rows:
            continue
        if int(mat.shape[1]) != int(cand_vec.shape[0]):
            raise RuntimeError(
                f"Vendor name embedding dimension mismatch for {location} "
                f"(cache_dim={mat.shape[1]}, cand_dim={cand_vec.shape[0]})"
            )
        scores = mat @ cand_vec
        for i, score in enumerate(scores):
            if float(score) >= threshold:
                row = rows[i]
                name_val = row[IDX_NAME] if row[IDX_NAME] is not None else ""
                matches.append({
                    "id": row[IDX_ID],
                    "Name": name_val,
                    "field_value": name_val,
                    "location": location,
                    "row": i + 1,
                })
    return matches


def match_vendor_numeric_variant(
    candidate_value: str,
    col_idx: int,
    db_rows: list[tuple[Any, ...]],
    approval_rows: list[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    """
    Check a candidate numeric field against main DB and approval view rows using
    the same normalization as the duplicate engine (strip specials, strip leading zeros).
    Returns match dicts: {id, Name, field_value, location, row}.
    """
    cand_norm = normalize_numeric_field(candidate_value)
    if not cand_norm:
        return []

    matches: list[dict[str, Any]] = []
    for rows, location in [(db_rows, "db"), (approval_rows, "approval")]:
        for i, row in enumerate(rows):
            raw = row[col_idx] if col_idx < len(row) else None
            if normalize_numeric_field(raw) == cand_norm:
                matches.append({
                    "id": row[IDX_ID],
                    "Name": row[IDX_NAME] if row[IDX_NAME] is not None else "",
                    "field_value": str(raw) if raw is not None else "",
                    "location": location,
                    "row": i + 1,
                })
    return matches


# ── Public API ─────────────────────────────────────────────────────────────────

def rebuild_vendor_embeddings_cache(
    rows: list[tuple[Any, ...]],
    *,
    embed_model: str | None = None,
    embed_batch: int | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Force-recompute and save vendor name embeddings. Called by the update-embeddings API."""
    model = embed_model or EMBED_MODEL
    batch = embed_batch or EMBED_BATCH
    cache = Path(cache_path) if cache_path else Path(EMBED_VENDOR_CACHE_FILE)

    names = [str(r[IDX_NAME]) if r[IDX_NAME] is not None else "" for r in rows]
    total = len(names)
    logger.info("Vendor embeddings: force COMPUTE %s rows → %s", total, cache)

    mat = _embed_vendor_names(names, model=model, batch_size=batch, cache_path=cache, force_recompute=True)

    cache_meta = cache.with_suffix(cache.suffix + ".meta.json")
    meta: dict[str, Any] = {}
    if cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "total_records": total,
        "embedding_dim": int(mat.shape[1]) if mat.size else 0,
        "cache_file": str(cache),
        "metadata_file": str(cache_meta),
        "model": model,
        "rows_in_metadata": int(meta.get("rows", total)),
    }


def run_vendor_master_duplicate_engine(
    rows: list[tuple[Any, ...]],
    *,
    embed_model: str | None = None,
    embed_batch: int | None = None,
    cache_path: str | Path | None = None,
    name_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Run the full vendor master duplicate detection pipeline.

    Returns a dict matching VendorMasterDuplicateEngineResponse:
      total_records, duplicates_by_NAME, duplicates_by_CNIC, duplicates_by_NTN,
      duplicates_by_STRN, duplicates_by_ACCOUNT_NO, duplicates_by_IBAN.
    """
    model = embed_model or EMBED_MODEL
    batch = embed_batch or EMBED_BATCH
    cache = Path(cache_path) if cache_path else Path(EMBED_VENDOR_CACHE_FILE)
    threshold = name_threshold if name_threshold is not None else VENDOR_NAME_TEXT_THRESHOLD

    total = len(rows)
    logger.info("Vendor duplicate engine: %s rows | name_threshold=%.3f", total, threshold)

    if total == 0:
        empty: dict[str, Any] = {"duplicate_groups": 0, "duplicate_records": 0, "groups": {}}
        return {
            "total_records": 0,
            "duplicates_by_NAME": empty,
            "duplicates_by_CNIC": empty,
            "duplicates_by_NTN": empty,
            "duplicates_by_STRN": empty,
            "duplicates_by_ACCOUNT_NO": empty,
            "duplicates_by_IBAN": empty,
        }

    # ── Name duplicates (embedding) ────────────────────────────────────────────
    names = [str(r[IDX_NAME]) if r[IDX_NAME] is not None else "" for r in rows]
    logger.info("Vendor engine: embedding %s names (model=%s)", total, model)
    mat = _embed_vendor_names(names, model=model, batch_size=batch, cache_path=cache)
    name_groups = find_name_duplicate_groups(mat, text_threshold=threshold)
    logger.info("Vendor engine: NAME duplicate groups=%s", len(name_groups))
    result_name = _build_field_result(rows, name_groups)

    # ── Numeric field duplicates ───────────────────────────────────────────────
    field_results: dict[str, dict[str, Any]] = {}
    labels = {
        "CNIC": ("CNIC", IDX_CNIC),
        "NTN": ("NTN", IDX_NTN),
        "STRN": ("STRN", IDX_STRN),
        "ACCOUNT_NO": ("Account No", IDX_ACCOUNT_NO),
        "IBAN": ("IBAN", IDX_IBAN),
    }
    for key, (label, col_idx) in labels.items():
        groups = find_numeric_duplicate_groups(rows, col_idx=col_idx)
        logger.info("Vendor engine: %s duplicate groups=%s", label, len(groups))
        field_results[key] = _build_field_result(rows, groups, field_col_idx=col_idx, field_label=label)

    return {
        "total_records": total,
        "duplicates_by_NAME": result_name,
        "duplicates_by_CNIC": field_results["CNIC"],
        "duplicates_by_NTN": field_results["NTN"],
        "duplicates_by_STRN": field_results["STRN"],
        "duplicates_by_ACCOUNT_NO": field_results["ACCOUNT_NO"],
        "duplicates_by_IBAN": field_results["IBAN"],
    }
