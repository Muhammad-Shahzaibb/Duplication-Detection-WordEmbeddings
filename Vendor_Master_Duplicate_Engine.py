"""
Vendor Master duplicate detection engine.

Fields checked **independently** (no combination logic):

  - Name       : embedding cosine similarity (threshold from cleansing-engine query param)
  - Address    : embedding cosine similarity (same threshold as Name on cleansing engine)
  - CNIC       : normalized exact match
  - NTN        : normalized exact match
  - STRN       : normalized exact match
  - Account No : normalized exact match
  - IBAN       : normalized exact match

Tuple layout from fetch_vendor_master_rows_from_view:
  index 0 = id, 1 = Name, 2 = CNIC, 3 = NTN, 4 = STRN, 5 = Account No, 6 = IBAN, 7 = Address
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from Config import (
    EMBED_VENDOR_ADDRESS_CACHE_FILE,
    EMBED_VENDOR_CACHE_FILE,
    VENDOR_MASTER_ROWS_JSON,
    VENDOR_MASTER_ROWS_JSONL,
)
from embeddings import (
    EMBED_BATCH,
    EMBED_MODEL,
    build_faiss_index,
    embed_texts_local,
    load_embedding_cache,
)
from logging_setup import get_logger

logger = get_logger("style_textile.vendor_engine")

_vendor_cache_rebuild_lock = threading.Lock()


def _staging_path(production: Path) -> Path:
    return production.with_name(f"{production.stem}.staging{production.suffix}")


def _embedding_meta_path(npy_path: Path) -> Path:
    return npy_path.with_suffix(npy_path.suffix + ".meta.json")


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass

# ── Tuple column indices ───────────────────────────────────────────────────────
IDX_ID = 0
IDX_NAME = 1
IDX_CNIC = 2
IDX_NTN = 3
IDX_STRN = 4
IDX_ACCOUNT_NO = 5
IDX_IBAN = 6
IDX_ADDRESS = 7

NUMERIC_FIELDS: list[tuple[str, int]] = [
    ("CNIC", IDX_CNIC),
    ("NTN", IDX_NTN),
    ("STRN", IDX_STRN),
    ("Account No", IDX_ACCOUNT_NO),
    ("IBAN", IDX_IBAN),
]


def _vendor_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _vendor_field_text(row: tuple[Any, ...], col_idx: int) -> str:
    """Stripped display text for a vendor column; ``None`` / blank → ``""``."""
    if col_idx >= len(row):
        return ""
    return _vendor_str(row[col_idx])


def _nonempty_text_indices(rows: list[tuple[Any, ...]], col_idx: int) -> set[int]:
    """Row indices with a non-empty text value (empty / null rows are excluded from embedding dup groups)."""
    return {i for i, row in enumerate(rows) if _vendor_field_text(row, col_idx)}


def vendor_row_to_cache_payload(row: tuple[Any, ...]) -> dict[str, Any]:
    """One index-aligned vendor row snapshot for ``vendor_final_rows.jsonl``."""
    return {
        "id": row[IDX_ID],
        "Name": _vendor_str(row[IDX_NAME]),
        "CNIC": _vendor_str(row[IDX_CNIC]),
        "NTN": _vendor_str(row[IDX_NTN]),
        "STRN": _vendor_str(row[IDX_STRN]),
        "Account No": _vendor_str(row[IDX_ACCOUNT_NO]),
        "IBAN": _vendor_str(row[IDX_IBAN]),
        "Address": _vendor_str(row[IDX_ADDRESS]) if len(row) > IDX_ADDRESS else "",
    }


def _cache_payload_to_tuple(rec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rec.get("id"),
        rec.get("Name", ""),
        rec.get("CNIC", ""),
        rec.get("NTN", ""),
        rec.get("STRN", ""),
        rec.get("Account No", ""),
        rec.get("IBAN", ""),
        rec.get("Address", ""),
    )


def write_vendor_row_cache_json(
    rows: list[tuple[Any, ...]],
    *,
    jsonl_path: str | Path,
    json_path: str | Path,
) -> tuple[Path, Path]:
    jsonl_path = Path(jsonl_path)
    json_path = Path(json_path)
    payload = [vendor_row_to_cache_payload(r) for r in rows]
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in payload:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return jsonl_path, json_path


def load_vendor_row_cache(jsonl_path: str | Path | None = None) -> list[tuple[Any, ...]]:
    path = Path(jsonl_path or VENDOR_MASTER_ROWS_JSONL)
    if not path.exists():
        raise FileNotFoundError(
            f"Vendor row cache not found: {path}. "
            "Run /Vendor-Master-update-embeddings to build the cache bundle."
        )
    rows: list[tuple[Any, ...]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_cache_payload_to_tuple(json.loads(line)))
    return rows


def load_vendor_main_db_cache(
    *,
    name_cache_path: str | Path | None = None,
    address_cache_path: str | Path | None = None,
    jsonl_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], list[tuple[Any, ...]]]:
    """
    Load the vendor main DB cache bundle: name embeddings, address embeddings,
    metadata, and row snapshots. No live database fetch.
    """
    name_cache = Path(name_cache_path or EMBED_VENDOR_CACHE_FILE)
    address_cache = Path(address_cache_path or EMBED_VENDOR_ADDRESS_CACHE_FILE)
    name_mat, name_meta = load_embedding_cache(name_cache)
    address_mat, address_meta = load_embedding_cache(address_cache)
    row_cache = load_vendor_row_cache(jsonl_path)

    n = min(int(name_mat.shape[0]), int(address_mat.shape[0]), len(row_cache))
    if int(name_mat.shape[0]) != len(row_cache) or int(address_mat.shape[0]) != len(row_cache):
        logger.warning(
            "Vendor cache row mismatch (name=%s address=%s row_cache=%s). "
            "Using aligned prefix of %s rows.",
            int(name_mat.shape[0]),
            int(address_mat.shape[0]),
            len(row_cache),
            n,
        )
    return (
        np.asarray(name_mat[:n], dtype=np.float32),
        np.asarray(address_mat[:n], dtype=np.float32),
        name_meta,
        address_meta,
        row_cache[:n],
    )


def _publish_vendor_cache_bundle(
    *,
    name_matrix: np.ndarray,
    address_matrix: np.ndarray,
    staging_name_npy: Path,
    staging_address_npy: Path,
    staging_jsonl: Path,
    staging_json: Path,
    production_name_npy: Path,
    production_address_npy: Path,
    production_jsonl: Path,
    production_json: Path,
) -> None:
    """Publish completed staging bundle to production paths (Windows-safe)."""
    production_name_npy.parent.mkdir(parents=True, exist_ok=True)
    staging_name_meta = _embedding_meta_path(staging_name_npy)
    staging_address_meta = _embedding_meta_path(staging_address_npy)
    production_name_meta = _embedding_meta_path(production_name_npy)
    production_address_meta = _embedding_meta_path(production_address_npy)

    np.save(production_name_npy, np.asarray(name_matrix, dtype=np.float32))
    np.save(production_address_npy, np.asarray(address_matrix, dtype=np.float32))
    shutil.copy2(staging_name_meta, production_name_meta)
    shutil.copy2(staging_address_meta, production_address_meta)
    shutil.copy2(staging_jsonl, production_jsonl)
    shutil.copy2(staging_json, production_json)

    for path in (
        staging_name_npy,
        staging_name_meta,
        staging_address_npy,
        staging_address_meta,
        staging_jsonl,
        staging_json,
    ):
        _unlink_quiet(path)

    logger.info(
        "Vendor cache bundle published: %s | %s | %s",
        production_name_npy,
        production_address_npy,
        production_jsonl,
    )


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


# ── Text embedding duplicate groups (Name, Address) ───────────────────────────

def find_embedding_text_duplicate_groups(
    mat: np.ndarray,
    *,
    text_threshold: float,
    eligible_indices: set[int] | None = None,
) -> list[list[int]]:
    """
    Group row indices where pairwise cosine similarity >= text_threshold.
    Used for vendor Name and Address embedding duplicate detection.

    When ``eligible_indices`` is set, rows outside that set are skipped (e.g. empty
    Name / Address — same rule as numeric fields in ``find_numeric_duplicate_groups``).
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
        if eligible_indices is not None and i not in eligible_indices:
            continue
        for j in range(i + 1, n):
            if eligible_indices is not None and j not in eligible_indices:
                continue
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
    """Load vendor main DB **name** embeddings + row snapshot from disk (no live DB)."""
    name_mat, _address_mat, _name_meta, _address_meta, rows = load_vendor_main_db_cache()
    return name_mat, rows


def load_vendor_main_address_embeddings_reuse_if_present() -> tuple[np.ndarray, list[tuple[Any, ...]]]:
    """Load vendor main DB **address** embeddings + row snapshot from disk (no live DB)."""
    _name_mat, address_mat, _name_meta, _address_meta, rows = load_vendor_main_db_cache()
    return address_mat, rows


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


def embed_vendor_approval_addresses_at_runtime() -> tuple[np.ndarray, list[tuple[Any, ...]]]:
    """Fetch approval view and embed Addresses in memory (not saved to disk)."""
    from Db_View import fetch_vendor_master_rows_from_approval_view

    rows = fetch_vendor_master_rows_from_approval_view()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []

    addresses = [str(r[IDX_ADDRESS]) if len(r) > IDX_ADDRESS and r[IDX_ADDRESS] is not None else "" for r in rows]
    logger.info("Vendor approval view: computing %s runtime address embeddings (not cached)", len(addresses))
    mat = embed_texts_local(addresses, model_id=EMBED_MODEL)
    return np.asarray(mat, dtype=np.float32), rows


def _match_vendor_text_field_variant(
    candidate: str,
    db_rows: list[tuple[Any, ...]],
    db_mat: np.ndarray,
    approval_rows: list[tuple[Any, ...]],
    ap_mat: np.ndarray,
    *,
    col_idx: int,
    threshold: float,
    field_label: str,
) -> list[dict[str, Any]]:
    """Check candidate text against main DB (cached) and approval (runtime) embeddings."""
    if not _vendor_str(candidate):
        return []

    cand_vec = embed_texts_local([candidate], model_id=EMBED_MODEL, batch_size=1)
    if cand_vec.ndim != 2 or cand_vec.shape[0] != 1:
        raise RuntimeError(f"Unexpected embedding shape for candidate {field_label}")
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
                f"Vendor {field_label} embedding dimension mismatch for {location} "
                f"(cache_dim={mat.shape[1]}, cand_dim={cand_vec.shape[0]})"
            )
        scores = mat @ cand_vec
        for i, score in enumerate(scores):
            if float(score) >= threshold:
                row = rows[i]
                field_val = _vendor_field_text(row, col_idx)
                if not field_val:
                    continue
                matches.append({
                    "id": row[IDX_ID],
                    "Name": row[IDX_NAME] if row[IDX_NAME] is not None else "",
                    "field_value": field_val,
                    "location": location,
                    "row": i + 1,
                })
    return matches


def match_vendor_name_variant(
    candidate_name: str,
    db_rows: list[tuple[Any, ...]],
    db_mat: np.ndarray,
    approval_rows: list[tuple[Any, ...]],
    ap_mat: np.ndarray,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    return _match_vendor_text_field_variant(
        candidate_name,
        db_rows,
        db_mat,
        approval_rows,
        ap_mat,
        col_idx=IDX_NAME,
        threshold=threshold,
        field_label="name",
    )


def match_vendor_address_variant(
    candidate_address: str,
    db_rows: list[tuple[Any, ...]],
    db_mat: np.ndarray,
    approval_rows: list[tuple[Any, ...]],
    ap_mat: np.ndarray,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    return _match_vendor_text_field_variant(
        candidate_address,
        db_rows,
        db_mat,
        approval_rows,
        ap_mat,
        col_idx=IDX_ADDRESS,
        threshold=threshold,
        field_label="address",
    )


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
    name_cache_path: str | Path | None = None,
    address_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Recompute vendor name + address embeddings and publish the full cache bundle.

    While embedding runs, readers keep using production files. Staging paths
    (``*.staging``) are published only when both embedding passes complete.
    """
    with _vendor_cache_rebuild_lock:
        model = embed_model or EMBED_MODEL
        batch = embed_batch or EMBED_BATCH
        production_name_npy = Path(name_cache_path or EMBED_VENDOR_CACHE_FILE)
        production_address_npy = Path(address_cache_path or EMBED_VENDOR_ADDRESS_CACHE_FILE)
        production_jsonl = VENDOR_MASTER_ROWS_JSONL
        production_json = VENDOR_MASTER_ROWS_JSON

        staging_name_npy = _staging_path(production_name_npy)
        staging_address_npy = _staging_path(production_address_npy)
        staging_jsonl = _staging_path(production_jsonl)
        staging_json = _staging_path(production_json)
        for path in (
            staging_name_npy,
            _embedding_meta_path(staging_name_npy),
            staging_address_npy,
            _embedding_meta_path(staging_address_npy),
            staging_jsonl,
            staging_json,
        ):
            _unlink_quiet(path)

        total = len(rows)
        names = [str(r[IDX_NAME]) if r[IDX_NAME] is not None else "" for r in rows]
        addresses = [
            str(r[IDX_ADDRESS]) if len(r) > IDX_ADDRESS and r[IDX_ADDRESS] is not None else ""
            for r in rows
        ]
        name_records = [{"text": n or None, "numeric": None} for n in names]
        address_records = [{"text": a or None, "numeric": None} for a in addresses]

        logger.info(
            "Vendor embeddings: refreshing cache bundle for %s rows (staging; production unchanged)",
            total,
        )
        print(f"\n[Update embeddings] Refreshing Vendor Master cache for {total} rows...")
        print(f"         Model    : {model}")
        print(f"         Batch    : {batch}")
        print(f"         Name staging    : {staging_name_npy}")
        print(f"         Address staging : {staging_address_npy}")
        print(f"         Row cache staging: {staging_jsonl}")

        try:
            write_vendor_row_cache_json(rows, jsonl_path=staging_jsonl, json_path=staging_json)

            _name_index, name_mat = build_faiss_index(
                name_records,
                model=model,
                batch_size=batch,
                cache_path=staging_name_npy,
                reuse_only=False,
                force_recompute=True,
            )
            del _name_index

            _address_index, address_mat = build_faiss_index(
                address_records,
                model=model,
                batch_size=batch,
                cache_path=staging_address_npy,
                reuse_only=False,
                force_recompute=True,
            )
            del _address_index

            _publish_vendor_cache_bundle(
                name_matrix=name_mat,
                address_matrix=address_mat,
                staging_name_npy=staging_name_npy,
                staging_address_npy=staging_address_npy,
                staging_jsonl=staging_jsonl,
                staging_json=staging_json,
                production_name_npy=production_name_npy,
                production_address_npy=production_address_npy,
                production_jsonl=production_jsonl,
                production_json=production_json,
            )
            print(f"[Update embeddings] Published Vendor Master cache bundle")
        except Exception:
            for path in (
                staging_name_npy,
                _embedding_meta_path(staging_name_npy),
                staging_address_npy,
                _embedding_meta_path(staging_address_npy),
                staging_jsonl,
                staging_json,
            ):
                _unlink_quiet(path)
            raise

        name_meta_path = _embedding_meta_path(production_name_npy.resolve())
        address_meta_path = _embedding_meta_path(production_address_npy.resolve())
        name_meta: dict[str, Any] = {}
        address_meta: dict[str, Any] = {}
        if name_meta_path.exists():
            name_meta = json.loads(name_meta_path.read_text(encoding="utf-8"))
        if address_meta_path.exists():
            address_meta = json.loads(address_meta_path.read_text(encoding="utf-8"))

        return {
            "total_records": total,
            "embedding_dim": int(name_mat.shape[1]) if name_mat.size else 0,
            "cache_file": str(production_name_npy.resolve()),
            "metadata_file": str(name_meta_path),
            "row_cache_file": str(production_jsonl.resolve()),
            "model": model,
            "rows_in_metadata": int(name_meta.get("rows", total)),
            "address_embedding_dim": int(address_mat.shape[1]) if address_mat.size else 0,
            "address_cache_file": str(production_address_npy.resolve()),
            "address_metadata_file": str(address_meta_path),
            "address_rows_in_metadata": int(address_meta.get("rows", total)),
        }


def run_vendor_master_duplicate_engine(
    *,
    name_threshold: float,
) -> dict[str, Any]:
    """
    Duplicate detection on the main DB using the on-disk vendor cache bundle only.

    Reads ``vendor_embeddings_cache.npy``, ``vendor_address_embeddings_cache.npy``,
    and ``vendor_final_rows.jsonl`` from the last ``/Vendor-Master-update-embeddings``
    run. No live database fetch.
    """
    threshold = name_threshold

    print(f"\n[Vendor duplicate engine] Loading cache bundle (no live DB)...")
    print(f"         Name embeddings : {EMBED_VENDOR_CACHE_FILE}")
    print(f"         Address embeddings: {EMBED_VENDOR_ADDRESS_CACHE_FILE}")
    print(f"         Row cache       : {VENDOR_MASTER_ROWS_JSONL}")

    name_mat, address_mat, _name_meta, _address_meta, rows = load_vendor_main_db_cache()
    total = len(rows)
    logger.info(
        "Vendor duplicate engine: %s cached rows | embedding_threshold=%.3f (name + address)",
        total, threshold,
    )

    if total == 0:
        empty: dict[str, Any] = {"duplicate_groups": 0, "duplicate_records": 0, "groups": {}}
        return {
            "total_records": 0,
            "duplicates_by_NAME": empty,
            "duplicates_by_ADDRESS": empty,
            "duplicates_by_CNIC": empty,
            "duplicates_by_NTN": empty,
            "duplicates_by_STRN": empty,
            "duplicates_by_ACCOUNT_NO": empty,
            "duplicates_by_IBAN": empty,
        }

    print(f"[Vendor duplicate engine] Cache loaded — {total} rows, dim={name_mat.shape[1]}")

    name_groups = find_embedding_text_duplicate_groups(
        name_mat,
        text_threshold=threshold,
        eligible_indices=_nonempty_text_indices(rows, IDX_NAME),
    )
    logger.info("Vendor engine: NAME duplicate groups=%s", len(name_groups))
    result_name = _build_field_result(rows, name_groups)

    address_groups = find_embedding_text_duplicate_groups(
        address_mat,
        text_threshold=threshold,
        eligible_indices=_nonempty_text_indices(rows, IDX_ADDRESS),
    )
    logger.info("Vendor engine: ADDRESS duplicate groups=%s", len(address_groups))
    result_address = _build_field_result(
        rows, address_groups, field_col_idx=IDX_ADDRESS, field_label="Address"
    )

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
        "duplicates_by_ADDRESS": result_address,
        "duplicates_by_CNIC": field_results["CNIC"],
        "duplicates_by_NTN": field_results["NTN"],
        "duplicates_by_STRN": field_results["STRN"],
        "duplicates_by_ACCOUNT_NO": field_results["ACCOUNT_NO"],
        "duplicates_by_IBAN": field_results["IBAN"],
    }
