import json
from pathlib import Path
from typing import Any

import numpy as np

from Config import (
    DUPLICATE_ENGINE_TEXT_THRESHOLD,
    ITEM_MASTER_MINIMIZED_JSON,
    ITEM_MASTER_MINIMIZED_JSONL,
)
from Db_View import fetch_item_master_rows_from_approval_view
from embeddings import (
    EMBED_BATCH,
    EMBED_CACHE_FILE,
    EMBED_MODEL,
    build_embedding_text,
    build_faiss_index,
    embed_texts_local,
    find_duplicate_groups_by_text_and_numeric,
)
from jsonify import clean_str, row_to_schema_json, schema_records_to_minimized, write_minimized_embedding_input_json
from logging_setup import get_logger

logger = get_logger("style_textile.engine")


def _write_minimized_json_before_embed(minimized: list[dict[str, Any]], *, cache_path: str | Path) -> tuple[Path, Path]:
    jl, jp = ITEM_MASTER_MINIMIZED_JSONL, ITEM_MASTER_MINIMIZED_JSON
    paths = write_minimized_embedding_input_json(minimized, jsonl_path=jl, json_path=jp)
    logger.info("Wrote minimized JSON (pre-embed): %s | %s", paths[0], paths[1])
    return paths


def _column_match_status(values: list[str]) -> str:
    """Return 'exact' if all non-empty comparisons match; 'different' if any differ (2+ rows)."""
    if len(values) < 2:
        return "exact"
    normalized = [v.strip().casefold() for v in values]
    return "exact" if len(set(normalized)) == 1 else "different"


def _duplicate_group_column_status(rows_out: list[dict[str, Any]]) -> dict[str, str]:
    """Per-column exact/different for a duplicate cluster."""
    return {
        "ITEM_TYPE": _column_match_status([str(r.get("ITEM_TYPE", "")) for r in rows_out]),
        "MAINGROUP": _column_match_status([str(r.get("MAINGROUP", "")) for r in rows_out]),
        "SUBGROUP": _column_match_status([str(r.get("SUBGROUP", "")) for r in rows_out]),
        "ITEMDESC": "different",
    }


def embed_item_master_approval_view_at_runtime() -> tuple[np.ndarray, list[tuple[Any, ...]]]:
    """
    Fetch the approval view and embed all rows in memory (not written to disk).
    Returns (matrix, view_tuples) with one vector per tuple row.
    """
    tuples = fetch_item_master_rows_from_approval_view()
    if not tuples:
        return np.zeros((0, 0), dtype=np.float32), []

    records = [
        row_to_schema_json(item_description=desc, item_type=it, main_group=mg, sub_group=sg)
        for it, mg, sg, desc in tuples
    ]
    minimized = schema_records_to_minimized(records)
    texts = [build_embedding_text(r) for r in minimized]
    logger.info("Approval view: computing %s runtime embeddings (not cached)", len(texts))
    mat = embed_texts_local(texts, model_id=EMBED_MODEL)
    return np.asarray(mat, dtype=np.float32), tuples


def rebuild_item_master_embeddings_cache(
    records: list[dict[str, str]],
    *,
    embed_model: str | None = None,
    embed_batch: int | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Recompute embeddings from Item Master schema rows (same regex minimization as duplicate detection).
    Ignores any existing cache and overwrites ``embeddings_cache.npy`` and ``*.meta.json``.
    """
    model = embed_model if embed_model is not None else EMBED_MODEL
    batch = embed_batch if embed_batch is not None else EMBED_BATCH
    cache = cache_path if cache_path is not None else EMBED_CACHE_FILE
    minimized = schema_records_to_minimized(records)
    total = len(minimized)
    print(f"\n[Update embeddings] Refreshing cache for {total} rows...")
    print(f"         Model    : {model}")
    print(f"         Batch    : {batch}")
    print(f"         Cache    : {cache}")
    _write_minimized_json_before_embed(minimized, cache_path=cache)
    _index, mat = build_faiss_index(
        minimized,
        model=model,
        batch_size=batch,
        cache_path=cache,
        reuse_only=False,
        force_recompute=True,
    )
    cache_p = Path(cache).resolve() if cache else None
    meta_p = cache_p.with_suffix(cache_p.suffix + ".meta.json") if cache_p else None
    meta: dict[str, Any] = {}
    if meta_p is not None and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    return {
        "total_records": total,
        "embedding_dim": int(mat.shape[1]) if mat.size else 0,
        "cache_file": str(cache_p) if cache_p else "",
        "metadata_file": str(meta_p) if meta_p else "",
        "model": model,
        "text_digest": str(meta.get("text_digest", "")),
        "rows_in_metadata": int(meta.get("rows", total)),
    }


def run_item_master_duplicate_engine(
    records: list[dict[str, str]],
    *,
    embed_model: str | None = None,
    embed_batch: int | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: Step 1 regex minimization → Step 2 embeddings + duplicate groups.

    Always reuses on-disk embedding cache when present (any row/model mismatch is ignored;
    rows are aligned to min(cache_rows, view_rows)). Does not recompute embeddings — use
    ``/Item-Master-update-embeddings`` to refresh the cache.

    Returns a plain dict suitable for ItemMasterDuplicateEngineResponse.
    """
    model = embed_model if embed_model is not None else EMBED_MODEL
    batch = embed_batch if embed_batch is not None else EMBED_BATCH
    cache = cache_path if cache_path is not None else EMBED_CACHE_FILE

    minimized = schema_records_to_minimized(records)
    total = len(minimized)
    if total == 0:
        return {
            "total_records": 0,
            "valid_records": 0,
            "duplicate_records": 0,
            "Data_quality_score": 0.0,
            "duplicates": {},
        }

    print(f"\n[Step 2] Loading embeddings from cache (force reuse if present)...")
    print(f"         Model    : {model}")
    print(f"         Cache    : {cache}")
    print("         Cache mode: reuse-if-present (no recompute on duplicate engine)")

    _write_minimized_json_before_embed(minimized, cache_path=cache)
    _index, mat = build_faiss_index(
        minimized,
        model=model,
        batch_size=batch,
        cache_path=cache,
        reuse_if_present=True,
    )

    n_aligned = min(int(mat.shape[0]), len(minimized))
    if int(mat.shape[0]) != len(minimized):
        logger.warning(
            "Duplicate engine: cache/view row mismatch (cache_vectors=%s view_rows=%s). "
            "Using aligned prefix of %s rows.",
            int(mat.shape[0]),
            len(minimized),
            n_aligned,
        )
        print(
            f"[Step 2] Cache/view row mismatch — using aligned prefix ({n_aligned} rows)"
        )
    mat = np.asarray(mat[:n_aligned], dtype=np.float32)
    minimized = minimized[:n_aligned]
    total = len(minimized)

    print(f"[Step 2] FAISS index built — {mat.shape[0]} vectors of dim {mat.shape[1]}")
    print("[Step 2] Searching for duplicate groups...")

    numerics = [r.get("numeric") or "" for r in minimized]
    groups = find_duplicate_groups_by_text_and_numeric(
        np.asarray(mat, dtype=np.float32),
        numerics,
        text_threshold=DUPLICATE_ENGINE_TEXT_THRESHOLD,
    )
    print(f"[Step 2] Found {len(groups)} duplicate group(s) (text_threshold={DUPLICATE_ENGINE_TEXT_THRESHOLD})")

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
            row_payload: dict[str, Any] = {
                "row#": m + 1,
                "ITEM_TYPE": clean_str(rec.get("_item_type", "")),
                "MAINGROUP": clean_str(rec.get("_main_group", "")),
                "SUBGROUP": clean_str(rec.get("_sub_group", "")),
                "ITEMDESC": clean_str(rec.get("_item_description", "")),
            }
            if "_item_code" in rec:
                row_payload["ITEM_CODE"] = clean_str(rec.get("_item_code", ""))
            if "_uom" in rec:
                row_payload["UOM"] = clean_str(rec.get("_uom", ""))
            rows_out.append(row_payload)
        duplicates[dup_id] = {
            "status": _duplicate_group_column_status(rows_out),
            "records": rows_out,
        }

    return {
        "total_records": total,
        "valid_records": valid_records,
        "duplicate_records": duplicate_record_count,
        "Data_quality_score": data_quality_score,
        "duplicates": duplicates,
    }

