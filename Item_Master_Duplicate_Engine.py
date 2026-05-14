import json
from pathlib import Path
from typing import Any

import numpy as np

from embeddings import EMBED_BATCH, EMBED_CACHE_FILE, EMBED_MODEL, build_faiss_index, find_exact_duplicate_groups
from jsonify import clean_str, row_to_schema_json, schema_records_to_minimized


def load_or_build_embeddings_matrix_for_schema_records(
    records: list[dict[str, str]],
    *,
    cache_path: str | Path,
    embed_model: str | None = None,
    embed_batch: int | None = None,
) -> np.ndarray:
    """
    Return the embedding matrix for the given schema rows, reusing ``cache_path`` + ``.meta.json``
    when they match current rows/model/content; otherwise compute and persist.
    """
    model = embed_model if embed_model is not None else EMBED_MODEL
    batch = embed_batch if embed_batch is not None else EMBED_BATCH
    minimized = schema_records_to_minimized(records)
    _index, mat = build_faiss_index(
        minimized,
        model=model,
        batch_size=batch,
        cache_path=str(cache_path),
        reuse_only=False,
        force_recompute=False,
    )
    return np.asarray(mat, dtype=np.float32)


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
                    "ITEM_TYPE": clean_str(rec.get("item_type", "")),
                    "MAINGROUP": clean_str(rec.get("main_group", "")),
                    "SUBGROUP": clean_str(rec.get("sub_group", "")),
                    "ITEMDESC": clean_str(rec.get("_item_description", "")),
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

