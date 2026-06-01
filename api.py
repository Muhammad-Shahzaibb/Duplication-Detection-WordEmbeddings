"""
FastAPI application: Item Master duplicate engine APIs.
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Config import DUPLICATE_ENGINE_TEXT_THRESHOLD, VARIANT_CHECK_TEXT_THRESHOLD
from Db_View import fetch_item_master_rows_from_approval_view, fetch_item_master_rows_from_view
from Item_Master_Duplicate_Engine import (
    load_or_build_embeddings_matrix_for_schema_records,
    rebuild_item_master_embeddings_cache,
    row_to_schema_json,
    run_item_master_duplicate_engine,
)
from embeddings import (
    EMBED_APPROVAL_CACHE_FILE,
    EMBED_CACHE_FILE,
    EMBED_MODEL,
    build_embedding_text,
    describe_embedding_cache_action,
    embed_texts_local,
    find_duplicate_groups_by_text_and_numeric,
    find_variant_matches_with_threshold,
    load_embedding_cache,
)
from jsonify import regex_extract_attributes, schema_records_to_minimized
from logging_setup import get_logger, setup_logging
from Schemas import (
    BulkItemResult,
    IntraBulkDuplicateGroup,
    ItemMasterBulkDuplicateCheckRequest,
    ItemMasterBulkDuplicateCheckResponse,
    ItemMasterDuplicateEngineResponse,
    ItemMasterUpdateEmbeddingsResponse,
    ItemMasterVariantDuplicateCheckRequest,
    ItemMasterVariantDuplicateCheckResponse,
    VariantDuplicateMatch,
)

setup_logging()
logger = get_logger("style_textile.api")

app = FastAPI(
    title="STYLE TEXTILE AI BACKEND",
    version="1.0.0",
    openapi_tags=[{"name": "ITEM MASTER APIS", "description": "Item Master data and duplicate detection."}],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_tuple_numerics(tuples: list[tuple]) -> list[str]:
    """Extract the *numeric* part of ITEMDESC from each view tuple (it, mg, sg, desc)."""
    return [
        (regex_extract_attributes(str(t[3]) if t[3] is not None else "").get("numeric") or "")
        for t in tuples
    ]


def _threshold_variant_matches(
    mat: np.ndarray,
    tuples: list[tuple],
    row_numerics: list[str],
    cand_vec: np.ndarray,
    cand_numeric: str,
    *,
    location: str,
    text_threshold: float,
) -> list[VariantDuplicateMatch]:
    """
    Return matches where:
      - numeric parts match exactly (case-insensitive, stripped), AND
      - text cosine similarity >= text_threshold.
    """
    idxs = find_variant_matches_with_threshold(
        mat, row_numerics, cand_vec, cand_numeric,
        text_threshold=text_threshold,
    )
    out: list[VariantDuplicateMatch] = []
    for i in idxs:
        desc = "" if tuples[i][3] is None else str(tuples[i][3])
        out.append(
            VariantDuplicateMatch(
                ITEMDESC=desc,
                location=location,  # type: ignore[arg-type]
                row=i + 1,
            )
        )
    return out


@app.post(
    "/Item-Master-duplicate-engine",
    response_model=ItemMasterDuplicateEngineResponse,
    summary="Run duplicate detection on Item Master view",
    tags=["ITEM MASTER APIS"],
)
def item_master_duplicate_engine() -> ItemMasterDuplicateEngineResponse:
    logger.info("POST /Item-Master-duplicate-engine — start")
    tuples = fetch_item_master_rows_from_view(include_item_code=True)
    logger.info("Main view rows fetched: %s", len(tuples))
    records = [
        row_to_schema_json(
            item_description=desc, item_type=it, main_group=mg, sub_group=sg, item_code=code
        )
        for it, mg, sg, desc, code in tuples
    ]
    payload = run_item_master_duplicate_engine(records)
    logger.info(
        "POST /Item-Master-duplicate-engine — done | total=%s duplicate_records=%s groups=%s",
        payload.get("total_records"),
        payload.get("duplicate_records"),
        len(payload.get("duplicates", {})),
    )
    return ItemMasterDuplicateEngineResponse.model_validate(payload)


@app.post(
    "/Item-Master-update-embeddings",
    response_model=ItemMasterUpdateEmbeddingsResponse,
    summary="Refresh Item Master embedding cache from the DB view",
    tags=["ITEM MASTER APIS"],
)
def item_master_update_embeddings() -> ItemMasterUpdateEmbeddingsResponse:
    logger.info("POST /Item-Master-update-embeddings — start (db cache, force COMPUTE)")
    tuples = fetch_item_master_rows_from_view()
    logger.info("Main view rows fetched: %s", len(tuples))
    records = [
        row_to_schema_json(item_description=desc, item_type=it, main_group=mg, sub_group=sg)
        for it, mg, sg, desc in tuples
    ]
    payload = rebuild_item_master_embeddings_cache(records)
    logger.info(
        "POST /Item-Master-update-embeddings — done | rows=%s dim=%s cache=%s",
        payload.get("total_records"),
        payload.get("embedding_dim"),
        payload.get("cache_file"),
    )
    return ItemMasterUpdateEmbeddingsResponse.model_validate(payload)


@app.post(
    "/Item-Master-update-approval-embeddings",
    response_model=ItemMasterUpdateEmbeddingsResponse,
    summary="Refresh approval queue embedding cache from the approval view",
    tags=["ITEM MASTER APIS"],
)
def item_master_update_approval_embeddings() -> ItemMasterUpdateEmbeddingsResponse:
    logger.info("POST /Item-Master-update-approval-embeddings — start (approval cache, force COMPUTE)")
    tuples = fetch_item_master_rows_from_approval_view()
    logger.info("Approval view rows fetched: %s", len(tuples))
    records = [
        row_to_schema_json(item_description=desc, item_type=it, main_group=mg, sub_group=sg)
        for it, mg, sg, desc in tuples
    ]
    payload = rebuild_item_master_embeddings_cache(records, cache_path=EMBED_APPROVAL_CACHE_FILE)
    logger.info(
        "POST /Item-Master-update-approval-embeddings — done | rows=%s dim=%s cache=%s",
        payload.get("total_records"),
        payload.get("embedding_dim"),
        payload.get("cache_file"),
    )
    return ItemMasterUpdateEmbeddingsResponse.model_validate(payload)


@app.post(
    "/Item-Master-check-duplicate-variant",
    response_model=ItemMasterVariantDuplicateCheckResponse,
    summary="Check if a single candidate ITEMDESC is an exact duplicate (cosine==1)",
    tags=["ITEM MASTER APIS"],
)
def item_master_check_duplicate_variant(
    req: ItemMasterVariantDuplicateCheckRequest,
) -> ItemMasterVariantDuplicateCheckResponse:
    logger.info("POST /Item-Master-check-duplicate-variant — start | ITEMDESC=%r", req.ITEMDESC)

    candidate_schema = row_to_schema_json(item_description=req.ITEMDESC)
    candidate_min = schema_records_to_minimized([candidate_schema])[0]
    candidate_text = build_embedding_text(candidate_min)
    cand_numeric = candidate_min.get("numeric") or ""
    logger.info("Candidate embedding text: %r | numeric: %r", candidate_text, cand_numeric)

    cand_vec = embed_texts_local([candidate_text], model_id=EMBED_MODEL, batch_size=1)
    if cand_vec.ndim != 2 or cand_vec.shape[0] != 1:
        raise RuntimeError("Unexpected embedding output shape for candidate row")
    cand_vec = cand_vec[0]
    logger.info("Candidate vector embedded (dim=%s)", cand_vec.shape[0])

    matches: list[VariantDuplicateMatch] = []

    # --- Main DB embeddings (reuse only) ---
    logger.info("DB embeddings [%s]: loading cache (reuse only, no compute)", EMBED_CACHE_FILE)
    mat_main, meta_main = load_embedding_cache(EMBED_CACHE_FILE)
    main_tuples = fetch_item_master_rows_from_view()
    logger.info("DB embeddings: REUSED from cache | path=%s rows=%s", EMBED_CACHE_FILE, len(main_tuples))

    if str(meta_main.get("model", "")) != EMBED_MODEL:
        raise RuntimeError(
            "Main embedding cache model does not match the active embedding model. "
            "Run /Item-Master-update-embeddings to refresh the cache."
        )
    if int(meta_main.get("rows", -1)) != len(main_tuples):
        raise RuntimeError(
            "Main embedding cache row-count does not match current view rows. "
            "Run /Item-Master-update-embeddings to refresh the cache."
        )
    if int(meta_main.get("dim", -1)) != int(cand_vec.shape[0]):
        raise RuntimeError(
            "Main embedding cache dimension does not match candidate embedding. "
            "Run /Item-Master-update-embeddings to refresh the cache."
        )

    main_numerics = _extract_tuple_numerics(main_tuples)
    db_matches = _threshold_variant_matches(
        mat_main, main_tuples, main_numerics, cand_vec, cand_numeric,
        location="db",
        text_threshold=VARIANT_CHECK_TEXT_THRESHOLD,
    )
    matches.extend(db_matches)
    logger.info(
        "DB embeddings: matches=%s (text_threshold=%.2f, cand_numeric=%r)",
        len(db_matches), VARIANT_CHECK_TEXT_THRESHOLD, cand_numeric,
    )

    # --- Approval embeddings (reuse or compute) ---
    approval_tuples = fetch_item_master_rows_from_approval_view()
    if not approval_tuples:
        logger.info("Approval embeddings: skipped (approval view has 0 rows)")
    else:
        approval_records = [
            row_to_schema_json(item_description=desc, item_type=it, main_group=mg, sub_group=sg)
            for it, mg, sg, desc in approval_tuples
        ]
        approval_min = schema_records_to_minimized(approval_records)
        planned = describe_embedding_cache_action(
            approval_min,
            cache_path=EMBED_APPROVAL_CACHE_FILE,
            model=EMBED_MODEL,
        )
        logger.info(
            "Approval embeddings [%s]: planned action=%s | approval_rows=%s",
            EMBED_APPROVAL_CACHE_FILE,
            planned,
            len(approval_tuples),
        )

        mat_ap, action = load_or_build_embeddings_matrix_for_schema_records(
            approval_records,
            cache_path=EMBED_APPROVAL_CACHE_FILE,
        )
        logger.info(
            "Approval embeddings: %s | path=%s rows=%s",
            "REUSED from cache" if action == "reuse" else "COMPUTED and saved",
            EMBED_APPROVAL_CACHE_FILE,
            len(approval_tuples),
        )

        _, meta_ap = load_embedding_cache(EMBED_APPROVAL_CACHE_FILE)
        if str(meta_ap.get("model", "")) != EMBED_MODEL:
            raise RuntimeError(
                "Approval embedding cache model does not match the active embedding model. "
                "Run /Item-Master-update-approval-embeddings to refresh."
            )
        if int(meta_ap.get("rows", -1)) != len(approval_tuples):
            raise RuntimeError("Approval embedding cache row-count does not match approval view rows.")
        if int(meta_ap.get("dim", -1)) != int(cand_vec.shape[0]):
            raise RuntimeError("Approval embedding cache dimension does not match candidate embedding.")

        approval_numerics = _extract_tuple_numerics(approval_tuples)
        ap_matches = _threshold_variant_matches(
            mat_ap, approval_tuples, approval_numerics, cand_vec, cand_numeric,
            location="approval",
            text_threshold=VARIANT_CHECK_TEXT_THRESHOLD,
        )
        matches.extend(ap_matches)
        logger.info(
            "Approval embeddings: matches=%s (text_threshold=%.2f, cand_numeric=%r)",
            len(ap_matches), VARIANT_CHECK_TEXT_THRESHOLD, cand_numeric,
        )

    if not matches:
        logger.info("POST /Item-Master-check-duplicate-variant — done | status=unique")
        return ItemMasterVariantDuplicateCheckResponse(status="unique", matches=[])

    logger.info(
        "POST /Item-Master-check-duplicate-variant — done | status=duplicate matches=%s (db=%s approval=%s)",
        len(matches),
        sum(1 for m in matches if m.location == "db"),
        sum(1 for m in matches if m.location == "approval"),
    )
    return ItemMasterVariantDuplicateCheckResponse(status="duplicate", matches=matches)


@app.post(
    "/Item-Master-check-duplicate-bulk",
    response_model=ItemMasterBulkDuplicateCheckResponse,
    summary="Bulk duplicate check: intra-batch dedup then match against DB and approval caches",
    tags=["ITEM MASTER APIS"],
)
def item_master_check_duplicate_bulk(
    req: ItemMasterBulkDuplicateCheckRequest,
) -> ItemMasterBulkDuplicateCheckResponse:
    """
    Two-step bulk duplicate check:

    1. Embed all submitted ITEMDESC values in **real-time** (no cache reuse).
       Detect exact-duplicate groups within the batch itself.
    2. For each unique representative, check against the **main DB** embeddings (cache reuse only)
       and the **approval** embeddings (build/reuse same as variant check).
    """
    submitted = req.ITEMDESC
    total_submitted = len(submitted)
    logger.info("POST /Item-Master-check-duplicate-bulk — start | submitted=%s", total_submitted)

    # ── Step 1: embed the entire bulk in real-time (no cache) ──────────────────
    bulk_schema = [row_to_schema_json(item_description=d) for d in submitted]
    bulk_min = schema_records_to_minimized(bulk_schema)
    bulk_texts = [build_embedding_text(r) for r in bulk_min]

    logger.info("Bulk: computing %s real-time embeddings (no cache)", total_submitted)
    bulk_mat = embed_texts_local(bulk_texts, model_id=EMBED_MODEL)
    logger.info("Bulk: embeddings done | dim=%s", bulk_mat.shape[1] if bulk_mat.ndim == 2 else "?")

    # ── Find intra-bulk duplicate groups (engine threshold + exact numeric) ────
    bulk_numerics = [r.get("numeric") or "" for r in bulk_min]
    logger.info(
        "Bulk intra-batch: using text_threshold=%.2f + exact numeric match",
        DUPLICATE_ENGINE_TEXT_THRESHOLD,
    )
    intra_groups_raw = find_duplicate_groups_by_text_and_numeric(
        bulk_mat, bulk_numerics, text_threshold=DUPLICATE_ENGINE_TEXT_THRESHOLD
    )

    # Track which submitted indices are "extra" duplicates (not the representative)
    # Representative = first index in each group; others are duplicates of it.
    extra_indices: set[int] = set()
    intra_bulk_groups: list[IntraBulkDuplicateGroup] = []
    for group in intra_groups_raw:
        rep_idx = group[0]
        dup_idxs = group[1:]
        extra_indices.update(dup_idxs)
        intra_bulk_groups.append(
            IntraBulkDuplicateGroup(
                representative=submitted[rep_idx],
                duplicates=[submitted[i] for i in dup_idxs],
            )
        )

    logger.info(
        "Bulk intra-batch: %s duplicate group(s), %s extra (removed) values",
        len(intra_bulk_groups),
        len(extra_indices),
    )

    # Unique indices: all submitted except the extra duplicates
    unique_indices = [i for i in range(total_submitted) if i not in extra_indices]
    unique_count = len(unique_indices)
    logger.info("Bulk: %s unique descriptions proceeding to DB/approval check", unique_count)

    # ── Step 2: load main DB cache once for all unique items ───────────────────
    logger.info("DB embeddings [%s]: loading cache (reuse only)", EMBED_CACHE_FILE)
    mat_main, meta_main = load_embedding_cache(EMBED_CACHE_FILE)
    main_tuples = fetch_item_master_rows_from_view()
    logger.info("DB embeddings: REUSED | rows=%s", len(main_tuples))

    if str(meta_main.get("model", "")) != EMBED_MODEL:
        raise RuntimeError(
            "Main embedding cache model mismatch. Run /Item-Master-update-embeddings to refresh."
        )
    if int(meta_main.get("rows", -1)) != len(main_tuples):
        raise RuntimeError(
            "Main embedding cache row-count does not match current view rows. "
            "Run /Item-Master-update-embeddings to refresh."
        )
    if int(meta_main.get("dim", -1)) != int(bulk_mat.shape[1]):
        raise RuntimeError(
            "Main embedding cache dimension does not match bulk embedding. "
            "Run /Item-Master-update-embeddings to refresh."
        )

    # ── Load/build approval cache once ────────────────────────────────────────
    mat_ap: np.ndarray | None = None
    approval_tuples: list[tuple] = []
    approval_tuples = fetch_item_master_rows_from_approval_view()
    if not approval_tuples:
        logger.info("Approval embeddings: skipped (0 rows in approval view)")
    else:
        approval_records = [
            row_to_schema_json(item_description=desc, item_type=it, main_group=mg, sub_group=sg)
            for it, mg, sg, desc in approval_tuples
        ]
        approval_min = schema_records_to_minimized(approval_records)
        planned = describe_embedding_cache_action(
            approval_min, cache_path=EMBED_APPROVAL_CACHE_FILE, model=EMBED_MODEL
        )
        logger.info(
            "Approval embeddings [%s]: planned action=%s | rows=%s",
            EMBED_APPROVAL_CACHE_FILE, planned, len(approval_tuples),
        )
        mat_ap, ap_action = load_or_build_embeddings_matrix_for_schema_records(
            approval_records, cache_path=EMBED_APPROVAL_CACHE_FILE
        )
        logger.info(
            "Approval embeddings: %s | rows=%s",
            "REUSED from cache" if ap_action == "reuse" else "COMPUTED and saved",
            len(approval_tuples),
        )
        _, meta_ap = load_embedding_cache(EMBED_APPROVAL_CACHE_FILE)
        if str(meta_ap.get("model", "")) != EMBED_MODEL:
            raise RuntimeError(
                "Approval embedding cache model mismatch. "
                "Run /Item-Master-update-approval-embeddings to refresh."
            )
        if int(meta_ap.get("rows", -1)) != len(approval_tuples):
            raise RuntimeError("Approval embedding cache row-count does not match approval view rows.")
        if int(meta_ap.get("dim", -1)) != int(bulk_mat.shape[1]):
            raise RuntimeError("Approval embedding cache dimension does not match bulk embedding.")

    # ── Pre-compute per-row numerics for DB and approval (done once, not per item) ──
    main_row_numerics = _extract_tuple_numerics(main_tuples)
    approval_row_numerics = _extract_tuple_numerics(approval_tuples) if approval_tuples else []
    logger.info(
        "Bulk DB/approval checks: text_threshold=%.2f + exact numeric match",
        VARIANT_CHECK_TEXT_THRESHOLD,
    )

    # ── Step 2: check each unique description ─────────────────────────────────
    results: list[BulkItemResult] = []
    for i in unique_indices:
        cand_vec = bulk_mat[i]
        desc = submitted[i]
        cand_numeric = bulk_min[i].get("numeric") or ""

        matches: list[VariantDuplicateMatch] = _threshold_variant_matches(
            mat_main, main_tuples, main_row_numerics, cand_vec, cand_numeric,
            location="db",
            text_threshold=VARIANT_CHECK_TEXT_THRESHOLD,
        )
        if mat_ap is not None:
            matches.extend(
                _threshold_variant_matches(
                    mat_ap, approval_tuples, approval_row_numerics, cand_vec, cand_numeric,
                    location="approval",
                    text_threshold=VARIANT_CHECK_TEXT_THRESHOLD,
                )
            )

        results.append(
            BulkItemResult(
                ITEMDESC=desc,
                status="duplicate" if matches else "unique",
                matches=matches,
            )
        )

    duplicate_results = sum(1 for r in results if r.status == "duplicate")
    logger.info(
        "POST /Item-Master-check-duplicate-bulk — done | submitted=%s unique=%s "
        "intra_groups=%s db/approval_duplicates=%s",
        total_submitted, unique_count, len(intra_bulk_groups), duplicate_results,
    )

    return ItemMasterBulkDuplicateCheckResponse(
        total_submitted=total_submitted,
        unique_count=unique_count,
        intra_bulk_duplicate_groups=intra_bulk_groups,
        results=results,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
