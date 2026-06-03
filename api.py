"""
FastAPI application: Item Master duplicate engine APIs.
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Config import (
    DUPLICATE_ENGINE_TEXT_THRESHOLD,
    VARIANT_CHECK_TEXT_THRESHOLD,
    VENDOR_VARIANT_CHECK_NAME_THRESHOLD,
)
from Db_View import (
    fetch_item_master_rows_from_view,
    fetch_vendor_master_rows_from_approval_view,
    fetch_vendor_master_rows_from_view,
)
from Item_Master_Duplicate_Engine import (
    embed_item_master_approval_view_at_runtime,
    rebuild_item_master_embeddings_cache,
    row_to_schema_json,
    run_item_master_duplicate_engine,
)
from embeddings import (
    EMBED_CACHE_FILE,
    EMBED_MODEL,
    build_embedding_text,
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
    VendorAccountNoVariantCheckRequest,
    VendorCNICVariantCheckRequest,
    VendorIBANVariantCheckRequest,
    VendorMasterDuplicateEngineResponse,
    VendorMasterUpdateEmbeddingsResponse,
    VendorNameVariantCheckRequest,
    VendorNTNVariantCheckRequest,
    VendorSTRNVariantCheckRequest,
    VendorVariantDuplicateCheckResponse,
    VendorVariantMatch,
)
from Vendor_Master_Duplicate_Engine import (
    IDX_CNIC,
    IDX_IBAN,
    IDX_ACCOUNT_NO,
    IDX_NTN,
    IDX_STRN,
    embed_vendor_approval_names_at_runtime,
    load_vendor_main_embeddings_reuse_if_present,
    match_vendor_name_variant,
    match_vendor_numeric_variant,
    rebuild_vendor_embeddings_cache,
    run_vendor_master_duplicate_engine,
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
    if int(meta_main.get("dim", -1)) != int(cand_vec.shape[0]):
        raise RuntimeError(
            "Main embedding cache dimension does not match candidate embedding. "
            "Run /Item-Master-update-embeddings to refresh the cache."
        )

    # Force-reuse mode: do not fail on row-count mismatch; align defensively to avoid index errors.
    if int(mat_main.shape[0]) != len(main_tuples):
        logger.warning(
            "DB embeddings: row mismatch (cache_vectors=%s view_rows=%s). "
            "Proceeding in force-reuse mode using the aligned prefix only. "
            "Run /Item-Master-update-embeddings to realign if needed.",
            int(mat_main.shape[0]),
            len(main_tuples),
        )
    n_main = min(int(mat_main.shape[0]), len(main_tuples))
    mat_main = np.asarray(mat_main[:n_main], dtype=np.float32)
    main_tuples = main_tuples[:n_main]

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

    # --- Approval embeddings (runtime only; never cached) ---
    mat_ap, approval_tuples = embed_item_master_approval_view_at_runtime()
    if approval_tuples:
        if int(mat_ap.shape[1]) != int(cand_vec.shape[0]):
            raise RuntimeError("Approval runtime embedding dimension does not match candidate embedding.")
        approval_numerics = _extract_tuple_numerics(approval_tuples)
        ap_matches = _threshold_variant_matches(
            mat_ap, approval_tuples, approval_numerics, cand_vec, cand_numeric,
            location="approval",
            text_threshold=VARIANT_CHECK_TEXT_THRESHOLD,
        )
        matches.extend(ap_matches)
        logger.info(
            "Approval embeddings (runtime): rows=%s matches=%s (text_threshold=%.2f, cand_numeric=%r)",
            len(approval_tuples),
            len(ap_matches),
            VARIANT_CHECK_TEXT_THRESHOLD,
            cand_numeric,
        )
    else:
        logger.info("Approval embeddings: skipped (approval view has 0 rows)")

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
    summary="Bulk duplicate check: intra-batch dedup then match against DB cache and approval view",
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
       and **approval** embeddings (computed once per request from the approval view, not cached).
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
    if int(meta_main.get("dim", -1)) != int(bulk_mat.shape[1]):
        raise RuntimeError(
            "Main embedding cache dimension does not match bulk embedding. "
            "Run /Item-Master-update-embeddings to refresh."
        )

    # Force-reuse mode: do not fail on row-count mismatch; align defensively to avoid index errors.
    if int(mat_main.shape[0]) != len(main_tuples):
        logger.warning(
            "Bulk DB embeddings: row mismatch (cache_vectors=%s view_rows=%s). "
            "Proceeding in force-reuse mode using the aligned prefix only. "
            "Run /Item-Master-update-embeddings to realign if needed.",
            int(mat_main.shape[0]),
            len(main_tuples),
        )
    n_main = min(int(mat_main.shape[0]), len(main_tuples))
    mat_main = np.asarray(mat_main[:n_main], dtype=np.float32)
    main_tuples = main_tuples[:n_main]

    # ── Approval embeddings once per request (runtime only; not cached) ───────
    mat_ap, approval_tuples = embed_item_master_approval_view_at_runtime()
    if approval_tuples:
        if int(mat_ap.shape[1]) != int(bulk_mat.shape[1]):
            raise RuntimeError("Approval runtime embedding dimension does not match bulk embedding.")
        logger.info(
            "Bulk approval embeddings (runtime): rows=%s dim=%s",
            len(approval_tuples),
            int(mat_ap.shape[1]),
        )
    else:
        logger.info("Approval embeddings: skipped (0 rows in approval view)")

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
        if approval_tuples:
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


@app.post(
    "/Vendor-Master-duplicate-engine",
    response_model=VendorMasterDuplicateEngineResponse,
    summary="Run duplicate detection on Vendor Master view (6 fields checked independently)",
    tags=["VENDOR MASTER APIS"],
)
def vendor_master_duplicate_engine() -> VendorMasterDuplicateEngineResponse:
    """
    Fetches all rows from the Vendor Master view and checks 6 fields independently:

    - **Name**       — embedding cosine similarity (VENDOR_NAME_TEXT_THRESHOLD)
    - **CNIC**       — normalized exact match (strip special chars + leading zeros)
    - **NTN**        — normalized exact match
    - **STRN**       — normalized exact match
    - **Account No** — normalized exact match
    - **IBAN**       — normalized exact match

    Each field result lists its own duplicate groups with row#, id, Name, and the field value.
    """
    logger.info("POST /Vendor-Master-duplicate-engine — start")
    rows = fetch_vendor_master_rows_from_view()
    logger.info("Vendor view rows fetched: %s", len(rows))
    payload = run_vendor_master_duplicate_engine(rows)
    logger.info(
        "POST /Vendor-Master-duplicate-engine — done | total=%s "
        "NAME_groups=%s CNIC_groups=%s NTN_groups=%s STRN_groups=%s ACCT_groups=%s IBAN_groups=%s",
        payload.get("total_records"),
        payload.get("duplicates_by_NAME", {}).get("duplicate_groups", 0),
        payload.get("duplicates_by_CNIC", {}).get("duplicate_groups", 0),
        payload.get("duplicates_by_NTN", {}).get("duplicate_groups", 0),
        payload.get("duplicates_by_STRN", {}).get("duplicate_groups", 0),
        payload.get("duplicates_by_ACCOUNT_NO", {}).get("duplicate_groups", 0),
        payload.get("duplicates_by_IBAN", {}).get("duplicate_groups", 0),
    )
    return VendorMasterDuplicateEngineResponse.model_validate(payload)


@app.post(
    "/Vendor-Master-update-embeddings",
    response_model=VendorMasterUpdateEmbeddingsResponse,
    summary="Refresh Vendor Master name embedding cache from the vendor view",
    tags=["VENDOR MASTER APIS"],
)
def vendor_master_update_embeddings() -> VendorMasterUpdateEmbeddingsResponse:
    """
    Forces a full recomputation of the Vendor Name embedding cache
    (``cache/vendor_embeddings_cache.npy``).  Call this after bulk vendor data changes.
    """
    logger.info("POST /Vendor-Master-update-embeddings — start (force COMPUTE)")
    rows = fetch_vendor_master_rows_from_view()
    logger.info("Vendor view rows fetched: %s", len(rows))
    payload = rebuild_vendor_embeddings_cache(rows)
    logger.info(
        "POST /Vendor-Master-update-embeddings — done | rows=%s dim=%s cache=%s",
        payload.get("total_records"),
        payload.get("embedding_dim"),
        payload.get("cache_file"),
    )
    return VendorMasterUpdateEmbeddingsResponse.model_validate(payload)


def _vendor_numeric_variant_response(
    field_label: str,
    candidate_value: str,
    col_idx: int,
) -> VendorVariantDuplicateCheckResponse:
    """Shared logic for the 5 numeric vendor variant check APIs."""
    logger.info(
        "POST /Vendor-Master-check-duplicate-%s — start | value=%r",
        field_label, candidate_value,
    )
    db_rows = fetch_vendor_master_rows_from_view()
    approval_rows = fetch_vendor_master_rows_from_approval_view()
    logger.info(
        "Vendor %s check: db_rows=%s approval_rows=%s",
        field_label, len(db_rows), len(approval_rows),
    )
    raw_matches = match_vendor_numeric_variant(candidate_value, col_idx, db_rows, approval_rows)
    matches = [VendorVariantMatch.model_validate(m) for m in raw_matches]
    status = "duplicate" if matches else "unique"
    logger.info(
        "POST /Vendor-Master-check-duplicate-%s — done | status=%s matches=%s",
        field_label, status, len(matches),
    )
    return VendorVariantDuplicateCheckResponse(status=status, matches=matches)


@app.post(
    "/Vendor-Master-check-duplicate-CNIC",
    response_model=VendorVariantDuplicateCheckResponse,
    summary="Check if a candidate CNIC already exists in Vendor Master or approval view",
    tags=["VENDOR MASTER APIS"],
)
def vendor_master_check_duplicate_cnic(
    req: VendorCNICVariantCheckRequest,
) -> VendorVariantDuplicateCheckResponse:
    return _vendor_numeric_variant_response("CNIC", req.CNIC, IDX_CNIC)


@app.post(
    "/Vendor-Master-check-duplicate-NTN",
    response_model=VendorVariantDuplicateCheckResponse,
    summary="Check if a candidate NTN already exists in Vendor Master or approval view",
    tags=["VENDOR MASTER APIS"],
)
def vendor_master_check_duplicate_ntn(
    req: VendorNTNVariantCheckRequest,
) -> VendorVariantDuplicateCheckResponse:
    return _vendor_numeric_variant_response("NTN", req.NTN, IDX_NTN)


@app.post(
    "/Vendor-Master-check-duplicate-STRN",
    response_model=VendorVariantDuplicateCheckResponse,
    summary="Check if a candidate STRN already exists in Vendor Master or approval view",
    tags=["VENDOR MASTER APIS"],
)
def vendor_master_check_duplicate_strn(
    req: VendorSTRNVariantCheckRequest,
) -> VendorVariantDuplicateCheckResponse:
    return _vendor_numeric_variant_response("STRN", req.STRN, IDX_STRN)


@app.post(
    "/Vendor-Master-check-duplicate-AccountNo",
    response_model=VendorVariantDuplicateCheckResponse,
    summary="Check if a candidate Account No already exists in Vendor Master or approval view",
    tags=["VENDOR MASTER APIS"],
)
def vendor_master_check_duplicate_account_no(
    req: VendorAccountNoVariantCheckRequest,
) -> VendorVariantDuplicateCheckResponse:
    return _vendor_numeric_variant_response("AccountNo", req.AccountNo, IDX_ACCOUNT_NO)


@app.post(
    "/Vendor-Master-check-duplicate-IBAN",
    response_model=VendorVariantDuplicateCheckResponse,
    summary="Check if a candidate IBAN already exists in Vendor Master or approval view",
    tags=["VENDOR MASTER APIS"],
)
def vendor_master_check_duplicate_iban(
    req: VendorIBANVariantCheckRequest,
) -> VendorVariantDuplicateCheckResponse:
    return _vendor_numeric_variant_response("IBAN", req.IBAN, IDX_IBAN)


@app.post(
    "/Vendor-Master-check-duplicate-Name",
    response_model=VendorVariantDuplicateCheckResponse,
    summary="Check if a candidate vendor Name is a duplicate (embedding cosine similarity)",
    tags=["VENDOR MASTER APIS"],
)
def vendor_master_check_duplicate_name(
    req: VendorNameVariantCheckRequest,
) -> VendorVariantDuplicateCheckResponse:
    logger.info("POST /Vendor-Master-check-duplicate-Name — start | Name=%r", req.Name)

    # Main DB: load cached embeddings, reuse regardless of row-count mismatch
    db_mat, db_rows = load_vendor_main_embeddings_reuse_if_present()
    logger.info("Vendor Name check: db_rows=%s (cache aligned)", len(db_rows))

    # Approval: embed at runtime, never saved
    ap_mat, approval_rows = embed_vendor_approval_names_at_runtime()
    logger.info("Vendor Name check: approval_rows=%s (runtime)", len(approval_rows))

    raw_matches = match_vendor_name_variant(
        req.Name,
        db_rows,
        db_mat,
        approval_rows,
        ap_mat,
        threshold=VENDOR_VARIANT_CHECK_NAME_THRESHOLD,
    )
    matches = [VendorVariantMatch.model_validate(m) for m in raw_matches]
    status = "duplicate" if matches else "unique"
    logger.info(
        "POST /Vendor-Master-check-duplicate-Name — done | threshold=%.3f status=%s matches=%s (db=%s approval=%s)",
        VENDOR_VARIANT_CHECK_NAME_THRESHOLD,
        status, len(matches),
        sum(1 for m in matches if m.location == "db"),
        sum(1 for m in matches if m.location == "approval"),
    )
    return VendorVariantDuplicateCheckResponse(status=status, matches=matches)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
