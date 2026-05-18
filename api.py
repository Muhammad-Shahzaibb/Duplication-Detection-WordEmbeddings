"""
FastAPI application: Item Master duplicate engine APIs.
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI

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
    load_embedding_cache,
)
from jsonify import schema_records_to_minimized
from logging_setup import get_logger, setup_logging
from Schemas import (
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

EXACT_COSINE_EPS = 1e-7


def _exact_variant_matches(
    mat: np.ndarray,
    tuples: list[tuple],
    vec: np.ndarray,
    *,
    location: str,
) -> list[VariantDuplicateMatch]:
    scores = np.asarray(mat @ vec, dtype=np.float32)
    idxs = np.where(scores >= (1.0 - EXACT_COSINE_EPS))[0].tolist()
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
    tuples = fetch_item_master_rows_from_view()
    logger.info("Main view rows fetched: %s", len(tuples))
    records = [
        row_to_schema_json(item_description=desc, item_type=it, main_group=mg, sub_group=sg)
        for it, mg, sg, desc in tuples
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
    logger.info("Candidate embedding text: %r", candidate_text)

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

    db_matches = _exact_variant_matches(mat_main, main_tuples, cand_vec, location="db")
    matches.extend(db_matches)
    logger.info("DB embeddings: exact matches=%s", len(db_matches))

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

        ap_matches = _exact_variant_matches(mat_ap, approval_tuples, cand_vec, location="approval")
        matches.extend(ap_matches)
        logger.info("Approval embeddings: exact matches=%s", len(ap_matches))

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
