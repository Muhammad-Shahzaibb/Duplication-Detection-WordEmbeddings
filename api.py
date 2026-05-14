"""
FastAPI application: Item Master duplicate engine (single endpoint).
"""
from __future__ import annotations

import numpy as np
from typing import Literal
from fastapi import FastAPI

from Db_View import fetch_item_master_rows_from_approval_view, fetch_item_master_rows_from_view
from Item_Master_Duplicate_Engine import (
    load_or_build_embeddings_matrix_for_schema_records,
    rebuild_item_master_embeddings_cache,
    row_to_schema_json,
    run_item_master_duplicate_engine,
)
from embeddings import EMBED_APPROVAL_CACHE_FILE, EMBED_CACHE_FILE, EMBED_MODEL, build_embedding_text, embed_texts_local, load_embedding_cache
from jsonify import schema_records_to_minimized
from Schemas import (
    ItemMasterDuplicateEngineResponse,
    ItemMasterUpdateEmbeddingsResponse,
    ItemMasterVariantDuplicateCheckRequest,
    ItemMasterVariantDuplicateCheckResponse,
)

app = FastAPI(
    title="STYLE TEXTILE AI BACKEND",
    version="1.0.0",
    openapi_tags=[{"name": "ITEM MASTER APIS", "description": "Item Master data and duplicate detection."}],
)


@app.post(
    "/Item-Master-duplicate-engine",
    response_model=ItemMasterDuplicateEngineResponse,
    summary="Run duplicate detection on Item Master view",
    tags=["ITEM MASTER APIS"],
)
def item_master_duplicate_engine() -> ItemMasterDuplicateEngineResponse:
    tuples = fetch_item_master_rows_from_view()
    records = [
        row_to_schema_json(
            item_type=it,
            main_group=mg,
            sub_group=sg,
            item_description=desc,
        )
        for it, mg, sg, desc in tuples
    ]
    payload = run_item_master_duplicate_engine(records)
    return ItemMasterDuplicateEngineResponse.model_validate(payload)


@app.post(
    "/Item-Master-update-embeddings",
    response_model=ItemMasterUpdateEmbeddingsResponse,
    summary="Refresh Item Master embedding cache from the DB view",
    tags=["ITEM MASTER APIS"],
)
def item_master_update_embeddings() -> ItemMasterUpdateEmbeddingsResponse:
    """
    Pull latest rows from the Item Master view, run the same JSON minimization as duplicate detection,
    recompute embeddings, and overwrite ``embeddings_cache.npy`` and its ``.meta.json`` sidecar.
    """
    tuples = fetch_item_master_rows_from_view()
    records = [
        row_to_schema_json(
            item_type=it,
            main_group=mg,
            sub_group=sg,
            item_description=desc,
        )
        for it, mg, sg, desc in tuples
    ]
    payload = rebuild_item_master_embeddings_cache(records)
    return ItemMasterUpdateEmbeddingsResponse.model_validate(payload)


@app.post(
    "/Item-Master-update-approval-embeddings",
    response_model=ItemMasterUpdateEmbeddingsResponse,
    summary="Refresh approval queue embedding cache from the approval view",
    tags=["ITEM MASTER APIS"],
)
def item_master_update_approval_embeddings() -> ItemMasterUpdateEmbeddingsResponse:
    """
    Pull latest rows from the approval Item Master view (``ITEM_MASTER_APPROVAL_VIEW``, default
    ``vw_item_master_items``), run the same JSON minimization as duplicate detection, recompute
    embeddings, and overwrite ``Approval_embedding_cache.npy`` and its ``.meta.json`` sidecar.
    """
    tuples = fetch_item_master_rows_from_approval_view()
    records = [
        row_to_schema_json(
            item_type=it,
            main_group=mg,
            sub_group=sg,
            item_description=desc,
        )
        for it, mg, sg, desc in tuples
    ]
    payload = rebuild_item_master_embeddings_cache(records, cache_path=EMBED_APPROVAL_CACHE_FILE)
    return ItemMasterUpdateEmbeddingsResponse.model_validate(payload)


@app.post(
    "/Item-Master-check-duplicate-variant",
    response_model=ItemMasterVariantDuplicateCheckResponse,
    summary="Check if a single candidate row is an exact duplicate (cosine==1) of existing Item Master rows",
    tags=["ITEM MASTER APIS"],
)
def item_master_check_duplicate_variant(
    req: ItemMasterVariantDuplicateCheckRequest,
) -> ItemMasterVariantDuplicateCheckResponse:
    """
    Real-time duplicate check for *one* candidate row.

    - Embeds the candidate row immediately.
    - Compares against the **main** Item Master embedding cache (must be up to date; no recompute here).
    - If the approval view has rows, also compares against **approval** embeddings
      (``Approval_embedding_cache.npy`` + metadata), building/reusing that cache as needed.
    - Returns `duplicate` if any exact (cosine==1 within float tolerance) matches exist in either pool,
      with an array of matched original ITEMDESC values (main + approval).
    """
    exact_eps = 1e-7

    def _exact_match_descs(mat: np.ndarray, tuples: list[tuple], vec: np.ndarray) -> list[str]:
        scores = np.asarray(mat @ vec, dtype=np.float32)
        idxs = np.where(scores >= (1.0 - exact_eps))[0].tolist()
        out: list[str] = []
        for i in idxs:
            out.append("" if tuples[i][3] is None else str(tuples[i][3]))
        return out

    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # 1) Build schema record and minimized record for the single candidate row (no file write).
    candidate_schema = row_to_schema_json(
        item_type=req.ITEM_TYPE,
        main_group=req.MAINGROUP,
        sub_group=req.SUBGROUP,
        item_description=req.ITEMDESC,
    )
    candidate_min = schema_records_to_minimized([candidate_schema])[0]

    # 2) Embed the candidate row (normalized embeddings -> cosine = dot product).
    candidate_text = build_embedding_text(candidate_min)
    cand_vec = embed_texts_local([candidate_text], model_id=EMBED_MODEL, batch_size=1)
    if cand_vec.ndim != 2 or cand_vec.shape[0] != 1:
        raise RuntimeError("Unexpected embedding output shape for candidate row")
    cand_vec = cand_vec[0]  # shape (dim,)

    # 3) Main DB view: load cache only (no recompute).
    mat_main, meta_main = load_embedding_cache(EMBED_CACHE_FILE)
    main_tuples = fetch_item_master_rows_from_view()
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

    matched_main = _exact_match_descs(mat_main, main_tuples, cand_vec)

    # 4) Approval view: only if there are pending rows; load or build approval cache.
    approval_tuples = fetch_item_master_rows_from_approval_view()
    matched_approval: list[str] = []
    if approval_tuples:
        approval_records = [
            row_to_schema_json(
                item_type=it,
                main_group=mg,
                sub_group=sg,
                item_description=desc,
            )
            for it, mg, sg, desc in approval_tuples
        ]
        mat_ap = load_or_build_embeddings_matrix_for_schema_records(
            approval_records,
            cache_path=EMBED_APPROVAL_CACHE_FILE,
        )
        _, meta_ap = load_embedding_cache(EMBED_APPROVAL_CACHE_FILE)
        if str(meta_ap.get("model", "")) != EMBED_MODEL:
            raise RuntimeError(
                "Approval embedding cache model does not match the active embedding model. "
                "Run /Item-Master-update-approval-embeddings to refresh, or delete Approval_embedding_cache.npy (+ .meta.json) to rebuild."
            )
        if int(meta_ap.get("rows", -1)) != len(approval_tuples):
            raise RuntimeError("Approval embedding cache row-count does not match approval view rows.")
        if int(meta_ap.get("dim", -1)) != int(cand_vec.shape[0]):
            raise RuntimeError("Approval embedding cache dimension does not match candidate embedding.")
        matched_approval = _exact_match_descs(mat_ap, approval_tuples, cand_vec)

    matched_descs = _dedupe_preserve_order(matched_main + matched_approval)

    if not matched_descs:
        return ItemMasterVariantDuplicateCheckResponse(status="unique", location="", ITEMDESC=[])

    if matched_main and matched_approval:
        loc: Literal["", "db", "approval", "both"] = "both"
    elif matched_main:
        loc = "db"
    else:
        loc = "approval"

    return ItemMasterVariantDuplicateCheckResponse(status="duplicate", location=loc, ITEMDESC=matched_descs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
