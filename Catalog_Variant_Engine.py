"""
Runtime embedding duplicate checks for Main Code, Sub Code, and UOM catalog views.
No cache read/write — views are small; embeddings are computed per request.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from Config import CATALOG_COL_ID, UOM_COL, UOM_VIEW
from Db_View import fetch_catalog_text_view_rows
from embeddings import EMBED_MODEL, embed_texts_local
from logging_setup import get_logger
from uom_synonyms import canonical_uom

logger = get_logger("style_textile.catalog_variant")


def _parse_catalog_rows(rows: list[tuple[Any, ...]]) -> tuple[list[int], list[str]]:
    """Return (view ids, text values) from (id, text) or legacy (text,) rows."""
    ids: list[int] = []
    texts: list[str] = []
    for row in rows:
        if len(row) >= 2:
            rid = row[0]
            raw = row[1]
            ids.append(int(rid) if rid is not None else 0)
        else:
            raw = row[0]
            ids.append(len(ids) + 1)
        texts.append("" if raw is None else str(raw).strip())
    return ids, texts


def check_catalog_text_variant(
    candidate: str,
    *,
    view: str,
    col_text: str,
    match_value_key: str,
    threshold: float,
    col_id: str | None = None,
) -> dict[str, Any]:
    """
    Embed candidate + all view rows at runtime; return duplicate/unique + matches.

    Each match uses the view ``id`` as row# (ORDER BY id). No embeddings are cached.
    """
    cand = (candidate or "").strip()
    if not cand:
        return {"status": "unique", "matches": []}

    cid = col_id if col_id is not None else CATALOG_COL_ID
    rows = fetch_catalog_text_view_rows(view=view, col_text=col_text, col_id=cid)
    row_ids, texts = _parse_catalog_rows(rows)
    logger.info(
        "Catalog variant check view=%s rows=%s threshold=%.3f candidate=%r",
        view,
        len(texts),
        threshold,
        cand,
    )

    cand_vec = embed_texts_local([cand], model_id=EMBED_MODEL, batch_size=1)
    if cand_vec.ndim != 2 or cand_vec.shape[0] != 1:
        raise RuntimeError("Unexpected embedding output shape for catalog candidate")
    cand_vec = cand_vec[0]

    if not texts:
        return {"status": "unique", "matches": []}

    mat = embed_texts_local(texts, model_id=EMBED_MODEL)
    mat = np.asarray(mat, dtype=np.float32)
    if int(mat.shape[1]) != int(cand_vec.shape[0]):
        raise RuntimeError("Catalog view embedding dimension does not match candidate embedding.")

    scores = mat @ cand_vec
    matches: list[dict[str, Any]] = []
    for i, score in enumerate(scores):
        if float(score) >= threshold:
            matches.append({
                match_value_key: texts[i],
                "row": row_ids[i],
            })

    status = "duplicate" if matches else "unique"
    logger.info("Catalog variant check view=%s — status=%s matches=%s", view, status, len(matches))
    return {"status": status, "matches": matches}


def check_uom_variant(
    candidate: str,
    *,
    threshold: float,
    match_value_key: str = "UOMDescription",
    col_id: str | None = None,
) -> dict[str, Any]:
    """
    UOM duplicate check: synonym/canonical match first, then embedding similarity.

    1. Map candidate and every view row through ``canonical_uom`` (alias dictionary).
    2. If any row shares the same canonical code → duplicate (no embeddings).
    3. Otherwise embed candidate + all view texts at runtime and apply cosine threshold.
    """
    cand = (candidate or "").strip()
    if not cand:
        return {"status": "unique", "matches": []}

    cid = col_id if col_id is not None else CATALOG_COL_ID
    rows = fetch_catalog_text_view_rows(view=UOM_VIEW, col_text=UOM_COL, col_id=cid)
    row_ids, texts = _parse_catalog_rows(rows)
    cand_canon = canonical_uom(cand)

    logger.info(
        "UOM variant check rows=%s candidate=%r canonical=%r threshold=%.3f",
        len(texts),
        cand,
        cand_canon,
        threshold,
    )

    if not cand_canon:
        return {"status": "unique", "matches": []}

    # ── Step 1: synonym / canonical match ─────────────────────────────────────
    synonym_matches: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        if not text:
            continue
        row_canon = canonical_uom(text)
        if row_canon and row_canon == cand_canon:
            synonym_matches.append({
                match_value_key: text,
                "row": row_ids[i],
            })

    if synonym_matches:
        logger.info(
            "UOM variant: synonym duplicate | canonical=%r matches=%s",
            cand_canon,
            len(synonym_matches),
        )
        return {"status": "duplicate", "matches": synonym_matches}

    # ── Step 2: embedding fallback ──────────────────────────────────────────────
    if not texts:
        return {"status": "unique", "matches": []}

    logger.info("UOM variant: no synonym match — running embedding fallback")
    cand_vec = embed_texts_local([cand], model_id=EMBED_MODEL, batch_size=1)
    if cand_vec.ndim != 2 or cand_vec.shape[0] != 1:
        raise RuntimeError("Unexpected embedding output shape for UOM candidate")
    cand_vec = cand_vec[0]

    mat = embed_texts_local(texts, model_id=EMBED_MODEL)
    mat = np.asarray(mat, dtype=np.float32)
    if int(mat.shape[1]) != int(cand_vec.shape[0]):
        raise RuntimeError("UOM embedding dimension does not match candidate embedding.")

    scores = mat @ cand_vec
    embedding_matches: list[dict[str, Any]] = []
    for i, score in enumerate(scores):
        if float(score) >= threshold:
            embedding_matches.append({
                match_value_key: texts[i],
                "row": row_ids[i],
            })

    status = "duplicate" if embedding_matches else "unique"
    logger.info(
        "UOM variant: embedding %s | matches=%s",
        status,
        len(embedding_matches),
    )
    return {"status": status, "matches": embedding_matches}
