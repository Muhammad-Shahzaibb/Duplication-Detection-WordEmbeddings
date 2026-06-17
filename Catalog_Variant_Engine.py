"""
Runtime duplicate checks for Main Code, Sub Code, and UOM catalog views.

View rows are normalized (no spell) once per request. The candidate is checked
without spell correction first; SymSpell runs only on a second pass if needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from catalog_fuzzy import find_fuzzy_catalog_matches
from Config import CATALOG_COL_ID, UOM_COL, UOM_VIEW
from Db_View import fetch_catalog_text_view_rows
from embeddings import EMBED_MODEL, embed_texts_local
from item_spell import normalize_variant_text, preprocess_variant_text, variant_check_passes
from jsonify import clean_str
from logging_setup import get_logger
from uom_synonyms import canonical_uom

logger = get_logger("style_textile.catalog_variant")


def _normalize_variant_texts(raw_texts: list[str]) -> list[str]:
    """Normalize view texts for embedding (no spell — matches cache / first pass)."""
    return [normalize_variant_text(t) if t else "" for t in raw_texts]


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


def _embedding_matches(
    mat: np.ndarray,
    cand_vec: np.ndarray,
    *,
    texts: list[str],
    row_ids: list[int],
    match_value_key: str,
    threshold: float,
) -> list[dict[str, Any]]:
    scores = mat @ cand_vec
    matches: list[dict[str, Any]] = []
    for i, score in enumerate(scores):
        if float(score) >= threshold:
            matches.append({
                match_value_key: texts[i],
                "row": row_ids[i],
            })
    return matches


def check_catalog_text_variant(
    candidate: str,
    *,
    view: str,
    col_text: str,
    match_value_key: str,
    threshold: float,
    col_id: str | None = None,
    fuzzy_fallback: bool = False,
) -> dict[str, Any]:
    """
    Embed candidate + all view rows at runtime; return duplicate/unique + matches.

    View embeddings are computed once (normalize only, no spell). The candidate
    is tried without spell first; SymSpell + re-embed runs only if no match.

    When ``fuzzy_fallback=True`` (Main/Sub code APIs), token-aligned Levenshtein
    runs only on the spell-correction pass if embedding similarity finds nothing.
    """
    raw_cand = clean_str(candidate)
    if not raw_cand:
        return {"status": "unique", "matches": []}

    passes = variant_check_passes(raw_cand)
    if not passes:
        return {"status": "unique", "matches": []}

    cid = col_id if col_id is not None else CATALOG_COL_ID
    rows = fetch_catalog_text_view_rows(view=view, col_text=col_text, col_id=cid)
    row_ids, texts = _parse_catalog_rows(rows)
    if not texts:
        return {"status": "unique", "matches": []}

    embed_texts = _normalize_variant_texts(texts)
    mat = embed_texts_local(embed_texts, model_id=EMBED_MODEL)
    mat = np.asarray(mat, dtype=np.float32)

    for pass_idx, (used_spell, prepared_cand) in enumerate(passes):
        pass_label = "spell" if used_spell else "no-spell"
        logger.info(
            "Catalog variant check view=%s pass=%s rows=%s threshold=%.3f candidate=%r prepared=%r",
            view,
            pass_label,
            len(texts),
            threshold,
            raw_cand,
            prepared_cand,
        )

        cand_vec = embed_texts_local([prepared_cand], model_id=EMBED_MODEL, batch_size=1)
        if cand_vec.ndim != 2 or cand_vec.shape[0] != 1:
            raise RuntimeError("Unexpected embedding output shape for catalog candidate")
        cand_vec = cand_vec[0]
        if int(mat.shape[1]) != int(cand_vec.shape[0]):
            raise RuntimeError("Catalog view embedding dimension does not match candidate embedding.")

        matches = _embedding_matches(
            mat,
            cand_vec,
            texts=texts,
            row_ids=row_ids,
            match_value_key=match_value_key,
            threshold=threshold,
        )
        if matches:
            logger.info(
                "Catalog variant check view=%s — duplicate | pass=%s matches=%s",
                view,
                pass_label,
                len(matches),
            )
            return {"status": "duplicate", "matches": matches}

        if fuzzy_fallback and used_spell:
            fuzzy_matches = find_fuzzy_catalog_matches(
                prepared_cand,
                embed_texts,
                raw_texts=texts,
                row_ids=row_ids,
                match_value_key=match_value_key,
            )
            if fuzzy_matches:
                logger.info(
                    "Catalog variant fuzzy fallback view=%s — duplicate | pass=%s matches=%s",
                    view,
                    pass_label,
                    len(fuzzy_matches),
                )
                return {"status": "duplicate", "matches": fuzzy_matches}

    logger.info("Catalog variant check view=%s — status=unique", view)
    return {"status": "unique", "matches": []}


def check_uom_variant(
    candidate: str,
    *,
    threshold: float,
    match_value_key: str = "UOMDescription",
    col_id: str | None = None,
) -> dict[str, Any]:
    """
    UOM duplicate check: synonym/canonical match first, then embedding similarity.

    View rows are loaded once. Each pass tries synonym match then embedding for
    the candidate (no spell first, spell fallback second).
    """
    raw_cand = clean_str(candidate)
    if not raw_cand:
        return {"status": "unique", "matches": []}

    passes = variant_check_passes(raw_cand)
    if not passes:
        return {"status": "unique", "matches": []}

    cid = col_id if col_id is not None else CATALOG_COL_ID
    rows = fetch_catalog_text_view_rows(view=UOM_VIEW, col_text=UOM_COL, col_id=cid)
    row_ids, texts = _parse_catalog_rows(rows)

    embed_texts = _normalize_variant_texts(texts)
    mat: np.ndarray | None = None
    if texts:
        mat = np.asarray(embed_texts_local(embed_texts, model_id=EMBED_MODEL), dtype=np.float32)

    for used_spell, prepared_cand in passes:
        pass_label = "spell" if used_spell else "no-spell"
        cand_canon = canonical_uom(prepared_cand)
        logger.info(
            "UOM variant check pass=%s rows=%s candidate=%r prepared=%r canonical=%r threshold=%.3f",
            pass_label,
            len(texts),
            raw_cand,
            prepared_cand,
            cand_canon,
            threshold,
        )

        if not cand_canon:
            continue

        synonym_matches: list[dict[str, Any]] = []
        for i, text in enumerate(texts):
            if not text:
                continue
            row_canon = canonical_uom(normalize_variant_text(text))
            if row_canon and row_canon == cand_canon:
                synonym_matches.append({
                    match_value_key: text,
                    "row": row_ids[i],
                })

        if synonym_matches:
            logger.info(
                "UOM variant: synonym duplicate | pass=%s canonical=%r matches=%s",
                pass_label,
                cand_canon,
                len(synonym_matches),
            )
            return {"status": "duplicate", "matches": synonym_matches}

        if not texts or mat is None:
            continue

        cand_vec = embed_texts_local([prepared_cand], model_id=EMBED_MODEL, batch_size=1)
        if cand_vec.ndim != 2 or cand_vec.shape[0] != 1:
            raise RuntimeError("Unexpected embedding output shape for UOM candidate")
        cand_vec = cand_vec[0]
        if int(mat.shape[1]) != int(cand_vec.shape[0]):
            raise RuntimeError("UOM embedding dimension does not match candidate embedding.")

        embedding_matches = _embedding_matches(
            mat,
            cand_vec,
            texts=texts,
            row_ids=row_ids,
            match_value_key=match_value_key,
            threshold=threshold,
        )
        if embedding_matches:
            logger.info(
                "UOM variant: embedding duplicate | pass=%s matches=%s",
                pass_label,
                len(embedding_matches),
            )
            return {"status": "duplicate", "matches": embedding_matches}

    logger.info("UOM variant: unique after all passes")
    return {"status": "unique", "matches": []}
