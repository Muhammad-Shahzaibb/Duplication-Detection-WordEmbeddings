"""
Token-aligned fuzzy matching for Main Code / Sub Code catalog variant checks.

Used only when spell + normalize + embedding similarity finds no duplicate.
Allows up to 2 character edits per token (1 for very short tokens), with the same
token count required for multi-word names so unrelated words do not align.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz.distance import Levenshtein

from logging_setup import get_logger

logger = get_logger("style_textile.catalog_fuzzy")

# Max Levenshtein edits per token (long tokens).
_MAX_TOKEN_EDITS = 2
# Shorter tokens are easier to false-match (e.g. "AB" vs "CD"), so allow only 1 edit.
_MAX_TOKEN_EDITS_SHORT = 1
_SHORT_TOKEN_MAX_LEN = 3


def _max_edits_for_token(token: str) -> int:
    return _MAX_TOKEN_EDITS_SHORT if len(token) <= _SHORT_TOKEN_MAX_LEN else _MAX_TOKEN_EDITS


def _within_edit_budget(left: str, right: str, max_edits: int) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > max_edits:
        return False
    return Levenshtein.distance(left, right) <= max_edits


def fuzzy_catalog_text_match(candidate: str, row_text: str) -> bool:
    """
    Return True when ``row_text`` is a fuzzy duplicate of ``candidate``.

    Rules:
    - Compare case-insensitively on preprocessed strings.
    - Single-token names: Levenshtein distance <= 2 (<= 1 if token length <= 3).
    - Multi-token names: same number of tokens; each aligned token is exact or
      within the per-token edit budget. Different word counts are never duplicates.
    """
    left = (candidate or "").casefold().strip()
    right = (row_text or "").casefold().strip()
    if not left or not right:
        return False
    if left == right:
        return True

    left_tokens = left.split()
    right_tokens = right.split()
    if len(left_tokens) != len(right_tokens):
        return False

    if len(left_tokens) == 1:
        budget = _max_edits_for_token(left_tokens[0])
        return _within_edit_budget(left_tokens[0], right_tokens[0], budget)

    for lt, rt in zip(left_tokens, right_tokens):
        if lt == rt:
            continue
        budget = min(_max_edits_for_token(lt), _max_edits_for_token(rt))
        if not _within_edit_budget(lt, rt, budget):
            return False
    return True


def find_fuzzy_catalog_matches(
    candidate: str,
    prepared_texts: list[str],
    *,
    raw_texts: list[str],
    row_ids: list[int],
    match_value_key: str,
) -> list[dict[str, Any]]:
    """Scan catalog rows with token-aligned fuzzy rules; return match payloads."""
    matches: list[dict[str, Any]] = []
    for i, prepared in enumerate(prepared_texts):
        if not prepared:
            continue
        if fuzzy_catalog_text_match(candidate, prepared):
            matches.append({
                match_value_key: raw_texts[i],
                "row": row_ids[i],
            })
    return matches
