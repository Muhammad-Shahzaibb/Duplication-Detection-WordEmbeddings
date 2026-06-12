"""
SymSpell-based spelling correction for Item Master variant check (candidate only).

Applied to text-like tokens before normalization/embedding — never alters numeric tokens.
"""
from __future__ import annotations

import importlib.resources
import re
from typing import TYPE_CHECKING

from jsonify import COLORS, DIM_LABELS, MATERIALS, clean_str
from logging_setup import get_logger

if TYPE_CHECKING:
    from symspellpy import SymSpell

logger = get_logger("style_textile.spell")

_DOMAIN_TOKENS: frozenset[str] = frozenset(MATERIALS | COLORS | DIM_LABELS)

_uom_skip_tokens: frozenset[str] | None = None

_sym_spell: SymSpell | None = None


def _token_lookup_key(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.casefold())


def _get_uom_skip_tokens() -> frozenset[str]:
    """Known UOM labels/abbreviations — never SymSpell-correct these tokens."""
    global _uom_skip_tokens
    if _uom_skip_tokens is None:
        from uom_synonyms import ALIAS_TO_CANONICAL, UOM_CANONICAL_GROUPS

        keys: set[str] = set(ALIAS_TO_CANONICAL.keys())
        for canonical, labels in UOM_CANONICAL_GROUPS.items():
            keys.add(_token_lookup_key(canonical))
            for label in labels:
                keys.add(_token_lookup_key(label))
                for part in label.split():
                    keys.add(_token_lookup_key(part))
        _uom_skip_tokens = frozenset(keys)
    return _uom_skip_tokens


def _get_sym_spell() -> SymSpell:
    global _sym_spell
    if _sym_spell is None:
        from symspellpy import SymSpell

        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = importlib.resources.files("symspellpy").joinpath(
            "frequency_dictionary_en_82_765.txt"
        )
        if not dict_path.is_file():
            raise FileNotFoundError(
                "SymSpell frequency dictionary not found in symspellpy package. "
                "Run: pip install symspellpy"
            )
        sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
        _sym_spell = sym
        logger.info("SymSpell dictionary loaded for variant-check spelling correction")
    return _sym_spell


def _lookup_token(tok: str) -> str:
    """Uppercase token stripped of punctuation for dictionary / domain lookup."""
    return re.sub(r"[().#,:;]", "", tok).upper()


def correct_itemdesc_spelling(desc: str) -> str:
    """
    Spell-correct candidate ITEMDESC text tokens (variant check only).

    - Skips tokens containing a digit (numeric codes, dimensions).
    - Skips known domain tokens (materials, colors, dimension labels).
    - Leaves unknown / already-correct tokens unchanged.
  """
    from symspellpy import Verbosity

    s = clean_str(desc)
    if not s:
        return s

    sym = _get_sym_spell()
    corrected: list[str] = []
    for tok in s.split():
        t = tok.rstrip(",;")
        if not t:
            continue
        if re.search(r"\d", t):
            corrected.append(t)
            continue

        lookup = _lookup_token(t)
        if not lookup or lookup in _DOMAIN_TOKENS:
            corrected.append(t)
            continue
        if _token_lookup_key(t) in _get_uom_skip_tokens():
            corrected.append(t)
            continue
        if len(lookup) <= 2:
            corrected.append(t)
            continue

        suggestions = sym.lookup(lookup.lower(), Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions and suggestions[0].distance > 0:
            corrected.append(suggestions[0].term.upper())
        else:
            corrected.append(t)

    return " ".join(corrected)


def preprocess_variant_text(text: str) -> str:
    """
    Variant-check pipeline: spell-correct then normalize (spaces, punctuation, hyphens).

    Used for Item Master ITEMDESC and catalog variant APIs (main code, sub code, UOM).
    """
    from jsonify import normalize_item_description

    s = clean_str(text)
    if not s:
        return s
    s = correct_itemdesc_spelling(s)
    return normalize_item_description(s)
