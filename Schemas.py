"""
Pydantic response models for the Item Master duplicate engine API.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DuplicateGroupPayload(BaseModel):
    """One duplicate cluster: fixed discriminator column + member rows."""

    status: Literal["ITEMDESC"] = Field(
        default="ITEMDESC",
        description="Column used to assess variation within the group (fixed: ITEMDESC).",
    )
    records: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rows in the group: row#, ITEM_TYPE, MAINGROUP, SUBGROUP, ITEMDESC",
    )


class ItemMasterUpdateEmbeddingsResponse(BaseModel):
    """Result of recomputing Item Master embeddings from the DB view."""

    total_records: int = Field(..., description="Number of rows embedded after regex minimization.")
    embedding_dim: int = Field(..., description="Vector dimension (0 if no rows).")
    cache_file: str = Field(default="", description="Absolute path to embeddings_cache.npy.")
    metadata_file: str = Field(default="", description="Absolute path to embeddings_cache.npy.meta.json.")
    model: str = Field(..., description="Sentence-transformers model id used for encoding.")
    text_digest: str = Field(
        default="",
        description="SHA-256 digest of per-row embedding texts (matches cache metadata).",
    )
    rows_in_metadata: int = Field(
        ...,
        description="Row count recorded in metadata (same as total_records when cache saved).",
    )


class ItemMasterDuplicateEngineResponse(BaseModel):
    total_records: int
    valid_records: int
    duplicate_records: int
    Data_quality_score: float = Field(
        ...,
        description="100 × valid_records / total_records (0–100). Higher means fewer duplicate rows vs total.",
    )
    duplicates: dict[str, DuplicateGroupPayload] = Field(
        default_factory=dict,
        description="Keys DUP_1, DUP_2, …; each value has status=ITEMDESC and records[]",
    )


class ItemMasterVariantDuplicateCheckRequest(BaseModel):
    ITEM_TYPE: str = Field(..., description="Candidate ITEM_TYPE")
    MAINGROUP: str = Field(..., description="Candidate MAINGROUP")
    SUBGROUP: str = Field(..., description="Candidate SUBGROUP")
    ITEMDESC: str = Field(..., description="Candidate ITEMDESC")


class ItemMasterVariantDuplicateCheckResponse(BaseModel):
    status: Literal["duplicate", "unique"] = Field(..., description="duplicate if any exact cosine==1 match exists.")
    location: Literal["", "db", "approval", "both"] = Field(
        default="",
        description='Where duplicate(s) were found: "db" (main embedding cache), "approval", "both", or "" when unique.',
    )
    ITEMDESC: list[str] = Field(
        default_factory=list,
        description="Original ITEMDESC values for all exact matches in main and/or approval (empty when unique).",
    )
