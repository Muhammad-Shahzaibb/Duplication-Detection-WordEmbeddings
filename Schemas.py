"""
Pydantic response models for the Item Master duplicate engine API.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ColumnMatchStatus = Literal["exact", "different"]


class DuplicateGroupColumnStatus(BaseModel):
    """Per-column variation within one duplicate cluster."""

    ITEM_TYPE: ColumnMatchStatus = Field(..., description="exact if all ITEM_TYPE values in the group match.")
    MAINGROUP: ColumnMatchStatus = Field(..., description="exact if all MAINGROUP values in the group match.")
    SUBGROUP: ColumnMatchStatus = Field(..., description="exact if all SUBGROUP values in the group match.")
    ITEMDESC: ColumnMatchStatus = Field(
        default="different",
        description="Always different (duplicates are detected on ITEMDESC embedding only).",
    )


class DuplicateGroupPayload(BaseModel):
    """One duplicate cluster: per-column status + member rows."""

    status: DuplicateGroupColumnStatus
    records: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rows in the group: row#, ITEM_TYPE, MAINGROUP, SUBGROUP, ITEMDESC, ITEM_CODE (main view only)",
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
        description="Keys DUP_1, DUP_2, …; each value has per-column status (exact/different) and records[]",
    )


class ItemMasterVariantDuplicateCheckRequest(BaseModel):
    ITEMDESC: str = Field(..., description="Candidate ITEMDESC (duplicate check uses description only)")


class VariantDuplicateMatch(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ITEMDESC: str
    location: Literal["db", "approval"] = Field(..., description="Main DB cache or approval queue cache.")
    row: int = Field(..., serialization_alias="row#", description="1-based row index in that view (db or approval).")


class ItemMasterVariantDuplicateCheckResponse(BaseModel):
    status: Literal["duplicate", "unique"] = Field(..., description="duplicate if any exact cosine==1 match exists.")
    matches: list[VariantDuplicateMatch] = Field(
        default_factory=list,
        description="Each exact match: ITEMDESC, location (db|approval), row# in that source.",
    )


# ─── Bulk duplicate check ──────────────────────────────────────────────────────


class ItemMasterBulkDuplicateCheckRequest(BaseModel):
    ITEMDESC: list[str] = Field(
        ...,
        min_length=1,
        description="List of candidate ITEMDESC values to check in bulk.",
    )


class IntraBulkDuplicateGroup(BaseModel):
    """A set of submitted descriptions that are exact duplicates of each other."""

    representative: str = Field(..., description="The description kept as the unique entry.")
    duplicates: list[str] = Field(
        ...,
        description="All submitted values that are exact duplicates of the representative (representative excluded).",
    )


class BulkItemResult(BaseModel):
    """Check result for one unique ITEMDESC from the bulk input."""

    ITEMDESC: str
    status: Literal["duplicate", "unique"]
    matches: list[VariantDuplicateMatch] = Field(
        default_factory=list,
        description="Exact matches found in DB and/or approval (empty when unique).",
    )


class ItemMasterBulkDuplicateCheckResponse(BaseModel):
    total_submitted: int = Field(..., description="Total ITEMDESC values received.")
    unique_count: int = Field(..., description="Distinct descriptions after intra-bulk deduplication.")
    intra_bulk_duplicate_groups: list[IntraBulkDuplicateGroup] = Field(
        default_factory=list,
        description="Groups of descriptions that were duplicates of each other within the submitted batch.",
    )
    results: list[BulkItemResult] = Field(
        default_factory=list,
        description="Per-unique-description match result against DB and approval caches.",
    )
