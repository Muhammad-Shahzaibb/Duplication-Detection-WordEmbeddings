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
    location: Literal["db", "approval"] = Field(..., description="Main DB (cached) or approval queue (live view).")
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
        description="Per-unique-description match result against DB cache and approval view.",
    )


# ─── Vendor Master ─────────────────────────────────────────────────────────────


class VendorDuplicateGroup(BaseModel):
    """One duplicate cluster for a vendor field."""

    records: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Each record: id, Name, and the field value being compared "
            "(field omitted for NAME groups)."
        ),
    )


class VendorFieldResult(BaseModel):
    """Duplicate detection result for one vendor field."""

    duplicate_groups: int = Field(..., description="Number of duplicate clusters found.")
    duplicate_records: int = Field(
        ...,
        description=(
            "Count of extra duplicate rows across all groups (per group: size - 1; "
            "one row per group is treated as the unique representative)."
        ),
    )
    groups: dict[str, VendorDuplicateGroup] = Field(
        default_factory=dict,
        description="Keys DUP_1, DUP_2, …; each value has a records[] list.",
    )


class VendorMasterDuplicateEngineResponse(BaseModel):
    """Full output of the Vendor Master duplicate engine."""

    total_records: int = Field(..., description="Total vendor rows fetched from the view.")
    duplicates_by_NAME: VendorFieldResult = Field(..., description="Duplicate groups detected by Name similarity.")
    duplicates_by_CNIC: VendorFieldResult = Field(..., description="Duplicate groups detected by CNIC (normalized).")
    duplicates_by_NTN: VendorFieldResult = Field(..., description="Duplicate groups detected by NTN (normalized).")
    duplicates_by_STRN: VendorFieldResult = Field(..., description="Duplicate groups detected by STRN (normalized).")
    duplicates_by_ACCOUNT_NO: VendorFieldResult = Field(..., description="Duplicate groups detected by Account No (normalized).")
    duplicates_by_IBAN: VendorFieldResult = Field(..., description="Duplicate groups detected by IBAN (normalized).")


class VendorMasterUpdateEmbeddingsResponse(BaseModel):
    """Result of recomputing Vendor Master name embeddings."""

    total_records: int = Field(..., description="Number of vendor rows embedded.")
    embedding_dim: int = Field(..., description="Vector dimension (0 if no rows).")
    cache_file: str = Field(default="", description="Absolute path to vendor_embeddings_cache.npy.")
    metadata_file: str = Field(default="", description="Absolute path to vendor_embeddings_cache.npy.meta.json.")
    model: str = Field(..., description="Sentence-transformers model used.")
    rows_in_metadata: int = Field(..., description="Row count recorded in saved metadata.")


# ─── Vendor Master variant check ───────────────────────────────────────────────


class VendorVariantMatch(BaseModel):
    """One matching vendor row returned by a variant check API."""

    model_config = ConfigDict(populate_by_name=True)

    id: Any = Field(..., description="Vendor row id from the view.")
    Name: str = Field(..., description="Vendor Name.")
    field_value: str = Field(
        default="",
        description="Matched field value (the Name for name-check; raw field value for numeric checks).",
    )
    location: Literal["db", "approval"] = Field(
        ..., description="Main DB view or approval view (live)."
    )
    row: int = Field(
        ..., serialization_alias="row#", description="1-based row index in that view."
    )


class VendorVariantDuplicateCheckResponse(BaseModel):
    """Output shared by all 6 vendor variant check APIs."""

    status: Literal["duplicate", "unique"] = Field(
        ..., description="duplicate if any match found; unique otherwise."
    )
    matches: list[VendorVariantMatch] = Field(
        default_factory=list,
        description="Matching vendor rows from main DB and/or approval view.",
    )


# ── Per-field request models ───────────────────────────────────────────────────

class VendorNameVariantCheckRequest(BaseModel):
    Name: str = Field(..., description="Candidate vendor Name to check for duplicates.")


class VendorCNICVariantCheckRequest(BaseModel):
    CNIC: str = Field(..., description="Candidate CNIC value to check for duplicates.")


class VendorNTNVariantCheckRequest(BaseModel):
    NTN: str = Field(..., description="Candidate NTN value to check for duplicates.")


class VendorSTRNVariantCheckRequest(BaseModel):
    STRN: str = Field(..., description="Candidate STRN (Sales Tax No) value to check for duplicates.")


class VendorAccountNoVariantCheckRequest(BaseModel):
    AccountNo: str = Field(..., description="Candidate Account No value to check for duplicates.")


class VendorIBANVariantCheckRequest(BaseModel):
    IBAN: str = Field(..., description="Candidate IBAN value to check for duplicates.")
