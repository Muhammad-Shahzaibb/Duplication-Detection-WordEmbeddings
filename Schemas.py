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
