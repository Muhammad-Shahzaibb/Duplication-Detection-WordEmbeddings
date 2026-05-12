"""
FastAPI application: Item Master duplicate engine (single endpoint).
"""
from __future__ import annotations

from fastapi import FastAPI

from Db_View import fetch_item_master_rows_from_view
from Item_Master_Duplicate_Engine import row_to_schema_json, run_item_master_duplicate_engine
from Schemas import ItemMasterDuplicateEngineResponse

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
