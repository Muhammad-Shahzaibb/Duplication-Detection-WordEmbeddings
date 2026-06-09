"""
Application configuration: paths and environment (including .env).
"""
from __future__ import annotations

import os
from pathlib import Path

APP_DIR: Path = Path(__file__).resolve().parent

# Embedding caches, metadata (.meta.json), and pre-embed minimized JSON.
CACHE_DIR: Path = APP_DIR / "cache"

EMBED_CACHE_FILE: Path = CACHE_DIR / "embeddings_cache.npy"

# Vendor Master name embedding cache.
EMBED_VENDOR_CACHE_FILE: Path = CACHE_DIR / "vendor_embeddings_cache.npy"

# Item Master row cache (text, numeric, display columns; index-aligned with embeddings_cache.npy).
# Written only on /Item-Master-update-embeddings (or scheduler); read by duplicate engine + variant check.
ITEM_MASTER_MINIMIZED_JSONL: Path = CACHE_DIR / "final_rows.jsonl"
ITEM_MASTER_MINIMIZED_JSON: Path = CACHE_DIR / "final_rows.json"
PG_HOST = os.environ.get("PGHOST", "163.61.91.149")
PG_PORT = int(os.environ.get("PGPORT", "30010"))
PG_DATABASE = os.environ.get("PGDATABASE", "Style")
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD", "postgres")
PG_SCHEMA = os.environ.get("PGSCHEMA", "public")
PG_CONNECT_TIMEOUT = int(os.environ.get("PG_CONNECT_TIMEOUT", "30"))

# Item Master view name (override via ITEM_MASTER_VIEW)
ITEM_MASTER_VIEW = os.environ.get("ITEM_MASTER_VIEW", "vw_item_master_view2")

# Vendor Master view name (override via VENDOR_MASTER_VIEW)
VENDOR_MASTER_VIEW = os.environ.get("VENDOR_MASTER_VIEW", "vw_vendor_master_view")

# Vendor Master approval view (override via VENDOR_MASTER_APPROVAL_VIEW)
VENDOR_MASTER_APPROVAL_VIEW = os.environ.get("VENDOR_MASTER_APPROVAL_VIEW", "vw_vendor_master_view_approval")

# ORDER BY for Vendor Master view (defaults to "id" for stable cache alignment)
VENDOR_MASTER_ORDER_BY = os.environ.get("VENDOR_MASTER_ORDER_BY", "").strip()

# VENDOR_NAME_TEXT_THRESHOLD — used by /Vendor-Master-duplicate-engine (cleansing engine).
# VENDOR_VARIANT_CHECK_NAME_THRESHOLD — used by /Vendor-Master-check-duplicate-Name only.
VENDOR_NAME_TEXT_THRESHOLD = float(os.environ.get("VENDOR_NAME_TEXT_THRESHOLD", "0.90"))
VENDOR_VARIANT_CHECK_NAME_THRESHOLD = float(
    os.environ.get("VENDOR_VARIANT_CHECK_NAME_THRESHOLD", "0.97")
)

# Approval Item Master view (override via ITEM_MASTER_APPROVAL_VIEW)
ITEM_MASTER_APPROVAL_VIEW = os.environ.get("ITEM_MASTER_APPROVAL_VIEW", "vw_item_master_items")

# ORDER BY clause (comma-separated, **without** leading "ORDER BY") for stable row order vs embedding cache.
# If empty, Db_View builds: ITEM_TYPE, MAINGROUP, SUBGROUP, ITEMDESC NULLS LAST.
# Override if your view has a stable id, e.g.: ITEM_MASTER_ORDER_BY='"ITEM_ID" NULLS LAST'
ITEM_MASTER_ORDER_BY = os.environ.get("ITEM_MASTER_ORDER_BY", "").strip()

# ── Cosine similarity thresholds ─────────────────────────────────────────────
# Embeddings are built on the TEXT part of ITEMDESC only.
# NUMERIC part must match exactly (case-insensitive, stripped).
#
# DUPLICATE_ENGINE_TEXT_THRESHOLD  — used by /Item-Master-duplicate-engine
#                                    and by the intra-bulk step in /Item-Master-check-duplicate-bulk.
# VARIANT_CHECK_TEXT_THRESHOLD     — used by /Item-Master-check-duplicate-variant
#                                    and by the DB/approval step in /Item-Master-check-duplicate-bulk.
# Approval queue embeddings are computed at runtime only (not cached on disk).
DUPLICATE_ENGINE_TEXT_THRESHOLD = float(os.environ.get("DUPLICATE_ENGINE_TEXT_THRESHOLD", "0.985"))
VARIANT_CHECK_TEXT_THRESHOLD = float(os.environ.get("VARIANT_CHECK_TEXT_THRESHOLD", "0.75"))

# Main code / sub code / UOM catalog views (runtime embeddings only; not cached).
ITEM_MAIN_CODE_VIEW = os.environ.get("ITEM_MAIN_CODE_VIEW", "vw_item_main_code")
ITEM_MAIN_CODE_COL = os.environ.get("ITEM_MAIN_CODE_COL", "ItemMainCode_Name")
ITEM_SUB_CODE_VIEW = os.environ.get("ITEM_SUB_CODE_VIEW", "vw_item_sub_code")
ITEM_SUB_CODE_COL = os.environ.get("ITEM_SUB_CODE_COL", "ItemSubCode_Name")
UOM_VIEW = os.environ.get("UOM_VIEW", "vw_uom")
UOM_COL = os.environ.get("UOM_COL", "UOM_Description")
CATALOG_COL_ID = os.environ.get("CATALOG_COL_ID", "id")
# Per-catalog cosine thresholds (runtime variant checks; not cached).
MAIN_CODE_VARIANT_CHECK_TEXT_THRESHOLD = float(
    os.environ.get("MAIN_CODE_VARIANT_CHECK_TEXT_THRESHOLD", "0.97")
)
SUB_CODE_VARIANT_CHECK_TEXT_THRESHOLD = float(
    os.environ.get("SUB_CODE_VARIANT_CHECK_TEXT_THRESHOLD", "0.97")
)
UOM_VARIANT_CHECK_TEXT_THRESHOLD = float(
    os.environ.get("UOM_VARIANT_CHECK_TEXT_THRESHOLD", "0.97")
)


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from .env in APP_DIR or cwd (does not override existing env)."""
    for env_path in (Path.cwd() / ".env", APP_DIR / ".env"):
        try:
            if not env_path.exists():
                continue
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if not k or k in os.environ:
                    continue
                v = v.strip()
                # Only unwrap when the *entire* value is quoted (preserve `"id" NULLS LAST`).
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                    v = v[1:-1]
                os.environ[k] = v
        except Exception:
            continue

    # Refresh derived settings after .env load
    global PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD, PG_SCHEMA, PG_CONNECT_TIMEOUT
    global ITEM_MASTER_VIEW, ITEM_MASTER_APPROVAL_VIEW, ITEM_MASTER_ORDER_BY
    global DUPLICATE_ENGINE_TEXT_THRESHOLD, VARIANT_CHECK_TEXT_THRESHOLD
    global VENDOR_MASTER_VIEW, VENDOR_MASTER_ORDER_BY, VENDOR_NAME_TEXT_THRESHOLD
    global VENDOR_VARIANT_CHECK_NAME_THRESHOLD, VENDOR_MASTER_APPROVAL_VIEW
    global ITEM_MAIN_CODE_VIEW, ITEM_MAIN_CODE_COL, ITEM_SUB_CODE_VIEW, ITEM_SUB_CODE_COL
    global UOM_VIEW, UOM_COL, CATALOG_COL_ID
    global MAIN_CODE_VARIANT_CHECK_TEXT_THRESHOLD, SUB_CODE_VARIANT_CHECK_TEXT_THRESHOLD
    global UOM_VARIANT_CHECK_TEXT_THRESHOLD
    PG_HOST = os.environ.get("PGHOST", PG_HOST)
    PG_PORT = int(os.environ.get("PGPORT", str(PG_PORT)))
    PG_DATABASE = os.environ.get("PGDATABASE", PG_DATABASE)
    PG_USER = os.environ.get("PGUSER", PG_USER)
    PG_PASSWORD = os.environ.get("PGPASSWORD", PG_PASSWORD)
    PG_SCHEMA = os.environ.get("PGSCHEMA", PG_SCHEMA)
    PG_CONNECT_TIMEOUT = int(os.environ.get("PG_CONNECT_TIMEOUT", str(PG_CONNECT_TIMEOUT)))
    ITEM_MASTER_VIEW = os.environ.get("ITEM_MASTER_VIEW", ITEM_MASTER_VIEW)
    ITEM_MASTER_APPROVAL_VIEW = os.environ.get("ITEM_MASTER_APPROVAL_VIEW", ITEM_MASTER_APPROVAL_VIEW)
    ITEM_MASTER_ORDER_BY = os.environ.get("ITEM_MASTER_ORDER_BY", ITEM_MASTER_ORDER_BY).strip()
    DUPLICATE_ENGINE_TEXT_THRESHOLD = float(
        os.environ.get("DUPLICATE_ENGINE_TEXT_THRESHOLD", str(DUPLICATE_ENGINE_TEXT_THRESHOLD))
    )
    VARIANT_CHECK_TEXT_THRESHOLD = float(
        os.environ.get("VARIANT_CHECK_TEXT_THRESHOLD", str(VARIANT_CHECK_TEXT_THRESHOLD))
    )
    VENDOR_MASTER_VIEW = os.environ.get("VENDOR_MASTER_VIEW", VENDOR_MASTER_VIEW)
    VENDOR_MASTER_APPROVAL_VIEW = os.environ.get("VENDOR_MASTER_APPROVAL_VIEW", VENDOR_MASTER_APPROVAL_VIEW)
    VENDOR_MASTER_ORDER_BY = os.environ.get("VENDOR_MASTER_ORDER_BY", VENDOR_MASTER_ORDER_BY).strip()
    VENDOR_NAME_TEXT_THRESHOLD = float(
        os.environ.get("VENDOR_NAME_TEXT_THRESHOLD", str(VENDOR_NAME_TEXT_THRESHOLD))
    )
    VENDOR_VARIANT_CHECK_NAME_THRESHOLD = float(
        os.environ.get(
            "VENDOR_VARIANT_CHECK_NAME_THRESHOLD",
            str(VENDOR_VARIANT_CHECK_NAME_THRESHOLD),
        )
    )
    ITEM_MAIN_CODE_VIEW = os.environ.get("ITEM_MAIN_CODE_VIEW", ITEM_MAIN_CODE_VIEW)
    ITEM_MAIN_CODE_COL = os.environ.get("ITEM_MAIN_CODE_COL", ITEM_MAIN_CODE_COL)
    ITEM_SUB_CODE_VIEW = os.environ.get("ITEM_SUB_CODE_VIEW", ITEM_SUB_CODE_VIEW)
    ITEM_SUB_CODE_COL = os.environ.get("ITEM_SUB_CODE_COL", ITEM_SUB_CODE_COL)
    UOM_VIEW = os.environ.get("UOM_VIEW", UOM_VIEW)
    UOM_COL = os.environ.get("UOM_COL", UOM_COL)
    CATALOG_COL_ID = os.environ.get("CATALOG_COL_ID", CATALOG_COL_ID)
    MAIN_CODE_VARIANT_CHECK_TEXT_THRESHOLD = float(
        os.environ.get(
            "MAIN_CODE_VARIANT_CHECK_TEXT_THRESHOLD",
            str(MAIN_CODE_VARIANT_CHECK_TEXT_THRESHOLD),
        )
    )
    SUB_CODE_VARIANT_CHECK_TEXT_THRESHOLD = float(
        os.environ.get(
            "SUB_CODE_VARIANT_CHECK_TEXT_THRESHOLD",
            str(SUB_CODE_VARIANT_CHECK_TEXT_THRESHOLD),
        )
    )
    UOM_VARIANT_CHECK_TEXT_THRESHOLD = float(
        os.environ.get(
            "UOM_VARIANT_CHECK_TEXT_THRESHOLD",
            str(UOM_VARIANT_CHECK_TEXT_THRESHOLD),
        )
    )


load_dotenv()
