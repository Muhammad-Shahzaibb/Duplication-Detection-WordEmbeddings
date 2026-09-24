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

# Vendor Master name / address embedding caches + row snapshot (index-aligned).
EMBED_VENDOR_CACHE_FILE: Path = CACHE_DIR / "vendor_embeddings_cache.npy"
EMBED_VENDOR_ADDRESS_CACHE_FILE: Path = CACHE_DIR / "vendor_address_embeddings_cache.npy"
VENDOR_MASTER_ROWS_JSONL: Path = CACHE_DIR / "vendor_final_rows.jsonl"
VENDOR_MASTER_ROWS_JSON: Path = CACHE_DIR / "vendor_final_rows.json"

# Item Master row cache (text, numeric, display columns; index-aligned with embeddings_cache.npy).
# Published atomically at the end of /Item-Master-update-embeddings (staging *.staging files during build).
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

# Approval Item Master view (override via ITEM_MASTER_APPROVAL_VIEW)
ITEM_MASTER_APPROVAL_VIEW = os.environ.get("ITEM_MASTER_APPROVAL_VIEW", "vw_item_master_items")

# ORDER BY clause (comma-separated, **without** leading "ORDER BY") for stable row order vs embedding cache.
# If empty, Db_View builds: ITEM_TYPE, MAINGROUP, SUBGROUP, ITEMDESC NULLS LAST.
# Override if your view has a stable id, e.g.: ITEM_MASTER_ORDER_BY='"ITEM_ID" NULLS LAST'
ITEM_MASTER_ORDER_BY = os.environ.get("ITEM_MASTER_ORDER_BY", "").strip()

# Main code / sub code / UOM catalog views (runtime embeddings only; not cached).
ITEM_MAIN_CODE_VIEW = os.environ.get("ITEM_MAIN_CODE_VIEW", "vw_item_main_code")
ITEM_MAIN_CODE_COL = os.environ.get("ITEM_MAIN_CODE_COL", "ItemMainCode_Name")
ITEM_SUB_CODE_VIEW = os.environ.get("ITEM_SUB_CODE_VIEW", "vw_item_sub_code")
ITEM_SUB_CODE_COL = os.environ.get("ITEM_SUB_CODE_COL", "ItemSubCode_Name")
UOM_VIEW = os.environ.get("UOM_VIEW", "vw_uom")
UOM_COL = os.environ.get("UOM_COL", "UOM_Description")
CATALOG_COL_ID = os.environ.get("CATALOG_COL_ID", "id")

# Hugging Face model id (stored in embedding cache metadata; used when no local path).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L12-v2")
# Optional: folder with modules.json (Docker/K8s offline). Empty = download from Hub on first use.
EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "").strip()


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
    global VENDOR_MASTER_VIEW, VENDOR_MASTER_ORDER_BY, VENDOR_MASTER_APPROVAL_VIEW
    global ITEM_MAIN_CODE_VIEW, ITEM_MAIN_CODE_COL, ITEM_SUB_CODE_VIEW, ITEM_SUB_CODE_COL
    global UOM_VIEW, UOM_COL, CATALOG_COL_ID, EMBED_MODEL, EMBED_MODEL_PATH
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
    VENDOR_MASTER_VIEW = os.environ.get("VENDOR_MASTER_VIEW", VENDOR_MASTER_VIEW)
    VENDOR_MASTER_APPROVAL_VIEW = os.environ.get("VENDOR_MASTER_APPROVAL_VIEW", VENDOR_MASTER_APPROVAL_VIEW)
    VENDOR_MASTER_ORDER_BY = os.environ.get("VENDOR_MASTER_ORDER_BY", VENDOR_MASTER_ORDER_BY).strip()
    ITEM_MAIN_CODE_VIEW = os.environ.get("ITEM_MAIN_CODE_VIEW", ITEM_MAIN_CODE_VIEW)
    ITEM_MAIN_CODE_COL = os.environ.get("ITEM_MAIN_CODE_COL", ITEM_MAIN_CODE_COL)
    ITEM_SUB_CODE_VIEW = os.environ.get("ITEM_SUB_CODE_VIEW", ITEM_SUB_CODE_VIEW)
    ITEM_SUB_CODE_COL = os.environ.get("ITEM_SUB_CODE_COL", ITEM_SUB_CODE_COL)
    UOM_VIEW = os.environ.get("UOM_VIEW", UOM_VIEW)
    UOM_COL = os.environ.get("UOM_COL", UOM_COL)
    CATALOG_COL_ID = os.environ.get("CATALOG_COL_ID", CATALOG_COL_ID)
    EMBED_MODEL = os.environ.get("EMBED_MODEL", EMBED_MODEL)
    EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", EMBED_MODEL_PATH).strip()


load_dotenv()
