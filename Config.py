"""
Application configuration: paths and environment (including .env).
"""
from __future__ import annotations

import os
from pathlib import Path

APP_DIR: Path = Path(__file__).resolve().parent

# Default embedding cache lives next to the deployed app (first run creates; then reuse).
EMBED_CACHE_FILE: Path = APP_DIR / "embeddings_cache.npy"

# Approval queue (items not yet in main Item Master) — same 4 columns as main view.
EMBED_APPROVAL_CACHE_FILE: Path = APP_DIR / "Approval_embedding_cache.npy"

# Minimized JSON (text + numeric, same as embedding input) written before embedding.
ITEM_MASTER_MINIMIZED_JSONL: Path = APP_DIR / "final_rows.jsonl"
ITEM_MASTER_MINIMIZED_JSON: Path = APP_DIR / "final_rows.json"
ITEM_MASTER_APPROVAL_MINIMIZED_JSONL: Path = APP_DIR / "Approval_final_rows.jsonl"
ITEM_MASTER_APPROVAL_MINIMIZED_JSON: Path = APP_DIR / "Approval_final_rows.json"
PG_HOST = os.environ.get("PGHOST", "163.61.91.149")
PG_PORT = int(os.environ.get("PGPORT", "30010"))
PG_DATABASE = os.environ.get("PGDATABASE", "Style")
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD", "postgres")
PG_SCHEMA = os.environ.get("PGSCHEMA", "public")

# Item Master view name (override via ITEM_MASTER_VIEW)
ITEM_MASTER_VIEW = os.environ.get("ITEM_MASTER_VIEW", "vw_item_master_view2")

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
DUPLICATE_ENGINE_TEXT_THRESHOLD = float(os.environ.get("DUPLICATE_ENGINE_TEXT_THRESHOLD", "0.985"))
VARIANT_CHECK_TEXT_THRESHOLD = float(os.environ.get("VARIANT_CHECK_TEXT_THRESHOLD", "0.97"))


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
                v = v.strip().strip('"').strip("'")
                os.environ[k] = v
        except Exception:
            continue

    # Refresh derived settings after .env load
    global PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD, PG_SCHEMA
    global ITEM_MASTER_VIEW, ITEM_MASTER_APPROVAL_VIEW, ITEM_MASTER_ORDER_BY
    global DUPLICATE_ENGINE_TEXT_THRESHOLD, VARIANT_CHECK_TEXT_THRESHOLD
    PG_HOST = os.environ.get("PGHOST", PG_HOST)
    PG_PORT = int(os.environ.get("PGPORT", str(PG_PORT)))
    PG_DATABASE = os.environ.get("PGDATABASE", PG_DATABASE)
    PG_USER = os.environ.get("PGUSER", PG_USER)
    PG_PASSWORD = os.environ.get("PGPASSWORD", PG_PASSWORD)
    PG_SCHEMA = os.environ.get("PGSCHEMA", PG_SCHEMA)
    ITEM_MASTER_VIEW = os.environ.get("ITEM_MASTER_VIEW", ITEM_MASTER_VIEW)
    ITEM_MASTER_APPROVAL_VIEW = os.environ.get("ITEM_MASTER_APPROVAL_VIEW", ITEM_MASTER_APPROVAL_VIEW)
    ITEM_MASTER_ORDER_BY = os.environ.get("ITEM_MASTER_ORDER_BY", ITEM_MASTER_ORDER_BY).strip()
    DUPLICATE_ENGINE_TEXT_THRESHOLD = float(
        os.environ.get("DUPLICATE_ENGINE_TEXT_THRESHOLD", str(DUPLICATE_ENGINE_TEXT_THRESHOLD))
    )
    VARIANT_CHECK_TEXT_THRESHOLD = float(
        os.environ.get("VARIANT_CHECK_TEXT_THRESHOLD", str(VARIANT_CHECK_TEXT_THRESHOLD))
    )


load_dotenv()
