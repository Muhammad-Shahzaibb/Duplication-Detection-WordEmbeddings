"""
Application configuration: paths and environment (including .env).
"""
from __future__ import annotations

import os
from pathlib import Path

APP_DIR: Path = Path(__file__).resolve().parent

# Default embedding cache lives next to the deployed app (first run creates; then reuse).
EMBED_CACHE_FILE: Path = APP_DIR / "embeddings_cache.npy"

# Postgres defaults (override via .env or environment)
PG_HOST = os.environ.get("PGHOST", "163.61.91.149")
PG_PORT = int(os.environ.get("PGPORT", "30010"))
PG_DATABASE = os.environ.get("PGDATABASE", "Style")
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD", "postgres")
PG_SCHEMA = os.environ.get("PGSCHEMA", "public")

# Item Master view name (override via ITEM_MASTER_VIEW)
ITEM_MASTER_VIEW = os.environ.get("ITEM_MASTER_VIEW", "vw_item_master_view2")


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
    global PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD, PG_SCHEMA, ITEM_MASTER_VIEW
    PG_HOST = os.environ.get("PGHOST", PG_HOST)
    PG_PORT = int(os.environ.get("PGPORT", str(PG_PORT)))
    PG_DATABASE = os.environ.get("PGDATABASE", PG_DATABASE)
    PG_USER = os.environ.get("PGUSER", PG_USER)
    PG_PASSWORD = os.environ.get("PGPASSWORD", PG_PASSWORD)
    PG_SCHEMA = os.environ.get("PGSCHEMA", PG_SCHEMA)
    ITEM_MASTER_VIEW = os.environ.get("ITEM_MASTER_VIEW", ITEM_MASTER_VIEW)


load_dotenv()
