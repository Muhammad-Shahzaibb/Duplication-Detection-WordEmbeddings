"""
Background scheduler: refresh Vendor then Item Master embedding caches on a wall-clock interval.

Runs every N hours on the clock (e.g. interval=2 → 00:00, 02:00, 04:00, …, 14:00, 16:00).
Does not run immediately on server startup unless EMBEDDING_SCHEDULER_RUN_ON_STARTUP=true.

Each cycle:
  1. POST /Vendor-Master-update-embeddings equivalent
  2. Wait EMBEDDING_SCHEDULER_GAP_MINUTES
  3. POST /Item-Master-update-embeddings equivalent

Started automatically when the FastAPI app boots (see api.py lifespan).
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta
from typing import Any

from Db_View import fetch_item_master_rows_from_view, fetch_vendor_master_rows_from_view
from Item_Master_Duplicate_Engine import rebuild_item_master_embeddings_cache, row_to_schema_json
from Vendor_Master_Duplicate_Engine import rebuild_vendor_embeddings_cache
from logging_setup import get_logger

logger = get_logger("style_textile.scheduler")

INTERVAL_HOURS = max(1, int(float(os.environ.get("EMBEDDING_SCHEDULER_INTERVAL_HOURS", "4"))))
INTERVAL_SECONDS = INTERVAL_HOURS * 3600
GAP_SECONDS = int(float(os.environ.get("EMBEDDING_SCHEDULER_GAP_MINUTES", "30")) * 60)
SCHEDULER_ENABLED = os.environ.get("EMBEDDING_SCHEDULER_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
RUN_ON_STARTUP = os.environ.get("EMBEDDING_SCHEDULER_RUN_ON_STARTUP", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)

_stop_event = threading.Event()
_thread: threading.Thread | None = None


def _format_run_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _next_aligned_run(after: datetime | None = None) -> datetime:
    """Next wall-clock slot every INTERVAL_HOURS (e.g. 12:00, 14:00, 16:00 when interval=2)."""
    now = after if after is not None else datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    step = timedelta(hours=INTERVAL_HOURS)
    slot_index = int((now - midnight).total_seconds() // step.total_seconds())
    candidate = midnight + step * slot_index
    if candidate <= now:
        candidate += step
    return candidate


def _wait_until(target: datetime) -> bool:
    """Block until target time. Returns True if the scheduler was stopped."""
    while not _stop_event.is_set():
        remaining = (target - datetime.now()).total_seconds()
        if remaining <= 0:
            return False
        if _stop_event.wait(remaining):
            return True
    return True


def _log_next_run(target: datetime) -> None:
    logger.info("Scheduler: next run at %s", _format_run_time(target))


def _run_vendor_master_update_embeddings() -> dict[str, Any]:
    """Same logic as POST /Vendor-Master-update-embeddings."""
    logger.info("Scheduler: Vendor-Master-update-embeddings — start")
    rows = fetch_vendor_master_rows_from_view()
    logger.info("Scheduler: vendor view rows fetched: %s", len(rows))
    payload = rebuild_vendor_embeddings_cache(rows)
    logger.info(
        "Scheduler: Vendor-Master-update-embeddings — done | rows=%s dim=%s cache=%s",
        payload.get("total_records"),
        payload.get("embedding_dim"),
        payload.get("cache_file"),
    )
    return payload


def _run_item_master_update_embeddings() -> dict[str, Any]:
    """Same logic as POST /Item-Master-update-embeddings."""
    logger.info("Scheduler: Item-Master-update-embeddings — start")
    tuples = fetch_item_master_rows_from_view(include_item_code=True)
    logger.info("Scheduler: item view rows fetched: %s", len(tuples))
    records = [
        row_to_schema_json(
            item_description=desc,
            item_type=it,
            main_group=mg,
            sub_group=sg,
            item_code=code,
            uom=uom,
            doc_no=doc_no,
        )
        for it, mg, sg, desc, code, uom, doc_no in tuples
    ]
    payload = rebuild_item_master_embeddings_cache(records)
    logger.info(
        "Scheduler: Item-Master-update-embeddings — done | rows=%s dim=%s cache=%s",
        payload.get("total_records"),
        payload.get("embedding_dim"),
        payload.get("cache_file"),
    )
    return payload


def _run_embedding_update_cycle() -> None:
    """Vendor first, gap, then Item (sequential, never parallel)."""
    logger.info(
        "Scheduler: embedding update cycle — start (vendor → %s min gap → item)",
        GAP_SECONDS // 60,
    )
    try:
        _run_vendor_master_update_embeddings()
    except Exception:
        logger.exception("Scheduler: vendor embedding update failed")

    item_at = datetime.now() + timedelta(seconds=GAP_SECONDS)
    logger.info(
        "Scheduler: vendor update finished — item master will start at %s (%s min gap)",
        _format_run_time(item_at),
        GAP_SECONDS // 60,
    )
    if _stop_event.wait(GAP_SECONDS):
        logger.info("Scheduler: cycle interrupted during vendor/item gap")
        return

    try:
        _run_item_master_update_embeddings()
    except Exception:
        logger.exception("Scheduler: item embedding update failed")
        return

    logger.info("Scheduler: embedding update cycle — complete")


def _scheduler_loop() -> None:
    cycles_per_day = 24 // INTERVAL_HOURS
    logger.info(
        "Scheduler: started | every %sh on the clock | gap=%smin | run_on_startup=%s | enabled=%s (%s cycles/24h)",
        INTERVAL_HOURS,
        GAP_SECONDS // 60,
        RUN_ON_STARTUP,
        SCHEDULER_ENABLED,
        cycles_per_day,
    )

    if RUN_ON_STARTUP:
        logger.info("Scheduler: running immediately (EMBEDDING_SCHEDULER_RUN_ON_STARTUP=true)")
        _run_embedding_update_cycle()
        if _stop_event.is_set():
            return

    while not _stop_event.is_set():
        next_run = _next_aligned_run()
        if not RUN_ON_STARTUP:
            logger.info("Scheduler: not running on startup — waiting for next scheduled slot")
        _log_next_run(next_run)
        if _wait_until(next_run):
            break
        _run_embedding_update_cycle()

    logger.info("Scheduler: stopped")


def start_embedding_scheduler() -> None:
    """Start the background embedding refresh thread (no-op if disabled or already running)."""
    global _thread
    if not SCHEDULER_ENABLED:
        logger.info("Scheduler: disabled (EMBEDDING_SCHEDULER_ENABLED=false)")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_scheduler_loop, name="embedding-scheduler", daemon=True)
    _thread.start()


def stop_embedding_scheduler() -> None:
    """Signal the scheduler thread to stop and wait briefly for it to exit."""
    _stop_event.set()
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=5.0)
