"""
Item Master data pull from Postgres view only (no duplicate logic).
"""
from __future__ import annotations

from typing import Any

from Config import (
    ITEM_MASTER_APPROVAL_VIEW,
    ITEM_MASTER_ORDER_BY,
    ITEM_MASTER_VIEW,
    PG_DATABASE,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_USER,
)


def _sql_quote_ident(name: str) -> str:
    """Double-quote a PostgreSQL identifier (handles mixed case)."""
    n = name.strip()
    if n.startswith('"') and n.endswith('"'):
        return n
    return '"' + n.replace('"', '""') + '"'


def _view_order_by_clause(
    *,
    col_item_type: str,
    col_main_group: str,
    col_sub_group: str,
    col_item_description: str,
) -> str:
    """Stable ORDER BY for cache row alignment (override via Config.ITEM_MASTER_ORDER_BY)."""
    if ITEM_MASTER_ORDER_BY:
        return ITEM_MASTER_ORDER_BY
    return (
        f"{_sql_quote_ident(col_item_type)} NULLS LAST, "
        f"{_sql_quote_ident(col_main_group)} NULLS LAST, "
        f"{_sql_quote_ident(col_sub_group)} NULLS LAST, "
        f"{_sql_quote_ident(col_item_description)} NULLS LAST"
    )


def fetch_item_master_rows_from_view(
    *,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
    schema: str | None = None,
    view: str | None = None,
    rows_limit: int | None = None,
    col_item_type: str = "ITEM_TYPE",
    col_main_group: str = "MAINGROUP",
    col_sub_group: str = "SUBGROUP",
    col_item_description: str = "ITEMDESC",
    col_item_code: str = "ITEM_CODE",
    include_item_code: bool = False,
) -> list[tuple[Any, ...]]:
    """
    Return rows as (ITEM_TYPE, MAINGROUP, SUBGROUP, ITEMDESC) tuples in **deterministic** order
    (ORDER BY on the four columns, or ``ITEM_MASTER_ORDER_BY`` from env / Config).

    When ``include_item_code=True`` (main view / duplicate-engine only), each tuple also includes
    ITEM_CODE as the fifth element.
    """
    h = host or PG_HOST
    p = int(port or PG_PORT)
    db = dbname or PG_DATABASE
    u = user or PG_USER
    pw = password or PG_PASSWORD
    sch = schema or PG_SCHEMA
    v = view or ITEM_MASTER_VIEW

    try:
        import psycopg  # type: ignore

        conn = psycopg.connect(
            host=h,
            port=p,
            dbname=db,
            user=u,
            password=pw,
            connect_timeout=15,
        )
    except Exception:
        try:
            import psycopg2  # type: ignore

            conn = psycopg2.connect(
                host=h,
                port=p,
                dbname=db,
                user=u,
                password=pw,
                connect_timeout=15,
            )
        except Exception as e:
            raise RuntimeError(
                "Could not connect to Postgres. Install: pip install psycopg[binary] or psycopg2-binary"
            ) from e

    ident = f'"{sch}"."{v}"'
    order_by = _view_order_by_clause(
        col_item_type=col_item_type,
        col_main_group=col_main_group,
        col_sub_group=col_sub_group,
        col_item_description=col_item_description,
    )
    limit_sql = " LIMIT %s" if rows_limit is not None else ""
    if include_item_code:
        select_cols = (
            f'"{col_item_type}", "{col_main_group}", "{col_sub_group}", '
            f'"{col_item_description}", "{col_item_code}"'
        )
        expected_cols = 5
    else:
        select_cols = (
            f'"{col_item_type}", "{col_main_group}", "{col_sub_group}", "{col_item_description}"'
        )
        expected_cols = 4
    sql = f"SELECT {select_cols} FROM {ident} ORDER BY {order_by}{limit_sql}"

    out: list[tuple[Any, ...]] = []
    try:
        cur = conn.cursor()
        try:
            if rows_limit is not None:
                cur.execute(sql, (int(rows_limit),))
            else:
                cur.execute(sql)
            fetch_size = 5000
            while True:
                batch = cur.fetchmany(fetch_size)
                if not batch:
                    break
                for row in batch:
                    if len(row) != expected_cols:
                        raise ValueError(f"Expected {expected_cols} columns, got {len(row)}")
                    out.append(tuple(row))
        finally:
            try:
                cur.close()
            except Exception:
                pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return out


def fetch_item_master_rows_from_approval_view(
    **kwargs: Any,
) -> list[tuple[Any, Any, Any, Any]]:
    """Same columns/order as main Item Master view, from the approval queue view."""
    return fetch_item_master_rows_from_view(**kwargs, view=ITEM_MASTER_APPROVAL_VIEW)
