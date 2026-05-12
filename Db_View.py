"""
Item Master data pull from Postgres view only (no duplicate logic).
"""
from __future__ import annotations

from typing import Any

from Config import (
    ITEM_MASTER_VIEW,
    PG_DATABASE,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_USER,
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
) -> list[tuple[Any, Any, Any, Any]]:
    """
    Return rows as (ITEM_TYPE, MAINGROUP, SUBGROUP, ITEMDESC) tuples in view order.
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
    limit_sql = " LIMIT %s" if rows_limit is not None else ""
    sql = (
        f'SELECT "{col_item_type}", "{col_main_group}", "{col_sub_group}", "{col_item_description}" '
        f"FROM {ident}{limit_sql}"
    )

    out: list[tuple[Any, Any, Any, Any]] = []
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
                    if len(row) != 4:
                        raise ValueError(f"Expected 4 columns, got {len(row)}")
                    out.append((row[0], row[1], row[2], row[3]))
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
