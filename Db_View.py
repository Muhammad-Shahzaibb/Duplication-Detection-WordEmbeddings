"""
Item Master and Vendor Master data pull from Postgres views (no duplicate logic).
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
    VENDOR_MASTER_APPROVAL_VIEW,
    VENDOR_MASTER_ORDER_BY,
    VENDOR_MASTER_VIEW,
)


def _sql_quote_ident(name: str) -> str:
    """Double-quote a PostgreSQL identifier (handles mixed case)."""
    n = name.strip()
    if n.startswith('"') and n.endswith('"'):
        return n
    return '"' + n.replace('"', '""') + '"'


def _view_order_by_clause(
    *,
    conn: Any,
    schema: str,
    view: str,
    col_id: str,
    col_item_type: str,
    col_main_group: str,
    col_sub_group: str,
    col_item_description: str,
) -> str:
    """Stable ORDER BY for cache row alignment (override via Config.ITEM_MASTER_ORDER_BY)."""
    if ITEM_MASTER_ORDER_BY:
        return ITEM_MASTER_ORDER_BY
    # Prefer stable primary key ordering when present (user requirement).
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s AND lower(column_name) = lower(%s)
                LIMIT 1
                """,
                (schema, view, col_id),
            )
            if cur.fetchone() is not None:
                return f"{_sql_quote_ident(col_id)} NULLS LAST"
        finally:
            try:
                cur.close()
            except Exception:
                pass
    except Exception:
        pass
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
    col_id: str = "id",
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
        conn=conn,
        schema=sch,
        view=v,
        col_id=col_id,
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


def fetch_vendor_master_rows_from_approval_view(
    **kwargs: Any,
) -> list[tuple[Any, ...]]:
    """Same columns/order as main Vendor Master view, from the approval view."""
    return fetch_vendor_master_rows_from_view(**kwargs, view=VENDOR_MASTER_APPROVAL_VIEW)


def _pg_connect(
    host: str, port: int, dbname: str, user: str, password: str
) -> Any:
    """Try psycopg3 then psycopg2."""
    try:
        import psycopg  # type: ignore
        return psycopg.connect(
            host=host, port=port, dbname=dbname, user=user, password=password, connect_timeout=15
        )
    except Exception:
        try:
            import psycopg2  # type: ignore
            return psycopg2.connect(
                host=host, port=port, dbname=dbname, user=user, password=password, connect_timeout=15
            )
        except Exception as e:
            raise RuntimeError(
                "Could not connect to Postgres. Install: pip install psycopg[binary] or psycopg2-binary"
            ) from e


def fetch_vendor_master_rows_from_view(
    *,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
    schema: str | None = None,
    view: str | None = None,
    rows_limit: int | None = None,
    col_id: str = "id",
    col_name: str = "Name",
    col_cnic: str = "CNIC",
    col_ntn: str = "NTN",
    col_strn: str = "SalesTaxNo",
    col_account_no: str = "AccountNo",
    col_iban: str = "IbanNO",
) -> list[tuple[Any, ...]]:
    """
    Return vendor rows as (id, Name, CNIC, NTN, SalesTaxNo, AccountNo, IbanNO) tuples
    in **deterministic** order (``VENDOR_MASTER_ORDER_BY`` from env/Config, or ``id`` by default).

    Tuple indices: 0=id, 1=Name, 2=CNIC, 3=NTN, 4=SalesTaxNo, 5=AccountNo, 6=IbanNO
    """
    h = host or PG_HOST
    p = int(port or PG_PORT)
    db = dbname or PG_DATABASE
    u = user or PG_USER
    pw = password or PG_PASSWORD
    sch = schema or PG_SCHEMA
    v = view or VENDOR_MASTER_VIEW

    conn = _pg_connect(h, p, db, u, pw)

    ident = f'"{sch}"."{v}"'
    order_by = VENDOR_MASTER_ORDER_BY or _sql_quote_ident(col_id)
    limit_sql = " LIMIT %s" if rows_limit is not None else ""
    select_cols = (
        f'{_sql_quote_ident(col_id)}, {_sql_quote_ident(col_name)}, '
        f'{_sql_quote_ident(col_cnic)}, {_sql_quote_ident(col_ntn)}, '
        f'{_sql_quote_ident(col_strn)}, {_sql_quote_ident(col_account_no)}, '
        f'{_sql_quote_ident(col_iban)}'
    )
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
