"""
Storage abstraction layer.

Defines the interface that convergence uses to read/write data.
Implementations: LanceStorage, PostgresStorage.

All methods work with PyArrow Tables as the interchange format —
processors always receive and return Arrow data regardless of backend.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pyarrow as pa


@dataclass
class StorageStats:
    exists: bool
    row_count: int
    columns: list[str]
    # Backend-specific health metric (fragments for Lance, table size for PG)
    fragmentation: int = 0


class StorageBackend(ABC):
    """
    Abstract storage backend for the convergence engine.

    All data passes through as PyArrow Tables. This keeps processors
    storage-agnostic — they always work with Arrow arrays.
    """

    @abstractmethod
    def exists(self) -> bool:
        """Does the dataset/table exist?"""

    @abstractmethod
    def stats(self) -> StorageStats:
        """Get dataset stats: row count, columns, fragmentation."""

    @abstractmethod
    def get_columns(self) -> list[str]:
        """Return list of column names in the dataset."""

    @abstractmethod
    def count_rows(self) -> int:
        """Total row count."""

    @abstractmethod
    def null_count(self, column: str) -> int:
        """Count of NULL values in a specific column."""

    @abstractmethod
    def column_exists(self, column: str) -> bool:
        """Does this column exist in the schema?"""

    @abstractmethod
    def read(
        self,
        columns: list[str] | None = None,
        filter_expr: str | None = None,
    ) -> pa.Table:
        """
        Read rows as a PyArrow Table.

        columns: subset of columns to read (None = all)
        filter_expr: SQL-like filter string, e.g. "member_id IN ('a','b')"
        """

    @abstractmethod
    def read_null_rows(self, column: str, extra_columns: list[str]) -> pa.Table:
        """
        Read rows where `column` IS NULL.
        Returns a table with `column` + `extra_columns`.
        """

    @abstractmethod
    def append(self, table: pa.Table) -> None:
        """Append rows. Does not check for duplicates."""

    @abstractmethod
    def upsert(self, table: pa.Table, primary_key: str) -> None:
        """
        Upsert: insert new rows, update existing by primary key.
        Existing columns not in `table` are preserved as-is.
        New rows get NULLs for columns not in `table`.
        """

    @abstractmethod
    def merge_column(self, table: pa.Table, primary_key: str, column: str) -> None:
        """
        Merge a single derived column back by primary key.
        `table` has [primary_key, column]. Updates only `column` for matching rows.
        """

    @abstractmethod
    def add_column(self, column: str, dtype: pa.DataType) -> None:
        """Add a new column (all NULLs) to the schema."""

    @abstractmethod
    def delete_rows(self, filter_expr: str) -> int:
        """Delete rows matching filter. Returns count deleted."""

    @abstractmethod
    def replace_rows(self, old_filter: str, new_rows: pa.Table) -> None:
        """Delete rows matching filter, then append new_rows. Atomic where possible."""

    @abstractmethod
    def compact(self, threshold: int = 20) -> bool:
        """Run storage maintenance (compaction/vacuum). Returns True if it ran."""

    @abstractmethod
    def drop(self) -> None:
        """Delete the entire dataset/table."""


# ---------------------------------------------------------------------------
# Lance implementation
# ---------------------------------------------------------------------------

class LanceStorage(StorageBackend):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def _open(self):
        import lance
        return lance.dataset(self.dataset_path)

    def exists(self) -> bool:
        try:
            self._open()
            return True
        except Exception:
            return False

    def stats(self) -> StorageStats:
        if not self.exists():
            return StorageStats(exists=False, row_count=0, columns=[], fragmentation=0)
        ds = self._open()
        return StorageStats(
            exists=True,
            row_count=ds.count_rows(),
            columns=ds.schema.names,
            fragmentation=len(ds.get_fragments()),
        )

    def get_columns(self) -> list[str]:
        return self._open().schema.names

    def count_rows(self) -> int:
        return self._open().count_rows()

    def null_count(self, column: str) -> int:
        ds = self._open()
        tbl = ds.to_table(columns=[column])
        return tbl.column(column).null_count

    def column_exists(self, column: str) -> bool:
        return column in self._open().schema.names

    def read(self, columns=None, filter_expr=None) -> pa.Table:
        ds = self._open()
        kwargs = {}
        if columns:
            kwargs["columns"] = columns
        if filter_expr:
            kwargs["filter"] = filter_expr
        return ds.to_table(**kwargs)

    def read_null_rows(self, column: str, extra_columns: list[str]) -> pa.Table:
        ds = self._open()
        all_cols = list(set([column] + extra_columns))
        tbl = ds.to_table(columns=all_cols)
        null_mask = tbl.column(column).is_null()
        return tbl.filter(null_mask)

    def append(self, table: pa.Table) -> None:
        import lance
        if not self.exists():
            lance.write_dataset(table, self.dataset_path, mode="overwrite")
        else:
            # Align schema: add NULL columns for missing fields
            ds = self._open()
            for col_name in ds.schema.names:
                if col_name not in table.schema.names:
                    col_type = ds.schema.field(col_name).type
                    table = table.append_column(col_name, pa.nulls(len(table), type=col_type))
            cols_to_write = [c for c in ds.schema.names if c in table.schema.names]
            table = table.select(cols_to_write)
            lance.write_dataset(table, self.dataset_path, mode="append")

    def upsert(self, table: pa.Table, primary_key: str) -> None:
        import lance
        if not self.exists():
            lance.write_dataset(table, self.dataset_path, mode="overwrite")
            return

        ds = self._open()
        new_ids = set(table.column(primary_key).to_pylist())
        existing_ids = set(ds.to_table(columns=[primary_key]).column(primary_key).to_pylist())
        overlap = new_ids & existing_ids

        if overlap:
            id_list = ", ".join(f"'{v}'" for v in overlap)
            ds.delete(f"{primary_key} IN ({id_list})")

        self.append(table)

    def merge_column(self, table: pa.Table, primary_key: str, column: str) -> None:
        ds = self._open()
        ds.merge(table, left_on=primary_key, right_on=primary_key)

    def add_column(self, column: str, dtype: pa.DataType) -> None:
        ds = self._open()
        pk = ds.schema.names[0]
        ids = ds.to_table(columns=[pk])
        merge_table = pa.table({
            pk: ids.column(pk),
            column: pa.nulls(len(ids), type=dtype),
        })
        ds.merge(merge_table, left_on=pk, right_on=pk)

    def delete_rows(self, filter_expr: str) -> int:
        ds = self._open()
        before = ds.count_rows()
        ds.delete(filter_expr)
        return before - self._open().count_rows()

    def replace_rows(self, old_filter: str, new_rows: pa.Table) -> None:
        import lance
        ds = self._open()
        ds.delete(old_filter)
        lance.write_dataset(new_rows, self.dataset_path, mode="append")

    def compact(self, threshold: int = 20) -> bool:
        if not self.exists():
            return False
        ds = self._open()
        if len(ds.get_fragments()) > threshold:
            ds.compact_files()
            return True
        return False

    def drop(self) -> None:
        import shutil
        from pathlib import Path
        path = Path(self.dataset_path)
        if path.exists():
            shutil.rmtree(path)


# ---------------------------------------------------------------------------
# Postgres implementation
# ---------------------------------------------------------------------------

class PostgresStorage(StorageBackend):
    """
    PostgreSQL backend. Uses psycopg for connection, reads/writes
    through PyArrow for zero-copy where possible.

    Expects a table named `{table_name}` in the given database.
    Auto-creates the table on first append if it doesn't exist.
    """

    def __init__(self, connection_string: str, table_name: str):
        self.connection_string = connection_string
        self.table_name = table_name
        self._ensure_connection()

    def _ensure_connection(self):
        import psycopg
        self._conn = psycopg.connect(self.connection_string, autocommit=True)

    def _execute(self, query: str, params=None):
        import psycopg
        try:
            with self._conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall(), [desc[0] for desc in cur.description]
                return None, None
        except psycopg.OperationalError:
            self._ensure_connection()
            with self._conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall(), [desc[0] for desc in cur.description]
                return None, None

    def _table_exists(self) -> bool:
        rows, _ = self._execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
            (self.table_name,),
        )
        return rows[0][0]

    @staticmethod
    def _arrow_type_to_pg(dtype: pa.DataType) -> str:
        """Map Arrow type to Postgres column type."""
        if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
            return "TEXT"
        elif pa.types.is_int32(dtype):
            return "INTEGER"
        elif pa.types.is_int64(dtype):
            return "BIGINT"
        elif pa.types.is_float32(dtype):
            return "REAL"
        elif pa.types.is_float64(dtype):
            return "DOUBLE PRECISION"
        elif pa.types.is_boolean(dtype):
            return "BOOLEAN"
        elif pa.types.is_list(dtype):
            # Store Arrow lists as JSONB (e.g., embeddings)
            return "JSONB"
        else:
            return "TEXT"

    def _create_table_from_arrow(self, table: pa.Table, primary_key: str | None = None):
        cols = []
        for field in table.schema:
            pg_type = self._arrow_type_to_pg(field.type)
            col_def = f'"{field.name}" {pg_type}'
            if field.name == primary_key:
                col_def += " PRIMARY KEY"
            cols.append(col_def)
        ddl = f'CREATE TABLE IF NOT EXISTS "{self.table_name}" ({", ".join(cols)})'
        self._execute(ddl)

    def _rows_to_arrow(self, rows, col_names) -> pa.Table:
        """Convert psycopg rows to Arrow table."""
        if not rows:
            return pa.table({name: pa.array([], type=pa.string()) for name in col_names})
        columns = {}
        for i, name in enumerate(col_names):
            values = [row[i] for row in rows]
            # Let Arrow infer types from Python values
            columns[name] = pa.array(values)
        return pa.table(columns)

    def exists(self) -> bool:
        return self._table_exists()

    def stats(self) -> StorageStats:
        if not self._table_exists():
            return StorageStats(exists=False, row_count=0, columns=[], fragmentation=0)
        rows, _ = self._execute(f'SELECT count(*) FROM "{self.table_name}"')
        row_count = rows[0][0]
        cols = self.get_columns()
        # Fragmentation = dead tuple ratio (from pg_stat)
        dead, _ = self._execute(
            "SELECT COALESCE(n_dead_tup, 0) FROM pg_stat_user_tables WHERE relname = %s",
            (self.table_name,),
        )
        frag = dead[0][0] if dead else 0
        return StorageStats(exists=True, row_count=row_count, columns=cols, fragmentation=frag)

    def get_columns(self) -> list[str]:
        rows, _ = self._execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position",
            (self.table_name,),
        )
        return [r[0] for r in rows]

    def count_rows(self) -> int:
        rows, _ = self._execute(f'SELECT count(*) FROM "{self.table_name}"')
        return rows[0][0]

    def null_count(self, column: str) -> int:
        rows, _ = self._execute(
            f'SELECT count(*) FROM "{self.table_name}" WHERE "{column}" IS NULL'
        )
        return rows[0][0]

    def column_exists(self, column: str) -> bool:
        return column in self.get_columns()

    def read(self, columns=None, filter_expr=None) -> pa.Table:
        col_clause = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        query = f'SELECT {col_clause} FROM "{self.table_name}"'
        if filter_expr:
            query += f" WHERE {filter_expr}"
        rows, col_names = self._execute(query)
        if columns and not col_names:
            col_names = columns
        return self._rows_to_arrow(rows or [], col_names or [])

    def read_null_rows(self, column: str, extra_columns: list[str]) -> pa.Table:
        all_cols = list(set([column] + extra_columns))
        col_clause = ", ".join(f'"{c}"' for c in all_cols)
        rows, col_names = self._execute(
            f'SELECT {col_clause} FROM "{self.table_name}" WHERE "{column}" IS NULL'
        )
        return self._rows_to_arrow(rows or [], col_names or all_cols)

    def append(self, table: pa.Table) -> None:
        import json
        if not self._table_exists():
            self._create_table_from_arrow(table)

        cols = table.schema.names
        placeholders = ", ".join(["%s"] * len(cols))
        col_names = ", ".join(f'"{c}"' for c in cols)
        query = f'INSERT INTO "{self.table_name}" ({col_names}) VALUES ({placeholders})'

        rows_data = table.to_pylist()
        with self._conn.cursor() as cur:
            for row in rows_data:
                values = []
                for c in cols:
                    v = row[c]
                    # Convert lists/arrays to JSON for JSONB columns
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v)
                    elif hasattr(v, 'tolist'):
                        v = json.dumps(v.tolist())
                    values.append(v)
                cur.execute(query, values)

    def upsert(self, table: pa.Table, primary_key: str) -> None:
        import json
        if not self._table_exists():
            self._create_table_from_arrow(table, primary_key=primary_key)
            self.append(table)
            return

        cols = table.schema.names
        col_names = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join(["%s"] * len(cols))
        update_cols = [c for c in cols if c != primary_key]
        update_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)

        query = (
            f'INSERT INTO "{self.table_name}" ({col_names}) VALUES ({placeholders}) '
            f'ON CONFLICT ("{primary_key}") DO UPDATE SET {update_clause}'
        )

        rows_data = table.to_pylist()
        with self._conn.cursor() as cur:
            for row in rows_data:
                values = []
                for c in cols:
                    v = row[c]
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v)
                    elif hasattr(v, 'tolist'):
                        v = json.dumps(v.tolist())
                    values.append(v)
                cur.execute(query, values)

    def merge_column(self, table: pa.Table, primary_key: str, column: str) -> None:
        """UPDATE ... SET column = val WHERE pk = pk_val. No small file problem."""
        import json
        rows_data = table.to_pylist()
        query = f'UPDATE "{self.table_name}" SET "{column}" = %s WHERE "{primary_key}" = %s'
        with self._conn.cursor() as cur:
            for row in rows_data:
                v = row[column]
                if isinstance(v, (list, dict)):
                    v = json.dumps(v)
                elif hasattr(v, 'tolist'):
                    v = json.dumps(v.tolist())
                cur.execute(query, (v, row[primary_key]))

    def add_column(self, column: str, dtype: pa.DataType) -> None:
        pg_type = self._arrow_type_to_pg(dtype)
        self._execute(f'ALTER TABLE "{self.table_name}" ADD COLUMN IF NOT EXISTS "{column}" {pg_type}')

    def delete_rows(self, filter_expr: str) -> int:
        rows, _ = self._execute(
            f'DELETE FROM "{self.table_name}" WHERE {filter_expr} RETURNING 1'
        )
        return len(rows) if rows else 0

    def replace_rows(self, old_filter: str, new_rows: pa.Table) -> None:
        self.delete_rows(old_filter)
        self.append(new_rows)

    def compact(self, threshold: int = 20) -> bool:
        """Run VACUUM ANALYZE — Postgres equivalent of compaction."""
        # Check dead tuple ratio
        stats = self.stats()
        if stats.fragmentation > threshold:
            self._execute(f'VACUUM ANALYZE "{self.table_name}"')
            return True
        return False

    def drop(self) -> None:
        self._execute(f'DROP TABLE IF EXISTS "{self.table_name}" CASCADE')


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_storage(config) -> StorageBackend:
    """
    Create the appropriate storage backend based on config or env vars.

    If STORAGE_BACKEND=postgres and DATABASE_URL is set, use Postgres.
    Otherwise, use Lance (default).
    """
    backend = os.environ.get("STORAGE_BACKEND", "lance").lower()

    if backend == "postgres":
        db_url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/convergence")
        table_name = config.name.replace("-", "_")
        return PostgresStorage(connection_string=db_url, table_name=table_name)
    else:
        return LanceStorage(dataset_path=config.dataset_path)
