"""Isolated warehouses. Only trusted code can mutate raw sources."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import duckdb
import pandas as pd

from .task import identifier


class DestinationAdapter(Protocol):
    namespace: str

    def create(self) -> DestinationAdapter: ...
    def write_source(self, name: str, frame: pd.DataFrame) -> None: ...
    def query(self, sql: str) -> pd.DataFrame: ...
    def table(self, name: str, schema: str = "raw") -> pd.DataFrame: ...
    def list_tables(self) -> list[str]: ...
    def dbt_profile(self) -> dict: ...
    def cleanup(self) -> None: ...
    def close(self) -> None: ...


class DuckDBAdapter:
    def __init__(
        self,
        path: Path | str,
        *,
        query_timeout: float = 15,
        memory_limit: str = "512MB",
    ):
        self.path = Path(path).absolute()
        self.namespace = self.path.stem
        self.query_timeout = query_timeout
        self.memory_limit = memory_limit
        self.connection = None

    def create(self) -> DuckDBAdapter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ValueError("Episode databases must start empty")
        self.connection = duckdb.connect(
            str(self.path),
            config={
                "enable_external_access": False,
                "autoload_known_extensions": False,
                "autoinstall_known_extensions": False,
                "threads": 1,
                "memory_limit": self.memory_limit,
            },
        )
        self.connection.execute("CREATE SCHEMA raw")
        self.connection.execute("CREATE SCHEMA transform")
        return self

    def _connect(self):
        if self.connection is None:
            raise RuntimeError("Warehouse is not open")
        return self.connection

    def write_source(self, name: str, frame: pd.DataFrame) -> None:
        identifier(name)
        for column in frame.columns:
            identifier(column)
        connection = self._connect()
        connection.register("__trusted_source", frame)
        try:
            connection.execute(
                f'CREATE OR REPLACE TABLE raw."{name}" AS SELECT * FROM __trusted_source'
            )
        finally:
            connection.unregister("__trusted_source")

    def query(self, sql: str) -> pd.DataFrame:
        connection = self._connect()
        timer = threading.Timer(self.query_timeout, connection.interrupt)
        timer.start()
        try:
            return connection.execute(sql).fetchdf()
        finally:
            timer.cancel()
            timer.join()

    def table(self, name: str, schema: str = "raw") -> pd.DataFrame:
        identifier(name)
        if schema not in {"raw", "transform"}:
            raise ValueError("Unknown logical schema")
        return self.query(f'SELECT * FROM {schema}."{name}"')

    def list_tables(self) -> list[str]:
        rows = self.query(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_schema IN ('raw', 'transform') ORDER BY 1, 2"
        )
        return [f"{r.table_schema}.{r.table_name}" for r in rows.itertuples()]

    def dbt_profile(self) -> dict:
        return {
            "type": "duckdb",
            "path": str(self.path),
            "schema": "transform",
            "threads": 1,
            "settings": {
                "enable_external_access": False,
                "threads": 1,
                "autoload_known_extensions": False,
                "autoinstall_known_extensions": False,
                "memory_limit": self.memory_limit,
            },
        }

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def reopen(self) -> None:
        self.connection = duckdb.connect(
            str(self.path),
            config={
                "enable_external_access": False,
                "autoload_known_extensions": False,
                "autoinstall_known_extensions": False,
                "threads": 1,
                "memory_limit": self.memory_limit,
            },
        )

    def cleanup(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)
        self.path.with_suffix(self.path.suffix + ".wal").unlink(missing_ok=True)
