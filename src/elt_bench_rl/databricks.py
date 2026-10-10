"""OAuth M2M SQL warehouse adapter with independently owned episode schemas.

Constructing an adapter never connects. Cloud execution starts at ``create``.
The service principal must be restricted to the configured training catalog.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from .errors import InfrastructureError, PolicyError
from .task import identifier


@dataclass(frozen=True)
class DatabricksConfig:
    host: str
    http_path: str
    catalog: str
    client_id: str = field(repr=False)
    client_secret: str = field(repr=False)
    timeout: float = 30

    def __post_init__(self):
        parsed = urlsplit(self.host if "://" in self.host else "https://" + self.host)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("DATABRICKS_HOST must be an HTTPS workspace hostname")
        object.__setattr__(self, "host", parsed.hostname)
        identifier(self.catalog)
        if not self.http_path.startswith("/sql/1.0/warehouses/") or any(
            c.isspace() or ord(c) < 32 for c in self.http_path
        ):
            raise ValueError("DATABRICKS_HTTP_PATH must identify a SQL warehouse")
        if not self.client_id or not self.client_secret:
            raise InfrastructureError("Databricks OAuth M2M credentials are missing")
        if not 0 < self.timeout <= 600:
            raise ValueError("Databricks timeout must be in (0, 600]")

    @classmethod
    def from_env(cls):
        names = (
            "DATABRICKS_HOST",
            "DATABRICKS_HTTP_PATH",
            "DATABRICKS_CATALOG",
            "DATABRICKS_CLIENT_ID",
            "DATABRICKS_CLIENT_SECRET",
        )
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise InfrastructureError(
                "Missing Databricks configuration: " + ", ".join(missing)
            )
        return cls(*(os.environ[name] for name in names))


class DatabricksAdapter:
    airbyte_repository = "airbyte/destination-databricks"

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        config: DatabricksConfig,
        namespace: str | None = None,
        connection_factory=None,
    ):
        # path is accepted for the destination factory; UUID also isolates replay.
        self.namespace = identifier(namespace or "elt_" + uuid.uuid4().hex)
        self.raw_schema = identifier(self.namespace + "_raw")
        self.transform_schema = identifier(self.namespace + "_transform")
        self.config = config
        self.connection = None
        self._factory = connection_factory
        self._owned: set[str] = set()

    def _connect(self):
        if self.connection is not None:
            return self.connection
        try:
            if self._factory is not None:
                self.connection = self._factory(self.config)
            else:
                from databricks import sql
                from databricks.sdk.core import Config, oauth_service_principal

                cfg = Config(
                    host="https://" + self.config.host,
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                )
                self.connection = sql.connect(
                    server_hostname=self.config.host,
                    http_path=self.config.http_path,
                    credentials_provider=lambda: oauth_service_principal(cfg),
                    _socket_timeout=self.config.timeout,
                    _retry_stop_after_attempts_count=2,
                    session_configuration={
                        "STATEMENT_TIMEOUT": str(max(1, int(self.config.timeout)))
                    },
                    use_cloud_fetch=False,
                )
        except Exception:  # noqa: BLE001 -- scrub all vendor exception messages at the credential boundary.
            raise InfrastructureError("Databricks SQL connection failed") from None
        return self.connection

    def _execute(
        self, sql: str, *, fetch: bool = False, candidate: bool = False
    ) -> pd.DataFrame:
        cursor = None
        try:
            cursor = self._connect().cursor()
            cursor.execute(sql)
            if fetch:
                return pd.DataFrame(
                    cursor.fetchall(), columns=[col[0] for col in cursor.description]
                )
            return pd.DataFrame()
        except InfrastructureError:
            raise
        except Exception as error:  # noqa: BLE001 -- vendor errors can include sensitive configuration.
            # Structured connector exception classes, never model-controlled text.
            from databricks.sql.exc import (
                DataError,
                ProgrammingError,
                ServerOperationError,
            )

            state = getattr(error, "sql_state", None)
            code = getattr(error, "error_code", None)
            if (
                state == "42501"
                or isinstance(state, str)
                and state[:2] in {"08", "28", "53", "58"}
                or code
                in {
                    "PERMISSION_DENIED",
                    "INSUFFICIENT_PERMISSIONS",
                    "INVALID_AUTHENTICATION",
                }
            ):
                raise InfrastructureError(
                    "Databricks authentication, permission or service failure"
                ) from None
            if candidate and isinstance(
                error, (DataError, ProgrammingError, ServerOperationError)
            ):
                raise ValueError("Warehouse rejected the candidate query") from None
            raise InfrastructureError("Databricks SQL execution failed") from None
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:  # noqa: BLE001,S110 -- preserve the primary sanitized query failure.
                    pass

    def _schema(self, schema: str) -> str:
        if schema not in {"raw", "transform"}:
            raise PolicyError("Unknown logical warehouse schema")
        return self.raw_schema if schema == "raw" else self.transform_schema

    def _qualified_schema(self, schema: str) -> str:
        return f"`{self.config.catalog}`.`{schema}`"

    def create(self):
        if self._owned:
            raise ValueError("Episode schemas have already been created")
        try:
            for schema in (self.raw_schema, self.transform_schema):
                # Never adopt another episode's existing schema.
                self._execute("CREATE SCHEMA " + self._qualified_schema(schema))
                self._owned.add(schema)
        except Exception:
            self.cleanup()
            raise
        return self

    def write_source(self, name: str, frame: pd.DataFrame) -> None:
        raise InfrastructureError("Databricks raw loading requires an Airbyte sync")

    def _rewrite(self, sql: str) -> str:
        try:
            trees = sqlglot.parse(sql, read="databricks")
        except sqlglot.errors.ParseError:
            raise PolicyError("Invalid Databricks SQL") from None
        if len(trees) != 1 or not isinstance(trees[0], exp.Query):
            raise PolicyError("Only a single read-only warehouse query is allowed")
        tree = trees[0]
        for scope in traverse_scope(tree):
            for table in scope.tables:
                if not isinstance(table.this, exp.Identifier):
                    raise PolicyError("Table functions are not allowed")
                if (
                    not table.db
                    and not table.catalog
                    and table.name in scope.cte_sources
                ):
                    continue
                identifier(table.name)
                if table.catalog and table.catalog != self.config.catalog:
                    raise PolicyError("Cross-catalog access is not allowed")
                schema = table.db or "transform"
                if schema in {"raw", "transform"}:
                    physical = self._schema(schema)
                elif schema in self._owned:
                    physical = schema
                else:
                    raise PolicyError("Cross-episode schema access is not allowed")
                table.set("db", exp.to_identifier(physical, quoted=True))
                table.set(
                    "catalog", exp.to_identifier(self.config.catalog, quoted=True)
                )
        return tree.sql(dialect="databricks")

    def query(self, sql: str) -> pd.DataFrame:
        return self._execute(self._rewrite(sql), fetch=True, candidate=True)

    def table(self, name: str, schema: str = "raw") -> pd.DataFrame:
        identifier(name)
        self._schema(schema)
        return self.query(f"SELECT * FROM {schema}.`{name}`")

    def list_tables(self) -> list[str]:
        result = []
        for logical in ("raw", "transform"):
            frame = self._execute(
                "SHOW TABLES IN " + self._qualified_schema(self._schema(logical)),
                fetch=True,
            )
            names = frame.get(
                "tableName", frame.get("table_name", pd.Series(dtype="str"))
            )
            result.extend(f"{logical}.{identifier(str(name))}" for name in names)
        return sorted(result)

    def dbt_profile(self) -> dict:
        # Trusted caller writes this to mode 0600, deletes it, excludes it from bundles.
        return {
            "type": "databricks",
            "auth_type": "oauth",
            "host": self.config.host,
            "http_path": self.config.http_path,
            "catalog": self.config.catalog,
            "schema": self.transform_schema,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "threads": 1,
            "connect_timeout": self.config.timeout,
            "connect_retries": 1,
            "connection_parameters": {
                "_socket_timeout": self.config.timeout,
                "_retry_stop_after_attempts_count": 2,
            },
            "session_properties": {
                "STATEMENT_TIMEOUT": str(max(1, int(self.config.timeout)))
            },
        }

    def airbyte_configuration(self) -> dict:
        """Trusted destination configuration; never put this in model feedback."""
        return {
            "hostname": self.config.host,
            "http_path": self.config.http_path,
            "port": "443",
            "database": self.config.catalog,
            "schema": self.raw_schema,
            "accept_terms": True,
            "purge_staging_data": True,
            "authentication": {
                "auth_type": "OAUTH",
                "client_id": self.config.client_id,
                "secret": self.config.client_secret,
            },
        }

    def close(self) -> None:
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 -- never expose credential-bearing vendor messages.
                raise InfrastructureError("Databricks SQL close failed") from None

    def reopen(self) -> None:
        self._connect()

    def check_health(self) -> bool:
        """Trusted connection probe; query rejection cannot impersonate auth failure."""
        self._execute("SELECT 1", fetch=True)
        return True

    def cleanup(self) -> None:
        failed = False
        for schema in sorted(self._owned):
            try:
                self._execute(
                    "DROP SCHEMA " + self._qualified_schema(schema) + " CASCADE"
                )
                self._owned.remove(schema)
            except InfrastructureError:
                failed = True
        try:
            self.close()
        except InfrastructureError:
            failed = True
        if failed:
            raise InfrastructureError(
                "Databricks cleanup incomplete; owned schemas retained for retry"
            )
