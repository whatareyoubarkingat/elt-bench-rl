"""Authenticated, bounded Airbyte 2.4 control-plane operations.

The Configuration API implements check/discover/sync. Declarative publication
uses the public API and reconciles builder drafts left by OSS state-conflicts.
HTTP errors never expose bodies. Missing CDK image metadata is an explicit
infrastructure failure; ``scripts/repair_airbyte_manifest_seed.py`` repairs it.
Mocks test the protocol; only a succeeded real sync plus fresh warehouse reads
can produce a load receipt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
import yaml

from .errors import InfrastructureError
from .ingestion import (
    AIRBYTE_METADATA_COLUMNS,
    LoadReceipt,
    coerce_source,
    content_digest,
)
from .task import SourceSpec, identifier


class AirbyteError(InfrastructureError):
    def __init__(self, operation: str, status: int | None = None):
        self.operation, self.status = operation, status
        suffix = f" (HTTP {status})" if status is not None else ""
        super().__init__("Airbyte " + operation + " failed" + suffix)


@dataclass(frozen=True)
class AirbyteConfig:
    base_url: str
    workspace_id: str = field(repr=False)
    client_id: str = field(repr=False)
    client_secret: str = field(repr=False)
    request_timeout: float = 120
    job_timeout: float = 900
    poll_interval: float = 3
    cancel_timeout: float = 60
    read_retries: int = 2

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError(
                "AIRBYTE_URL must be a base HTTP(S) URL without credentials"
            )
        if parsed.scheme == "http" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("Remote Airbyte requires HTTPS")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if not all((self.workspace_id, self.client_id, self.client_secret)):
            raise InfrastructureError(
                "Airbyte credentials or workspace configuration missing"
            )
        if not (
            0 < self.request_timeout <= 300
            and 0 < self.job_timeout <= 7200
            and 0 < self.poll_interval <= 60
            and 0 < self.cancel_timeout <= 120
            and 0 <= self.read_retries <= 5
        ):
            raise ValueError("Airbyte timeouts or retries are outside bounded limits")

    @classmethod
    def from_env(cls):
        names = ("AIRBYTE_WORKSPACE_ID", "AIRBYTE_CLIENT_ID", "AIRBYTE_CLIENT_SECRET")
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise InfrastructureError(
                "Missing Airbyte configuration: " + ", ".join(missing)
            )
        return cls(
            os.environ.get("AIRBYTE_URL", "http://localhost:8000"),
            *(os.environ[name] for name in names),
        )


@dataclass(frozen=True)
class JobSummary:
    job_id: int
    status: str
    attempts: int


class AirbyteClient:
    def __init__(
        self,
        config: AirbyteConfig,
        *,
        session=None,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.session.trust_env = False
        self._clock, self._sleep = clock, sleep
        self._token = None
        self._expires = 0.0
        self._deadline: float | None = None

    def set_deadline(self, deadline: float | None) -> None:
        if deadline is not None and (
            not isinstance(deadline, (int, float)) or not math.isfinite(deadline)
        ):
            raise ValueError("Expected a finite monotonic deadline")
        self._deadline = deadline

    def _timeouts(self) -> tuple[float, float]:
        if self._deadline is None:
            return 5, self.config.request_timeout
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            raise AirbyteError("episode deadline exceeded")
        return min(5, remaining / 2), min(self.config.request_timeout, remaining / 2)

    def _pause(self, delay: float) -> None:
        if self._deadline is not None:
            remaining = self._deadline - self._clock()
            if remaining <= 0:
                raise AirbyteError("episode deadline exceeded")
            delay = min(delay, remaining)
        self._sleep(delay)

    def _authenticate(self):
        try:
            response = self.session.request(
                "POST",
                self.config.base_url + "/api/public/v1/applications/token",
                json={
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "grant_type": "client_credentials",
                },
                timeout=self._timeouts(),
                allow_redirects=False,
            )
            if response.status_code != 200:
                raise AirbyteError("authentication", response.status_code)
            payload = response.json()
            token = payload.get("access_token")
            if not isinstance(token, str) or not token:
                raise AirbyteError("authentication response")
            self._token = token
            self._expires = self._clock() + max(
                1, float(payload.get("expires_in", 180)) - 15
            )
        except AirbyteError:
            raise
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            raise AirbyteError("authentication transport or response") from None

    def request(self, method: str, path: str, *, body=None, read_only=False):
        if not path.startswith("/api/") or "?" in path or "#" in path:
            raise ValueError("Invalid fixed Airbyte API path")
        retries = self.config.read_retries if read_only else 0
        refreshed = False
        attempt = 0
        while True:
            timeout = self._timeouts()
            if self._token is None or self._clock() >= self._expires:
                self._authenticate()
                timeout = self._timeouts()
            try:
                response = self.session.request(
                    method,
                    self.config.base_url + path,
                    json=body,
                    headers={"Authorization": "Bearer " + self._token},
                    timeout=timeout,
                    allow_redirects=False,
                )
            except requests.RequestException:
                if attempt < retries:
                    attempt += 1
                    self._pause(min(attempt, 3))
                    continue
                raise AirbyteError(
                    "transport (mutation outcome may be unknown)"
                    if not read_only
                    else "read transport"
                ) from None
            status = response.status_code
            if status == 401 and not refreshed:
                refreshed = True
                self._token = None
                continue  # An unauthorized request has not performed the mutation.
            if (status == 429 or 500 <= status < 600) and attempt < retries:
                attempt += 1
                self._pause(min(attempt, 3))
                continue
            if status not in {200, 201, 202, 204}:
                raise AirbyteError("API request", status)
            if status == 204:
                return {}
            try:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError
                return payload
            except (ValueError, TypeError):
                raise AirbyteError("invalid JSON response", status) from None

    def rpc(self, family: str, method: str, body: dict, *, read_only=False):
        return self.request(
            "POST", f"/api/v1/{family}/{method}", body=body, read_only=read_only
        )

    def health(self) -> bool:
        return (
            self.request("GET", "/api/v1/health", read_only=True).get("available")
            is True
        )

    def definitions(self, kind: str) -> list[dict]:
        if kind not in {"source", "destination"}:
            raise ValueError("Definition kind must be source or destination")
        payload = self.rpc(
            kind + "_definitions",
            "list_for_workspace",
            {"workspaceId": self.config.workspace_id},
            read_only=True,
        )
        return payload.get(kind + "Definitions", [])

    def definition(self, kind: str, repository: str) -> dict:
        matches = [
            item
            for item in self.definitions(kind)
            if item.get("dockerRepository") == repository
        ]
        if len(matches) != 1:
            raise AirbyteError("required connector definition unavailable or ambiguous")
        return matches[0]

    def _declarative_path(self, definition_id: str | None = None):
        base = f"/api/public/v1/workspaces/{self.config.workspace_id}/definitions/declarative_sources"
        return base if definition_id is None else base + "/" + definition_id

    @staticmethod
    def _digest(manifest: dict) -> str:
        return hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _existing_manifest(self, name: str, manifest: dict) -> str | None:
        payload = self.request("GET", self._declarative_path(), read_only=True)
        matches = [item for item in payload.get("data", []) if item.get("name") == name]
        if len(matches) > 1:
            raise AirbyteError("ambiguous declarative definition")
        if matches:
            definition_id = matches[0].get("id") or matches[0].get("sourceDefinitionId")
            if not isinstance(definition_id, str):
                raise AirbyteError("invalid declarative definition identity")
            actual = self.request(
                "GET", self._declarative_path(definition_id), read_only=True
            )
            if self._digest(actual.get("manifest", {})) != self._digest(manifest):
                raise AirbyteError("immutable declarative definition manifest mismatch")
            return definition_id
        return None

    def _builder_projects(self, name: str, manifest: dict) -> list[dict]:
        projects = self.rpc(
            "connector_builder_projects",
            "list",
            {"workspaceId": self.config.workspace_id},
            read_only=True,
        ).get("projects", [])
        matches = [item for item in projects if item.get("name") == name]
        for item in matches:
            project_id = item.get("builderProjectId")
            if not isinstance(project_id, str):
                raise AirbyteError("builder project response missing identity")
            actual = self.rpc(
                "connector_builder_projects",
                "get_with_manifest",
                {
                    "workspaceId": self.config.workspace_id,
                    "builderProjectId": project_id,
                },
                read_only=True,
            )
            stored = actual.get("declarativeManifest", {}).get("manifest")
            if self._digest(stored or {}) != self._digest(manifest):
                raise AirbyteError("immutable builder project manifest mismatch")
        return sorted(matches, key=lambda item: item["builderProjectId"])

    def register_manifest(self, manifest: dict, *, name_prefix="elt-bench-rl") -> str:
        """Reuse only equal manifests; repair OSS public-create conflict via builder.

        Definitions are immutable, content-addressed shared caches. Never update
        another active episode's definition to resolve a version conflict.
        """
        if manifest.get("type") != "DeclarativeSource":
            raise ValueError("Expected a trusted declarative source manifest")
        name = name_prefix + "-" + self._digest(manifest)[:20]
        existing = self._existing_manifest(name, manifest)
        if existing:
            return existing
        # Public create can persist a draft and still return 409. Inspect before
        # every mutation, including a fresh invocation after an earlier failure.
        matches = self._builder_projects(name, manifest)
        published = [item for item in matches if item.get("sourceDefinitionId")]
        if published:
            return published[0]["sourceDefinitionId"]
        try:
            self.rpc(
                "connector_builder_projects",
                "get_base_image",
                {"manifest": manifest},
                read_only=True,
            )
        except AirbyteError as error:
            raise AirbyteError(
                "declarative CDK base image unavailable; run the manifest seed preflight",
                error.status,
            ) from None
        if not matches:
            try:
                result = self.request(
                    "POST",
                    self._declarative_path(),
                    body={"name": name, "manifest": manifest},
                )
                definition_id = result.get("id") or result.get("sourceDefinitionId")
                if not isinstance(definition_id, str):
                    raise AirbyteError("declarative create response missing identity")
                return definition_id
            except AirbyteError as error:
                if error.status != 409:
                    raise
            existing = self._existing_manifest(name, manifest)
            if existing:
                return existing
            matches = self._builder_projects(name, manifest)
            published = [item for item in matches if item.get("sourceDefinitionId")]
            if published:
                return published[0]["sourceDefinitionId"]
        created = not matches
        if matches:
            # Equal duplicate drafts can be safely reused; do not delete another
            # caller's draft or mutate a shared published definition.
            project_id = matches[0]["builderProjectId"]
        else:
            result = self.rpc(
                "connector_builder_projects",
                "create",
                {
                    "workspaceId": self.config.workspace_id,
                    "builderProject": {"name": name, "draftManifest": manifest},
                },
            )
            project_id = result.get("builderProjectId")
        if not isinstance(project_id, str):
            raise AirbyteError("builder project response missing identity")
        spec = manifest.get("spec", {}).get("connection_specification")
        if not isinstance(spec, dict):
            raise TypeError("Declarative manifest requires a connection specification")
        try:
            result = self.rpc(
                "connector_builder_projects",
                "publish",
                {
                    "workspaceId": self.config.workspace_id,
                    "builderProjectId": project_id,
                    "name": name,
                    "initialDeclarativeManifest": {
                        "version": 1,
                        "description": "ELT-Bench RL immutable source manifest",
                        "manifest": manifest,
                        # Native builder expects the full Airbyte protocol SPEC.
                        "spec": {
                            "connectionSpecification": copy.deepcopy(spec),
                            "documentationUrl": "https://docs.airbyte.com/platform/connector-development/config-based/low-code-cdk-overview",
                        },
                    },
                },
            )
            definition_id = result.get("sourceDefinitionId")
            if not isinstance(definition_id, str):
                raise AirbyteError("builder publish response missing identity")
            return definition_id
        except AirbyteError:
            # Publication may have succeeded before a transport error: reconcile.
            existing = self._existing_manifest(name, manifest)
            if existing:
                return existing
            if created:
                self.rpc(
                    "connector_builder_projects",
                    "delete",
                    {
                        "workspaceId": self.config.workspace_id,
                        "builderProjectId": project_id,
                    },
                )
            raise

    def create_source(self, name: str, definition_id: str, configuration: dict) -> str:
        result = self.rpc(
            "sources",
            "create",
            {
                "name": name,
                "workspaceId": self.config.workspace_id,
                "sourceDefinitionId": definition_id,
                "connectionConfiguration": configuration,
            },
        )
        value = result.get("sourceId")
        if not isinstance(value, str):
            raise AirbyteError("source creation missing identity")
        return value

    def check_source(self, source_id: str) -> None:
        result = self.rpc("sources", "check_connection", {"sourceId": source_id})
        if result.get("status") != "succeeded":
            raise AirbyteError("source check")

    def discover(self, source_id: str) -> tuple[dict, str | None]:
        result = self.rpc(
            "sources", "discover_schema", {"sourceId": source_id, "disableCache": True}
        )
        catalog = result.get("catalog")
        if not isinstance(catalog, dict) or not catalog.get("streams"):
            raise AirbyteError("source discovery")
        return catalog, result.get("catalogId")

    def create_destination(
        self, name: str, definition_id: str, configuration: dict
    ) -> str:
        result = self.rpc(
            "destinations",
            "create",
            {
                "name": name,
                "workspaceId": self.config.workspace_id,
                "destinationDefinitionId": definition_id,
                "connectionConfiguration": configuration,
            },
        )
        value = result.get("destinationId")
        if not isinstance(value, str):
            raise AirbyteError("destination creation missing identity")
        return value

    def check_destination(self, destination_id: str) -> None:
        result = self.rpc(
            "destinations", "check_connection", {"destinationId": destination_id}
        )
        if result.get("status") != "succeeded":
            raise AirbyteError("destination check")

    def create_connection(
        self,
        name: str,
        source_id: str,
        destination_id: str,
        catalog: dict,
        raw_schema: str,
        *,
        source_name: str,
        catalog_id: str | None = None,
    ) -> str:
        identifier(raw_schema)
        catalog = copy.deepcopy(catalog)
        selected = []
        for item in catalog.get("streams", []):
            stream = item.get("stream", {})
            if stream.get("name") != source_name:
                continue
            if "full_refresh" not in stream.get("supportedSyncModes", ["full_refresh"]):
                raise AirbyteError("full refresh unsupported by source")
            item["config"] = {
                "syncMode": "full_refresh",
                "destinationSyncMode": "overwrite",
                "selected": True,
                "aliasName": source_name,
                "cursorField": [],
                "primaryKey": [],
            }
            selected.append(item)
        if len(selected) != 1:
            raise AirbyteError("expected source stream absent or ambiguous")
        body = {
            "name": name,
            "sourceId": source_id,
            "destinationId": destination_id,
            "status": "active",
            "scheduleType": "manual",
            "namespaceDefinition": "customformat",
            "namespaceFormat": raw_schema,
            "prefix": "",
            "syncCatalog": {"streams": selected},
        }
        if catalog_id:
            body["sourceCatalogId"] = catalog_id
        result = self.rpc("connections", "create", body)
        value = result.get("connectionId")
        if not isinstance(value, str):
            raise AirbyteError("connection creation missing identity")
        return value

    def sync(self, connection_id: str) -> int:
        result = self.rpc("connections", "sync", {"connectionId": connection_id})
        job_id = result.get("job", {}).get("id")
        if not isinstance(job_id, int) or isinstance(job_id, bool):
            raise AirbyteError("sync response missing job identity")
        return job_id

    def job(self, job_id: int) -> JobSummary:
        result = self.rpc("jobs", "get", {"id": job_id}, read_only=True)
        status = result.get("job", {}).get("status")
        if status not in {
            "pending",
            "running",
            "incomplete",
            "failed",
            "succeeded",
            "cancelled",
        }:
            raise AirbyteError("unknown job state")
        return JobSummary(job_id, status, len(result.get("attempts", [])))

    def connection_jobs(self, connection_id: str) -> list[JobSummary]:
        """Recover a sync whose submission response was lost, without resubmitting."""
        result = self.rpc(
            "jobs",
            "list",
            {
                "configTypes": ["sync"],
                "configId": connection_id,
                "pagination": {"pageSize": 100, "rowOffset": 0},
                "orderByField": "createdAt",
                "orderByMethod": "DESC",
            },
            read_only=True,
        )
        rows = result.get("jobs", [])
        if not isinstance(rows, list) or len(rows) >= 100:
            raise AirbyteError("bounded job reconciliation unavailable")
        summaries = []
        for row in rows:
            job = row.get("job", row)
            job_id, status = job.get("id"), job.get("status")
            if (
                not isinstance(job_id, int)
                or isinstance(job_id, bool)
                or status
                not in {
                    "pending",
                    "running",
                    "incomplete",
                    "failed",
                    "succeeded",
                    "cancelled",
                }
            ):
                raise AirbyteError("invalid reconciliation job identity or state")
            summaries.append(JobSummary(job_id, status, len(row.get("attempts", []))))
        return summaries

    def poll(self, job_id: int) -> JobSummary:
        deadline = self._clock() + self.config.job_timeout
        previous = self._deadline
        self._deadline = min(deadline, previous) if previous is not None else deadline
        try:
            while self._clock() < self._deadline:
                result = self.job(job_id)
                if result.status == "succeeded":
                    return result
                if result.status in {"failed", "cancelled", "incomplete"}:
                    raise AirbyteError("sync ended " + result.status)
                self._pause(self.config.poll_interval)
            # Owned job remains tracked by the ingestor. Its bounded cleanup
            # gets a fresh cancellation budget even after the episode expires.
            raise AirbyteError("sync timeout; owned job retained for cleanup")
        finally:
            self._deadline = previous

    def poll_cancel_terminal(self, job_id: int) -> JobSummary:
        deadline = self._clock() + self.config.cancel_timeout
        previous = self._deadline
        self._deadline = min(deadline, previous) if previous is not None else deadline
        try:
            while self._clock() < self._deadline:
                result = self.job(job_id)
                if result.status in {"cancelled", "failed", "succeeded"}:
                    return result
                self._pause(self.config.poll_interval)
            raise AirbyteError("cancellation not terminal; owned resources retained")
        finally:
            self._deadline = previous

    def cancel(self, job_id: int) -> JobSummary:
        # A cancel response acknowledges a request; it does not prove quiescence.
        current = self.job(job_id)
        if current.status in {"cancelled", "failed", "succeeded"}:
            return current
        self.rpc("jobs", "cancel", {"id": job_id})
        return self.poll_cancel_terminal(job_id)

    def delete(self, kind: str, resource_id: str) -> None:
        if kind not in {"source", "destination", "connection"}:
            raise ValueError("Unsupported cleanup resource")
        try:
            self.rpc(kind + "s", "delete", {kind + "Id": resource_id})
        except AirbyteError as error:
            if error.status != 404:
                raise

    def reconcile_actor(self, kind: str, name: str, identity: dict) -> str | None:
        if kind not in {"source", "destination", "connection"}:
            raise ValueError("Unsupported creation ownership kind")
        payload = self.rpc(
            kind + "s",
            "list",
            {"workspaceId": self.config.workspace_id},
            read_only=True,
        )
        rows = payload.get(kind + "s", [])
        matches = [
            row
            for row in rows
            if row.get("name") == name
            and all(row.get(key) == value for key, value in identity.items())
        ]
        if len(matches) > 1:
            raise AirbyteError("ambiguous actor creation ownership")
        if not matches:
            return None
        value = matches[0].get(kind + "Id")
        if not isinstance(value, str):
            raise AirbyteError("reconciled actor lacks identity")
        return value


class AirbyteIngestor:
    name = "airbyte"

    def __init__(
        self,
        client: AirbyteClient,
        *,
        project_root: Path | str,
        destination_configuration=None,
    ):
        self.client = client
        self.project_root = Path(project_root).resolve()
        self._destination_configuration = destination_configuration
        self._destinations: dict[str, str] = {}
        self._validated_destinations: set[str] = set()
        self._sources: dict[tuple[str, str], str] = {}
        self._connections: dict[tuple[str, str], str] = {}
        self._resources: dict[str, list[tuple[str, str]]] = {}
        self._jobs: dict[str, set[int]] = {}
        self._pending_sync: dict[str, dict[str, int]] = {}
        self._last_job: dict[str, int] = {}
        self._pending_actors: dict[str, list[dict]] = {}

    @classmethod
    def from_env(cls, *, project_root: Path | str):
        return cls(AirbyteClient(AirbyteConfig.from_env()), project_root=project_root)

    def extract(self, source: SourceSpec) -> pd.DataFrame:
        """Inspect the verified public source mirror; this does not load data.

        Actual loading always invokes check/discover/sync, never this cache.
        ``expected_path`` is the public source CSV, never verifier ground truth.
        """
        if source.expected_path is None:
            raise InfrastructureError("Public source inspection mirror unavailable")
        try:
            if source.expected_path.stat().st_size > 8_000_000:
                raise InfrastructureError(
                    "Public source inspection mirror exceeds byte limit"
                )
            return coerce_source(pd.read_csv(source.expected_path), source)
        except (OSError, pd.errors.ParserError):
            raise InfrastructureError(
                "Public source inspection mirror unavailable"
            ) from None

    def _remember(self, namespace: str, kind: str, value: str) -> str:
        self._resources.setdefault(namespace, []).append((kind, value))
        return value

    def _create_owned(
        self, namespace: str, kind: str, name: str, identity: dict, create
    ) -> str:
        pending = self._pending_actors.setdefault(namespace, [])
        intent = {"kind": kind, "name": name, "identity": dict(identity)}
        if intent in pending:
            value = self.client.reconcile_actor(kind, name, identity)
            if value is None:
                raise AirbyteError(
                    "earlier actor creation unresolved; ownership intent retained"
                )
        else:
            pending.append(intent)
            try:
                value = create()
            except AirbyteError as error:
                if error.status in {400, 401, 403, 404, 422}:
                    pending.remove(intent)
                    raise
                try:
                    value = self.client.reconcile_actor(kind, name, identity)
                except AirbyteError:
                    raise error from None
                if value is None:
                    raise error from None
        pending.remove(intent)
        return self._remember(namespace, kind, value)

    def _source(self, source: SourceSpec, namespace: str) -> str:
        key = (namespace, source.name)
        if key in self._sources:
            return self._sources[key]
        if source.kind == "csv":
            definition = self.client.definition("source", "airbyte/source-file")
            url = source.official_locator
            if not url or urlsplit(url).scheme != "https":
                raise InfrastructureError(
                    "Cloud file source requires its trusted official HTTPS locator"
                )
            definition_id = definition["sourceDefinitionId"]
            configuration = {
                "dataset_name": source.name,
                "format": "csv",
                "url": url,
                "provider": {"storage": "HTTPS"},
            }
        elif source.kind == "http":
            # This installed manifest is trusted and its elt-api URL is Pod-reachable.
            manifest = yaml.safe_load(
                (self.project_root / "configs/airbyte-trains.yaml").read_text()
            )
            stream = manifest.get("definitions", {}).get("streams", {}).get(source.name)
            expected_path = urlsplit(source.locator).path
            if (
                stream is None
                or stream.get("retriever", {}).get("requester", {}).get("path")
                != expected_path
            ):
                raise InfrastructureError(
                    "No trusted Airbyte manifest for the requested HTTP source"
                )
            definition_id = self.client.register_manifest(manifest)
            configuration = {}
        else:
            raise ValueError("Unsupported cloud source kind")
        name = namespace + "-" + source.name
        source_id = self._create_owned(
            namespace,
            "source",
            name,
            {"sourceDefinitionId": definition_id},
            lambda: self.client.create_source(name, definition_id, configuration),
        )

        self._sources[key] = source_id
        return source_id

    def set_deadline(self, deadline: float | None) -> None:
        self.client.set_deadline(deadline)

    def _destination(self, adapter) -> str:
        namespace = adapter.namespace
        if namespace in self._destinations:
            destination_id = self._destinations[namespace]
        else:
            definition = self.client.definition(
                "destination",
                getattr(
                    adapter, "airbyte_repository", "airbyte/destination-databricks"
                ),
            )
            configuration = (
                self._destination_configuration(adapter)
                if self._destination_configuration
                else self._databricks_configuration(adapter)
            )
            definition_id = definition["destinationDefinitionId"]
            destination_id = self._create_owned(
                namespace,
                "destination",
                namespace,
                {"destinationDefinitionId": definition_id},
                lambda: self.client.create_destination(
                    namespace, definition_id, configuration
                ),
            )
            self._destinations[namespace] = destination_id
        if namespace not in self._validated_destinations:
            self.client.check_destination(destination_id)
            self._validated_destinations.add(namespace)
        return destination_id

    @staticmethod
    def _databricks_configuration(adapter) -> dict:
        hook = getattr(adapter, "airbyte_configuration", None)
        if hook is not None:
            return hook()
        config = adapter.config
        return {
            "hostname": config.host,
            "http_path": config.http_path,
            "port": "443",
            "database": config.catalog,
            "schema": adapter.raw_schema,
            "accept_terms": True,
            "purge_staging_data": True,
            "authentication": {
                "auth_type": "OAUTH",
                "client_id": config.client_id,
                "secret": config.client_secret,
            },
        }

    def load(self, source: SourceSpec, adapter) -> LoadReceipt:
        key = (adapter.namespace, source.name)
        if key not in self._connections:
            destination_id = self._destination(adapter)
            source_id = self._source(source, adapter.namespace)
            self.client.check_source(source_id)
            catalog, catalog_id = self.client.discover(source_id)
            name = adapter.namespace + "-" + source.name
            connection_id = self._create_owned(
                adapter.namespace,
                "connection",
                name,
                {
                    "sourceId": source_id,
                    "destinationId": destination_id,
                    "namespaceFormat": adapter.raw_schema,
                },
                lambda: self.client.create_connection(
                    name,
                    source_id,
                    destination_id,
                    catalog,
                    adapter.raw_schema,
                    source_name=source.name,
                    catalog_id=catalog_id,
                ),
            )
            self._connections[key] = connection_id
        connection_id = self._connections[key]
        self._pending_sync.setdefault(adapter.namespace, {})[connection_id] = (
            self._last_job.get(connection_id, -1)
        )
        try:
            job_id = self.client.sync(connection_id)
        except AirbyteError as error:
            if error.status in {400, 401, 403, 404, 422}:
                self._pending_sync[adapter.namespace].pop(connection_id, None)
            raise
        self._last_job[connection_id] = job_id
        self._pending_sync[adapter.namespace].pop(connection_id, None)
        self._jobs.setdefault(adapter.namespace, set()).add(job_id)
        self.client.poll(job_id)
        self._jobs[adapter.namespace].discard(job_id)
        # Do not manufacture rows/counts from cached source/reference data.
        frame = adapter.table(source.name, "raw")
        names = [str(column).lower() for column in frame.columns]
        if len(set(names)) != len(names):
            raise AirbyteError("loaded source has ambiguous column casing")
        frame = frame.copy()
        frame.columns = names
        columns = [column.name for column in source.columns]
        extras = [column for column in frame.columns if column not in columns]
        if any(column not in AIRBYTE_METADATA_COLUMNS for column in extras):
            raise AirbyteError("loaded source contains unexpected data columns")
        try:
            frame = coerce_source(frame.drop(columns=extras), source)
        except (ValueError, TypeError, KeyError):
            raise AirbyteError("loaded source schema mismatch") from None
        return LoadReceipt(
            source.name,
            self.name,
            len(frame),
            tuple(frame.columns),
            content_digest(frame),
            str(job_id),
        )

    def cleanup(self, namespace: str | None = None) -> None:
        # Cancellation is cleanup, with its own bounded budget. An expired
        # rollout deadline must not strand a running cloud writer.
        previous = self.client._deadline
        self.client.set_deadline(
            self.client._clock() + self.client.config.cancel_timeout
        )
        try:
            self._cleanup(namespace)
        finally:
            self.client.set_deadline(previous)

    def _cleanup(self, namespace: str | None = None) -> None:
        failed = False
        targets = (
            [namespace]
            if namespace is not None
            else list(
                set(self._resources)
                | set(self._jobs)
                | set(self._pending_sync)
                | set(self._pending_actors)
            )
        )
        for target in targets:
            for intent in list(self._pending_actors.get(target, [])):
                try:
                    value = self.client.reconcile_actor(
                        intent["kind"], intent["name"], intent["identity"]
                    )
                    if value is None:
                        failed = True
                        continue
                    self._remember(target, intent["kind"], value)
                    self._pending_actors[target].remove(intent)
                except AirbyteError:
                    failed = True
            for connection_id, previous_job in list(
                self._pending_sync.get(target, {}).items()
            ):
                try:
                    jobs = self.client.connection_jobs(connection_id)
                    new = [job for job in jobs if job.job_id > previous_job]
                    if not new:
                        # An uncertain request can still be executing server-side.
                        # Keep schemas/actors until its created job is observable.
                        failed = True
                        continue
                    self._jobs.setdefault(target, set()).update(
                        job.job_id
                        for job in jobs
                        if job.status not in {"succeeded", "failed", "cancelled"}
                    )
                    self._pending_sync[target].pop(connection_id)
                except AirbyteError:
                    failed = True
            for job_id in list(self._jobs.get(target, set())):
                try:
                    self.client.cancel(job_id)
                    self._jobs[target].discard(job_id)
                except AirbyteError:
                    # Do not drop a namespace while its sync may still write.
                    failed = True
            if (
                self._jobs.get(target)
                or self._pending_sync.get(target)
                or self._pending_actors.get(target)
            ):
                continue
            self._jobs.pop(target, None)
            resources = self._resources.get(target, [])
            # Connections precede actors, including partially-created configurations.
            order = {"connection": 0, "source": 1, "destination": 2}
            for kind, value in sorted(resources, key=lambda item: order[item[0]]):
                try:
                    self.client.delete(kind, value)
                    resources.remove((kind, value))
                except AirbyteError:
                    failed = True
            if not resources:
                self._resources.pop(target, None)
                self._pending_sync.pop(target, None)
                self._pending_actors.pop(target, None)
                for pair, connection_id in self._connections.items():
                    if pair[0] == target:
                        self._last_job.pop(connection_id, None)
                self._destinations.pop(target, None)
                self._validated_destinations.discard(target)
                self._sources = {
                    key: value
                    for key, value in self._sources.items()
                    if key[0] != target
                }
                self._connections = {
                    key: value
                    for key, value in self._connections.items()
                    if key[0] != target
                }
        if failed:
            raise AirbyteError("cleanup incomplete; owned resources retained for retry")

    def cleanup_ownership(self, namespace: str) -> dict:
        """Credential-free recovery metadata; caller may save it mode 0600."""
        return {
            "namespace": namespace,
            "resources": list(self._resources.get(namespace, [])),
            "jobs": sorted(self._jobs.get(namespace, set())),
            "pending_actors": copy.deepcopy(self._pending_actors.get(namespace, [])),
            "pending_sync": dict(self._pending_sync.get(namespace, {})),
        }
