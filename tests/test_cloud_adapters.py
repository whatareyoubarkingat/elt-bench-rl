"""Offline protocol/ownership tests. They never imply a real cloud sync."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import requests

from elt_bench_rl.airbyte import (
    AirbyteClient,
    AirbyteConfig,
    AirbyteError,
    AirbyteIngestor,
    JobSummary,
)
from elt_bench_rl.databricks import DatabricksAdapter, DatabricksConfig
from elt_bench_rl.errors import InfrastructureError, PolicyError
from elt_bench_rl.task import ColumnSpec, SourceSpec


class DataError(Exception):
    """Test-only DB-API exception: cloud extras are not required by mocks."""


class ProgrammingError(Exception):
    pass


class ServerOperationError(Exception):
    pass


class RequestError(Exception):
    pass


@pytest.fixture(autouse=True)
def stub_vendor_exceptions(monkeypatch):
    """Exercise classification using an injected exception boundary, not SDK I/O.

    All warehouse connections below come from connection_factory. This fixture
    is restored after each test and keeps the complete mock suite runnable with
    the default local dependencies, even when Databricks is entirely absent.
    """
    package = ModuleType("databricks")
    package.__path__ = []
    sql = ModuleType("databricks.sql")
    sql.__path__ = []
    exceptions = ModuleType("databricks.sql.exc")
    for exception in (DataError, ProgrammingError, ServerOperationError, RequestError):
        setattr(exceptions, exception.__name__, exception)
    package.sql = sql
    sql.exc = exceptions
    for name, module in (
        ("databricks", package),
        ("databricks.sql", sql),
        ("databricks.sql.exc", exceptions),
    ):
        monkeypatch.setitem(sys.modules, name, module)


class Response:
    def __init__(self, status=200, payload=None):
        self.status_code, self.payload = status, payload or {}

    def json(self):
        return self.payload


class Session:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"][0] > 0
        if url.endswith("/applications/token"):
            return Response(
                payload={"access_token": "private-bearer", "expires_in": 180}
            )
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def client(steps, **kwargs):
    cfg = AirbyteConfig(
        "http://localhost:8000",
        "workspace",
        "private-client",
        "private-secret",
        job_timeout=5,
        poll_interval=1,
    )
    session = Session(steps)
    return AirbyteClient(cfg, session=session, sleep=lambda _: None, **kwargs), session


def test_airbyte_auth_refresh_and_read_retry_have_bounds():
    api, session = client(
        [Response(401), Response(503), Response(payload={"available": True})]
    )
    assert api.health()
    assert len(session.calls) == 5  # two auth calls, three reads
    assert session.trust_env is False
    assert not session.steps


def test_airbyte_uncertain_mutation_never_retried_or_leaks_secret():
    api, session = client([requests.ConnectionError("private-secret private-bearer")])
    with pytest.raises(AirbyteError) as error:
        api.create_source("source", "definition", {"password": "private-secret"})
    assert "private-secret" not in str(error.value)
    assert "private-bearer" not in str(error.value)
    assert len(session.calls) == 2
    assert "private-secret" not in repr(api.config)


def manifest():
    return {
        "type": "DeclarativeSource",
        "version": "6.33.4",
        "streams": [],
        "spec": {"connection_specification": {"type": "object", "properties": {}}},
    }


def test_public_conflict_uses_native_builder_publish_protocol():
    api, session = client(
        [
            Response(payload={"data": []}),
            Response(payload={"projects": []}),
            Response(payload={"baseImage": "image@sha256:verified"}),
            Response(409),
            Response(payload={"data": []}),
            Response(payload={"projects": []}),
            Response(201, {"builderProjectId": "project"}),
            Response(payload={"sourceDefinitionId": "definition"}),
        ]
    )
    assert api.register_manifest(manifest()) == "definition"
    assert session.calls[-2][1].endswith("/connector_builder_projects/create")
    publish = session.calls[-1]
    assert publish[1].endswith("/connector_builder_projects/publish")
    assert publish[2]["json"]["initialDeclarativeManifest"]["version"] == 1
    assert publish[2]["json"]["initialDeclarativeManifest"]["spec"][
        "connectionSpecification"
    ] == {
        "type": "object",
        "properties": {},
    }


def test_registration_recovers_equal_duplicate_drafts_without_public_post():
    api, session = client([])
    name = "elt-bench-rl-" + api._digest(manifest())[:20]
    api.session.steps = [
        Response(payload={"data": []}),
        Response(
            payload={
                "projects": [
                    {"name": name, "builderProjectId": "b"},
                    {"name": name, "builderProjectId": "a"},
                ]
            }
        ),
        Response(payload={"declarativeManifest": {"manifest": manifest()}}),
        Response(payload={"declarativeManifest": {"manifest": manifest()}}),
        Response(payload={"baseImage": "image@sha256:verified"}),
        Response(payload={"sourceDefinitionId": "definition"}),
    ]
    assert api.register_manifest(manifest()) == "definition"
    assert session.calls[-1][2]["json"]["builderProjectId"] == "a"
    assert not any(
        method == "POST" and url.endswith("/declarative_sources")
        for method, url, _ in session.calls
    )


def test_missing_base_image_fails_before_any_registration_mutation():
    api, session = client(
        [
            Response(payload={"data": []}),
            Response(payload={"projects": []}),
            Response(500),
            Response(500),
            Response(500),
        ]
    )
    with pytest.raises(AirbyteError, match="CDK base image unavailable"):
        api.register_manifest(manifest())
    assert not any(
        url.endswith(("/create", "/publish"))
        or (method == "POST" and url.endswith("/declarative_sources"))
        for method, url, _ in session.calls
    )


def test_manifest_reuse_checks_exact_bytes_not_only_name():
    api, _ = client([])
    name = "elt-bench-rl-" + api._digest(manifest())[:20]
    api.session.steps = [
        Response(payload={"data": [{"name": name, "id": "definition"}]}),
        Response(payload={"manifest": {"type": "wrong"}}),
    ]
    with pytest.raises(AirbyteError, match="manifest mismatch"):
        api.register_manifest(manifest())


def test_selected_catalog_uses_only_declared_stream_and_is_not_mutated():
    api, session = client([Response(payload={"connectionId": "connection"})])
    original = {
        "streams": [
            {"stream": {"name": "cars", "supportedSyncModes": ["full_refresh"]}},
            {"stream": {"name": "other"}},
        ]
    }
    api.create_connection(
        "episode", "source", "destination", original, "episode_raw", source_name="cars"
    )
    body = session.calls[-1][2]["json"]
    assert len(body["syncCatalog"]["streams"]) == 1
    assert (
        body["syncCatalog"]["streams"][0]["config"]["destinationSyncMode"]
        == "overwrite"
    )
    assert body["namespaceFormat"] == "episode_raw"
    assert "config" not in original["streams"][0]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, delay):
        self.now += delay


def test_poll_timeout_keeps_its_job_for_bounded_ingestor_cleanup():
    clock = Clock()
    api, session = client(
        [
            Response(payload={"job": {"status": "running"}, "attempts": []})
            for _ in range(5)
        ],
        clock=clock,
    )
    api._sleep = clock.sleep
    # Authentication also reads the monotonic clock; pre-cache a bearer.
    api._token, api._expires = "private-bearer", 100
    with pytest.raises(AirbyteError, match="timeout"):
        api.poll(123)
    assert session.calls[-1][1].endswith("/jobs/get")
    assert session.calls[-1][2]["json"] == {"id": 123}
    assert clock.now == 5


def test_cancel_acknowledgement_waits_for_terminal_state():
    api, session = client(
        [
            Response(payload={"job": {"status": "running"}}),
            Response(payload={}),
            Response(payload={"job": {"status": "running"}}),
            Response(payload={"job": {"status": "cancelled"}}),
        ]
    )
    assert api.cancel(123).status == "cancelled"
    assert [url.rsplit("/", 1)[-1] for _, url, _ in session.calls[1:]] == [
        "get",
        "cancel",
        "get",
        "get",
    ]


def test_episode_deadline_caps_requests_retry_sleep_and_polling():
    clock = Clock()
    api, session = client(
        [Response(payload={"job": {"status": "running"}})], clock=clock
    )
    api._sleep = clock.sleep
    api.set_deadline(0.5)
    with pytest.raises(AirbyteError, match="timeout"):
        api.poll(4)
    assert clock.now == 0.5
    assert all(sum(call[2]["timeout"]) <= 0.5 for call in session.calls)
    before = len(session.calls)
    with pytest.raises(AirbyteError, match="deadline"):
        api.health()
    assert len(session.calls) == before


@pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete"])
def test_poll_failure_never_returns_load_success(status):
    api, _ = client([Response(payload={"job": {"status": status}, "attempts": []})])
    with pytest.raises(AirbyteError, match="sync ended"):
        api.poll(4)


class Cursor:
    description = (("id",),)

    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql):
        self.connection.sql.append(sql)
        if self.connection.failure and self.connection.failure[0] in sql:
            raise self.connection.failure[1]

    def fetchall(self):
        return [(1,)]

    def close(self):
        pass


class Connection:
    def __init__(self, failure=None):
        self.sql = []
        self.failure = failure
        self.closed = False

    def cursor(self):
        return Cursor(self)

    def close(self):
        self.closed = True


def warehouse(*, failure=None, namespace="episode"):
    cfg = DatabricksConfig(
        "https://example.cloud.databricks.com",
        "/sql/1.0/warehouses/123",
        "training",
        "private-client",
        "private-secret",
    )
    connection = Connection(failure)
    adapter = DatabricksAdapter(
        Path("/unused/warehouse.duckdb"),
        config=cfg,
        namespace=namespace,
        connection_factory=lambda _: connection,
    )
    return adapter, connection


def test_cloud_constructor_is_lazy_and_replay_uses_unique_schemas():
    adapter, connection = warehouse()
    assert connection.sql == []
    fresh1 = DatabricksAdapter("warehouse.duckdb", config=adapter.config)
    fresh2 = DatabricksAdapter("warehouse.duckdb", config=adapter.config)
    assert fresh1.namespace != fresh2.namespace
    assert "private-secret" not in repr(adapter.config)


def test_create_plain_ddl_query_ast_rewrite_and_owned_cleanup():
    adapter, connection = warehouse()
    adapter.create()
    assert connection.sql[:2] == [
        "CREATE SCHEMA `training`.`episode_raw`",
        "CREATE SCHEMA `training`.`episode_transform`",
    ]
    frame = adapter.query("WITH cars AS (SELECT * FROM raw.cars) SELECT id FROM cars")
    assert frame.to_dict("records") == [{"id": 1}]
    assert "`training`.`episode_raw`.cars" in connection.sql[-1]
    assert "FROM cars" in connection.sql[-1]
    adapter.close()
    adapter.reopen()
    assert adapter.namespace == "episode"
    adapter.cleanup()
    assert "DROP SCHEMA `training`.`episode_raw` CASCADE" in connection.sql
    assert "DROP SCHEMA `training`.`episode_transform` CASCADE" in connection.sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM other.cars",
        "SELECT * FROM other.raw.cars",
        "SELECT * FROM episode_other.cars",
        "DROP SCHEMA raw",
        "SELECT 1; SELECT 2",
        "SELECT * FROM read_files('https://example.com/private')",
    ],
)
def test_query_rejects_cross_namespace_and_non_select(sql):
    adapter, connection = warehouse()
    with pytest.raises((PolicyError, ValueError)):
        adapter.query(sql)
    assert connection.sql == []


def test_partial_create_does_not_drop_preexisting_foreign_schema():
    adapter, connection = warehouse(
        failure=(
            "CREATE SCHEMA `training`.`episode_transform`",
            RuntimeError("private-secret already exists"),
        )
    )
    with pytest.raises(InfrastructureError) as error:
        adapter.create()
    assert "private-secret" not in str(error.value)
    assert "DROP SCHEMA `training`.`episode_raw` CASCADE" in connection.sql
    assert "DROP SCHEMA `training`.`episode_transform` CASCADE" not in connection.sql


def test_query_errors_are_candidate_failures_and_transport_stays_infrastructure():
    adapter, _ = warehouse(
        failure=("SELECT", ServerOperationError("private-secret missing table"))
    )
    with pytest.raises(ValueError, match="rejected"):
        adapter.query("SELECT * FROM raw.cars")
    adapter, _ = warehouse(failure=("SELECT", RequestError("private-secret network")))
    with pytest.raises(InfrastructureError) as error:
        adapter.query("SELECT 1")
    assert "private-secret" not in str(error.value)


def test_structured_sqlstate_permission_is_infrastructure_and_literal_text_is_not():
    error = ServerOperationError("private-secret permission")
    error.sql_state = "42501"
    adapter, _ = warehouse(failure=("SELECT", error))
    with pytest.raises(InfrastructureError, match="permission"):
        adapter.query("SELECT * FROM raw.cars")
    error = ServerOperationError("PERMISSION_DENIED injected literal")
    error.sql_state = "22000"
    adapter, _ = warehouse(failure=("SELECT", error))
    with pytest.raises(ValueError, match="candidate query"):
        adapter.query("SELECT * FROM raw.cars")


def test_trusted_dbt_profile_has_oauth_and_statement_timeout():
    adapter, _ = warehouse()
    profile = adapter.dbt_profile()
    assert profile["type"] == "databricks"
    assert profile["auth_type"] == "oauth"
    assert profile["schema"] == "episode_transform"
    assert profile["session_properties"]["STATEMENT_TIMEOUT"] == "30"
    airbyte_config = AirbyteIngestor._databricks_configuration(adapter)
    assert airbyte_config["authentication"]["auth_type"] == "OAUTH"
    assert airbyte_config["authentication"]["secret"] == adapter.config.client_secret
    assert "client_secret" not in airbyte_config["authentication"]


class FakeControlPlane:
    def __init__(self, *, failed=False):
        self.calls = []
        self.failed = failed
        self.cancel_failed = False
        self._deadline = None
        self._clock = lambda: 0
        self.config = SimpleNamespace(cancel_timeout=60)

    def set_deadline(self, value):
        self._deadline = value

    def cancel(self, value):
        self.calls.append(("jobs", "cancel", {"id": value}))
        if self.cancel_failed:
            raise AirbyteError("cancellation not terminal")

    def definition(self, kind, repository):
        return {kind + "DefinitionId": kind + "-definition"}

    def create_destination(self, *args):
        return "destination"

    def check_destination(self, value):
        pass

    def create_source(self, *args):
        return "source"

    def check_source(self, value):
        pass

    def discover(self, value):
        return {"streams": []}, None

    def create_connection(self, *args, **kwargs):
        return "connection"

    def sync(self, value):
        return 99

    def poll(self, value):
        if self.failed:
            raise AirbyteError("sync failed")

    def delete(self, kind, value):
        self.calls.append(("delete", kind, value))

    def rpc(self, family, method, body):
        self.calls.append((family, method, body))
        return {}


def test_ingestion_receipt_reads_actual_warehouse_and_cleanup_cancels_before_actors(
    tmp_path,
):
    api = FakeControlPlane()
    ingestor = AirbyteIngestor(
        api, project_root=tmp_path, destination_configuration=lambda _: {}
    )
    source = SourceSpec(
        "train",
        "csv",
        "unused",
        (ColumnSpec("id", dtype="BIGINT"),),
        official_locator="https://example.com/train.csv",
    )
    adapter, _ = warehouse()
    adapter.table = lambda *args: pd.DataFrame(
        {"id": [1, 2], "_airbyte_meta": ["{}", "{}"]}
    )
    receipt = ingestor.load(source, adapter)
    assert receipt.rows == 2
    assert receipt.columns == ("id",)
    assert receipt.external_job_id == "99"
    api.failed = True
    with pytest.raises(AirbyteError):
        ingestor.load(source, adapter)
    ingestor.cleanup("episode")
    assert api.calls[0] == ("jobs", "cancel", {"id": 99})
    assert [item[1] for item in api.calls[1:]] == [
        "connection",
        "source",
        "destination",
    ]


def test_nonterminal_cancel_preserves_owned_resources_without_deleting_actors(tmp_path):
    api = FakeControlPlane()
    api.cancel_failed = True
    api._deadline = -1  # Cleanup has a separate budget after episode expiry.
    ingestor = AirbyteIngestor(api, project_root=tmp_path)
    ingestor._jobs = {"episode": {99}}
    ingestor._resources = {
        "episode": [("connection", "connection"), ("source", "source")]
    }
    with pytest.raises(AirbyteError, match="retained"):
        ingestor.cleanup("episode")
    assert ingestor._jobs == {"episode": {99}}
    assert len(ingestor._resources["episode"]) == 2
    assert api.calls == [("jobs", "cancel", {"id": 99})]
    assert api._deadline == -1


def test_missing_config_never_contacts_cloud(monkeypatch):
    for name in (
        "DATABRICKS_HOST",
        "DATABRICKS_HTTP_PATH",
        "DATABRICKS_CATALOG",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "AIRBYTE_CLIENT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(InfrastructureError, match="Missing Databricks"):
        DatabricksConfig.from_env()
    with pytest.raises(InfrastructureError, match="Missing Airbyte"):
        AirbyteConfig.from_env()


def test_destination_hook_selects_repository_without_changing_ingestor(tmp_path):
    class OtherDestination:
        namespace = "other_episode"
        airbyte_repository = "airbyte/destination-postgres"

        def airbyte_configuration(self):
            return {"database": "training"}

    class ControlPlane(FakeControlPlane):
        def definition(self, kind, repository):
            assert kind == "destination"
            assert repository == "airbyte/destination-postgres"
            return {"destinationDefinitionId": "postgres-definition"}

        def create_destination(self, name, definition_id, configuration):
            assert configuration == {"database": "training"}
            return "owned-postgres"

    ingestor = AirbyteIngestor(ControlPlane(), project_root=tmp_path)
    assert ingestor._destination(OtherDestination()) == "owned-postgres"


def test_actor_creation_lost_response_reconciles_owned_identity(tmp_path):
    api, _ = client(
        [
            Response(
                payload={
                    "sources": [
                        {
                            "sourceId": "owned-source",
                            "name": "episode-train",
                            "sourceDefinitionId": "definition",
                        }
                    ]
                }
            )
        ]
    )
    ingestor = AirbyteIngestor(api, project_root=tmp_path)
    calls = []

    def create():
        calls.append(1)
        raise AirbyteError("transport (mutation outcome unknown)")

    result = ingestor._create_owned(
        "episode",
        "source",
        "episode-train",
        {"sourceDefinitionId": "definition"},
        create,
    )
    assert result == "owned-source"
    assert calls == [1]
    assert ingestor.cleanup_ownership("episode")["resources"] == [
        ("source", "owned-source")
    ]
    assert not ingestor.cleanup_ownership("episode")["pending_actors"]


def test_unresolved_actor_creation_never_blindly_recreates_and_cleanup_recovers(
    tmp_path,
):
    api, _ = client(
        [
            Response(payload={"sources": []}),
            Response(payload={"sources": []}),
            Response(
                payload={
                    "sources": [
                        {
                            "sourceId": "owned-source",
                            "name": "episode-train",
                            "sourceDefinitionId": "definition",
                        }
                    ]
                }
            ),
            Response(204),
        ]
    )
    ingestor = AirbyteIngestor(api, project_root=tmp_path)
    calls = []

    def create():
        calls.append(1)
        raise AirbyteError("transport (mutation outcome unknown)")

    for _ in range(2):
        with pytest.raises(AirbyteError):
            ingestor._create_owned(
                "episode",
                "source",
                "episode-train",
                {"sourceDefinitionId": "definition"},
                create,
            )
    assert calls == [1]
    assert ingestor.cleanup_ownership("episode")["pending_actors"] == [
        {
            "kind": "source",
            "name": "episode-train",
            "identity": {"sourceDefinitionId": "definition"},
        }
    ]
    ingestor.cleanup("episode")
    assert not ingestor.cleanup_ownership("episode")["pending_actors"]
    assert not ingestor.cleanup_ownership("episode")["resources"]


def test_job_submission_lost_response_reconciles_new_job_before_deleting_connection(
    tmp_path,
):
    api = FakeControlPlane()
    api.connection_jobs = lambda _: [JobSummary(100, "running", 0)]
    ingestor = AirbyteIngestor(api, project_root=tmp_path)
    ingestor._pending_sync = {"episode": {"connection": 99}}
    ingestor._resources = {"episode": [("connection", "connection")]}
    ingestor.cleanup("episode")
    assert api.calls == [
        ("jobs", "cancel", {"id": 100}),
        ("delete", "connection", "connection"),
    ]


def test_unknown_job_submission_without_new_job_preserves_ownership(tmp_path):
    api = FakeControlPlane()
    api.connection_jobs = lambda _: [JobSummary(99, "succeeded", 1)]
    ingestor = AirbyteIngestor(api, project_root=tmp_path)
    ingestor._pending_sync = {"episode": {"connection": 99}}
    ingestor._resources = {"episode": [("connection", "connection")]}
    with pytest.raises(AirbyteError):
        ingestor.cleanup("episode")
    assert not api.calls
    assert ingestor.cleanup_ownership("episode")["pending_sync"] == {"connection": 99}


def test_actor_reconciliation_rejects_similar_name_and_definition_mismatch():
    api, _ = client(
        [
            Response(
                payload={
                    "destinations": [
                        {
                            "name": "foreign-episode",
                            "destinationDefinitionId": "definition",
                            "destinationId": "foreign",
                        },
                        {
                            "name": "episode",
                            "destinationDefinitionId": "other-definition",
                            "destinationId": "foreign",
                        },
                    ]
                }
            )
        ]
    )
    assert (
        api.reconcile_actor(
            "destination", "episode", {"destinationDefinitionId": "definition"}
        )
        is None
    )
