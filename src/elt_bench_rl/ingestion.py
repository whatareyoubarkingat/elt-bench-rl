"""Real CSV/HTTP ingestion; never pre-populates an episode database."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import pandas as pd
import requests

from .errors import InfrastructureError
from .task import SourceSpec

AIRBYTE_METADATA_COLUMNS = frozenset(
    {
        "_airbyte_raw_id",
        "_airbyte_extracted_at",
        "_airbyte_loaded_at",
        "_airbyte_meta",
        "_airbyte_generation_id",
        "_airbyte_emitted_at",
        "_airbyte_ab_id",
        "_airbyte_normalized_at",
    }
)


@dataclass(frozen=True)
class LoadReceipt:
    source: str
    loader: str
    rows: int
    columns: tuple[str, ...]
    content_sha256: str
    external_job_id: str | None = None


def content_digest(frame: pd.DataFrame) -> str:
    # Receipt is diagnostic. The verifier always reads fresh warehouse data.
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def coerce_source(frame: pd.DataFrame, source: SourceSpec) -> pd.DataFrame:
    expected = [c.name for c in source.columns]
    if set(frame.columns) != set(expected) or len(frame.columns) != len(expected):
        raise ValueError(
            "Extracted source schema differs from the public specification"
        )
    frame = frame[expected].copy()
    for column in source.columns:
        if column.dtype == "BIGINT":
            frame[column.name] = pd.to_numeric(
                frame[column.name], errors="raise"
            ).astype("Int64")
        else:
            frame[column.name] = frame[column.name].astype("string")
    return frame


class LocalIngestor:
    name = "local_csv_http"

    def __init__(
        self, *, timeout: float = 10, max_bytes: int = 8_000_000, retries: int = 1
    ):
        self.timeout, self.max_bytes, self.retries = timeout, max_bytes, retries

    def extract(self, source: SourceSpec) -> pd.DataFrame:
        if source.kind == "csv":
            from pathlib import Path

            path = Path(source.locator)
            try:
                if path.stat().st_size > self.max_bytes:
                    raise InfrastructureError("Source exceeds extraction byte limit")
                frame = pd.read_csv(io.BytesIO(path.read_bytes()))
            except OSError as error:
                raise InfrastructureError(
                    "Verified CSV source mirror unavailable"
                ) from error
        elif source.kind == "http":
            # Endpoint comes exclusively from trusted TaskSpec. No model URLs.
            for attempt in range(self.retries + 1):
                try:
                    with requests.Session() as session:
                        session.trust_env = False
                        with session.get(
                            source.locator,
                            timeout=self.timeout,
                            stream=True,
                            allow_redirects=False,
                        ) as response:
                            response.raise_for_status()
                            payload = bytearray()
                            for chunk in response.iter_content(65536):
                                payload.extend(chunk)
                                if len(payload) > self.max_bytes:
                                    raise InfrastructureError(
                                        "Source exceeds extraction byte limit"
                                    )
                    import json

                    records = json.loads(payload)
                    if not isinstance(records, list):
                        raise InfrastructureError(
                            "Source API returned an unexpected JSON shape"
                        )
                    frame = pd.DataFrame(records)
                    break
                except (requests.RequestException, OSError) as error:
                    if attempt == self.retries:
                        raise InfrastructureError(
                            "HTTP source unavailable after bounded retries"
                        ) from error
        else:
            raise ValueError("Unknown source kind")
        return coerce_source(frame, source)

    def load(self, source: SourceSpec, adapter) -> LoadReceipt:
        frame = self.extract(source)
        adapter.write_source(source.name, frame)
        return LoadReceipt(
            source.name,
            self.name,
            len(frame),
            tuple(frame.columns),
            content_digest(frame),
        )
