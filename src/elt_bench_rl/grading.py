"""Trusted, fresh-result comparison for ELT rewards.

Call this module outside the model sandbox. It never reads candidate CSV caches
and its diagnostics contain counts, not reference values or primary keys.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import InfrastructureError

OFFICIAL_BENCHMARK_REVISION = "fcf3129df49ce5ce63ff46b2bbbbba1eaf9ec055"
_OFFICIAL_STAGE2_SHA256 = (
    "4b0243916d43a7d706b5efc7a059a6cc636ab6c9abce20360a2d585e03862170"
)


@lru_cache(maxsize=1)
def _official_evaluator():
    path = (
        Path(__file__).resolve().parents[2]
        / "third_party/elt-bench/evaluation/eva_stage2.py"
    )
    try:
        source = path.read_bytes()
    except OSError as error:
        raise InfrastructureError("Pinned official evaluator unavailable") from error
    if hashlib.sha256(source).hexdigest() != _OFFICIAL_STAGE2_SHA256:
        raise InfrastructureError(
            "Official evaluator differs from the pinned benchmark"
        )
    spec = importlib.util.spec_from_file_location("_elt_bench_rl_official_stage2", path)
    module = importlib.util.module_from_spec(spec)
    # Execute the exact bytes verified above, not a second filesystem read.
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 -- only SHA-256-verified upstream bytes.
    return module


def official_comparison(
    actual: pd.DataFrame, expected: pd.DataFrame, keys: tuple[str, ...]
) -> bool:
    """Run the pinned official Stage-2 semantics on fresh query results.

    Match the driver's CSV/string/null normalization in memory, then call the
    actual upstream sort_by_keys/check_corretness functions. No candidate cache
    is read or written. This metric is deliberately separate from the stricter
    reward: the official evaluator ignores extra columns and tolerates numeric
    key differences, and does not enforce primary-key uniqueness.
    """
    evaluator = _official_evaluator()

    def normalize(frame):
        return pd.read_csv(
            io.StringIO(frame.to_csv(index=False)), dtype=str, keep_default_na=True
        )

    reference = normalize(expected)
    try:
        candidate = normalize(actual)
    except pd.errors.EmptyDataError:
        return False
    reference = evaluator.sort_by_keys(reference, list(keys))
    candidate = evaluator.sort_by_keys(candidate, list(keys))
    return bool(evaluator.check_corretness(reference, candidate)["match"])


def official_stage1_loading(
    actual_frames: Mapping[str, pd.DataFrame | None], expected_counts: Mapping[str, int]
) -> bool:
    """Official Stage-1 table-presence/row-count check on fresh base tables.

    The adapter must establish schema presence and supply base-table frames;
    missing tables are absent or None. Extra tables, columns and wrong values
    do not affect this official metric. Trusted receipts and content checks
    belong to the stronger loading reward, not to official Stage 1.
    """
    counts = {str(name).lower(): count for name, count in expected_counts.items()}
    if len(counts) != len(expected_counts):
        raise ValueError("Ambiguous expected table names")
    frames: dict[str, list[pd.DataFrame | None]] = defaultdict(list)
    for name, frame in actual_frames.items():
        frames[str(name).lower()].append(frame)
    for name, count in counts.items():
        candidates = frames.get(name, [])
        if len(candidates) != 1 or not isinstance(candidates[0], pd.DataFrame):
            return False
        if len(candidates[0]) != count:
            return False
    return True


@dataclass(frozen=True)
class Comparison:
    score: float
    strict: bool
    details: dict[str, Any]


def _is_null(value: Any) -> bool:
    if not pd.api.types.is_scalar(value):
        return False
    return bool(pd.isna(value))


def _key_component(value: Any) -> tuple[str, Any] | None:
    if _is_null(value) or not pd.api.types.is_scalar(value):
        return None
    # Keys are exact. Decimal avoids silently collapsing distinct large integer
    # identifiers through float conversion, while matching 1, "1.0", and "1e0".
    text = (
        str(int(value)) if isinstance(value, (bool, np.bool_)) else str(value).strip()
    )
    try:
        number = Decimal(text)
        if number.is_finite():
            return ("number", number)
    except InvalidOperation:
        pass
    return ("text", text.lower())


def _columns(frame: pd.DataFrame) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, column in enumerate(frame.columns):
        positions[str(column).lower()].append(index)
    return dict(positions)


def _rows(
    frame: pd.DataFrame,
    columns: dict[str, list[int]],
    keys: tuple[str, ...],
    side: str,
) -> tuple[dict[tuple, list[int]], int]:
    positions = [columns[key][0] for key in keys if len(columns.get(key, [])) == 1]
    valid_columns = len(positions) == len(keys)
    groups: dict[tuple, list[int]] = defaultdict(list)
    invalid = 0
    if valid_columns:
        key_values = frame.iloc[:, positions].itertuples(index=False, name=None)
    else:
        key_values = (None for _ in range(len(frame)))
    for index, values in enumerate(key_values):
        token = (
            tuple(_key_component(value) for value in values)
            if values is not None
            else ()
        )
        if not valid_columns or None in token:
            # Invalid keys cannot match another invalid key, even across frames.
            token = (("invalid", side, index),)
            invalid += 1
        groups[token].append(index)
    return dict(groups), invalid


def _numeric_column(series: pd.Series) -> pd.Series | None:
    converted = pd.to_numeric(series, errors="coerce")
    if converted.isna().sum() != series.isna().sum():
        return None
    if pd.api.types.is_complex_dtype(converted.dtype):
        return None
    return converted


def _cell_matches(expected: Any, actual: Any) -> bool:
    expected_null, actual_null = _is_null(expected), _is_null(actual)
    if expected_null or actual_null:
        return expected_null and actual_null
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    if not pd.api.types.is_scalar(expected) or not pd.api.types.is_scalar(actual):
        return False
    return bool(expected == actual)


def compare_frames(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: tuple[str, ...],
) -> Comparison:
    """Score fresh frames against a trusted reference, without exposing it.

    Column names ignore case. Keys normalize numeric spelling and trim/lower
    textual values, but never use numerical tolerance. A key with duplicate
    rows on either side receives no matching cells. Null or missing keys also
    cannot match. Non-key cells retain official ELT-Bench comparison semantics:
    an entirely numeric column uses rtol=1e-2/atol=1e-9; paired nulls match;
    text values compare after trim/lower.

    The denominator is the Cartesian product of the union of row keys and the
    union of column names, with maximum multiplicity on each side. Extra rows,
    columns and duplicates therefore cost cells instead of disappearing. Case
    collisions in column names are ambiguous and do not match. ``strict`` also
    requires identical row/column sets and valid, unique primary keys.

    A malformed trusted reference (missing configured key columns) raises
    ValueError, rather than becoming a training reward. Candidate mistakes
    produce a scored failure. Neither input is modified.
    """
    if not isinstance(actual, pd.DataFrame) or not isinstance(expected, pd.DataFrame):
        raise TypeError("actual and expected must be pandas DataFrames")
    normalized_keys = tuple(str(key).lower() for key in keys)
    if not normalized_keys or len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("keys must contain distinct primary-key columns")

    actual_columns, expected_columns = _columns(actual), _columns(expected)
    if any(key not in expected_columns for key in normalized_keys):
        raise ValueError("reference is missing configured primary-key columns")
    actual_rows, actual_invalid = _rows(
        actual, actual_columns, normalized_keys, "actual"
    )
    expected_rows, expected_invalid = _rows(
        expected, expected_columns, normalized_keys, "expected"
    )
    row_keys = actual_rows.keys() | expected_rows.keys()
    column_names = actual_columns.keys() | expected_columns.keys()
    union_rows = sum(
        max(len(actual_rows.get(key, [])), len(expected_rows.get(key, [])))
        for key in row_keys
    )
    union_columns = sum(
        max(len(actual_columns.get(column, [])), len(expected_columns.get(column, [])))
        for column in column_names
    )
    matched_rows = [
        (expected_rows[key][0], actual_rows[key][0])
        for key in actual_rows.keys() & expected_rows.keys()
        if len(actual_rows[key]) == len(expected_rows[key]) == 1
    ]

    matched_cells = 0
    expected_indices = [row[0] for row in matched_rows]
    actual_indices = [row[1] for row in matched_rows]
    for column in actual_columns.keys() & expected_columns.keys():
        if len(actual_columns[column]) != 1 or len(expected_columns[column]) != 1:
            continue
        if column in normalized_keys:
            # A matched composite key already establishes exact equality.
            matched_cells += len(matched_rows)
            continue
        actual_values = actual.iloc[:, actual_columns[column][0]]
        expected_values = expected.iloc[:, expected_columns[column][0]]
        actual_numeric = _numeric_column(actual_values)
        expected_numeric = _numeric_column(expected_values)
        numeric = actual_numeric is not None and expected_numeric is not None
        if numeric:
            actual_numbers = actual_numeric.iloc[actual_indices].to_numpy(
                dtype=float, na_value=np.nan
            )
            expected_numbers = expected_numeric.iloc[expected_indices].to_numpy(
                dtype=float, na_value=np.nan
            )
            # Preserve the official argument order, tolerances and paired NaNs.
            matched_cells += int(
                np.count_nonzero(
                    np.isclose(
                        expected_numbers,
                        actual_numbers,
                        rtol=1e-2,
                        atol=1e-9,
                        equal_nan=True,
                    )
                )
            )
            continue
        for expected_index, actual_index in matched_rows:
            matched_cells += int(
                _cell_matches(
                    expected_values.iloc[expected_index],
                    actual_values.iloc[actual_index],
                )
            )

    actual_duplicate_rows = sum(
        len(rows) for rows in actual_rows.values() if len(rows) > 1
    )
    expected_duplicate_rows = sum(
        len(rows) for rows in expected_rows.values() if len(rows) > 1
    )
    actual_duplicate_columns = sum(
        len(positions) for positions in actual_columns.values() if len(positions) > 1
    )
    expected_duplicate_columns = sum(
        len(positions) for positions in expected_columns.values() if len(positions) > 1
    )
    total_cells = union_rows * union_columns
    same_structure = (
        actual_rows.keys() == expected_rows.keys()
        and actual_columns.keys() == expected_columns.keys()
        and not any(
            (
                actual_invalid,
                expected_invalid,
                actual_duplicate_rows,
                expected_duplicate_rows,
                actual_duplicate_columns,
                expected_duplicate_columns,
            )
        )
    )
    strict = same_structure and matched_cells == total_cells
    score = matched_cells / total_cells if total_cells else float(strict)
    details = {
        "matched_cells": matched_cells,
        "total_cells": total_cells,
        "actual_rows": len(actual),
        "expected_rows": len(expected),
        "union_rows": union_rows,
        "actual_columns": len(actual.columns),
        "expected_columns": len(expected.columns),
        "union_columns": union_columns,
        "matched_key_rows": len(matched_rows),
        "missing_rows": sum(
            len(expected_rows[key]) for key in expected_rows.keys() - actual_rows.keys()
        ),
        "extra_rows": sum(
            len(actual_rows[key]) for key in actual_rows.keys() - expected_rows.keys()
        ),
        "missing_columns": sum(
            len(expected_columns[key])
            for key in expected_columns.keys() - actual_columns.keys()
        ),
        "extra_columns": sum(
            len(actual_columns[key])
            for key in actual_columns.keys() - expected_columns.keys()
        ),
        "actual_duplicate_key_rows": actual_duplicate_rows,
        "expected_duplicate_key_rows": expected_duplicate_rows,
        "actual_invalid_key_rows": actual_invalid,
        "expected_invalid_key_rows": expected_invalid,
        "actual_duplicate_columns": actual_duplicate_columns,
        "expected_duplicate_columns": expected_duplicate_columns,
    }
    return Comparison(score=float(score), strict=bool(strict), details=details)
