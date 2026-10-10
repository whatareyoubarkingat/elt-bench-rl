"""Reward correctness and adversarial structural cases, without real GT."""

import json

import numpy as np
import pandas as pd
import pytest

from elt_bench_rl.grading import (
    compare_frames,
    official_comparison,
    official_stage1_loading,
)


def reference():
    return pd.DataFrame({"id": [1, 2], "value": [100.0, 200.0]})


def test_perfect_reordered_frames_ignore_index_and_column_case():
    expected = reference()
    actual = pd.DataFrame({"VALUE": [200.0, 100.0], "ID": [2, 1]}, index=[42, 19])
    result = compare_frames(actual, expected, ("id",))
    assert result.strict and result.score == 1.0
    assert result.details["matched_cells"] == result.details["total_cells"] == 4
    pd.testing.assert_frame_equal(expected, reference())


@pytest.mark.parametrize("actual,score", [(100.9, 1.0), (102.0, 0.5), (1e-10, 0.5)])
def test_official_relative_tolerance(actual, score):
    result = compare_frames(
        pd.DataFrame({"id": [1], "v": [actual]}),
        pd.DataFrame({"id": [1], "v": [100]}),
        ("id",),
    )
    assert result.score == score
    assert result.strict == (score == 1.0)


def test_absolute_tolerance_and_numeric_strings():
    result = compare_frames(
        pd.DataFrame({"id": ["1.0"], "v": ["0.0000000005"]}),
        pd.DataFrame({"id": [1], "v": [0]}),
        ("id",),
    )
    assert result.strict


@pytest.mark.parametrize("null", [None, np.nan, pd.NA, pd.NaT])
def test_paired_null_values_match(null):
    expected = pd.DataFrame({"id": [1, 2], "v": [None, "West"]})
    actual = pd.DataFrame({"id": [1, 2], "v": [null, " west "]})
    assert compare_frames(actual, expected, ("id",)).strict


def test_unpaired_null_and_wrong_cell_receive_partial_score():
    expected = pd.DataFrame({"id": [1, 2], "v": [None, "east"]})
    actual = pd.DataFrame({"id": [1, 2], "v": ["non-null", "wrong"]})
    result = compare_frames(actual, expected, ("id",))
    assert not result.strict and result.score == 0.5


@pytest.mark.parametrize(
    "operation,score",
    [
        (lambda df: df.iloc[:1], 0.5),
        (lambda df: pd.concat([df, pd.DataFrame({"id": [3], "value": [300]})]), 2 / 3),
        (lambda df: df.drop(columns="value"), 0.5),
        (lambda df: df.assign(extra="ignored"), 2 / 3),
    ],
)
def test_missing_and_extra_rows_or_columns_cost_cells(operation, score):
    result = compare_frames(operation(reference()), reference(), ("id",))
    assert not result.strict and result.score == pytest.approx(score)


def test_duplicate_key_all_copies_are_unmatched_and_count_in_denominator():
    actual = pd.concat([reference(), reference().iloc[:1]], ignore_index=True)
    result = compare_frames(actual, reference(), ("id",))
    assert result.score == pytest.approx(1 / 3)
    assert not result.strict
    assert result.details["actual_duplicate_key_rows"] == 2
    assert result.details["union_rows"] == 3


def test_duplicate_case_colliding_columns_cannot_match():
    actual = pd.DataFrame(
        [[1, 100, 100], [2, 200, 200]], columns=["id", "value", "VALUE"]
    )
    result = compare_frames(actual, reference(), ("id",))
    assert result.score == pytest.approx(1 / 3)
    assert result.details["actual_duplicate_columns"] == 2
    assert not result.strict


def test_null_keys_do_not_match_each_other():
    frame = pd.DataFrame({"id": [None], "v": ["same"]})
    result = compare_frames(frame, frame, ("id",))
    assert result.score == 0 and not result.strict
    assert result.details["union_rows"] == 2


def test_missing_candidate_key_is_scored_failure():
    result = compare_frames(reference().drop(columns="id"), reference(), ("id",))
    assert result.score == 0 and not result.strict
    assert result.details["actual_invalid_key_rows"] == 2


def test_composite_string_keys_use_exact_trimmed_case_insensitive_alignment():
    expected = pd.DataFrame({"tenant": ["East", "West"], "id": [1, 1], "v": [10, 20]})
    actual = pd.DataFrame(
        {"tenant": [" west ", "EAST"], "id": ["1.0", "1e0"], "v": [20, 10]}
    )
    assert compare_frames(actual, expected, ("tenant", "id")).strict


def test_primary_keys_never_use_value_tolerance_or_float_precision():
    expected = pd.DataFrame({"id": [9007199254740992], "v": [1]})
    actual = pd.DataFrame({"id": [9007199254740993], "v": [1]})
    result = compare_frames(actual, expected, ("id",))
    assert result.score == 0 and not result.strict


def test_primary_key_is_not_affected_by_an_extra_malformed_key():
    expected = pd.DataFrame({"id": [1], "v": ["same"]})
    actual = pd.DataFrame({"id": ["1.0", "extra"], "v": ["same", "extra"]})
    assert compare_frames(actual, expected, ("id",)).score == 0.5


def test_nullable_numeric_extension_types_and_infinities():
    expected = pd.DataFrame(
        {"id": [1, 2, 3], "v": pd.Series([pd.NA, 10, np.inf], dtype="Float64")}
    )
    actual = pd.DataFrame(
        {"id": [3, 1, 2], "v": pd.Series([np.inf, pd.NA, 10.01], dtype="Float64")}
    )
    assert compare_frames(actual, expected, ("id",)).strict


def test_empty_tables_match_only_with_equal_structure():
    expected = reference().iloc[:0]
    result = compare_frames(expected.copy(), expected, ("id",))
    assert result.score == 1 and result.strict
    result = compare_frames(expected.assign(extra=[]), expected, ("id",))
    assert result.score == 0 and not result.strict


def test_ambiguous_reference_keys_cannot_produce_success():
    expected = pd.concat([reference(), reference().iloc[:1]], ignore_index=True)
    result = compare_frames(reference(), expected, ("id",))
    assert result.score == pytest.approx(1 / 3) and not result.strict


@pytest.mark.parametrize("keys", [(), ("id", "ID"), ("absent",)])
def test_invalid_reference_key_configuration_raises(keys):
    with pytest.raises(ValueError):
        compare_frames(reference(), reference(), keys)


def test_diagnostics_contain_only_counts_and_no_reference_values():
    expected = pd.DataFrame({"id": [123456789], "v": ["private-reference-value"]})
    result = compare_frames(
        pd.DataFrame({"id": [123456789], "v": ["wrong"]}), expected, ("id",)
    )
    encoded = json.dumps(result.details)
    assert "123456789" not in encoded and "private-reference-value" not in encoded
    assert all(isinstance(value, int) for value in result.details.values())


def test_official_stage2_normalization_and_reordered_fresh_frames():
    expected = pd.DataFrame({"id": [1, 2], "v": [None, "west"], "n": [100, 200]})
    actual = pd.DataFrame(
        {"ID": ["2.0", "1"], "v": [" West ", pd.NA], "n": ["200.9", "100.5"]}
    )
    assert official_comparison(actual, expected, ("id",))


@pytest.mark.parametrize(
    "actual,expected",
    [
        (reference().assign(extra=0), reference()),
        (
            pd.DataFrame({"id": [1, 1], "v": [100, 100]}),
            pd.DataFrame({"id": [1, 1], "v": [100, 100]}),
        ),
        (
            pd.DataFrame({"id": [100.5], "v": [100]}),
            pd.DataFrame({"id": [100], "v": [100]}),
        ),
    ],
)
def test_official_success_does_not_imply_strict_reward(actual, expected):
    assert official_comparison(actual, expected, ("id",))
    assert not compare_frames(actual, expected, ("id",)).strict


def test_official_comparison_uses_fresh_frames_every_time():
    actual, expected = reference(), reference()
    assert official_comparison(actual, expected, ("id",))
    actual.loc[0, "value"] = 999
    assert not official_comparison(actual, expected, ("id",))


def test_official_stage1_checks_counts_not_content():
    actual = pd.DataFrame({"wrong_column": ["wrong", "wrong"]})
    assert official_stage1_loading(
        {"TRAIN": actual, "extra": pd.DataFrame()}, {"train": 2}
    )
    assert not official_stage1_loading({"train": actual.iloc[:1]}, {"train": 2})
    assert not official_stage1_loading({}, {"train": 2})
    assert not official_stage1_loading({"train": None}, {"train": 2})


def test_official_evaluator_pin_is_verified(monkeypatch):
    from elt_bench_rl import grading
    from elt_bench_rl.errors import InfrastructureError

    monkeypatch.setattr(grading, "_OFFICIAL_STAGE2_SHA256", "not-the-source-hash")
    with pytest.raises(InfrastructureError, match="pinned benchmark"):
        official_comparison(reference(), reference(), ("id",))
