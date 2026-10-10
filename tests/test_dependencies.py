import pytest

from elt_bench_rl.dependencies import validate_dependencies
from elt_bench_rl.errors import PolicyError


def test_jinja_ref_chain_covers_sources():
    models = {
        "cars_summary": "SELECT train_id, COUNT(*) AS n FROM {{ source('raw', 'cars') }} GROUP BY train_id",
        "trains": "SELECT t.id, c.n FROM {{ source('raw', 'train') }} t JOIN {{ ref('cars_summary') }} c ON t.id = c.train_id",
    }
    original = models.copy()
    validate_dependencies(models, {"cars", "train"}, {"trains"})
    assert models == original


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 AS id",
        "SELECT 1 AS id UNION ALL SELECT 2 AS id",
        "WITH unused AS (SELECT * FROM raw.cars) SELECT 1 AS id",
    ],
)
def test_pure_constant_output_is_rejected(sql):
    with pytest.raises(PolicyError, match="depend on a task source"):
        validate_dependencies({"trains": sql}, {"cars"}, {"trains"})


@pytest.mark.parametrize(
    "models",
    [
        {"trains": "SELECT * FROM transform.trains"},
        {
            "trains": "SELECT * FROM transform.helper",
            "helper": "SELECT * FROM transform.trains",
        },
        {
            "trains": "SELECT * FROM raw.cars",
            "a": "SELECT * FROM transform.b",
            "b": "SELECT * FROM transform.a",
        },
    ],
)
def test_cycles_in_any_submitted_model_are_rejected(models):
    with pytest.raises(PolicyError, match="Cyclic"):
        validate_dependencies(models, {"cars"}, {"trains"})


def test_target_union_must_cover_every_task_source():
    with pytest.raises(PolicyError, match="every task source"):
        validate_dependencies(
            {"trains": "SELECT * FROM raw.train"}, {"train", "cars"}, {"trains"}
        )


def test_disconnected_non_target_model_does_not_supply_missing_source():
    models = {"trains": "SELECT * FROM raw.train", "unused": "SELECT * FROM raw.cars"}
    with pytest.raises(PolicyError, match="every task source"):
        validate_dependencies(models, {"train", "cars"}, {"trains"})


def test_unused_cte_does_not_supply_missing_source():
    sql = "WITH unused AS (SELECT * FROM raw.cars) SELECT * FROM raw.train"
    with pytest.raises(PolicyError, match="every task source"):
        validate_dependencies({"trains": sql}, {"train", "cars"}, {"trains"})


def test_multiple_requested_targets_can_collectively_cover_sources():
    validate_dependencies(
        {
            "cars_model": "SELECT * FROM raw.cars",
            "train_model": "SELECT * FROM raw.train",
        },
        {"train", "cars"},
        {"cars_model", "train_model"},
    )


@pytest.mark.parametrize(
    "sql",
    [
        "WITH used AS (SELECT * FROM raw.cars) SELECT * FROM used JOIN raw.train ON TRUE",
        "SELECT (SELECT COUNT(*) FROM raw.cars) AS n FROM raw.train",
        "SELECT * FROM raw.cars UNION ALL SELECT * FROM raw.train",
        "SELECT * FROM raw.train SEMI JOIN raw.cars ON train.id = cars.train_id",
    ],
)
def test_ctes_subqueries_unions_and_semi_join_references_count(sql):
    validate_dependencies({"trains": sql}, {"train", "cars"}, {"trains"})


def test_missing_target_is_rejected():
    with pytest.raises(PolicyError, match="every requested target"):
        validate_dependencies(
            {"helper": "SELECT * FROM raw.cars"}, {"cars"}, {"trains"}
        )


def test_databricks_dialect_and_quoted_names():
    validate_dependencies(
        {"trains": "SELECT * FROM raw.`train` JOIN raw.`cars` ON TRUE"},
        {"train", "cars"},
        {"trains"},
        dialect="databricks",
    )


def test_dummy_reference_is_a_documented_limit_of_syntactic_lineage():
    # A structural check cannot prove that output values depend on the inputs.
    validate_dependencies(
        {"trains": "SELECT 1 AS id FROM raw.cars LIMIT 1"}, {"cars"}, {"trains"}
    )


@pytest.mark.parametrize(
    "sql",
    [
        "WITH unused AS (SELECT * FROM transform.trains) SELECT * FROM raw.cars",
        "SELECT * FROM raw.cars -- {{ ref('trains') }}",
    ],
)
def test_unused_or_commented_ref_cannot_hide_dbt_cycle(sql):
    with pytest.raises(PolicyError, match="Cyclic"):
        validate_dependencies({"trains": sql}, {"cars"}, {"trains"})


def test_ambiguous_table_aliases_are_rejected():
    with pytest.raises(PolicyError, match="cannot be resolved"):
        validate_dependencies(
            {"trains": "SELECT * FROM raw.train AS t JOIN raw.cars AS t ON TRUE"},
            {"train", "cars"},
            {"trains"},
        )
