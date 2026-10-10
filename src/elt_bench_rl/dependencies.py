"""Minimum source-lineage requirements for logical pipeline submissions.

This rejects obvious constant answers and disconnected loading. It is not a
proof of semantic dependence: dummy joins or scalar reads can still satisfy a
syntactic lineage check. Clean replay and held-out tasks remain necessary.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, build_scope

from .errors import PolicyError
from .sql_policy import JINJA_REFERENCE, render_references, validate_sql
from .task import identifier


def _direct_dependencies(sql: str, dialect: str) -> set[tuple[str, str]]:
    root = build_scope(sqlglot.parse_one(sql, read=dialect))
    if root is None:
        raise PolicyError("Model does not have a query scope")
    dependencies: set[tuple[str, str]] = set()
    visited: set[int] = set()

    def visit(scope: Scope) -> None:
        if id(scope) in visited:
            return
        visited.add(id(scope))
        # Trigger the optimizer's duplicate-alias validation. Use references
        # below as well because selected_sources excludes SEMI/ANTI join RHS.
        _ = scope.selected_sources
        for name, _node in scope.references:
            source = scope.sources.get(name)
            if isinstance(source, Scope):
                visit(source)
            elif isinstance(source, exp.Table):
                schema, table = source.db.lower(), source.name.lower()
                if schema in {"raw", "transform"}:
                    dependencies.add((schema, table))
        for child in (*scope.union_scopes, *scope.subquery_scopes):
            visit(child)

    # Unused CTE definitions are not visited: their mere presence cannot meet
    # the required output models' source-coverage check.
    visit(root)
    return dependencies


def validate_dependencies(
    models: dict[str, str],
    sources: set[str],
    targets: set[str],
    dialect: str = "duckdb",
) -> None:
    """Require acyclic models and transitive raw-source coverage by targets.

    Every submitted model must ultimately reference a source; the union of the
    requested targets' dependencies must cover every task source. Literal
    dbt source/ref expressions and explicit raw/transform table names both
    count, with SQL scopes resolving CTEs and subqueries. Inputs are unchanged.
    """
    for name in set(models) | sources | targets:
        identifier(name)
    if not targets or not targets <= set(models):
        raise PolicyError("Submission must contain every requested target model")

    dependencies: dict[str, set[tuple[str, str]]] = {}
    cycle_dependencies: dict[str, set[str]] = {}
    for name, sql in models.items():
        rendered = render_references(sql, sources, set(models))
        validate_sql(rendered, sources, set(models), dialect=dialect)
        try:
            dependencies[name] = _direct_dependencies(rendered, dialect)
            cycle_dependencies[name] = {
                table.name.lower()
                for table in sqlglot.parse_one(rendered, read=dialect).find_all(
                    exp.Table
                )
                if table.db.lower() == "transform"
            }
            # dbt processes ref() even in unused CTEs and SQL comments; those
            # edges must not evade cycle rejection by being unreachable SQL.
            cycle_dependencies[name].update(
                match.group(3)
                for match in JINJA_REFERENCE.finditer(sql)
                if match.group(1) == "ref"
            )
        except sqlglot.errors.SqlglotError as error:
            raise PolicyError("Model dependencies cannot be resolved") from error

    visiting: set[str] = set()
    resolved: dict[str, set[str]] = {}
    cycle_checked: set[str] = set()

    def check_cycles(model: str) -> None:
        if model in visiting:
            raise PolicyError("Cyclic model dependencies are forbidden")
        if model in cycle_checked:
            return
        visiting.add(model)
        for dependency in cycle_dependencies[model]:
            check_cycles(dependency)
        visiting.remove(model)
        cycle_checked.add(model)

    for model in models:
        check_cycles(model)

    def raw_sources(model: str) -> set[str]:
        if model in visiting:
            raise PolicyError("Cyclic model dependencies are forbidden")
        if model in resolved:
            return resolved[model]
        visiting.add(model)
        found = set()
        for schema, dependency in dependencies[model]:
            if schema == "raw":
                found.add(dependency)
            else:
                found.update(raw_sources(dependency))
        visiting.remove(model)
        if not found:
            raise PolicyError("Every model must ultimately depend on a task source")
        resolved[model] = found
        return found

    for model in models:
        raw_sources(model)
    covered = set().union(*(resolved[target] for target in targets))
    if covered != sources:
        raise PolicyError(
            "Requested target models must collectively depend on every task source"
        )
