"""Capability checks for model SQL, before handing it to any engine."""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from .errors import PolicyError

SAFE_FUNCTIONS = {
    "ABS",
    "AVG",
    "CAST",
    "CEIL",
    "CEILING",
    "COALESCE",
    "CONCAT",
    "COUNT",
    "FLOOR",
    "GREATEST",
    "IF",
    "LEAST",
    "LENGTH",
    "LOWER",
    "MAX",
    "MIN",
    "NULLIF",
    "ROUND",
    "ROW_NUMBER",
    "SUM",
    "TRIM",
    "UPPER",
    "SUBSTRING",
    "CASE",
    "AND",
    "OR",
    "IFNULL",
    "TRY_CAST",
    "DENSE_RANK",
    "RANK",
}
JINJA_REFERENCE = re.compile(
    r"\{\{\s*(source|ref)\(\s*(['\"])([a-z][a-z0-9_]*)\2"
    r"(?:\s*,\s*(['\"])([a-z][a-z0-9_]*)\4)?\s*\)\s*\}\}"
)


def render_references(sql: str, sources: set[str], models: set[str]) -> str:
    def replace(match):
        kind, _, first, _, second = match.groups()
        if kind == "source" and first == "raw" and second in sources:
            return f'raw."{second}"'
        if kind == "ref" and first in models and second is None:
            return f'transform."{first}"'
        raise PolicyError("Unknown or disallowed dbt reference")

    rendered = JINJA_REFERENCE.sub(replace, sql)
    if any(token in rendered for token in ("{{", "}}", "{%", "%}", "{#", "#}")):
        raise PolicyError("Only literal source/ref Jinja expressions are allowed")
    return rendered


def validate_sql(
    sql: str, sources: set[str], models: set[str], *, dialect: str = "duckdb"
) -> str:
    if not isinstance(sql, str) or len(sql) > 32000:
        raise PolicyError("SQL must be a string of at most 32000 characters")
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.SqlglotError as error:
        raise ValueError("SQL parse error: " + str(error)[:500]) from error
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union)):
        raise PolicyError("Only one SELECT query is allowed")
    tree = statements[0]
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier) or table.catalog:
            raise PolicyError(
                "External tables, catalogs and table functions are forbidden"
            )
        schema, name = table.db.lower(), table.name.lower()
        if not schema and name in ctes:
            continue
        if (schema == "raw" and name in sources) or (
            schema == "transform" and name in models
        ):
            continue
        raise PolicyError(
            "Query can access only this episode's raw sources and transform models"
        )
    for func in tree.find_all(exp.Func):
        name = (
            func.name.upper()
            if isinstance(func, exp.Anonymous)
            else func.sql_name().upper()
        )
        if name not in SAFE_FUNCTIONS:
            raise PolicyError(
                "SQL function is not on the capability allowlist: " + name
            )
    # Block model-controlled DB settings, unbounded recursion and engine commands.
    if any(isinstance(n, (exp.Command, exp.Into, exp.Parameter)) for n in tree.walk()):
        raise PolicyError("Engine commands and parameters are forbidden")
    if any(node.args.get("recursive") for node in tree.find_all(exp.With)):
        raise PolicyError("Recursive CTEs are disabled")
    return sql
