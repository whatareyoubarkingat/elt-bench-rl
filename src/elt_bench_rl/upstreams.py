"""Resolve pinned, clean upstream checkouts outside the project, offline."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .errors import InfrastructureError

_URLS = {
    "elt-bench": "https://github.com/uiuc-kang-lab/ELT-Bench.git",
    "tinker-cookbook": "https://github.com/thinking-machines-lab/tinker-cookbook.git",
}
_PREPARE = "Run .venv/bin/python scripts/fetch_upstreams.py from the project root."


def _git(checkout: Path, *arguments: str) -> str:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(checkout), *arguments],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise InfrastructureError(
            f"Upstream Git verification unavailable. {_PREPARE}"
        ) from error
    if result.returncode:
        raise InfrastructureError(f"Upstream checkout verification failed. {_PREPARE}")
    return result.stdout.strip()


def resolve_upstream(project_root: Path | str, name: str) -> Path:
    """Check the manifest, external path, exact Git revision and clean state.

    This function never fetches, clones, updates or repairs a checkout. Each
    call verifies current state, so a previous successful check cannot mask a
    later modification. Git's optional index-refresh writes are disabled.
    """
    root = Path(project_root).resolve()
    if name not in _URLS:
        raise InfrastructureError("Unsupported upstream name")
    try:
        entry = json.loads((root / "configs/upstream-revisions.json").read_text())[name]
        revision = entry["revision"]
        if (
            not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or entry.get("url") != _URLS[name]
            or entry.get("cache_path") != f"{name}/{revision}"
        ):
            raise ValueError("Invalid upstream identity")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise InfrastructureError(
            f"Pinned upstream manifest is invalid. {_PREPARE}"
        ) from error

    configured = os.environ.get("ELT_BENCH_RL_UPSTREAM_CACHE")
    cache = (
        Path(configured).expanduser()
        if configured is not None
        else Path.home() / ".cache/elt-bench-rl/upstreams"
    )
    if not cache.is_absolute():
        raise InfrastructureError(
            "Upstream cache root must be an absolute external path"
        )
    cache = cache.resolve()
    if cache.is_relative_to(root):
        raise InfrastructureError("Upstream cache root must be outside the project")
    checkout = (cache / entry["cache_path"]).resolve()
    if checkout.is_relative_to(root) or not checkout.is_relative_to(cache):
        raise InfrastructureError(
            "Upstream checkout escapes the external cache boundary"
        )
    if not checkout.is_dir():
        raise InfrastructureError(f"Pinned upstream cache is missing. {_PREPARE}")
    if Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout:
        raise InfrastructureError(f"Upstream path is not a repository root. {_PREPARE}")
    if _git(checkout, "rev-parse", "HEAD") != revision:
        raise InfrastructureError(
            f"Upstream revision differs from the manifest pin. {_PREPARE}"
        )
    if _git(checkout, "remote", "get-url", "origin") != entry["url"]:
        raise InfrastructureError(
            f"Upstream origin differs from the official URL. {_PREPARE}"
        )
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise InfrastructureError(f"Pinned upstream checkout is not clean. {_PREPARE}")
    return checkout
