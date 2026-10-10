"""Fetch pinned upstream checkouts outside the project using only the standard library.

Existing cache targets are verified without modification. A failed first fetch
retains its newly created directory for inspection; no directory is deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_NAMES = ("elt-bench", "tinker-cookbook")


class CacheError(RuntimeError):
    def __init__(self, stage: str, *, returncode: int | None = None):
        super().__init__(stage)
        self.stage, self.returncode = stage, returncode


def git(path: Path, *arguments: str, timeout: int = 30) -> str:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(path),
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        raise CacheError("git_timeout") from None
    except OSError:
        raise CacheError("git_unavailable") from None
    if result.returncode:
        # Git output may include proxy credentials or authentication details.
        raise CacheError(f"git_{arguments[0]}", returncode=result.returncode)
    return result.stdout.strip()


def cache_root(project_root: Path) -> Path:
    value = os.environ.get("ELT_BENCH_RL_UPSTREAM_CACHE")
    if value is not None and (
        not value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise CacheError("invalid_cache_root")
    path = (
        Path(value).expanduser()
        if value is not None
        else Path.home() / ".cache/elt-bench-rl/upstreams"
    )
    if not path.is_absolute():
        raise CacheError("cache_root_must_be_absolute")
    path = path.resolve()
    if path.is_relative_to(project_root.resolve()):
        raise CacheError("cache_must_be_outside_project")
    return path


def load_upstreams(project_root: Path) -> dict[str, dict]:
    try:
        manifest = json.loads(
            (project_root / "configs/upstream-revisions.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, ValueError):
        raise CacheError("invalid_upstream_manifest") from None
    entries = {}
    for name in UPSTREAM_NAMES:
        entry = manifest.get(name) if isinstance(manifest, dict) else None
        if not isinstance(entry, dict):
            raise CacheError("missing_upstream_entry")
        revision, url = entry.get("revision"), entry.get("url")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise CacheError("invalid_upstream_revision")
        if not isinstance(url, str) or any(
            ord(character) < 32 or ord(character) == 127 for character in url
        ):
            raise CacheError("invalid_upstream_url")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "github.com"
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git", parsed.path)
        ):
            raise CacheError("invalid_upstream_url")
        if entry.get("cache_path") != f"{name}/{revision}":
            raise CacheError("invalid_upstream_cache_path")
        entries[name] = entry
    return entries


def verify_checkout(path: Path, entry: dict) -> None:
    if not path.is_dir():
        raise CacheError("cache_checkout_missing")
    if Path(git(path, "rev-parse", "--show-toplevel")).resolve() != path.resolve():
        raise CacheError("cache_is_not_repository_root")
    if git(path, "rev-parse", "--verify", "HEAD") != entry["revision"]:
        raise CacheError("cache_revision_mismatch")
    if git(path, "remote", "get-url", "origin") != entry["url"]:
        raise CacheError("cache_origin_mismatch")
    if git(path, "status", "--porcelain", "--untracked-files=all"):
        raise CacheError("cache_checkout_not_clean")


def fetch_checkout(path: Path, entry: dict, *, check_only: bool) -> bool:
    # Resolve again so a symlink cannot place source files inside the project.
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise CacheError("cache_must_be_outside_project")
    if not resolved.is_relative_to(cache_root(PROJECT_ROOT)):
        raise CacheError("cache_checkout_escapes_root")
    if path.exists() or path.is_symlink() or check_only:
        verify_checkout(path, entry)
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()  # Exclusive creation: never take ownership of an existing target.
    except FileExistsError:
        verify_checkout(path, entry)
        return False
    except OSError:
        raise CacheError("cache_directory_creation_failed") from None
    git(path, "init", "--quiet")
    git(path, "remote", "add", "origin", entry["url"])
    git(path, "fetch", "--quiet", "--depth=1", "origin", entry["revision"], timeout=300)
    git(path, "checkout", "--quiet", "--detach", entry["revision"])
    verify_checkout(path, entry)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify existing caches without fetching.",
    )
    parser.add_argument(
        "--name", choices=UPSTREAM_NAMES, help="Fetch or verify one upstream."
    )
    arguments = parser.parse_args()
    name = None
    try:
        root = cache_root(PROJECT_ROOT)
        entries = load_upstreams(PROJECT_ROOT)
        results = []
        for name in (arguments.name,) if arguments.name else UPSTREAM_NAMES:
            entry = entries[name]
            path = root / entry["cache_path"]
            fetched = fetch_checkout(path, entry, check_only=arguments.check_only)
            results.append(
                {
                    "name": name,
                    "revision": entry["revision"],
                    "cache_path": str(path),
                    "fetched": fetched,
                }
            )
        print(json.dumps({"status": "ready", "upstreams": results}, indent=2))
        return 0
    except CacheError as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "upstream": name,
                    "stage": error.stage,
                    "returncode": error.returncode,
                    "note": "Existing targets are never modified or deleted; failed new cache directories are retained.",
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
