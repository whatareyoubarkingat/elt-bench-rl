"""External-cache identity and boundary checks using local Git fixtures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from elt_bench_rl.errors import InfrastructureError
from elt_bench_rl.upstreams import resolve_upstream

URL = "https://github.com/uiuc-kang-lab/ELT-Bench.git"


def git(path, *arguments):
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-C",
            str(path),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root, cache = tmp_path / "project", tmp_path / "external-cache"
    (root / "configs").mkdir(parents=True)
    repo = cache / "staging"
    repo.mkdir(parents=True)
    git(repo, "init", "--quiet")
    (repo / "README.md").write_text("Pinned fixture.\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "--quiet", "-m", "Fixture")
    revision = git(repo, "rev-parse", "HEAD")
    git(repo, "remote", "add", "origin", URL)
    destination = cache / "elt-bench" / revision
    destination.parent.mkdir()
    repo.rename(destination)
    manifest = {
        "elt-bench": {
            "url": URL,
            "revision": revision,
            "cache_path": f"elt-bench/{revision}",
        }
    }
    (root / "configs/upstream-revisions.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("ELT_BENCH_RL_UPSTREAM_CACHE", str(cache))
    return root, cache, destination, manifest


def test_clean_exact_pin_resolves_without_network(checkout):
    root, _, repo, _ = checkout
    assert resolve_upstream(root, "elt-bench") == repo


def test_default_cache_is_outside_project(checkout, tmp_path, monkeypatch):
    root, cache, repo, _ = checkout
    home = tmp_path / "home"
    (home / ".cache/elt-bench-rl").mkdir(parents=True)
    (home / ".cache/elt-bench-rl/upstreams").symlink_to(cache, target_is_directory=True)
    monkeypatch.delenv("ELT_BENCH_RL_UPSTREAM_CACHE")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert resolve_upstream(root, "elt-bench") == repo


def test_missing_cache_explains_the_explicit_prepare_step(checkout):
    root, cache, repo, _ = checkout
    repo.rename(cache / "preserved-fixture")
    with pytest.raises(InfrastructureError, match="missing.*fetch_upstreams.py"):
        resolve_upstream(root, "elt-bench")


@pytest.mark.parametrize("change", ["short_revision", "traversal", "wrong_url"])
def test_invalid_manifest_is_rejected(checkout, change):
    root, _, _, manifest = checkout
    entry = manifest["elt-bench"]
    if change == "short_revision":
        entry["revision"] = entry["revision"][:7]
    elif change == "traversal":
        entry["cache_path"] = "../../project"
    else:
        entry["url"] = "https://example.invalid/fake.git"
    (root / "configs/upstream-revisions.json").write_text(json.dumps(manifest))
    with pytest.raises(InfrastructureError, match="manifest is invalid"):
        resolve_upstream(root, "elt-bench")


@pytest.mark.parametrize("symlink", [False, True])
def test_cache_inside_project_is_rejected(checkout, tmp_path, monkeypatch, symlink):
    root, _, _, _ = checkout
    path = root / "cache"
    path.mkdir()
    if symlink:
        alias = tmp_path / "cache-alias"
        alias.symlink_to(path, target_is_directory=True)
        path = alias
    monkeypatch.setenv("ELT_BENCH_RL_UPSTREAM_CACHE", str(path))
    with pytest.raises(InfrastructureError, match="outside the project"):
        resolve_upstream(root, "elt-bench")


@pytest.mark.parametrize("inside_project", [False, True])
def test_checkout_symlink_cannot_escape_the_external_cache(
    checkout, tmp_path, inside_project
):
    root, _, repo, _ = checkout
    moved = (root if inside_project else tmp_path) / "moved-upstream"
    repo.rename(moved)
    repo.symlink_to(moved, target_is_directory=True)
    with pytest.raises(InfrastructureError, match="external cache boundary"):
        resolve_upstream(root, "elt-bench")


def test_relative_cache_root_is_rejected(checkout, monkeypatch):
    root, _, _, _ = checkout
    monkeypatch.setenv("ELT_BENCH_RL_UPSTREAM_CACHE", "relative-cache")
    with pytest.raises(InfrastructureError, match="absolute external path"):
        resolve_upstream(root, "elt-bench")


def test_changed_head_is_rejected(checkout):
    root, _, repo, _ = checkout
    (repo / "README.md").write_text("Changed commit.\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "--quiet", "-m", "Changed")
    with pytest.raises(InfrastructureError, match="revision differs"):
        resolve_upstream(root, "elt-bench")


@pytest.mark.parametrize("tracked", [False, True])
def test_later_changes_cannot_reuse_a_cached_validation(checkout, tracked):
    root, _, repo, _ = checkout
    assert resolve_upstream(root, "elt-bench") == repo
    (repo / ("README.md" if tracked else "untracked.txt")).write_text("Changed.\n")
    with pytest.raises(InfrastructureError, match="not clean"):
        resolve_upstream(root, "elt-bench")


def test_wrong_origin_is_rejected(checkout):
    root, _, repo, _ = checkout
    git(repo, "remote", "set-url", "origin", "https://example.invalid/other.git")
    with pytest.raises(InfrastructureError, match="origin differs"):
        resolve_upstream(root, "elt-bench")


def test_nested_repository_path_is_rejected(checkout):
    root, _, repo, _ = checkout
    (repo / ".git").rename(repo.parent / ".git")
    with pytest.raises(InfrastructureError, match="not a repository root"):
        resolve_upstream(root, "elt-bench")


def test_unknown_upstream_name_is_rejected(checkout):
    root, _, _, _ = checkout
    with pytest.raises(InfrastructureError, match="Unsupported upstream"):
        resolve_upstream(root, "../elt-bench")
