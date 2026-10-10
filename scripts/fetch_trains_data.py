#!/usr/bin/env python3
"""Fetch and verify only the official ELT-Bench trains sources and destination GT.

Uses Python's standard library and the system curl command. Source data and
ground truth remain in separate ignored directories. Existing different files
are never overwritten; all downloads must match the pinned SHA-256 hashes.
"""
from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

BENCHMARK_REVISION = "fcf3129df49ce5ce63ff46b2bbbbba1eaf9ec055"
DATASET_REVISION = "52a48541547a4340e19e54386350740acca80146"
ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1-Gv5g_Yg_YrR-NxH4s2tSEK3VhJQc2Q0&export=download"
)
ARCHIVE_BYTES = 122545366
ASSETS = (
    {
        "path": "data/trains/train.csv",
        "url": "https://drive.google.com/uc?export=download&id=1w5_eaQI7_bxn5zQVEOwB5WsIAkAu7VCB",
        "bytes": 164,
        "sha256": "f3ed8d5d62147ae2f51accd86f4d960602deb5e5c0acb8313b084eceacb0ecf9",
        "rows": 20,
    },
    {
        "path": "data/trains/cars.csv",
        "url": ARCHIVE_URL,
        "bytes": 3289,
        "sha256": "d36ba10381d12eef1079d830f391f28c38158c0376c6a8d317c1102efcbfc3f8",
        "rows": 63,
        "zip_member": "data/trains/cars.csv",
    },
    {
        "path": "ground_truth/gt_snowflake/trains/trains.csv",
        "url": f"https://huggingface.co/datasets/tttjjj/elt_bench/resolve/{DATASET_REVISION}/gt_snowflake/trains/trains.csv",
        "bytes": 565,
        "sha256": "9b11c739c45c2ff131f99e06902bebe2a5ec44545d1934a534627bd0d7985996",
        "rows": 20,
    },
)


def curl(url: str, *options: str) -> bytes:
    result = subprocess.run(
        [
            "curl", "-fsSL", "--connect-timeout", "15", "--max-time", "60",
            "--proto", "=https", "--proto-redir", "=https", *options, url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"curl failed ({result.returncode}): {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def destination(root: Path, relative: str) -> Path:
    """Reject symlinks so downloaded files cannot replace unrelated content."""
    path = root / relative
    for component in [path, *path.parents]:
        if component == root:
            break
        if component.is_symlink():
            raise RuntimeError(f"Refusing symlink in asset path: {component}")
    if not path.resolve().is_relative_to(root):
        raise RuntimeError(f"Asset path escapes project root: {path}")
    return path


def verify(body: bytes, asset: dict) -> None:
    if len(body) != asset["bytes"] or hashlib.sha256(body).hexdigest() != asset["sha256"]:
        raise RuntimeError(f"Size or SHA-256 mismatch for {asset['path']}; existing files will not be overwritten")
    records = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
    if len(records) != asset["rows"]:
        raise RuntimeError(f"CSV row count mismatch for {asset['path']}")


def create_asset(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".elt-download-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
        # Hard-link creation is atomic and fails if any existing file appeared.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_cars() -> tuple[bytes, int]:
    warning = curl(ARCHIVE_URL, "--max-filesize", "65536").decode()
    match = re.search(r'name="uuid" value="([^"]+)"', warning)
    if match is None:
        raise RuntimeError("Official Drive ZIP confirmation response changed; inspect it before updating this script")
    confirmed_url = ARCHIVE_URL + "&confirm=t&uuid=" + match.group(1)
    headers = curl(confirmed_url, "-I").decode()
    lengths = re.findall(r"^content-length:\s*(\d+)", headers, re.M | re.I)
    if not lengths or int(lengths[-1]) != ARCHIVE_BYTES:
        raise RuntimeError("Official API ZIP size changed; refusing an unpinned archive")
    downloaded_bytes = 0

    def byte_range(start: int, end: int) -> bytes:
        nonlocal downloaded_bytes
        expected = end - start + 1
        if start < 0 or end >= ARCHIVE_BYTES or not 0 < expected <= 5_000_000:
            raise RuntimeError("ZIP metadata requested an unsafe byte range")
        response = curl(
            confirmed_url, "--range", f"{start}-{end}", "--max-filesize", str(expected),
            "-w", "\n%{http_code}",
        )
        body, status = response.rsplit(b"\n", 1)
        if status != b"206" or len(body) != expected:
            raise RuntimeError("Drive did not honor HTTP byte ranges; refusing a full ZIP download")
        downloaded_bytes += len(body)
        return body

    tail = byte_range(ARCHIVE_BYTES - 65536, ARCHIVE_BYTES - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("ZIP central directory not found")
    _, disk, cd_disk, disk_entries, entries, cd_bytes, cd_offset, comment_length = struct.unpack_from("<4s4H2LH", tail, eocd)
    if disk or cd_disk or entries == 65535 or disk_entries != entries or cd_bytes > 5_000_000:
        raise RuntimeError("Unsupported ZIP layout")
    tail_offset = ARCHIVE_BYTES - len(tail)
    if cd_offset >= tail_offset and cd_offset + cd_bytes <= ARCHIVE_BYTES:
        central = tail[cd_offset - tail_offset:cd_offset - tail_offset + cd_bytes]
    else:
        central = byte_range(cd_offset, cd_offset + cd_bytes - 1)
    target = None
    offset = 0
    for _ in range(entries):
        fields = struct.unpack_from("<4s6H3L5H2L", central, offset)
        signature, _, _, flags, method, _, _, crc, compressed, uncompressed, name_length, extra_length, comment_length, _, _, _, local_offset = fields
        if signature != b"PK\x01\x02":
            raise RuntimeError("Invalid ZIP central directory entry")
        name = central[offset + 46:offset + 46 + name_length].decode("utf-8")
        if name == ASSETS[1]["zip_member"]:
            target = (flags, method, crc, compressed, uncompressed, local_offset)
            break
        offset += 46 + name_length + extra_length + comment_length
    if target is None:
        raise RuntimeError("Official API ZIP does not contain data/trains/cars.csv")
    flags, method, crc, compressed, uncompressed, local_offset = target
    if flags & 1 or uncompressed != ASSETS[1]["bytes"] or not 0 < compressed < 1_000_000:
        raise RuntimeError("Unexpected ZIP member properties")
    local = struct.unpack("<4s5H3L2H", byte_range(local_offset, local_offset + 29))
    if local[0] != b"PK\x03\x04":
        raise RuntimeError("Invalid ZIP member header")
    data_offset = local_offset + 30 + local[-2] + local[-1]
    payload = byte_range(data_offset, data_offset + compressed - 1)
    if method == 8:
        decompressor = zlib.decompressobj(-15)
        body = decompressor.decompress(payload, uncompressed + 1)
        if not decompressor.eof or decompressor.unconsumed_tail:
            raise RuntimeError("ZIP member exceeded its expected uncompressed size")
    elif method == 0:
        body = payload
    else:
        raise RuntimeError(f"Unsupported ZIP compression method: {method}")
    if len(body) != uncompressed or binascii.crc32(body) & 0xffffffff != crc:
        raise RuntimeError("ZIP member CRC mismatch")
    return body, downloaded_bytes


def manifest(range_bytes: int | None = None) -> dict:
    result = {
        "schema_version": 1,
        "task_id": "trains",
        "benchmark_url": "https://github.com/uiuc-kang-lab/ELT-Bench",
        "benchmark_revision": BENCHMARK_REVISION,
        "dataset_revision": DATASET_REVISION,
        "sources": list(ASSETS[:2]),
        "ground_truth": list(ASSETS[2:]),
        "archive": {
            "url": ARCHIVE_URL,
            "bytes": ARCHIVE_BYTES,
            "download_method": "HTTP byte ranges; only cars.csv decompressed",
        },
    }
    if range_bytes is not None:
        result["archive"]["range_download_bytes"] = range_bytes
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-only", action="store_true", help="Verify existing assets and manifest without network or writes")
    parser.add_argument("--destination", choices=["databricks", "snowflake", "redshift"], default="databricks")
    args = parser.parse_args()
    # All three pinned trains references are byte-identical; fetch the requested
    # destination's actual asset instead of relabeling another warehouse's GT.
    ASSETS[2]["path"] = ASSETS[2]["path"].replace("gt_snowflake", f"gt_{args.destination}")
    ASSETS[2]["url"] = ASSETS[2]["url"].replace("gt_snowflake", f"gt_{args.destination}")
    root = args.project_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Project root does not exist: {root}")
    range_bytes = None
    verified = []
    for asset in ASSETS:
        path = destination(root, asset["path"])
        if path.is_file():
            verify(path.read_bytes(), asset)
        elif args.check_only:
            raise RuntimeError(f"Missing asset: {asset['path']}; run this script without --check-only")
        else:
            if shutil.which("curl") is None:
                raise RuntimeError("curl is required to download public task data")
            if "zip_member" in asset:
                body, range_bytes = fetch_cars()
            else:
                body = curl(asset["url"], "--max-filesize", "1000000")
            verify(body, asset)
            create_asset(path, body)
        verified.append({"path": asset["path"], "bytes": asset["bytes"], "rows": asset["rows"], "sha256": asset["sha256"]})
    manifest_path = destination(root, "data/manifest.json")
    expected = manifest(range_bytes)
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    # Preserve the original range-transfer measurement on idempotent checks.
    if range_bytes is None and existing:
        previous_range = existing.get("archive", {}).get("range_download_bytes")
        if isinstance(previous_range, int) and previous_range >= 0:
            expected["archive"]["range_download_bytes"] = previous_range
    if args.check_only:
        if existing != expected:
            raise RuntimeError("Missing or inconsistent data/manifest.json; rerun the downloader to regenerate verified public provenance")
    elif existing != expected:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=manifest_path.parent, prefix=".elt-manifest-", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(expected, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, manifest_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(json.dumps({"status": "verified", "task_id": "trains", "manifest": str(manifest_path), "assets": verified}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, struct.error, zlib.error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
