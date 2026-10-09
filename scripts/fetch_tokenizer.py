"""Fetch the pinned Qwen tokenizer files and verify their trusted SHA-256 digests."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKENIZER_DIRECTORY = Path("data/tokenizers/Qwen3.5-4B")
FILES = {
    "tokenizer.json": (12807982, "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    "tokenizer_config.json": (16710, "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"),
    "vocab.json": (6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
    "merges.txt": (3353259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
    "chat_template.jinja": (7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
    "config.json": (3161, "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"),
}


def verify_file(path: Path, size: int, digest: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular tokenizer file: {path}")
    if path.stat().st_size != size:
        raise ValueError(f"Existing file has an unexpected size; refusing to overwrite: {path}")
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"Existing file has an unexpected SHA-256; refusing to overwrite: {path}")


def verify_files(project_dir: Path) -> None:
    for name, (size, digest) in FILES.items():
        path = project_dir / TOKENIZER_DIRECTORY / name
        if not os.path.lexists(path):
            raise FileNotFoundError(f"Missing tokenizer file: {path}")
        verify_file(path, size, digest)


def _publish_verified(temporary: Path, target: Path) -> None:
    """Atomically publish a verified file without replacing an existing path."""
    if sys.platform == "darwin":
        rename = ctypes.CDLL(None, use_errno=True).renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        # macOS sys/stdio.h: RENAME_EXCL prevents replacement, including races.
        if rename(os.fsencode(temporary), os.fsencode(target), 0x00000004) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
    else:
        # Same-directory hard links give portable atomic no-clobber publication.
        os.link(temporary, target)
        temporary.unlink()


def _download(path: Path, size: int, digest: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{path.name}"
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=60) as response:
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > size:
                    raise ValueError(f"Download exceeds the pinned file size: {path.name}")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        verify_file(temporary, size, digest)
        try:
            _publish_verified(temporary, path)
        except FileExistsError:
            # A concurrent fetch may have finished first; verify that file.
            verify_file(path, size, digest)
    finally:
        temporary.unlink(missing_ok=True)


def fetch(project_dir: Path) -> None:
    destination = project_dir / TOKENIZER_DIRECTORY
    missing = []
    # Check every existing file before writing any downloads.
    for name, (size, digest) in FILES.items():
        path = destination / name
        if os.path.lexists(path):
            verify_file(path, size, digest)
        else:
            missing.append(name)
    destination.mkdir(parents=True, exist_ok=True)
    for name in missing:
        _download(destination / name, *FILES[name])
    verify_files(project_dir)
    manifest_path = project_dir / "data/tokenizer_manifest.json"
    if not manifest_path.exists():
        manifest = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "tokenizer_directory": TOKENIZER_DIRECTORY.as_posix(),
            "files": [
                {"path": (TOKENIZER_DIRECTORY / name).as_posix(), "size_bytes": size, "sha256": digest}
                for name, (size, digest) in FILES.items()
            ],
            "total_size_bytes": sum(size for size, _ in FILES.values()),
            "model_weights_downloaded": False,
        }
        try:
            with manifest_path.open("x") as output:
                output.write(json.dumps(manifest, indent=2) + "\n")
        except FileExistsError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify local files without network or writes.")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    try:
        if args.check_only:
            verify_files(project_dir)
        else:
            fetch(project_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Tokenizer verification failed: {error}\n")
    print(json.dumps({
        "status": "passed", "model_id": MODEL_ID, "revision": REVISION,
        "files_verified": len(FILES), "total_size_bytes": sum(size for size, _ in FILES.values()),
        "check_only": args.check_only, "model_weights_downloaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
