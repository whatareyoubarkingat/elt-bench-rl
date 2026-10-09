#!/usr/bin/env python3
"""Repair a loopback proxy inherited by the project's Airbyte Kind node.

Prints one JSON status object, never proxy values or subprocess output. Only the
named containerd drop-in is updated; NO_PROXY/no_proxy and other services remain
untouched. This script must run after the project Docker daemon is available.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unicodedata
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/with_docker.sh"
NODE = "airbyte-abctl-control-plane"
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
NODE_TEMPLATE = '{"running":{{json .State.Running}},"env":{{json .Config.Env}}}'
DAEMON_TEMPLATE = '{"http":{{json .HTTPProxy}},"https":{{json .HTTPSProxy}}}'

# Shell programs are fixed. Proxy-bearing content is passed only through stdin.
READ_DROPIN_SHELL = """set -eu
if [ -e /etc/systemd/system/containerd.service.d/elt-bench-proxy.conf ]; then
    printf 'present\\n'
    cat /etc/systemd/system/containerd.service.d/elt-bench-proxy.conf
else
    printf 'absent\\n'
fi
"""
WRITE_DROPIN_SHELL = """set -eu
mkdir -p /etc/systemd/system/containerd.service.d
umask 077
temporary=$(mktemp /etc/systemd/system/containerd.service.d/.elt-bench-proxy.XXXXXX)
trap 'rm -f "$temporary"' EXIT
cat > "$temporary"
chmod 600 "$temporary"
mv -f "$temporary" /etc/systemd/system/containerd.service.d/elt-bench-proxy.conf
"""


class StageError(Exception):
    def __init__(self, stage: str, returncode: int):
        self.stage = stage
        self.returncode = returncode
        super().__init__(stage)


def docker(arguments: list[str], stage: str, *, content: bytes | None = None,
           allow_failure: bool = False) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["bash", str(WRAPPER), "docker", *arguments],
            input=content, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise StageError(stage, 124) from None
    except OSError:
        raise StageError(stage, 127) from None
    if result.returncode and not allow_failure:
        raise StageError(stage, result.returncode)
    return result


def decode_json(content: bytes, stage: str) -> dict:
    try:
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, UnicodeError):
        raise StageError(stage, 2) from None


def report(status: str, reason: str, *, restarted: bool = False, **details: object) -> dict:
    return {"status": status, "reason": reason, "node": NODE,
            "restarted": restarted, **details}


def valid_proxy(value: str) -> bool:
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in value):
        return False
    try:
        parsed = urlsplit(value)
        return bool(parsed.scheme and parsed.hostname)
    except ValueError:
        return False


def proxy_hostname(value: str) -> str | None:
    if not valid_proxy(value):
        return None
    try:
        return urlsplit(value).hostname
    except ValueError:
        return None


def quoted_environment(name: str, value: str) -> str:
    if name not in PROXY_KEYS or not valid_proxy(value):
        raise StageError("validate_proxy", 2)
    # Environment= supports C-style quoted strings and systemd specifiers.
    # Dollar signs have no shell meaning here; percent signs must be doubled.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'Environment="{name}={escaped}"\n'


def configure() -> dict:
    inspected = docker(
        ["inspect", NODE, "--format", NODE_TEMPLATE], "inspect_node", allow_failure=True,
    )
    if inspected.returncode:
        return report("skipped", "node_unavailable")
    node = decode_json(inspected.stdout, "decode_node")
    if node.get("running") is not True:
        return report("skipped", "node_not_running")
    entries = node.get("env") or []
    if not isinstance(entries, list) or any(not isinstance(item, str) for item in entries):
        raise StageError("decode_node_environment", 2)
    original_environment = dict(item.split("=", 1) for item in entries if "=" in item)
    # NO_PROXY/no_proxy are retained in their original inherited environment.
    # The drop-in never defines them, and no manager environment is modified.

    daemon = decode_json(
        docker(["info", "--format", DAEMON_TEMPLATE], "inspect_daemon").stdout,
        "decode_daemon",
    )
    server_proxy = daemon.get("https") or daemon.get("http")
    if not server_proxy:
        return report("skipped", "no_server_proxy")
    if not isinstance(server_proxy, str) or not valid_proxy(server_proxy):
        raise StageError("validate_proxy", 2)
    desired = ("[Service]\n" + "".join(
        quoted_environment(name, server_proxy) for name in PROXY_KEYS
    )).encode("utf-8")

    current = docker(["exec", NODE, "sh", "-c", READ_DROPIN_SHELL], "read_dropin").stdout
    marker, separator, existing = current.partition(b"\n")
    if not separator or marker not in {b"present", b"absent"}:
        raise StageError("decode_dropin", 2)
    if marker == b"present" and existing == desired:
        return report("ready", "already_configured")

    original_proxies = [original_environment.get(name, "") for name in PROXY_KEYS]
    original_proxies = [value for value in original_proxies if value]
    if marker == b"absent":
        if not original_proxies or any(
            proxy_hostname(value) not in LOOPBACK_HOSTS for value in original_proxies
        ):
            return report("skipped", "original_proxy_not_loopback")
        if not any(value != server_proxy for value in original_proxies):
            return report("skipped", "original_proxy_matches_daemon")

    # EnvironmentFiles can override Environment= directives. Do not modify a
    # node with additional proxy controls; its configuration needs separate review.
    environment_files = docker(
        ["exec", NODE, "systemctl", "show", "containerd", "--property=EnvironmentFiles", "--value"],
        "check_environment_files",
    ).stdout.strip()
    if environment_files:
        return report("skipped", "environment_files_present")

    docker(["exec", "-i", NODE, "sh", "-c", WRITE_DROPIN_SHELL],
           "write_dropin", content=desired)
    docker(["exec", NODE, "systemctl", "daemon-reload"], "daemon_reload")
    docker(["exec", NODE, "systemctl", "restart", "containerd"], "restart_containerd")
    return report("configured", "proxy_updated", restarted=True)


def main() -> int:
    try:
        result = configure()
    except StageError as error:
        result = report("error", "command_failed", stage=error.stage, returncode=error.returncode)
    print(json.dumps(result, ensure_ascii=True))
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
