#!/usr/bin/env python3
"""Repair only a dangling resolv.conf symlink in the project's Colima VM.

Run before installing Airbyte. A repair restarts this VM's Docker daemon, so
project container/service health must be checked afterward. Existing effective
or custom resolver files are preserved. Output is one sanitized JSON report.
"""

from __future__ import annotations

import ipaddress
import json
import subprocess
import unicodedata


PROFILE = "elt-bench-rl"
NETPLAN_PATH = "/etc/netplan/50-cloud-init.yaml"
MAX_NETPLAN_BYTES = 2 * 1024 * 1024

# All guest shell programs and paths are fixed. Nameserver values never become
# shell source or command arguments; verified resolver content arrives on stdin.
READ_STATE_SHELL = """set -eu
if [ -L /etc/resolv.conf ] && [ ! -e /etc/resolv.conf ]; then
    printf 'dangling\\n'
elif [ -e /etc/resolv.conf ]; then
    printf 'valid\\n'
else
    printf 'missing\\n'
fi
"""
WRITE_RESOLVER_SHELL = """set -eu
if [ ! -L /etc/resolv.conf ] || [ -e /etc/resolv.conf ]; then
    printf 'preserved\\n'
    exit 0
fi
if [ -e /etc/resolv.conf.elt-bench-original ] || [ -L /etc/resolv.conf.elt-bench-original ]; then
    if [ ! -L /etc/resolv.conf.elt-bench-original ]; then
        printf 'backup_conflict\\n'
        exit 0
    fi
else
    cp --no-dereference --no-clobber /etc/resolv.conf /etc/resolv.conf.elt-bench-original
    if [ ! -L /etc/resolv.conf.elt-bench-original ]; then
        printf 'backup_conflict\\n'
        exit 0
    fi
fi
umask 022
temporary=$(mktemp /etc/.elt-bench-resolv.XXXXXX)
trap 'rm -f "$temporary"' EXIT
cat > "$temporary"
chmod 644 "$temporary"
if [ ! -L /etc/resolv.conf ] || [ -e /etc/resolv.conf ]; then
    printf 'preserved\\n'
    exit 0
fi
mv -f "$temporary" /etc/resolv.conf
printf 'written\\n'
"""


class StageError(Exception):
    def __init__(self, stage: str, returncode: int, *, requires_service_healthcheck: bool = False):
        self.stage = stage
        self.returncode = returncode
        self.requires_service_healthcheck = requires_service_healthcheck
        super().__init__(stage)


def ssh(arguments: list[str], stage: str, *, content: bytes | None = None,
        requires_service_healthcheck: bool = False) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["colima", "--profile", PROFILE, "ssh", "--", *arguments],
            input=content, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise StageError(stage, 124,
                         requires_service_healthcheck=requires_service_healthcheck) from None
    except OSError:
        raise StageError(stage, 127,
                         requires_service_healthcheck=requires_service_healthcheck) from None
    if result.returncode:
        raise StageError(stage, result.returncode,
                         requires_service_healthcheck=requires_service_healthcheck)
    return result


def report(status: str, reason: str, *, requires_service_healthcheck: bool = False,
           **details: object) -> dict:
    return {"status": status, "reason": reason, "profile": PROFILE,
            "requires_service_healthcheck": requires_service_healthcheck, **details}


def nameservers(content: bytes) -> list[str]:
    if len(content) > MAX_NETPLAN_BYTES:
        raise StageError("validate_netplan_size", 2)
    try:
        import yaml
    except ImportError:
        raise StageError("import_yaml", 2) from None
    try:
        configuration = yaml.safe_load(content)
    except (yaml.YAMLError, UnicodeError):
        raise StageError("parse_netplan", 2) from None
    if not isinstance(configuration, dict) or not isinstance(configuration.get("network"), dict):
        return []

    addresses: list[str] = []
    visited: set[int] = set()
    pending = [configuration["network"]]
    while pending:
        mapping = pending.pop()
        if id(mapping) in visited:
            continue
        visited.add(id(mapping))
        servers = mapping.get("nameservers")
        if isinstance(servers, dict) and isinstance(servers.get("addresses"), list):
            for candidate in servers["addresses"]:
                if not isinstance(candidate, str) or candidate != candidate.strip():
                    continue
                if "%" in candidate or any(
                    unicodedata.category(character) in {"Cc", "Cs"} for character in candidate
                ):
                    continue
                # Only literal IPs are accepted: no paths, CIDRs, hostnames,
                # interface-scoped suffixes, shell fragments, or DNS directives.
                try:
                    address = ipaddress.ip_address(candidate).compressed
                except ValueError:
                    continue
                if address not in addresses:
                    addresses.append(address)
        pending.extend(value for value in mapping.values() if isinstance(value, dict))
    return addresses


def configure() -> dict:
    state = ssh(["sudo", "sh", "-c", READ_STATE_SHELL], "inspect_resolver").stdout.strip()
    if state == b"valid":
        return report("skipped", "valid_resolver_preserved")
    if state == b"missing":
        return report("skipped", "resolver_not_a_dangling_symlink")
    if state != b"dangling":
        raise StageError("decode_resolver_state", 2)

    netplan = ssh(["sudo", "cat", NETPLAN_PATH], "read_netplan").stdout
    addresses = nameservers(netplan)
    if not addresses:
        return report("skipped", "no_valid_netplan_nameserver")
    content = ("".join(f"nameserver {address}\n" for address in addresses)
               + "options timeout:2 attempts:3\n").encode("ascii")
    written = ssh(["sudo", "sh", "-c", WRITE_RESOLVER_SHELL],
                  "write_resolver", content=content).stdout.strip()
    if written == b"preserved":
        return report("skipped", "resolver_changed_preserved")
    if written == b"backup_conflict":
        return report("skipped", "existing_backup_file_preserved")
    if written != b"written":
        raise StageError("decode_write_result", 2, requires_service_healthcheck=True)

    ssh(["sudo", "systemctl", "restart", "docker"], "restart_project_docker",
        requires_service_healthcheck=True)
    return report("configured", "dangling_resolver_repaired",
                  requires_service_healthcheck=True, nameserver_count=len(addresses))


def main() -> int:
    try:
        result = configure()
    except StageError as error:
        result = report("error", "command_failed", stage=error.stage, returncode=error.returncode,
                        requires_service_healthcheck=error.requires_service_healthcheck)
    print(json.dumps(result, ensure_ascii=True))
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
