# Local Airbyte declarative-manifest seed repair

The project's Airbyte 2.4.0 installation had only CDK major mappings 0–4.
Its trains manifest declares `6.33.4`, so publication could not resolve major 6.
The public definition API returned 409 and left a builder draft; publishing that
draft returned 500. This is consistent with the upstream
[fresh-install seed issue](https://github.com/airbytehq/airbyte/issues/84928).
Registration and warehouse synchronization are separate verification steps.

## Verified evidence

On 2026-10-10, a read-only query of this project's PostgreSQL container
(`airbyte-db-0`, namespace `airbyte-abctl`) established:

- Table: `public.declarative_manifest_image_version`.
- `major_version`: non-null integer, primary key.
- `image_version` and `image_sha`: non-null varchar.
- `created_at` and `updated_at`: non-null timestamptz with `CURRENT_TIMESTAMP`
  defaults.
- Existing mappings: `0 → 0.90.0`, `1 → 1.7.0`, `2 → 2.1.0`,
  `3 → 3.10.4`, `4 → 4.3.0`; major 6 was absent.

The [official DockerHub tag API](https://hub.docker.com/v2/repositories/airbyte/source-declarative-manifest/tags/6.33.4/)
returned tag `6.33.4`, active Linux amd64 and arm64 images, and the top-level OCI
image-index digest:

```text
sha256:8fd974de2be41d6acad43a3d43576f9940a04c4f289164a2b1a5846d6f676b65
```

This is the multi-platform digest, not a platform-specific child-image digest.
The script fetches the official tag metadata each time and rejects a changed
digest or missing platform; it never uses an invented SHA.

The [upstream handler source](https://github.com/airbytehq/airbyte-platform/blob/main/airbyte-commons-server/src/main/kotlin/io/airbyte/commons/server/handlers/ConnectorBuilderProjectsHandler.kt)
corroborates that `getImageVersionForManifest` looks up the manifest's CDK major,
and `getDeclarativeManifestBaseImage` forms `docker.io/<repository>:<version>@<sha>`.
This source was inspected on upstream main; the deployed database schema and
local API behavior are verified independently rather than assuming main equals
the deployed 2.4.0 source revision.

## Check and apply

From the repository root:

```bash
.venv/bin/python scripts/repair_airbyte_manifest_seed.py
.venv/bin/python scripts/repair_airbyte_manifest_seed.py --apply
.venv/bin/python scripts/repair_airbyte_manifest_seed.py --check-only
```

The default is read-only. `--apply` uses the existing project wrapper and
`colima-elt-bench-rl` context, checks the Kind labels and PostgreSQL pod identity,
verifies the table schema, and inserts only absent major 6. A bounded transaction
lock and `ON CONFLICT DO NOTHING` preserve an existing mapping. Before/after
snapshots are captured in the same transaction and compared afterward.
Existing different major-6 mappings are
reported as `skipped`, never overwritten. No restart, reinstall, cloud request,
connector deletion, or credential modification occurs.

Standard output is JSON containing only public image metadata, table state and
sanitized status. PostgreSQL uses its pod's existing environment internally;
the script neither prints its credentials nor retrieves Airbyte API secrets.

## Completed local verification

On 2026-10-10 at 01:58:07 UTC, `--apply` returned `configured` and
`inserted: true`. The seed majors became `[0, 1, 2, 3, 4, 6]`; all existing
rows, including their timestamps, were unchanged between the snapshots
captured in that transaction and compared afterward.
Both a subsequent default check and a repeated `--apply` returned `ready`
with `inserted: false`. The inserted mapping and its timestamps were identical
to the first apply result.

The first apply JSON is retained locally at
`.local/airbyte-seed-repair-apply.json` with mode `0600`; it is excluded from Git.
Focused mocked checks covered read-only missing-state handling, an existing
matching mapping, preservation of a different mapping, and the absent-row
insert path. The reproducible CI tests are in
`tests/test_manifest_seed_repair.py`; all 25 passed without Docker or network
access, and Ruff passed for the script and tests. Separate API verification
subsequently confirmed `get_base_image` resolution and successful publication
of the existing matching builder draft. This does not establish a warehouse
sync or credentialed rollout.

## Rollback and scope

Keep the apply JSON as a local audit artifact. If it reports `inserted: true`,
rollback can delete **only** major 6 with the exact tag, digest and `created_at`
from that output, after confirming no publication needs the new mapping. Never
delete an existing mapping merely because it matches these public fields.

```sql
DELETE FROM public.declarative_manifest_image_version
WHERE major_version = 6
  AND image_version = '6.33.4'
  AND image_sha = 'sha256:8fd974de2be41d6acad43a3d43576f9940a04c4f289164a2b1a5846d6f676b65'
  AND created_at = '<created_at recorded by this repair>';
```

This repair alone does not establish a Databricks sync or credentialed ELT
rollout. Recheck `connector_builder_projects/get_base_image`, then reuse the
existing matching builder draft for publication and verify the registered
source definition separately.
