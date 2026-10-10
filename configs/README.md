# Configuration

- `.env.example`: credential names; never commit a filled copy.
- `train.yaml`: bounded one-step Qwen3.5-4B defaults.
- `upstream-revisions.json`: benchmark/Cookbook/dataset/tokenizer pins.
- `airbyte-trains.yaml`: public declarative `cars` stream, using `elt-api`
  inside Airbyte's Kind network.
- `sources.compose.yml`: source-only mounts, no verifier or secrets.

Model-visible loading choices contain source names and mode, not credentials
or connection IDs. Offline CSV ingestion uses pinned bytes with provenance
in `data/manifest.json`; HTTP ingestion makes an actual request. The local
loader never claims to perform an Airbyte sync.
