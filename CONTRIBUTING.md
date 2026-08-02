# Contributing

GatedOps is a reference MLOps platform: a gated train/evaluate/promote/serve
loop with full model lineage. Contributions that strengthen the gates, the
lineage contract, or the pipeline are welcome.

## Ground Rules

- **Conventional Commits** are required (`feat:`, `fix:`, `docs:`, `test:`,
  `ci:`, `chore:`, `refactor:`) — keep history machine-readable.
- Every change must pass CI: lint (ruff), type check (mypy), unit tests, and —
  for anything touching the stack — the **compose-smoke** job, which boots the
  full compose stack and asserts that a scored request carries its lineage.
- The **gate semantics are sacred**: a model must not be able to reach the
  production alias without passing the gate. Changes to the gate engine must
  keep `demo bad` blocked (the gate demo job proves it).
- No secrets. Keep them out of code, configs, and history.

## Development Setup

```bash
# Python 3.12+
uv sync --frozen --extra dev

# Run the checks locally
uv run ruff check .
uv run mypy src
uv run pytest -q

# Gate demos (prove the contract works)
uv run python -m gatedops demo good   # trains, gates, promotes
uv run python -m gatedops demo bad    # gate must BLOCK this model
```

## Where Things Live

| Area | Path |
| :--- | :--- |
| Gate engine (thresholds, challenger/champion) | `src/gatedops/gate/` |
| Model manifests (sha256 hashing, schema) | `src/gatedops/manifest/` |
| Registry & promotion (MLflow) | `src/gatedops/registry/`, `src/gatedops/promote/` |
| Serving API (FastAPI, lineage in responses) | `src/gatedops/serve/` |
| Pipeline orchestration & CLI | `src/gatedops/pipelines/`, `src/gatedops/cli.py` |
| Tests | `tests/` |

## Making Changes

1. Fork the repository and create a branch (`git checkout -b feat/my-change`).
2. Make your change and add tests — the gate, manifest, promote-guard, and
   serve layers all have dedicated test modules.
3. Run the local checks (see above).
4. Open a pull request; CI runs automatically and the compose-smoke job
   validates the full stack on push.

## Reporting Security Issues

See [SECURITY.md](SECURITY.md) — report privately, never in a public issue.
