# GatedOps

_Reference MLOps platform: gated train/evaluate/promote/serve with full lineage._

[![CI](https://github.com/LiamCarPer/GatedOps/actions/workflows/ci.yml/badge.svg)](https://github.com/LiamCarPer/GatedOps/actions/workflows/ci.yml)
[![Train + gate demo](https://github.com/LiamCarPer/GatedOps/actions/workflows/train-eval-gate.yml/badge.svg)](https://github.com/LiamCarPer/GatedOps/actions/workflows/train-eval-gate.yml)

GatedOps is the controlled path that takes a model from training to production
serving: every release must pass a quality gate, every promotion is explicit,
and every served prediction carries its lineage back to the exact code, data,
and training run that produced it.

**Train -> evaluate -> gate -> register/promote -> serve -> lineage.**

It is not a predictive-maintenance product (that is AetherPdM) and not an edge
runtime. It is the operations layer — how a model is industrialized, as a
runnable reference implementation.

---

## The problem it solves

Most teams can train a model. Few have a system where:

- only a model that clears the quality bar reaches production,
- a bad model cannot be promoted (the gate proves it, in CI),
- every prediction can be traced to code + data + run,
- the deployment is the same locally, in CI, and against a production stack.

GatedOps is that system. It is deliberately generic: any
cloudpickle-serializable model that exposes `predict_proba` can be gated,
registered, promoted, and served through the same machinery.

## The loop

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ 1. TRAIN    │───▶│ 2. EVALUATE  │───▶│ 3. GATE         │
│ config+data │    │ metrics      │    │ pass / fail     │
│ reproducible│    │ vs champion  │    │ thresholds      │
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                │ fail → no promote
                                                ▼ pass
                       ┌──────────────┐    ┌─────────────┐
                       │ 4. REGISTER  │───▶│ 5. PROMOTE  │
                       │ MLflow       │    │ Staging→Prod│
                       │ version+tag  │    │ production  │
                       └──────────────┘    └─────┬───────┘
                                                 ▼
                       ┌──────────────┐    ┌─────────────┐
                       │ 6. SERVE     │───▶│ 7. LINEAGE  │
                       │ production   │    │ manifest in │
                       │ alias only   │    │ every score │
                       └──────────────┘    └─────────────┘
```

## What's in the box

| Piece | Location | What it enforces |
| --- | --- | --- |
| **Gate engine** | `src/gatedops/gate/` | Declarative rules: absolute thresholds plus a champion-comparison rule with `tolerance`/`min_delta` and explicit metric direction (`higher_is_better`). Verdicts are `PASS` / `FAIL` / `ERROR`, so a misconfigured gate is never mistaken for a failing model. Fails closed on missing metrics; the champion guard is vacuous until a first release exists. |
| **Lineage manifest** | `src/gatedops/manifest/` | A per-release record: `model_version`, SHA-256 `artifact_hash` of the model bytes, `run_id`, `git_sha`, `data_hash`, metrics, and the gate verdict. |
| **Guarded promotion** | `src/gatedops/promote/` | Refuses to promote unless the gate passed *and* the artifact in the registry hashes to exactly what was trained. Registry is behind a protocol — policy stays registry-agnostic. |
| **Scoring service** | `src/gatedops/serve/` | FastAPI app that serves **only** the production alias, re-verifies the artifact hash before loading, and echoes the manifest in every `/score` response. A background poller hot-swaps when the alias moves. |
| **Training pipeline** | `src/gatedops/pipelines/run.py` | The canonical loop as a single function/CLI: data -> fit -> evaluate -> gate -> register -> promote, emitting a manifest and tagging the model version. |
| **CLI** | `src/gatedops/cli.py` | `python -m gatedops run` and `python -m gatedops demo good|bad`. |
| **Reproducible deploy** | `Dockerfile`, `compose.yaml` | One image (dependency set pinned by `uv.lock`) that runs the MLflow registry and the API. `docker compose up` trains, gates, promotes, and serves. |

## Quickstart

### Locally (no Docker)

```bash
uv sync --extra dev

# Train, gate, and promote a good model into production
uv run python -m gatedops demo good

# Start the scoring API (serves the production alias)
uv run python -m uvicorn gatedops.serve.main:app --port 8000
```

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"tenure_years":2.5,"monthly_spend":49.9,"support_tickets":2,"usage_frequency":41.0,"engagement_score":0.4,"has_contract":0,"payment_delay":3.2}'
```

```json
{
  "prediction": 1,
  "probability": 0.984,
  "lineage": {
    "model_name": "churn-classifier",
    "model_version": "3",
    "artifact_hash": "be3e877b...",
    "git_sha": "a49d766",
    "run_id": "45aaa974...",
    "data_hash": "99550137..."
  }
}
```

Every score is auditable back to a specific version, artifact, commit, dataset,
and training run.

### With Docker (registry + training + serving)

```bash
docker compose up --build
```

Starts the MLflow registry, trains and promotes a model through the gate
(`init`), then serves it from the production alias:

- API: `POST http://localhost:8000/score`
- MLflow UI: http://localhost:5001 (browse versions, aliases, manifests)

## See it fail

The gate is the point of the project, so it is demoed both ways:

```bash
uv run python -m gatedops demo good   # gate PASS, promoted
uv run python -m gatedops demo bad    # gate FAIL, exit code 1, nothing promoted
```

```text
$ uv run python -m gatedops demo bad
run_id:      56e3963d70ef406086c404e554855f4b
version:     5
metrics:     accuracy=0.5705, precision=0.6530, recall=0.3163, f1=0.4262, false_alarm_rate=0.1710, roc_auc=0.6407
gate:        FAIL (1/4 checks passed)
promoted:    no
  FAIL threshold f1                 actual 0.4262 vs threshold 0.7000
  FAIL threshold false_alarm_rate   actual 0.1710 vs threshold 0.1000
  FAIL champion  f1                 challenger delta -0.4631 vs required >= -0.0200
```

The same run in CI fails the build (non-zero exit code), so a bad model can
never be promoted. The `train-eval-gate` workflow demonstrates it live — see
[the gate demo run](https://github.com/LiamCarPer/GatedOps/actions/runs/30761815637):
the `train-promote-good` job is green, the `gate-blocks-bad` job is red, and
the `GateReport` is attached as a downloadable artifact
(`gate-report-bad-model`). Trigger a fresh run anytime from the
[workflow page](https://github.com/LiamCarPer/GatedOps/actions/workflows/train-eval-gate.yml).

## Design decisions worth reading

- **Fail-closed gate.** A missing metric fails the run; a model cannot slip
  through because a check was forgotten.
- **Config errors are not model failures.** An empty or malformed gate returns
  `ERROR`, distinct from `FAIL`, so policy misconfiguration is never masked as
  model underperformance.
- **Metric direction is explicit.** `f1` is higher-is-better, `false_alarm_rate`
  is lower-is-better; the champion rule honors that, so a regression on FAR is
  caught as a regression.
- **Byte-exact lineage.** `artifact_hash` is the SHA-256 of the actual model
  bytes. Promotion and serving both re-check it, so what is gated is exactly
  what is deployed — no drift, no tampered artifacts.
- **Serve the alias, not a version.** The API loads whatever the
  `production` alias points at and hot-swaps on promotion. Promoting a new
  model does not require a redeploy.
- **Registry-agnostic policy.** Promotion logic talks to a small protocol
  (`version_artifact`, `set_production`); MLflow is today's adapter. The same
  guard could sit on top of another registry.

## Plugging in your own model

The platform machinery (gate, manifest, promote, serve) is model-agnostic.
Three places describe the model itself:

1. **Gate rules** — `configs/pipeline.yaml` (thresholds and the champion rule).
2. **Training recipe** — the estimator built in `gatedops/pipelines/run.py`
   (the demo uses a `StandardScaler + LogisticRegression` pipeline).
3. **Request contract** — the typed feature schema in
   `gatedops/serve/schemas.py`.

Anything that satisfies these contracts trains, gets gated, promoted, and
served with the same lineage guarantees.

## Honest limits

Out of scope by design (the platform is a demonstration, not a product):

- no multi-tenant SaaS, billing, or admin UI;
- no feature store, streaming, or lakehouse;
- no canary/shadow traffic or multi-cluster rollout;
- the demo model is synthetic churn data — the model is not the product, the
  path to production is;
- the API serves one model through the production alias (no multi-model
  routing);
- in-container runs have no `git`, so `git_sha` is either empty or set via the
  `GATEDOPS_GIT_SHA` environment variable.

## Portfolio context

- **AetherPdM** — a vertical application (industrial predictive maintenance).
- **GatedOps** — the horizontal release system: the same gates that would
  promote an AetherPdM model.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy src
uv run pytest -q
```

GitHub Actions runs three workflows:

- `ci` -- lint, type checking, tests, and a Docker Compose smoke test
  (builds the stack, scores a request, asserts the served lineage);
- `train-eval-gate` -- trains and promotes a good model, and on PRs /
  `workflow_dispatch` intentionally runs a bad model through the gate to show
  it being blocked, uploading the `GateReport` as an artifact.

## Roadmap

- wire an AetherPdM model through the same gate contract;
- canary/shadow routing and traffic splitting;
- a minimal Kubernetes Deployment on top of the Compose stack;
- multi-model serving with model-name routing.
