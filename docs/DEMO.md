# GatedOps demo — end-to-end walkthrough

This guide takes you through the whole loop the way a first-time visitor would:
train a model, gate it, promote it, serve it, and prove the lineage. It is the
same path the CI workflows exercise, but on your machine.

Two ways to run it:

- **Locally without Docker** (fastest, no containers) — the registry lives in
  a local SQLite database.
- **With Docker Compose** (the full stack: MLflow registry + API + an
  auto-seeding init job).

You do not need both. The local path covers everything; Compose additionally
shows the registry UI.

---

## 1. Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (the project uses `uv sync`; `uv.lock` is
  committed)

```bash
git clone https://github.com/LiamCarPer/GatedOps.git
cd GatedOps
uv sync --extra dev
```

---

## 2. Local path (no Docker)

### 2.1 Train, gate, and promote a good model

```bash
uv run python -m gatedops demo good
```

Expected outcome (identifiers vary per run):

```text
run_id:      e25d46da0b034046bf6cd61aeec447bd
version:     6
metrics:     accuracy=0.8935, precision=0.9160, recall=0.8641, f1=0.8893, false_alarm_rate=0.0777, roc_auc=0.9632
gate:        PASS (4/4 checks passed)
promoted:    Staging -> Production
```

What happened, in one line: the pipeline trained a `StandardScaler +
LogisticRegression` pipeline on a synthetic churn dataset, evaluated it on a
held-out split, ran the gate (three thresholds plus the champion rule), and —
because every check passed — registered the model and pointed the `production`
alias at it. The exit code is `0`.

### 2.2 Prove the gate blocks a bad model

```bash
uv run python -m gatedops demo bad
echo $?    # 1
```

Expected outcome:

```text
gate:        FAIL (1/4 checks passed)
promoted:    no
  FAIL threshold f1                 actual 0.4262 vs threshold 0.7000
  FAIL threshold false_alarm_rate   actual 0.1710 vs threshold 0.1000
  FAIL champion  f1                 challenger delta -0.4631 vs required >= -0.0200
```

The same dataset with a deliberately weaker signal produces a model below the
quality bar. The gate refuses it: nothing is promoted and the command exits
non-zero — which is exactly what stops a bad model in CI.

### 2.3 Serve the production model

```bash
uv run python -m uvicorn gatedops.serve.main:app --port 8000
```

In another terminal:

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "6",
  "artifact_hash": "be3e877b1a1d32c6dd8badea3024c94af9c0ee294d5dae78f3f2d86cca9d0cde",
  "loaded_at": "2026-08-09T08:54:47.738679+00:00"
}
```

The API loads **only** the version behind the `production` alias, and only
after re-verifying that the artifact bytes match the SHA-256 recorded in the
lineage manifest.

Score a request:

```bash
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"tenure_years":2.5,"monthly_spend":49.9,"support_tickets":2,"usage_frequency":41.0,"engagement_score":0.4,"has_contract":0,"payment_delay":3.2}'
```

```json
{
  "prediction": 1,
  "probability": 0.9839772504927619,
  "lineage": {
    "model_name": "churn-classifier",
    "model_version": "6",
    "artifact_hash": "be3e877b1a1d32c6dd8badea3024c94af9c0ee294d5dae78f3f2d86cca9d0cde",
    "git_sha": "b8bfd43",
    "run_id": "e25d46da0b034046bf6cd61aeec447bd",
    "data_hash": "995501372d1a58944263a0037da4f107926d7bf7a8746d4bc635ed1545ba6b34"
  }
}
```

Every score carries its lineage: the model version, the SHA-256 of the served
artifact bytes, the git commit the pipeline ran from, the training run, and a
hash of the dataset.

Other endpoints: `GET /manifest` (the full manifest of the served model) and
`POST /reload` (force re-resolve the alias, e.g. after a new promotion).

### 2.4 Browse the registry

The registry is a local SQLite database outside the repo (so cloud-synced
folders do not slow it down). To see where it lives and open the MLflow UI:

```bash
uv run python -c "from gatedops.config import default_tracking_uri; print(default_tracking_uri())"
uv run python -m mlflow server --backend-store-uri <that uri> --host 127.0.0.1 --port 5001
```

Open http://localhost:5001: the `churn-classifier` model has versioned runs;
the promoted version carries the `gatedops.manifest` and `gatedops.status`
tags.

---

## 3. Docker path (full stack)

Requires Docker with a running engine.

```bash
docker compose up --build
```

This starts three services:

| Service | Role |
| --- | --- |
| `mlflow` | MLflow tracking + registry server (UI at http://localhost:5001) |
| `init` | One-shot job: trains and promotes a good model through the gate |
| `api` | The scoring API (http://localhost:8000), starts only after `init` succeeded |

Once the API reports healthy, the score request is identical to the local
path:

```bash
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"tenure_years":2.5,"monthly_spend":49.9,"support_tickets":2,"usage_frequency":41.0,"engagement_score":0.4,"has_contract":0,"payment_delay":3.2}'
```

Tear down with `docker compose down -v`.

---

## 4. The gate demo in CI

The same loop runs in GitHub Actions:

- `ci` — lint, type checks, tests, and a Docker Compose smoke test (builds the
  stack, scores a request, asserts the lineage) on every push/PR.
- `train-eval-gate` — on `main` it trains and promotes a good model; on pull
  requests and `workflow_dispatch` it additionally runs a bad model through
  the gate. That second job **fails on purpose** and uploads the `GateReport`
  as an artifact — see
  [the demo run](https://github.com/LiamCarPer/GatedOps/actions/runs/30761815637).

---

## 5. The AetherPdM crossover

[AetherPdM](https://github.com/LiamCarPer/AetherPdM) — a bearing
predictive-maintenance platform — consumes this package as a pinned git
dependency (`gatedops@v0.2.0`). Its fault and anomaly models are promoted
through the same gate engine, and every promotion records a
`gatedops.manifest` tag. One contract, two domains.

To see it: train and promote an AetherPdM model (its README covers the
pipeline), then open its MLflow registry — the promoted versions carry the
`gatedops.manifest` tag, and its `/score` responses echo lineage for both the
anomaly and fault models.

---

## Troubleshooting

- **First run is slow.** The first contact with a fresh registry creates the
  MLflow schema; subsequent runs are fast.
- **Registry location.** The default tracking URI points outside the repo
  (`%LOCALAPPDATA%/gatedops/mlflow.db` on Windows, `$XDG_DATA_HOME`/cwd
  elsewhere) to avoid cloud-synced folders. Override with
  `--tracking-uri` on the CLI or `GATEDOPS_TRACKING_URI` for the API.
- **`git_sha` empty in containers.** The runtime image has no `git`; set
  `GATEDOPS_GIT_SHA` to record a revision explicitly.
- **Versions/ids differ from this guide.** Every run creates new ids and the
  version number depends on your registry's history — the shape of the output
  is what matters.
