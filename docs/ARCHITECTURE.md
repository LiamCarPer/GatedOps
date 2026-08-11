# Architecture notes

Design decisions that shape the code, written down so the reasoning survives
the demo.

## Model registry: stages -> aliases

MLflow deprecated model registry **stages** (Staging/Production) in 2.9 and
points new work at **aliases**. GatedOps uses the current API exclusively:

- **`staging` alias** — set at registration time; this is the *candidate* that
  the promotion gate evaluates.
- **`production` alias** — set only by `gatedops.promote.promote` after the
  gate passes and the artifact hash is re-verified. The scoring API serves
  exactly what this alias points at.

Consequences:

- promoting a new version is a metadata flip, so the API hot-swaps on the next
  poll (`serve/loader.py`) without a redeploy;
- nothing in the promotion or serving path touches the deprecated stage
  transition APIs.

[AetherPdM](https://github.com/LiamCarPer/AetherPdM) consumes `gatedops@v0.2.0`
and was migrated to the same alias contract (its promote path sets `staging`
at train time and `production` on promotion, and its inference engine loads by
the `production` alias).

## Serving-quality gates

The gate engine is metric-agnostic: any metric in the run's evaluation dict
can be gated (`ThresholdRule(metric="latency_p95_ms", op="<=", value=...)`).
The serving-quality demo closes the loop by measuring the ONNX runtime path
and gating on the results:

- `gatedops run --config configs/serving.yaml --benchmark`:
  1. trains and exports the pipeline to **ONNX** (`pipelines/benchmark.py`,
     skl2onnx) and logs `model.onnx` as the model artifact;
  2. benchmarks single-row inference on the test features with ONNX Runtime,
     producing `latency_p50/p95/p99_ms` and `throughput_rps`;
  3. merges those metrics into the run's metrics **before** gate evaluation,
     so they flow through the same gate -> manifest -> lineage machinery as
     classification metrics;
  4. the gate in `configs/serving.yaml` requires `latency_p95_ms <= 5` and
     `throughput_rps >= 1000` (measured ~0.04 ms / ~34k rps locally — the
     thresholds are generous enough for slow hardware, tight enough to be a
     real bar).

Because the gate fails closed on missing metrics, the serving thresholds live
in their own config: running `pipeline.yaml` without `--benchmark` never sees
them.

The FastAPI loader serves the ONNX artifact when present (feature order comes
from the request schema's `FEATURE_ORDER`), falling back to the cloudpickle
path otherwise. The Docker image installs core dependencies only, so the ONNX
serving path runs in dev/CI via the `serving` extra — a documented boundary,
not a hidden gap.

### Triton extension (documented, not built)

A GPU-serving variant would swap the runtime, not the platform:

- serve `model.onnx` from a `nvcr.io/nvidia/tritonserver` container (a new
  compose service, replacing the uvicorn inference path);
- the benchmark step would point at Triton's HTTP/gRPC endpoint (perf_analyzer
  or a small async client) instead of a local InferenceSession — the metric
  names stay identical, so the gate, manifest, and lineage are unchanged;
- the loader would proxy to Triton (`trt_client`) instead of importing
  onnxruntime.

The value of the current design is that the *contract* (metrics -> gate ->
manifest) is independent of the runtime, so the Triton swap is configuration
plus one adapter, not a redesign.

## Orchestration

The pipeline is a plain function (`run_pipeline`), so schedulers wrap it
rather than reimplement it:

- **Prefect 3 flow** (`pipelines/prefect_flow.py`, extra `orchestration`):
  `train -> evaluate -> gate` run inside a task with `promote_on_pass=False`;
  the `promote` task runs only when the gate report reads `PASS`, and it
  re-tags the manifest as `Production`. Deploy as a cron flow with
  `gated_flow.serve(...)`.
- **Airflow operator** (`airflow/operator.py`, extra `airflow`): a
  `BaseOperator` that runs the pipeline, pushes `gate_status` to XCom, and
  raises `AirflowFailException` on a rejected gate by default
  (`fail_on_reject=True`) — the "bad model fails the DAG" story. An example
  DAG lives in `examples/dags/gatedops_dag.py`.

Both are thin: no scheduling logic lives in the package, only the adapters.
