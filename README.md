# GatedOps

A reference MLOps platform: the controlled path that takes a model from training to
production serving, with quality gates, lineage, and reproducible deployment.

Train -> evaluate -> gate -> register/promote -> serve -> lineage.

## Run the whole thing with Docker

Requires Docker with a running engine.

```bash
docker compose up --build
```

This starts the MLflow registry, trains and promotes a model through the gate
(`init`), and then serves it from the production alias:

- API: `POST http://localhost:8000/score`
- MLflow UI: http://localhost:5001

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"tenure_years":2.5,"monthly_spend":49.9,"support_tickets":2,"usage_frequency":41.0,"engagement_score":0.4,"has_contract":0,"payment_delay":3.2}'
```

The response includes the prediction, the probability, and the lineage of the
model that produced it (version, artifact hash, run, data hash).

