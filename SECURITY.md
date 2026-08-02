# Security Policy

## Scope

GatedOps is a **reference MLOps platform**: a gated train/evaluate/promote/serve
loop with model lineage. It is a demonstration-quality implementation, not a
production service — but the integrity properties it demonstrates are
security-critical by design.

In scope for security reports:
- **Gate integrity:** any path that lets a model reach the production alias
  without passing the quality gate (a bypassed gate means an unreviewed model
  is served)
- **Manifest/lineage integrity:** sha256 manifest hashing, tag handling, or
  serve-side verification that can be forged or skipped
- **Registry/promotion logic:** MLflow registry interactions and promote
  guards that can be subverted
- **The serving API:** FastAPI endpoints (`/score`, `/health`) and model
  loading behavior

Out of scope:
- Vulnerabilities in third-party dependencies (MLflow, FastAPI, scikit-learn)
  — report those to their respective maintainers

## Reporting a Vulnerability

Please **do not** open a public issue for security-sensitive findings.

Report privately via GitHub's **Private Vulnerability Reporting** (Security
tab → Report a vulnerability), or open a standard issue if the finding is not
sensitive.

What to include:
- A clear description of the issue and its impact (e.g., "a model with a
  tampered manifest can be promoted")
- Steps to reproduce (ideally with the CLI or compose commands)
- Suggested remediation, if known

## Disclosure Policy

- **Acknowledgement:** you will be acknowledged for validated reports (unless you prefer to remain anonymous).
- **Response target:** an initial response within 5 business days.
- **Fix window:** validated critical/high findings are prioritized; fixes land through the CI pipeline (lint, type check, tests, compose-smoke).

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `main` | Yes — CI enforced (lint, mypy, pytest, compose-smoke, gate demo) |
| Latest tag (see [Releases](https://github.com/LiamCarPer/GatedOps/releases)) | Yes |
| Older tags | No — upgrade |

## Verification

Every fix is verified by the automated gates: lint (ruff), type check (mypy),
unit tests, the compose-smoke job (boots the stack and asserts a scored
request carries lineage), and the gate demo jobs (a good model is promoted,
a bad model is blocked).
