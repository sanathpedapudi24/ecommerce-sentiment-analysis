# MLOps

Scope: introductory lifecycle, versioning, reproducibility, automation (Modules 17–18).

## Lifecycle

`Data → Train → Evaluate → Register → Serve → Monitor → Retrain`

## Reproducibility

| Item | Rule |
|---|---|
| Random seed | `42`, set in `config.yaml`, passed to splits, models, searches |
| Dependencies | Pinned in `requirements.txt` (exact versions); Python 3.11 |
| Data | SHA-256 hash of `data/raw/reviews.csv` stored in `metadata.json` |
| Code | Git commit hash stored in `metadata.json`; training refuses to run with a dirty tree unless `--allow-dirty` |
| Splits | Saved in `data/processed/split_ids.json` |
| Environment | `make setup` builds the venv; optional `Dockerfile` for the API |

## Model versioning

Semantic versions `vMAJOR.MINOR.PATCH`.

- MAJOR: label scheme or input format changes
- MINOR: new algorithm or feature set
- PATCH: retrain on same configuration

`models/<version>/metadata.json`:
```json
{
  "model_version": "v1.0.0",
  "algorithm": "LogisticRegression",
  "params": {},
  "data_hash": "sha256:...",
  "git_commit": "abc1234",
  "seed": 42,
  "sklearn_version": "x.y.z",
  "trained_at": "ISO-8601",
  "train_rows": 0,
  "metrics": {"val_macro_f1": 0.0, "test_macro_f1": 0.0}
}
```

The same record is inserted into `model_runs`. Promotion = setting `is_active=1` via `make promote VERSION=v1.0.0`; promotion is blocked if test macro-F1 is below the current active model's.

## Automation

`Makefile` targets: `setup`, `data`, `eda`, `train`, `evaluate`, `promote`, `batch-score`, `export`, `serve`, `test`, `lint`, `all`.

GitHub Actions `.github/workflows/ci.yml` on push and pull request:
1. Install dependencies
2. `ruff` and `black --check`
3. `pytest` with coverage gate (≥ 70%)
4. Smoke train on a 500-row sample and assert pipeline artefact loads and predicts

## Monitoring

- `vw_api_usage` tracks request volume and mean confidence
- Alert condition (documented, checked manually on dashboard): mean confidence drops > 10 points week over week, or negative share shifts > 15 points
- Retrain trigger: alert condition met, or ≥ 20% new labelled reviews since last training

## Out of scope

Feature stores, Kubernetes, automated retraining schedulers, managed model registries.
