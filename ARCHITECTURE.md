# Architecture

## Components

```
              ┌────────────┐   ┌───────────┐   ┌─────────────┐
raw CSV  ───► │ Ingestion  │──►│  SQLite   │◄──│  Flask API  │◄── clients
              └────────────┘   │ reviews   │   │  /predict   │
                    │          │ predictions│  └──────▲──────┘
                    ▼          │ model_runs│          │ loads
              ┌────────────┐   │ views     │   ┌──────┴──────┐
              │ Training   │──►└─────┬─────┘   │ models/<v>/ │
              │ pipeline   │─────────┼────────►│ model.joblib│
              └────────────┘         │         └─────────────┘
                                     ▼
                              ┌─────────────┐
                              │ CSV exports │──► Power BI
                              └─────────────┘
```

## Data flow

1. Ingestion cleans the CSV and writes `reviews`.
2. Training reads `reviews` (train/val split), fits and tunes, writes `models/<version>/` and `model_runs`.
3. Batch scoring reads all reviews, writes `predictions` with `source='batch'`.
4. The API loads the active model, scores requests, writes `predictions` with `source='api'`.
5. `make export` dumps `vw_*` views to CSV; Power BI reads them.

## Repository layout

```
.
├── README.md
├── config.yaml
├── Makefile
├── requirements.txt
├── data/
│   ├── raw/                 # untouched input, gitignored
│   ├── processed/
│   └── sentiment.db         # gitignored
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── config.py
│   ├── data/      (load.py, clean.py, split.py)
│   ├── features/  (text.py)
│   ├── models/    (train.py, tune.py, evaluate.py, predict.py)
│   ├── db/        (schema.sql, repository.py, migrations/)
│   ├── api/       (app.py, schemas.py)
│   └── cli.py
├── models/<version>/
├── reports/
│   ├── figures/
│   ├── exports/
│   ├── evaluation.md
│   └── sentiment_dashboard.pbix
├── tests/
│   ├── test_clean.py
│   ├── test_text.py
│   ├── test_pipeline.py
│   ├── test_db.py
│   └── test_api.py
├── docs/                    # the requirement files
└── .github/workflows/ci.yml
```

## Design decisions

| Decision | Reason |
|---|---|
| One `Pipeline` object for preprocessing + vectoriser + model | Prevents train/serve skew; single artefact to version |
| SQLite | Zero setup, enough for ~25k rows; ANSI schema allows later move to PostgreSQL |
| CSV exports for Power BI | Works without ODBC drivers; refresh is deterministic |
| Notebooks for exploration only | Production logic lives in `src/` and is tested |
