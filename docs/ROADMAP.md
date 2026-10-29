# Roadmap

Maps each course module to a capstone deliverable. Capstone budget: 15 h in Week 8; earlier modules produce reusable components.

| Week | Module | Capstone deliverable |
|---|---|---|
| 1 | 1 Intro to ML/DS | Repo, uv env (`pyproject.toml`, `uv.lock`), `config.yaml` |
| 1 | 2 NumPy | Array utilities for metrics and sampling |
| 2 | 3 Pandas | `load.py`, `clean.py`, data quality report |
| 2 | 4 Matplotlib/Seaborn | EDA notebook and figures (FR-04) |
| 3 | 5 Scikit-learn, Supervised | Logistic Regression baseline |
| 3 | 6 Feature Engineering | Label derivation, text length features, stratified split |
| 4 | 7 Evaluation & Selection | Metrics module, CV harness, confusion matrices |
| 4 | 8 Clustering | Optional theme discovery on negative reviews |
| 5 | 9 Text & NLP | Preprocessing transformer, BoW and TF-IDF |
| 5 | 10 Naive Bayes | MultinomialNB baseline |
| 6 | 11 SQL | `schema.sql`, views |
| 6 | 12 Python + SQL | Repository layer, ingestion, batch scoring |
| 7 | 13 Flask | API endpoints and persistence |
| 7 | 14 Power BI | Dashboard pages 1–4 |
| 7 | 15 Ensembles | Random Forest and Gradient Boosting candidates |
| 7 | 16 Tuning | Grid and Randomized search |
| 8 | 17 Pipelines & Ethics | Single `Pipeline` artefact, `ETHICS.md`, reproducibility |
| 8 | 18 MLOps | Versioning, Makefile, CI |
| 8 | 19 Capstone (15 h) | Integration, tests, final report, demo |

## Capstone build order (Week 8, 15 h)

| Block | Hours | Task | Done when |
|---|---|---|---|
| 1 | 2 | Ingest data into SQL, run data quality report | `reviews` populated, counts match |
| 2 | 3 | Train baseline and candidates, tune, select | Metrics meet acceptance criterion 1–2 |
| 3 | 1.5 | Persist model, `model_runs`, batch scoring | `predictions` populated |
| 4 | 2.5 | Flask API with validation and persistence | API tests pass |
| 5 | 3 | Power BI dashboard | Four pages, totals reconcile |
| 6 | 1.5 | Tests, CI, Makefile | `make all` and CI green |
| 7 | 1.5 | Evaluation report, ethics notes, README, demo | Definition of done checked |
