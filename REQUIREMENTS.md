# Requirements

## 1. Objective

Classify e-commerce customer reviews as **negative / neutral / positive**, store reviews and predictions in a SQL database, expose the model through a Flask API, and report insights in a Power BI dashboard.

## 2. Scope

**In scope:** EDA, text preprocessing, Naive Bayes baseline, ensemble comparison, hyperparameter tuning, SQL storage, REST API, Power BI dashboard, tests, reproducible pipeline, documentation.

**Out of scope:** deep learning/transformers, real-time streaming, user authentication, cloud deployment, multilingual text.

## 3. Functional requirements

| ID | Requirement | Module |
|---|---|---|
| FR-01 | Load raw review CSV with pandas; validate columns, types, and missing values | 3 |
| FR-02 | Clean data: drop rows with empty review text, deduplicate, normalise whitespace | 3 |
| FR-03 | Derive sentiment label from rating (see `DATA.md`) | 6 |
| FR-04 | Produce EDA figures: label distribution, text length, rating by department, top terms per class (Matplotlib/Seaborn), saved to `reports/figures/` | 4 |
| FR-05 | Text preprocessing: lowercase, strip URLs/HTML/punctuation/digits, tokenise, stop-word removal (keep negations), lemmatise | 9 |
| FR-06 | Vectorise with Bag-of-Words and TF-IDF (unigram + bigram), compare both | 9 |
| FR-07 | Train Multinomial Naive Bayes baseline | 10 |
| FR-08 | Train and compare Logistic Regression, Random Forest, Gradient Boosting | 5, 15 |
| FR-09 | Tune the best candidates with GridSearchCV and RandomizedSearchCV using stratified 5-fold CV | 16 |
| FR-10 | Evaluate with accuracy, per-class precision/recall/F1, macro-F1, confusion matrix | 7 |
| FR-11 | (Optional) K-Means on TF-IDF to surface complaint themes; report silhouette score | 8 |
| FR-12 | Package preprocessing + vectoriser + classifier as one scikit-learn `Pipeline` serialised with joblib | 17 |
| FR-13 | Store raw reviews, model runs, and predictions in SQL | 11, 12 |
| FR-14 | Batch-score all stored reviews and write predictions to SQL | 12 |
| FR-15 | Flask API: single prediction, batch prediction, health, model info, stats (see `API.md`) | 13 |
| FR-16 | Persist every API prediction to SQL | 12, 13 |
| FR-17 | SQL views feeding the dashboard (see `DATABASE.md`) | 11 |
| FR-18 | Power BI dashboard with four pages (see `DASHBOARD.md`) | 14 |
| FR-19 | Version each trained model with metadata (see `MLOPS.md`) | 18 |
| FR-20 | One-command automation via Makefile and a CI workflow | 18 |

## 4. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | Reproducibility: fixed seed `42`; same data + code + seed gives identical metrics |
| NFR-02 | Single-prediction latency p95 < 200 ms locally |
| NFR-03 | Batch endpoint accepts up to 100 texts per request |
| NFR-04 | Model artefact < 100 MB |
| NFR-05 | No data leakage: split before fitting any vectoriser or encoder |
| NFR-06 | Test coverage ≥ 70% on `src/` |
| NFR-07 | All config (paths, seed, DB URL, model version) in `config.yaml` or environment variables, none hard-coded |
| NFR-08 | Code formatted with `black` and linted with `ruff` |
| NFR-09 | Fresh clone to running API in ≤ 5 commands, documented in `README.md` |

## 5. Acceptance criteria

1. Macro-F1 on the held-out test set ≥ 0.65 (3-class). Baseline and final numbers both reported.
2. Tuned model beats the untuned Naive Bayes baseline on macro-F1.
3. `POST /predict` returns a valid label and probabilities summing to 1 (±1e-6).
4. Every prediction made via the API appears in the `predictions` table.
5. Dashboard refreshes from the exported SQL views with no manual editing.
6. `make test` passes; CI is green.
7. All eleven documents in the index are present and consistent with the code.

## 6. Deliverables

Source repository, trained model artefact, SQLite database file, `.pbix` dashboard, EDA figures, evaluation report (`reports/evaluation.md`), these requirement documents.
